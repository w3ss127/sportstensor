import json
import os
import sys
import bittensor as bt
import aiohttp
import asyncio
import random
import math
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from common.data import MatchPrediction, League, ProbabilityChoice, get_league_from_string
from common.constants import LEAGUES_ALLOWING_DRAWS
from st.sport_prediction_model import SportPredictionModel

MINER_ENV_PATH = 'neurons/miner.env'
load_dotenv(dotenv_path=MINER_ENV_PATH)
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
if not ODDS_API_KEY:
    raise ValueError(f"ODDS_API_KEY not found in {MINER_ENV_PATH}")

API_URL = "https://api.the-odds-api.com/v4/sports/"

# Team name mappings for normalization
mismatch_teams_mapping = {
    "Orlando City SC": "Orlando City",
    "Inter Miami CF": "Inter Miami",
    "Atlanta United FC": "Atlanta United",
    "Montreal Impact": "CF Montréal",
    "D.C. United": "DC United",
    "Tottenham Hotspur": "Tottenham",
    "Columbus Crew SC": "Columbus Crew",
    "Minnesota United FC": "Minnesota United",
    "Vancouver Whitecaps FC": "Vancouver Whitecaps",
    "Leicester City": "Leicester",
    "West Ham United": "West Ham",
    "Brighton and Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "LA Galaxy": "L.A. Galaxy",
    # "Oakland Athletics": "Athletics",
}

SPORTS_TYPES = [
    {
        'sport_key': 'baseball_mlb',
        'region': 'us,eu',
    },
    {
        'sport_key': 'americanfootball_nfl',
        'region': 'us,eu'
    },
    {
        'sport_key': 'soccer_usa_mls',
        'region': 'us,eu'
    },
    {
        'sport_key': 'soccer_epl',
        'region': 'uk,eu'
    },
    {
        'sport_key': 'basketball_nba',
        'region': 'us,eu'
    },
]

league_mapping = {
    'NBA': 'NBA',
    'NFL': 'NFL',
    'MLS': 'MLS',
    'EPL': 'EPL',
    'MLB': 'MLB',
}

class SportstensorBaseModel(SportPredictionModel):
    def __init__(self, prediction: MatchPrediction):
        super().__init__(prediction)
        self.boost_min_percent = 0.1
        self.boost_max_percent = 0.2
        self.probability_cap = 0.95
        self.max_retries = 3
        self.retry_delay = 0.5
        self.timeout = 3

    async def fetch_odds(self, sport_key: str, region: str) -> Optional[dict]:
        """Fetch odds from the new API."""
        url = f"{API_URL}{sport_key}/odds/"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": region,
            "bookmakers": "pinnacle",
            "markets": "h2h"
        }
        async with aiohttp.ClientSession() as session:
            try:
                bt.logging.debug("Fetching odds from API...")
                async with session.get(url, params=params, timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        print(f"\n=== API Error ===\nStatus: {response.status}")
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"\n=== API Exception ===\n{str(e)}")
                return None

    def map_team_name(self, team_name: str) -> str:
        """Map team names using mismatch mapping."""
        return mismatch_teams_mapping.get(team_name, team_name)

    def odds_to_probabilities(self, home_odds: float, away_odds: float, draw_odds: Optional[float] = None) -> Dict[str, float]:
        """Convert odds to probabilities."""
        try:
            if home_odds is None or away_odds is None:
                print("Missing required odds values")
                return None

            # Convert odds to probabilities
            home_prob = 1 / home_odds if home_odds > 0 else 0
            away_prob = 1 / away_odds if away_odds > 0 else 0
            draw_prob = 1 / draw_odds if draw_odds and draw_odds > 0 else 0

            # Normalize probabilities
            total = home_prob + away_prob + draw_prob
            if total <= 0:
                print("Invalid odds values resulted in zero total probability")
                return None

            probabilities = {
                "home": home_prob / total,
                "away": away_prob / total,
            }

            if draw_odds:
                probabilities["draw"] = draw_prob / total
            
            return probabilities
        
        except Exception as e:
            bt.logging.error(f"Error converting odds to probabilities: {str(e)}")
            return None

    def get_stats(self, hometeam, awayteam, json_path="storage/mlb.json", recent_n=10):
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Stats file not found: {json_path}")

        with open(json_path, "r") as f:
            matches = json.load(f)

        # Filter only head-to-head games
        h2h_matches = [
            m for m in matches
            if {m["home_team"], m["away_team"]} == {hometeam, awayteam}
        ]

        total_matches = len(h2h_matches)
        team1_wins = team2_wins = 0
        team1_total_score = team2_total_score = 0
        team1_score_as_home = team2_score_as_away = 0
        team1_wins_as_home = team1_wins_as_away = 0
        team2_wins_as_home = team2_wins_as_away = 0
        margins_team1 = []
        margins_team2 = []
        win_streak_team1 = win_streak_team2 = 0

        # For recent stats
        recent_matches = h2h_matches[-recent_n:] if total_matches >= recent_n else h2h_matches
        
        recent_team1_wins = recent_team2_wins = 0
        recent_team1_score = recent_team2_score = 0

        # For win streak
        last_winner = None

        for match in h2h_matches:
            home = match["home_team"]
            away = match["away_team"]
            home_score = match["home_score"]
            away_score = match["away_score"]

            # Assign team1/team2
            if home == hometeam:
                team1_score = home_score
                team2_score = away_score
                team1_score_as_home += home_score
                team2_score_as_away += away_score
            else:
                team1_score = away_score
                team2_score = home_score
                team2_score_as_away += away_score
                team1_score_as_home += home_score  # This is rare, but for completeness

            team1_total_score += team1_score
            team2_total_score += team2_score

            # Win logic
            if team1_score > team2_score:
                team1_wins += 1
                margins_team1.append(team1_score - team2_score)
                if home == hometeam:
                    team1_wins_as_home += 1
                else:
                    team1_wins_as_away += 1
                if last_winner == hometeam or last_winner is None:
                    win_streak_team1 += 1
                    win_streak_team2 = 0
                else:
                    win_streak_team1 = 1
                    win_streak_team2 = 0
                last_winner = hometeam
            else:
                team2_wins += 1
                margins_team2.append(team2_score - team1_score)
                if home == awayteam:
                    team2_wins_as_home += 1
                else:
                    team2_wins_as_away += 1
                if last_winner == awayteam or last_winner is None:
                    win_streak_team2 += 1
                    win_streak_team1 = 0
                else:
                    win_streak_team2 = 1
                    win_streak_team1 = 0
                last_winner = awayteam

        # Recent stats
        for match in recent_matches:
            home = match["home_team"]
            away = match["away_team"]
            home_score = match["home_score"]
            away_score = match["away_score"]
            if home == hometeam:
                team1_score = home_score
                team2_score = away_score
            else:
                team1_score = away_score
                team2_score = home_score
            recent_team1_score += team1_score
            recent_team2_score += team2_score
            if team1_score > team2_score:
                recent_team1_wins += 1
            else:
                recent_team2_wins += 1

        average_margin_team1 = sum(margins_team1) / len(margins_team1) if margins_team1 else 0
        average_margin_team2 = sum(margins_team2) / len(margins_team2) if margins_team2 else 0

        return {
            "total_matches": total_matches,
            f"{hometeam}_wins": team1_wins,
            f"{awayteam}_wins": team2_wins,
            f"{hometeam}_total_score": team1_total_score,
            f"{awayteam}_total_score": team2_total_score,
            f"{hometeam}_score_as_home": team1_score_as_home,
            f"{awayteam}_score_as_away": team2_score_as_away,
            f"{hometeam}_wins_as_home": team1_wins_as_home,
            f"{hometeam}_wins_as_away": team1_wins_as_away,
            f"{awayteam}_wins_as_home": team2_wins_as_home,
            f"{awayteam}_wins_as_away": team2_wins_as_away,
            f"recent_{hometeam}_wins": recent_team1_wins,
            f"recent_{awayteam}_wins": recent_team2_wins,
            f"recent_{hometeam}_score": recent_team1_score,
            f"recent_{awayteam}_score": recent_team2_score,
            f"{hometeam}_avg_margin": average_margin_team1,
            f"{awayteam}_avg_margin": average_margin_team2,
            f"{hometeam}_current_win_streak": win_streak_team1,
            f"{awayteam}_current_win_streak": win_streak_team2,
        }

    def betting_algorithm(self, stats, hometeam, awayteam):
        total_matches = stats["total_matches"]
        if total_matches == 0:
            return {"winner": None, "probabilities": {hometeam: 0.5, awayteam: 0.5}}

        prob_range_from = 0
        prob_range_to = 0
        recent_n = 10

        use_recent = total_matches >= recent_n
        home_wins = stats[f"recent_{hometeam}_wins"] if use_recent else stats[f"{hometeam}_wins"]
        away_wins = stats[f"recent_{awayteam}_wins"] if use_recent else stats[f"{awayteam}_wins"]

        win_diff = abs(home_wins - away_wins)
        
        total = home_wins + away_wins
        home_weight = home_wins / total
        away_weight = away_wins / total
        
        winner = random.choices([hometeam, awayteam], weights=[home_weight, away_weight], k=1)[0]

        if(win_diff <= 2):
            prob_range_from = 0.51
            prob_range_to = 0.65
        elif(win_diff <= 6):
            prob_range_from = 0.65
            prob_range_to = 0.8
        else:
            prob_range_from = 0.8
            prob_range_to = 0.95

        winner_prob = round(random.uniform(prob_range_from, prob_range_to), 2)
        loser_prob = 1 - winner_prob
        
        return {
            "winner": winner,
            "probabilities": {
                hometeam: round(winner_prob, 2) if winner == hometeam else round(loser_prob, 2),
                awayteam: round(winner_prob, 2) if winner == awayteam else round(loser_prob, 2)
            }
        }
    
    async def make_prediction(self):
        """Synchronous wrapper for async prediction logic."""
        bt.logging.info(f"Predicting {self.prediction.league} game...")
        
        try:
            # Convert the league to enum if it's not already one
            if not isinstance(self.prediction.league, League):
                try:
                    league_enum = get_league_from_string(str(self.prediction.league))
                    if league_enum is None:
                        bt.logging.error(f"Unknown league: {self.prediction.league}. Returning.")
                    self.prediction.league = league_enum
                except ValueError as e:
                    bt.logging.error(f"Failed to convert league: {self.prediction.league}. Error: {e}")
            else:
                league_enum = self.prediction.league

            if not isinstance(self.prediction.league, League):
                bt.logging.error(f"Invalid league type: {type(self.prediction.league)}. Expected League enum.")
                return self.prediction
            
            # Dynamically determine sport_key
            league_to_sport_key = {
                "NBA": "basketball_nba",
                "NFL": "americanfootball_nfl",
                "MLS": "soccer_usa_mls",
                "EPL": "soccer_epl",
                "MLB": "baseball_mlb",
                "English Premier League": "soccer_epl",
                "American Major League Soccer": "soccer_usa_mls",
            }

            league_key = self.prediction.league.name
            sport_key = league_to_sport_key.get(league_key)

            if not sport_key:
                bt.logging.error(f"Unknown league: {league_key}. Unable to determine sport_key.")
                return self.prediction

            # Determine the region (optional customization for regions)
            region = "us,eu" if sport_key in ["baseball_mlb", "americanfootball_nfl", "basketball_nba"] else "uk,eu"
            
            # odds_data = await self.fetch_odds(sport_key, region)
            odds_data = []

            # if not odds_data:
            #     bt.logging.error("No odds data fetched.")
            #     return self.prediction

            home_team = self.map_team_name(self.prediction.homeTeamName)
            away_team = self.map_team_name(self.prediction.awayTeamName)

            # Find the match
            for odds in odds_data:
                if  odds["home_team"] == home_team and odds["away_team"] == away_team:
                    bookmaker = next((b for b in odds["bookmakers"] if b["key"] == "pinnacle"), None)
                    if not bookmaker:
                        bt.logging.error("No Pinnacle odds found")
                        continue

                    market = next((m for m in bookmaker["markets"] if m["key"] == "h2h"), None)
                    if not market:
                        bt.logging.error("No h2h market found")
                        continue

                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    home_odds = outcomes.get(home_team)
                    away_odds = outcomes.get(away_team)
                    draw_odds = outcomes.get("Draw") if self.prediction.league in LEAGUES_ALLOWING_DRAWS else None

                    bt.logging.debug(f"Raw odds: {outcomes}")

                    if home_odds is None or away_odds is None:
                        bt.logging.error("Missing odds for one or both teams")
                        continue

                    probabilities = self.odds_to_probabilities(home_odds, away_odds, draw_odds)
                    bt.logging.debug(f"Calculated probabilities: {probabilities}")

                    if probabilities:
                        # Find the highest probability outcome
                        max_prob = max(probabilities["home"], probabilities["away"], probabilities.get("draw", 0))

                        if max_prob == probabilities["home"]:
                            self.prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
                        elif max_prob == probabilities["away"]:
                            self.prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM
                        else:
                            self.prediction.probabilityChoice = ProbabilityChoice.DRAW

                        self.prediction.probability = max_prob + random.uniform(self.boost_min_percent, self.boost_max_percent)
                        bt.logging.info(f"Prediction made: {self.prediction.probabilityChoice} with probability {self.prediction.probability}")
                        return
            
            stats = self.get_stats(home_team, away_team)
            bt.logging.debug(f"Stats: {stats}")
            result = self.betting_algorithm(stats, home_team, away_team)
            bt.logging.debug(f"Betting algorithm result: {result}")

            if result["winner"] == home_team:
                self.prediction.probabilityChoice = ProbabilityChoice.HOMETEAM
                self.prediction.probability = result["probabilities"][home_team]
            elif result["winner"] == away_team:
                self.prediction.probabilityChoice = ProbabilityChoice.AWAYTEAM
                self.prediction.probability = result["probabilities"][away_team]
            else:
                self.prediction.probabilityChoice = random.choice([ProbabilityChoice.HOMETEAM, ProbabilityChoice.AWAYTEAM])
                self.prediction.probability = round(random.uniform(0.51, 0.55), 2)

            # json_result = self.prediction.model_dump_json(indent=2)
            # with open("storage/miner_response.json", "w") as f:
            #     json.dump(json_result, f, indent=2)
            # bt.logging.info(f"Sent Prediction Successfully: {json_result}")
            
            bt.logging.success(
                f"Match Prediction for {self.prediction.awayTeamName} at {self.prediction.homeTeamName} on {self.prediction.matchDate}: {self.prediction.probabilityChoice} ({self.prediction.get_predicted_team()}) with wp {round(self.prediction.probability, 4)}"
            )
            return
            
        except Exception as e:
            bt.logging.error(f"Failed to make prediction: {str(e)}")