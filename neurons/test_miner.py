import datetime as dt
from common.data import Sport, League, MatchPrediction
from st.sport_prediction_model import make_match_prediction
import bittensor as bt
import asyncio

bt.logging.set_trace(True)
bt.logging.set_debug(True)


async def mls():
    matchDate = "2024-08-25"
    match_prediction = MatchPrediction(
        matchId="9101",
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.SOCCER,
        league=League.MLS,
        homeTeamName="Inter Miami",
        awayTeamName="Miami Fusion",
    )

    match_prediction = await make_match_prediction(match_prediction)

    bt.logging.info(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
        {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with wp {round(match_prediction.probability, 4)}"
    )

    return match_prediction


async def mlb():
    matchDate = "2024-08-25"
    match_prediction = MatchPrediction(
        matchId="1234",
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.BASEBALL,
        league=League.MLB,
        homeTeamName="Baltimore Orioles",
        awayTeamName="Texas Rangers",
    )

    match_prediction = await make_match_prediction(match_prediction)

    bt.logging.info(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
        {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with wp {round(match_prediction.probability, 4)}"
    )

async def epl():
    matchDate = "2024-12-20"
    match_prediction = MatchPrediction(
        matchId="5678",
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.SOCCER,
        league=League.EPL,
        homeTeamName="Aston Villa",
        awayTeamName="Manchester City",
    )

    match_prediction = await make_match_prediction(match_prediction)

    bt.logging.info(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
        {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with wp {round(match_prediction.probability, 4)}"
    )

async def nfl():
    matchDate = "2024-12-20"
    match_prediction = MatchPrediction(
        matchId="1121",
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.AMERICAN_FOOTBALL,
        league=League.NFL,
        homeTeamName="Los Angeles Chargers",
        awayTeamName="Denver Broncos",
    )

    match_prediction = await make_match_prediction(match_prediction)

    bt.logging.info(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
        {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with wp {round(match_prediction.probability, 4)}"
    )

    return match_prediction

async def nba():
    matchDate = "2025-02-20"
    match_predictions = [
        MatchPrediction(
            matchId="3141",
            matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
            sport=Sport.BASKETBALL,
            league=League.NBA,
            homeTeamName="San Antonio Spurs",
            awayTeamName="Phoenix Suns",
        ),
        MatchPrediction(
            matchId="e6a39ecdde1417e008fed58878a66a55",
            matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
            sport=Sport.BASKETBALL,
            league=League.NBA,
            homeTeamName="Portland Trail Blazers",
            awayTeamName="Los Angeles Lakers",
        ),
    ]
    for match_prediction in match_predictions:
        match_prediction = await make_match_prediction(match_prediction)
        bt.logging.info(
            f"-------------------------------------------------------------------------------------- \n \
            Match: {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}\n \
            Prediction: {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with probability {match_prediction.probability}"
        )

    return match_predictions

if __name__ == "__main__":
    asyncio.run(mlb())
    #asyncio.run(epl())
    #asyncio.run(mls())
    #asyncio.run(nfl())
    #asyncio.run(nba())