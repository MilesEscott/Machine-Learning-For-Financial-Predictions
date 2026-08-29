
import pandas

def totalReturn(returns):
    return ((returns + 1).prod() - 1).iloc[0]

def annualisedReturn(returns, periodsPerYear=252):
    compoundedGrowth = (returns + 1).prod()
    noOfPeriods = returns.shape[0]
    return  (compoundedGrowth**(periodsPerYear/noOfPeriods) - 1).iloc[0]

def annualisedVolatility(returns, periodsPerYear=252):
    return (returns.std() * (periodsPerYear**0.5)).iloc[0]

def rawSharpsRatio(returns):
    return annualisedReturn(returns) / annualisedVolatility(returns)

def maxiumumDrawDown(returns, strategyName):
    wealthIndex = (returns+1).cumprod()
    wealthIndex.plot(title= "Wealth Index for " + strategyName)
    previousPeaks = wealthIndex.cummax()
    drawDowns = (wealthIndex - previousPeaks)/ previousPeaks

    return (drawDowns.min()).iloc[0]