// Copyright Paul A. 2007, 2010
// Copyright John Maddock 2006

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Simple example of computing probabilities and quantiles for
// a Bernoulli random variable representing the flipping of a coin.

// http://mathworld.wolfram.com/CoinTossing.html
// http://en.wikipedia.org/wiki/Bernoulli_trial
// Weisstein, Eric W. "Dice." From MathWorld--A Wolfram Web Resource.
// http://mathworld.wolfram.com/Dice.html
// http://en.wikipedia.org/wiki/Bernoulli_distribution
// http://mathworld.wolfram.com/BernoulliDistribution.html
//
// An idealized coin consists of a circular disk of zero thickness which,
// when thrown in the air and allowed to fall, will rest with either side face up
// ("heads" H or "tails" T) with equal probability. A coin is therefore a two-sided die.
// Despite slight differences between the sides and nonzero thickness of actual coins,
// the distribution of their tosses makes a good approximation to a p==1/2 Bernoulli distribution.

//[binomial_coinflip_example1

/*`An example of a [@http://en.wikipedia.org/wiki/Bernoulli_process Bernoulli process]
is coin flipping.
A variable in such a sequence may be called a Bernoulli variable.

This example shows using the Binomial distribution to predict the probability
of heads and tails when throwing a coin.

The number of correct answers (say heads),
X, is distributed as a binomial random variable
with binomial distribution parameters number of trials (flips) n = 10 and probability (success_fraction) of getting a head p = 0.5 (a 'fair' coin).

(Our coin is assumed fair, but we could easily change the success_fraction parameter p
from 0.5 to some other value to simulate an unfair coin,
say 0.6 for one with chewing gum on the tail,
so it is more likely to fall tails down and heads up).

First we need some includes and using statements to be able to use the binomial distribution, some std input and output, and get started:
*/

#include <boost/math/distributions/binomial.hpp>
  using boost::math::binomial;

#include <iostream>
  using std::cout;  using std::endl;  using std::left;
#include <iomanip>
  using std::setw;

int main()
{
  cout << "Using Binomial distribution to predict how many heads and tails." << endl;
  try
  {
/*`
See note [link coinflip_eg_catch with the catch block]
about why a try and catch block is always a good idea.

First, construct a binomial distribution with parameters success_fraction
1/2, and how many flips.
*/
    const double success_fraction = 0.5; // = 50% = 1/2 for a 'fair' coin.
    int flips = 10;
    binomial flip(flips, success_fraction);

    cout.precision(4);
/*`
 Then some examples of using Binomial moments (and echoing the parameters).
*/
    cout << "From " << flips << " one can expect to get on average "
      << mean(flip) << " heads (or tails)." << endl;
    cout << "Mode is " << mode(flip) << endl;
    cout << "Standard deviation is " << standard_deviation(flip) << endl;
    cout << "So about 2/3 will lie within 1 standard deviation and get between "
      <<  ceil(mean(flip) - standard_deviation(flip))  << " and "
      << floor(mean(flip) + standard_deviation(flip)) << " correct." << endl;
    cout << "Skewness is " << skewness(flip) << endl;
    // Skewness of binomial distributions is only zero (symmetrical)
    // if success_fraction is exactly one half,
    // for example, when flipping 'fair' coins.
    cout << "Skewness if success_fraction is " << flip.success_fraction()
      << " is " << skewness(flip) << endl << endl; // Expect zero for a 'fair' coin.
/*`
Now we show a variety of predictions on the probability of heads:
*/
    cout << "For " << flip.trials() << " coin flips: " << endl;
    cout << "Probability of getting no heads is " << pdf(flip, 0) << endl;
    cout << "Probability of getting at least one head is " << 1. - pdf(flip, 0) << endl;
/*`
When we want to calculate the probability for a range or values we can sum the PDF's:
*/
    cout << "Probability of getting 0 or 1 heads is "
      << pdf(flip, 0) + pdf(flip, 1) << endl; // sum of exactly == probabilities
/*`
Or we can use the cdf.
*/
    cout << "Probability of getting 0 or 1 (<= 1) heads is " << cdf(flip, 1) << endl;
    cout << "Probability of getting 9 or 10 heads is " << pdf(flip, 9) + pdf(flip, 10) << endl;
/*`
Note that using
*/
    cout << "Probability of getting 9 or 10 heads is " << 1. - cdf(flip, 8) << endl;
/*`
is less accurate than using the complement
*/
    cout << "Probability of getting 9 or 10 heads is " << cdf(complement(flip, 8)) << endl;
/*`
Since the subtraction may involve
[@http://docs.sun.com/source/806-3568/ncg_goldberg.html cancellation error],
where as `cdf(complement(flip, 8))`
does not use such a subtraction internally, and so does not exhibit the problem.

To get the probability for a range of heads, we can either add the pdfs for each number of heads
*/
    cout << "Probability of between 4 and 6 heads (4 or 5 or 6) is "
      //  P(X == 4) + P(X == 5) + P(X == 6)
      << pdf(flip, 4) + pdf(flip, 5) + pdf(flip, 6) << endl;
/*`
But this is probably less efficient than using the cdf
*/
    cout << "Probability of between 4 and 6 heads (4 or 5 or 6) is "
      // P(X <= 6) - P(X <= 3) == P(X < 4)
      << cdf(flip, 6) - cdf(flip, 3) << endl;
/*`
Certainly for a bigger range like, 3 to 7
*/
    cout << "Probability of between 3 and 7 heads (3, 4, 5, 6 or 7) is "
      // P(X <= 7) - P(X <= 2) == P(X < 3)
      << cdf(flip, 7) - cdf(flip, 2) << endl;
    cout << endl;

/*`
Finally, print two tables of probability for the /exactly/ and /at least/ a number of heads.
*/
    // Print a table of probability for the exactly a number of heads.
    cout << "Probability of getting exactly (==) heads" << endl;
    for (int successes = 0; successes <= flips; successes++)
    { // Say success means getting a head (or equally success means getting a tail).
      double probability = pdf(flip, successes);
      cout << left << setw(2) << successes << "     " << setw(10)
        << probability << " or 1 in " << 1. / probability
        << ", or " << probability * 100. << "%" << endl;
    } // for i
    cout << endl;

    // Tabulate the probability of getting between zero heads and 0 up to 10 heads.
    cout << "Probability of getting up to (<=) heads" << endl;
    for (int successes = 0; successes <= flips; successes++)
    { // Say success means getting a head
      // (equally success could mean getting a tail).
      double probability = cdf(flip, successes); // P(X <= heads)
      cout << setw(2) << successes << "        " << setw(10) << left
        << probability << " or 1 in " << 1. / probability << ", or "
        << probability * 100. << "%"<< endl;
    } // for i
/*`
The last (0 to 10 heads) must, of course, be 100% probability.
*/
    double probability = 0.3;
    double q = quantile(flip, probability);
    std::cout << "Quantile (flip, " << probability << ") = " << q << std::endl; // Quantile (flip, 0.3) = 3
    probability = 0.6;
    q = quantile(flip, probability);
    std::cout << "Quantile (flip, " << probability << ") = " << q << std::endl; // Quantile (flip, 0.6) = 5
  }
  catch(const std::exception& e)
  {
    //
    /*`
    [#coinflip_eg_catch]
    It is always essential to include try & catch blocks because
    default policies are to throw exceptions on arguments that
    are out of domain or cause errors like numeric-overflow.

    Lacking try & catch blocks, the program will abort, whereas the
    message below from the thrown exception will give some helpful
    clues as to the cause of the problem.
    */
    std::cout <<
      "\n""Message from thrown exception was:\n   " << e.what() << std::endl;
  }
//] [binomial_coinflip_example1]
  return 0;
} // int main()

// Output:

//[binomial_coinflip_example_output
/*`

[pre
Using Binomial distribution to predict how many heads and tails.
From 10 one can expect to get on average 5 heads (or tails).
Mode is 5
Standard deviation is 1.581
So about 2/3 will lie within 1 standard deviation and get between 4 and 6 correct.
Skewness is 0
Skewness if success_fraction is 0.5 is 0

For 10 coin flips:
Probability of getting no heads is 0.0009766
Probability of getting at least one head is 0.999
Probability of getting 0 or 1 heads is 0.01074
Probability of getting 0 or 1 (<= 1) heads is 0.01074
Probability of getting 9 or 10 heads is 0.01074
Probability of getting 9 or 10 heads is 0.01074
Probability of getting 9 or 10 heads is 0.01074
Probability of between 4 and 6 heads (4 or 5 or 6) is 0.6562
Probability of between 4 and 6 heads (4 or 5 or 6) is 0.6563
Probability of between 3 and 7 heads (3, 4, 5, 6 or 7) is 0.8906

Probability of getting exactly (==) heads
0      0.0009766  or 1 in 1024, or 0.09766%
1      0.009766   or 1 in 102.4, or 0.9766%
2      0.04395    or 1 in 22.76, or 4.395%
3      0.1172     or 1 in 8.533, or 11.72%
4      0.2051     or 1 in 4.876, or 20.51%
5      0.2461     or 1 in 4.063, or 24.61%
6      0.2051     or 1 in 4.876, or 20.51%
7      0.1172     or 1 in 8.533, or 11.72%
8      0.04395    or 1 in 22.76, or 4.395%
9      0.009766   or 1 in 102.4, or 0.9766%
10     0.0009766  or 1 in 1024, or 0.09766%

Probability of getting up to (<=) heads
0         0.0009766  or 1 in 1024, or 0.09766%
1         0.01074    or 1 in 93.09, or 1.074%
2         0.05469    or 1 in 18.29, or 5.469%
3         0.1719     or 1 in 5.818, or 17.19%
4         0.377      or 1 in 2.653, or 37.7%
5         0.623      or 1 in 1.605, or 62.3%
6         0.8281     or 1 in 1.208, or 82.81%
7         0.9453     or 1 in 1.058, or 94.53%
8         0.9893     or 1 in 1.011, or 98.93%
9         0.999      or 1 in 1.001, or 99.9%
10        1          or 1 in 1, or 100%
]
*/
//][/binomial_coinflip_example_output]
