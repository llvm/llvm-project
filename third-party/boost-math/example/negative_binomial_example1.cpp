// negative_binomial_example1.cpp

// Copyright Paul A. Bristow 2007, 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Example 1 of using negative_binomial distribution.

//[negative_binomial_eg1_1

/*`
Based on [@http://en.wikipedia.org/wiki/Negative_binomial_distribution
a problem by Dr. Diane Evans,
Professor of Mathematics at Rose-Hulman Institute of Technology].

Pat is required to sell candy bars to raise money for the 6th grade field trip.
There are thirty houses in the neighborhood,
and Pat is not supposed to return home until five candy bars have been sold.
So the child goes door to door, selling candy bars.
At each house, there is a 0.4 probability (40%) of selling one candy bar
and a 0.6 probability (60%) of selling nothing.

What is the probability mass (density) function (pdf) for selling the last (fifth)
candy bar at the nth house?

The Negative Binomial(r, p) distribution describes the probability of k failures
and r successes in k+r Bernoulli(p) trials with success on the last trial.
(A [@http://en.wikipedia.org/wiki/Bernoulli_distribution Bernoulli trial]
is one with only two possible outcomes, success of failure,
and p is the probability of success).
See also [@ http://en.wikipedia.org/wiki/Bernoulli_distribution Bernoulli distribution]
and [@http://www.math.uah.edu/stat/bernoulli/Introduction.xhtml Bernoulli applications].

In this example, we will deliberately produce a variety of calculations
and outputs to demonstrate the ways that the negative binomial distribution
can be implemented with this library: it is also deliberately over-commented.

First we need to #define macros to control the error and discrete handling policies.
For this simple example, we want to avoid throwing
an exception (the default policy) and just return infinity.
We want to treat the distribution as if it was continuous,
so we choose a discrete_quantile policy of real,
rather than the default policy integer_round_outwards.
*/
#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#define BOOST_MATH_DISCRETE_QUANTILE_POLICY real
/*`
After that we need some includes to provide easy access to the negative binomial distribution,
[caution It is vital to #include distributions etc *after* the above #defines]
and we need some std library iostream, of course.
*/
#include <boost/math/distributions/negative_binomial.hpp>
  // for negative_binomial_distribution
  using boost::math::negative_binomial; // typedef provides default type is double.
  using  ::boost::math::pdf; // Probability mass function.
  using  ::boost::math::cdf; // Cumulative density function.
  using  ::boost::math::quantile;

#include <iostream>
  using std::cout; using std::endl;
  using std::noshowpoint; using std::fixed; using std::right; using std::left;
#include <iomanip>
  using std::setprecision; using std::setw;

#include <limits>
  using std::numeric_limits;
//] [negative_binomial_eg1_1]

int main()
{
  cout <<"Selling candy bars - using the negative binomial distribution."
    << "\nby Dr. Diane Evans,"
    "\nProfessor of Mathematics at Rose-Hulman Institute of Technology,"
    << "\nsee http://en.wikipedia.org/wiki/Negative_binomial_distribution\n"
    << endl;
  cout << endl;
  cout.precision(5);
  // None of the values calculated have a useful accuracy as great this, but
  // INF shows wrongly with < 5 !
  // https://connect.microsoft.com/VisualStudio/feedback/ViewFeedback.aspx?FeedbackID=240227
//[negative_binomial_eg1_2
/*`
It is always sensible to use try and catch blocks because defaults policies are to
throw an exception if anything goes wrong.

A simple catch block (see below) will ensure that you get a
helpful error message instead of an abrupt program abort.
*/
  try
  {
/*`
Selling five candy bars means getting five successes, so successes r = 5.
The total number of trials (n, in this case, houses visited) this takes is therefore
  = successes + failures or k + r = k + 5.
*/
    double sales_quota = 5; // Pat's sales quota - successes (r).
/*`
At each house, there is a 0.4 probability (40%) of selling one candy bar
and a 0.6 probability (60%) of selling nothing.
*/
    double success_fraction = 0.4; // success_fraction (p) - so failure_fraction is 0.6.
/*`
The Negative Binomial(r, p) distribution describes the probability of k failures
and r successes in k+r Bernoulli(p) trials with success on the last trial.
(A [@http://en.wikipedia.org/wiki/Bernoulli_distribution Bernoulli trial]
is one with only two possible outcomes, success of failure,
and p is the probability of success).

We therefore start by constructing a negative binomial distribution
with parameters sales_quota (required successes) and probability of success.
*/
    negative_binomial nb(sales_quota, success_fraction); // type double by default.
/*`
To confirm, display the success_fraction & successes parameters of the distribution.
*/
    cout << "Pat has a sales per house success rate of " << success_fraction
      << ".\nTherefore he would, on average, sell " << nb.success_fraction() * 100
      << " bars after trying 100 houses." << endl;

    int all_houses = 30; // The number of houses on the estate.

    cout << "With a success rate of " << nb.success_fraction()
      << ", he might expect, on average,\n"
        "to need to visit about " << success_fraction * all_houses
        << " houses in order to sell all " << nb.successes() << " bars. " << endl;
/*`
[pre
Pat has a sales per house success rate of 0.4.
Therefore he would, on average, sell 40 bars after trying 100 houses.
With a success rate of 0.4, he might expect, on average,
to need to visit about 12 houses in order to sell all 5 bars.
]

The random variable of interest is the number of houses
that must be visited to sell five candy bars,
so we substitute k = n - 5 into a negative_binomial(5, 0.4)
and obtain the __pdf of the distribution of houses visited.
Obviously, the best possible case is that Pat makes sales on all the first five houses.

We calculate this using the pdf function:
*/
    cout << "Probability that Pat finishes on the " << sales_quota << "th house is "
      << pdf(nb, 5 - sales_quota) << endl; // == pdf(nb, 0)
/*`
Of course, he could not finish on fewer than 5 houses because he must sell 5 candy bars.
So the 5th house is the first that he could possibly finish on.

To finish on or before the 8th house, Pat must finish at the 5th, 6th, 7th or 8th house.
The probability that he will finish on *exactly* ( == ) on any house
is the Probability Density Function (pdf).
*/
    cout << "Probability that Pat finishes on the 6th house is "
      << pdf(nb, 6 - sales_quota) << endl;
    cout << "Probability that Pat finishes on the 7th house is "
      << pdf(nb, 7 - sales_quota) << endl;
    cout << "Probability that Pat finishes on the 8th house is "
      << pdf(nb, 8 - sales_quota) << endl;
/*`
[pre
Probability that Pat finishes on the 6th house is 0.03072
Probability that Pat finishes on the 7th house is 0.055296
Probability that Pat finishes on the 8th house is 0.077414
]

The sum of the probabilities for these houses is the Cumulative Distribution Function (cdf).
We can calculate it by adding the individual probabilities.
*/
    cout << "Probability that Pat finishes on or before the 8th house is sum "
      "\n" << "pdf(sales_quota) + pdf(6) + pdf(7) + pdf(8) = "
      // Sum each of the mass/density probabilities for houses sales_quota = 5, 6, 7, & 8.
      << pdf(nb, 5 - sales_quota) // 0 failures.
        + pdf(nb, 6 - sales_quota) // 1 failure.
        + pdf(nb, 7 - sales_quota) // 2 failures.
        + pdf(nb, 8 - sales_quota) // 3 failures.
      << endl;
/*`[pre
pdf(sales_quota) + pdf(6) + pdf(7) + pdf(8) = 0.17367
]

Or, usually better, by using the negative binomial *cumulative* distribution function.
*/
    cout << "\nProbability of selling his quota of " << sales_quota
      << " bars\non or before the " << 8 << "th house is "
      << cdf(nb, 8 - sales_quota) << endl;
/*`[pre
Probability of selling his quota of 5 bars on or before the 8th house is 0.17367
]*/
    cout << "\nProbability that Pat finishes exactly on the 10th house is "
      << pdf(nb, 10 - sales_quota) << endl;
    cout << "\nProbability of selling his quota of " << sales_quota
      << " bars\non or before the " << 10 << "th house is "
      << cdf(nb, 10 - sales_quota) << endl;
/*`
[pre
Probability that Pat finishes exactly on the 10th house is 0.10033
Probability of selling his quota of 5 bars on or before the 10th house is 0.3669
]*/
    cout << "Probability that Pat finishes exactly on the 11th house is "
      << pdf(nb, 11 - sales_quota) << endl;
    cout << "\nProbability of selling his quota of " << sales_quota
      << " bars\non or before the " << 11 << "th house is "
      << cdf(nb, 11 - sales_quota) << endl;
/*`[pre
Probability that Pat finishes on the 11th house is 0.10033
Probability of selling his quota of 5 candy bars
on or before the 11th house is 0.46723
]*/
    cout << "Probability that Pat finishes exactly on the 12th house is "
      << pdf(nb, 12 - sales_quota) << endl;

    cout << "\nProbability of selling his quota of " << sales_quota
      << " bars\non or before the " << 12 << "th house is "
      << cdf(nb, 12 - sales_quota) << endl;
/*`[pre
Probability that Pat finishes on the 12th house is 0.094596
Probability of selling his quota of 5 candy bars
on or before the 12th house is 0.56182
]
Finally consider the risk of Pat not selling his quota of 5 bars
even after visiting all the houses.
Calculate the probability that he /will/ sell on
or before the last house:
Calculate the probability that he would sell all his quota on the very last house.
*/
    cout << "Probability that Pat finishes on the " << all_houses
      << " house is " << pdf(nb, all_houses - sales_quota) << endl;
/*`
Probability of selling his quota of 5 bars on the 30th house is
[pre
Probability that Pat finishes on the 30 house is 0.00069145
]
when he'd be very unlucky indeed!

What is the probability that Pat exhausts all 30 houses in the neighborhood,
and *still* doesn't sell the required 5 candy bars?
*/
    cout << "\nProbability of selling his quota of " << sales_quota
      << " bars\non or before the " << all_houses << "th house is "
      << cdf(nb, all_houses - sales_quota) << endl;
/*`
[pre
Probability of selling his quota of 5 bars
on or before the 30th house is 0.99849
]

So the risk of failing even after visiting all the houses is 1 - this probability,
  ``1 - cdf(nb, all_houses - sales_quota``
But using this expression may cause serious inaccuracy,
so it would be much better to use the complement of the cdf:
So the risk of failing even at, or after, the 31th (non-existent) houses is 1 - this probability,
  ``1 - cdf(nb, all_houses - sales_quota)``
But using this expression may cause serious inaccuracy.
So it would be much better to use the __complement of the cdf (see __why_complements).
*/
    cout << "\nProbability of failing to sell his quota of " << sales_quota
      << " bars\neven after visiting all " << all_houses << " houses is "
      << cdf(complement(nb, all_houses - sales_quota)) << endl;
/*`
[pre
Probability of failing to sell his quota of 5 bars
even after visiting all 30 houses is 0.0015101
]
We can also use the quantile (percentile), the inverse of the cdf, to
predict which house Pat will finish on.  So for the 8th house:
*/
 double p = cdf(nb, (8 - sales_quota));
 cout << "Probability of meeting sales quota on or before 8th house is "<< p << endl;
/*`
[pre
Probability of meeting sales quota on or before 8th house is 0.174
]
*/
    cout << "If the confidence of meeting sales quota is " << p
        << ", then the finishing house is " << quantile(nb, p) + sales_quota << endl;

    cout<< " quantile(nb, p) = " << quantile(nb, p) << endl;
/*`
[pre
If the confidence of meeting sales quota is 0.17367, then the finishing house is 8
]
Demanding absolute certainty that all 5 will be sold,
implies an infinite number of trials.
(Of course, there are only 30 houses on the estate,
so he can't ever be *certain* of selling his quota).
*/
    cout << "If the confidence of meeting sales quota is " << 1.
        << ", then the finishing house is " << quantile(nb, 1) + sales_quota << endl;
    //  1.#INF == infinity.
/*`[pre
If the confidence of meeting sales quota is 1, then the finishing house is 1.#INF
]
And similarly for a few other probabilities:
*/
    cout << "If the confidence of meeting sales quota is " << 0.
        << ", then the finishing house is " << quantile(nb, 0.) + sales_quota << endl;

    cout << "If the confidence of meeting sales quota is " << 0.5
        << ", then the finishing house is " << quantile(nb, 0.5) + sales_quota << endl;

    cout << "If the confidence of meeting sales quota is " << 1 - 0.00151 // 30 th
        << ", then the finishing house is " << quantile(nb, 1 - 0.00151) + sales_quota << endl;
/*`
[pre
If the confidence of meeting sales quota is 0, then the finishing house is 5
If the confidence of meeting sales quota is 0.5, then the finishing house is 11.337
If the confidence of meeting sales quota is 0.99849, then the finishing house is 30
]

Notice that because we chose a discrete quantile policy of real,
the result can be an 'unreal' fractional house.

If the opposite is true, we don't want to assume any confidence, then this is tantamount
to assuming that all the first sales_quota trials will be successful sales.
*/
    cout << "If confidence of meeting quota is zero\n(we assume all houses are successful sales)"
      ", then finishing house is " << sales_quota << endl;
/*`
[pre
If confidence of meeting quota is zero (we assume all houses are successful sales), then finishing house is 5
If confidence of meeting quota is 0, then finishing house is 5
]
We can list quantiles for a few probabilities:
*/

    double ps[] = {0., 0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999, 1.};
    // Confidence as fraction = 1-alpha, as percent =  100 * (1-alpha[i]) %
    cout.precision(3);
    for (unsigned i = 0; i < sizeof(ps)/sizeof(ps[0]); i++)
    {
      cout << "If confidence of meeting quota is " << ps[i]
        << ", then finishing house is " << quantile(nb, ps[i]) + sales_quota
        << endl;
   }

/*`
[pre
If confidence of meeting quota is 0, then finishing house is 5
If confidence of meeting quota is 0.001, then finishing house is 5
If confidence of meeting quota is 0.01, then finishing house is 5
If confidence of meeting quota is 0.05, then finishing house is 6.2
If confidence of meeting quota is 0.1, then finishing house is 7.06
If confidence of meeting quota is 0.5, then finishing house is 11.3
If confidence of meeting quota is 0.9, then finishing house is 17.8
If confidence of meeting quota is 0.95, then finishing house is 20.1
If confidence of meeting quota is 0.99, then finishing house is 24.8
If confidence of meeting quota is 0.999, then finishing house is 31.1
If confidence of meeting quota is 1, then finishing house is 1.#INF
]

We could have applied a ceil function to obtain a 'worst case' integer value for house.
``ceil(quantile(nb, ps[i]))``

Or, if we had used the default discrete quantile policy, integer_outside, by omitting
``#define BOOST_MATH_DISCRETE_QUANTILE_POLICY real``
we would have achieved the same effect.

The real result gives some suggestion which house is most likely.
For example, compare the real and integer_outside for 95% confidence.

[pre
If confidence of meeting quota is 0.95, then finishing house is 20.1
If confidence of meeting quota is 0.95, then finishing house is 21
]
The real value 20.1 is much closer to 20 than 21, so integer_outside is pessimistic.
We could also use integer_round_nearest policy to suggest that 20 is more likely.

Finally, we can tabulate the probability for the last sale being exactly on each house.
*/
   cout << "\nHouse for " << sales_quota << "th (last) sale.  Probability (%)" << endl;
   cout.precision(5);
   for (int i = (int)sales_quota; i < all_houses+1; i++)
   {
     cout << left << setw(3) << i << "                             " << setw(8) << cdf(nb, i - sales_quota)  << endl;
   }
   cout << endl;
/*`
[pre
House for 5 th (last) sale.  Probability (%)
5                               0.01024
6                               0.04096
7                               0.096256
8                               0.17367
9                               0.26657
10                              0.3669
11                              0.46723
12                              0.56182
13                              0.64696
14                              0.72074
15                              0.78272
16                              0.83343
17                              0.874
18                              0.90583
19                              0.93039
20                              0.94905
21                              0.96304
22                              0.97342
23                              0.98103
24                              0.98655
25                              0.99053
26                              0.99337
27                              0.99539
28                              0.99681
29                              0.9978
30                              0.99849
]

As noted above, using a catch block is always a good idea, even if you do not expect to use it.
*/
  }
  catch(const std::exception& e)
  { // Since we have set an overflow policy of ignore_error,
    // an overflow exception should never be thrown.
     std::cout << "\nMessage from thrown exception was:\n " << e.what() << std::endl;
/*`
For example, without a ignore domain error policy, if we asked for ``pdf(nb, -1)`` for example, we would get:
[pre
Message from thrown exception was:
 Error in function boost::math::pdf(const negative_binomial_distribution<double>&, double):
 Number of failures argument is -1, but must be >= 0 !
]
*/
//] [/ negative_binomial_eg1_2]
  }
   return 0;
}  // int main()


/*

Output is:

Selling candy bars - using the negative binomial distribution.
by Dr. Diane Evans,
Professor of Mathematics at Rose-Hulman Institute of Technology,
see http://en.wikipedia.org/wiki/Negative_binomial_distribution
Pat has a sales per house success rate of 0.4.
Therefore he would, on average, sell 40 bars after trying 100 houses.
With a success rate of 0.4, he might expect, on average,
to need to visit about 12 houses in order to sell all 5 bars.
Probability that Pat finishes on the 5th house is 0.01024
Probability that Pat finishes on the 6th house is 0.03072
Probability that Pat finishes on the 7th house is 0.055296
Probability that Pat finishes on the 8th house is 0.077414
Probability that Pat finishes on or before the 8th house is sum
pdf(sales_quota) + pdf(6) + pdf(7) + pdf(8) = 0.17367
Probability of selling his quota of 5 bars
on or before the 8th house is 0.17367
Probability that Pat finishes exactly on the 10th house is 0.10033
Probability of selling his quota of 5 bars
on or before the 10th house is 0.3669
Probability that Pat finishes exactly on the 11th house is 0.10033
Probability of selling his quota of 5 bars
on or before the 11th house is 0.46723
Probability that Pat finishes exactly on the 12th house is 0.094596
Probability of selling his quota of 5 bars
on or before the 12th house is 0.56182
Probability that Pat finishes on the 30 house is 0.00069145
Probability of selling his quota of 5 bars
on or before the 30th house is 0.99849
Probability of failing to sell his quota of 5 bars
even after visiting all 30 houses is 0.0015101
Probability of meeting sales quota on or before 8th house is 0.17367
If the confidence of meeting sales quota is 0.17367, then the finishing house is 8
 quantile(nb, p) = 3
If the confidence of meeting sales quota is 1, then the finishing house is 1.#INF
If the confidence of meeting sales quota is 0, then the finishing house is 5
If the confidence of meeting sales quota is 0.5, then the finishing house is 11.337
If the confidence of meeting sales quota is 0.99849, then the finishing house is 30
If confidence of meeting quota is zero
(we assume all houses are successful sales), then finishing house is 5
If confidence of meeting quota is 0, then finishing house is 5
If confidence of meeting quota is 0.001, then finishing house is 5
If confidence of meeting quota is 0.01, then finishing house is 5
If confidence of meeting quota is 0.05, then finishing house is 6.2
If confidence of meeting quota is 0.1, then finishing house is 7.06
If confidence of meeting quota is 0.5, then finishing house is 11.3
If confidence of meeting quota is 0.9, then finishing house is 17.8
If confidence of meeting quota is 0.95, then finishing house is 20.1
If confidence of meeting quota is 0.99, then finishing house is 24.8
If confidence of meeting quota is 0.999, then finishing house is 31.1
If confidence of meeting quota is 1, then finishing house is 1.#J
House for 5th (last) sale.  Probability (%)
5                               0.01024
6                               0.04096
7                               0.096256
8                               0.17367
9                               0.26657
10                              0.3669
11                              0.46723
12                              0.56182
13                              0.64696
14                              0.72074
15                              0.78272
16                              0.83343
17                              0.874
18                              0.90583
19                              0.93039
20                              0.94905
21                              0.96304
22                              0.97342
23                              0.98103
24                              0.98655
25                              0.99053
26                              0.99337
27                              0.99539
28                              0.99681
29                              0.9978
30                              0.99849

*/
