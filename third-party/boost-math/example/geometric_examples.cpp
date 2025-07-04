// geometric_examples.cpp

// Copyright Paul A. Bristow 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file is written to be included from a Quickbook .qbk document.
// It can still be compiled by the C++ compiler, and run.
// Any output can also be added here as comment or included or pasted in elsewhere.
// Caution: this file contains Quickbook markup as well as code
// and comments: don't change any of the special comment markups!

// Examples of using the geometric distribution.

//[geometric_eg1_1
/*`
For this example, we will opt to #define two macros to control
the error and discrete handling policies.
For this simple example, we want to avoid throwing
an exception (the default policy) and just return infinity.
We want to treat the distribution as if it was continuous,
so we choose a discrete_quantile policy of real,
rather than the default policy integer_round_outwards.
*/
#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#define BOOST_MATH_DISCRETE_QUANTILE_POLICY real
/*`
[caution It is vital to #include distributions etc *after* the above #defines]
After that we need some includes to provide easy access to the negative binomial distribution,
and we need some std library iostream, of course.
*/
#include <boost/math/distributions/geometric.hpp>
  // for geometric_distribution
  using ::boost::math::geometric_distribution; //
  using ::boost::math::geometric; // typedef provides default type is double.
  using  ::boost::math::pdf; // Probability mass function.
  using  ::boost::math::cdf; // Cumulative density function.
  using  ::boost::math::quantile;

#include <boost/math/distributions/negative_binomial.hpp>
  // for negative_binomial_distribution
  using boost::math::negative_binomial; // typedef provides default type is double.

#include <boost/math/distributions/normal.hpp>
  // for negative_binomial_distribution
  using boost::math::normal; // typedef provides default type is double.

#include <iostream>
  using std::cout; using std::endl;
  using std::noshowpoint; using std::fixed; using std::right; using std::left;
#include <iomanip>
  using std::setprecision; using std::setw;

#include <limits>
  using std::numeric_limits;
//] [geometric_eg1_1]

int main()
{
  cout <<"Geometric distribution example" << endl;
  cout << endl;

  cout.precision(4); // But only show a few for this example.
  try
  {
//[geometric_eg1_2
/*`
It is always sensible to use try and catch blocks because defaults policies are to
throw an exception if anything goes wrong.

Simple try'n'catch blocks (see below) will ensure that you get a
helpful error message instead of an abrupt (and silent) program abort.

[h6 Throwing a dice]
The Geometric distribution describes the probability (/p/) of a number of failures
to get the first success in /k/ Bernoulli trials.
(A [@http://en.wikipedia.org/wiki/Bernoulli_distribution Bernoulli trial]
is one with only two possible outcomes, success of failure,
and /p/ is the probability of success).

Suppose an 'fair' 6-face dice is thrown repeatedly:
*/
    double success_fraction = 1./6; // success_fraction (p) = 0.1666
    // (so failure_fraction is 1 - success_fraction = 5./6 = 1- 0.1666 = 0.8333)

/*`If the dice is thrown repeatedly until the *first* time a /three/ appears.
The probability distribution of the number of times it is thrown *not* getting a /three/
 (/not-a-threes/ number of failures to get a /three/)
is a geometric distribution with the success_fraction = 1/6 = 0.1666[recur].

We therefore start by constructing a geometric distribution
with the one parameter success_fraction, the probability of success.
*/
    geometric g6(success_fraction); // type double by default.
/*`
To confirm, we can echo the success_fraction parameter of the distribution.
*/
    cout << "success fraction of a six-sided dice is " << g6.success_fraction() << endl;
/*`So the probability of getting a three at the first throw (zero failures) is
*/
    cout << pdf(g6, 0) << endl; // 0.1667
    cout << cdf(g6, 0) << endl; // 0.1667
/*`Note that the cdf and pdf are identical because the is only one throw.
If we want the probability of getting the first /three/ on the 2nd throw:
*/
    cout << pdf(g6, 1) << endl; // 0.1389

/*`If we want the probability of getting the first /three/ on the 1st or 2nd throw
(allowing one failure):
*/
    cout << "pdf(g6, 0) + pdf(g6, 1) = " << pdf(g6, 0) + pdf(g6, 1) << endl;
/*`Or more conveniently, and more generally,
we can use the Cumulative Distribution Function CDF.*/

    cout << "cdf(g6, 1) = " << cdf(g6, 1) << endl; // 0.3056

/*`If we allow many more (12) throws, the probability of getting our /three/ gets very high:*/
    cout << "cdf(g6, 12) = " << cdf(g6, 12) << endl; // 0.9065 or 90% probability.
/*`If we want to be much more confident, say 99%,
we can estimate the number of throws to be this sure
using the inverse or quantile.
*/
    cout << "quantile(g6, 0.99) = " << quantile(g6, 0.99) << endl; // 24.26
/*`Note that the value returned is not an integer:
if you want an integer result you should use either floor, round or ceil functions,
or use the policies mechanism.

See __understand_dis_quant.

The geometric distribution is related to the negative binomial
__spaces `negative_binomial_distribution(RealType r, RealType p);` with parameter /r/ = 1.
So we could get the same result using the negative binomial,
but using the geometric the results will be faster, and may be more accurate.
*/
    negative_binomial nb(1, success_fraction);
    cout << pdf(nb, 1) << endl; // 0.1389
    cout << cdf(nb, 1) << endl; // 0.3056
/*`We could also the complement to express the required probability
as 1 - 0.99 = 0.01 (and get the same result):
*/
    cout << "quantile(complement(g6, 1 - p))  " << quantile(complement(g6, 0.01)) << endl; // 24.26
/*`
Note too that Boost.Math geometric distribution is implemented as a continuous function.
Unlike other implementations (for example R) it *uses* the number of failures as a *real* parameter,
not as an integer. If you want this integer behaviour, you may need to enforce this by
rounding the parameter you pass, probably rounding down, to the nearest integer.
For example, R returns the success fraction probability for all values of failures
from 0 to 0.999999 thus:
[pre
__spaces R> formatC(pgeom(0.0001,0.5, FALSE), digits=17) "               0.5"
] [/pre]
So in Boost.Math the equivalent is
*/
    geometric g05(0.5);  // Probability of success = 0.5 or 50%
    // Output all potentially significant digits for the type, here double.

#ifdef BOOST_NO_CXX11_NUMERIC_LIMITS
  int max_digits10 = 2 + (boost::math::policies::digits<double, boost::math::policies::policy<> >() * 30103UL) / 100000UL;
  cout << "BOOST_NO_CXX11_NUMERIC_LIMITS is defined" << endl;
#else
  int max_digits10 = std::numeric_limits<double>::max_digits10;
#endif
  cout << "Show all potentially significant decimal digits std::numeric_limits<double>::max_digits10 = "
    << max_digits10 << endl;
  cout.precision(max_digits10); //

    cout << cdf(g05, 0.0001) << endl; // returns 0.5000346561579232, not exact 0.5.
/*`To get the R discrete behaviour, you simply need to round with,
for example, the `floor` function.
*/
    cout << cdf(g05, floor(0.0001)) << endl; // returns exactly 0.5
/*`
[pre
`> formatC(pgeom(0.9999999,0.5, FALSE), digits=17) [1] "              0.25"`
`> formatC(pgeom(1.999999,0.5, FALSE), digits=17)[1] "              0.25" k = 1`
`> formatC(pgeom(1.9999999,0.5, FALSE), digits=17)[1] "0.12500000000000003" k = 2`
] [/pre]
shows that R makes an arbitrary round-up decision at about 1e7 from the next integer above.
This may be convenient in practice, and could be replicated in C++ if desired.

[h6 Surveying customers to find one with a faulty product]
A company knows from warranty claims that 2% of their products will be faulty,
so the 'success_fraction' of finding a fault is 0.02.
It wants to interview a purchaser of faulty products to assess their 'user experience'.

To estimate how many customers they will probably need to contact
in order to find one who has suffered from the fault,
we first construct a geometric distribution with probability 0.02,
and then chose a confidence, say 80%, 95%, or 99% to finding a customer with a fault.
Finally, we probably want to round up the result to the integer above using the `ceil` function.
(We could also use a policy, but that is hardly worthwhile for this simple application.)

(This also assumes that each customer only buys one product:
if customers bought more than one item,
the probability of finding a customer with a fault obviously improves.)
*/
    cout.precision(5);
    geometric g(0.02); // On average, 2 in 100 products are faulty.
    double c = 0.95; // 95% confidence.
    cout << " quantile(g, " << c << ") = " << quantile(g, c) << endl;

    cout << "To be " << c * 100
      << "% confident of finding we customer with a fault, need to survey "
      <<  ceil(quantile(g, c)) << " customers." << endl; // 148
    c = 0.99; // Very confident.
    cout << "To be " << c * 100
      << "% confident of finding we customer with a fault, need to survey "
      <<  ceil(quantile(g, c)) << " customers." << endl; // 227
    c = 0.80; // Only reasonably confident.
    cout << "To be " << c * 100
      << "% confident of finding we customer with a fault, need to survey "
      <<  ceil(quantile(g, c)) << " customers." << endl; // 79

/*`[h6 Basket Ball Shooters]
According to Wikipedia, average pro basket ball players get
[@http://en.wikipedia.org/wiki/Free_throw free throws]
in the baskets 70 to 80 % of the time,
but some get as high as 95%, and others as low as 50%.
Suppose we want to compare the probabilities
of failing to get a score only on the first or on the fifth shot?
To start we will consider the average shooter, say 75%.
So we construct a geometric distribution
with success_fraction parameter 75/100 = 0.75.
*/
    cout.precision(2);
    geometric gav(0.75); // Shooter averages 7.5 out of 10 in the basket.
/*`What is probability of getting 1st try in the basket, that is with no failures? */
    cout << "Probability of score on 1st try = " << pdf(gav, 0) << endl; // 0.75
/*`This is, of course, the success_fraction probability 75%.
What is the probability that the shooter only scores on the fifth shot?
So there are 5-1 = 4 failures before the first success.*/
    cout << "Probability of score on 5th try = " << pdf(gav, 4) << endl; // 0.0029
/*`Now compare this with the poor and the best players success fraction.
We need to constructing new distributions with the different success fractions,
and then get the corresponding probability density functions values:
*/
    geometric gbest(0.95);
    cout << "Probability of score on 5th try = " << pdf(gbest, 4) << endl; // 5.9e-6
    geometric gmediocre(0.50);
    cout << "Probability of score on 5th try = " << pdf(gmediocre, 4) << endl; // 0.031
/*`So we can see the very much smaller chance (0.000006) of 4 failures by the best shooters,
compared to the 0.03 of the mediocre.*/

/*`[h6 Estimating failures]
Of course one man's failure is an other man's success.
So a fault can be defined as a 'success'.

If a fault occurs once after 100 flights, then one might naively say
that the risk of fault is obviously 1 in 100 = 1/100, a probability of 0.01.

This is the best estimate we can make, but while it is the truth,
it is not the whole truth,
for it hides the big uncertainty when estimating from a single event.
"One swallow doesn't make a summer."
To show the magnitude of the uncertainty, the geometric
(or the negative binomial) distribution can be used.

If we chose the popular 95% confidence in the limits, corresponding to an alpha of 0.05,
because we are calculating a two-sided interval, we must divide alpha by two.
*/
    double alpha = 0.05;
    double k = 100; // So frequency of occurrence is 1/100.
    cout << "Probability is failure is " << 1/k << endl;
    double t = geometric::find_lower_bound_on_p(k, alpha/2);
    cout << "geometric::find_lower_bound_on_p(" << int(k) << ", " << alpha/2 << ") = "
      << t << endl; // 0.00025
    t = geometric::find_upper_bound_on_p(k, alpha/2);
    cout << "geometric::find_upper_bound_on_p(" << int(k) << ", " << alpha/2 << ") = "
      << t << endl; // 0.037
/*`So while we estimate the probability is 0.01, it might lie between 0.0003 and 0.04.
Even if we relax our confidence to alpha = 90%, the bounds only contract to 0.0005 and 0.03.
And if we require a high confidence, they widen to 0.00005 to 0.05.
*/
    alpha = 0.1; // 90% confidence.
    t = geometric::find_lower_bound_on_p(k, alpha/2);
    cout << "geometric::find_lower_bound_on_p(" << int(k) << ", " << alpha/2 << ") = "
      << t << endl; // 0.0005
    t = geometric::find_upper_bound_on_p(k, alpha/2);
    cout << "geometric::find_upper_bound_on_p(" << int(k) << ", " << alpha/2 << ") = "
      << t << endl; // 0.03

    alpha = 0.01; // 99% confidence.
    t = geometric::find_lower_bound_on_p(k, alpha/2);
    cout << "geometric::find_lower_bound_on_p(" << int(k) << ", " << alpha/2 << ") = "
      << t << endl; // 5e-005
    t = geometric::find_upper_bound_on_p(k, alpha/2);
    cout << "geometric::find_upper_bound_on_p(" << int(k) << ", " << alpha/2 << ") = "
        << t << endl; // 0.052
/*`In real life, there will usually be more than one event (fault or success),
when the negative binomial, which has the necessary extra parameter, will be needed.
*/

/*`As noted above, using a catch block is always a good idea,
even if you hope not to use it!
*/
  }
  catch(const std::exception& e)
  { // Since we have set an overflow policy of ignore_error,
    // an overflow exception should never be thrown.
     std::cout << "\nMessage from thrown exception was:\n " << e.what() << std::endl;
/*`
For example, without a ignore domain error policy,
if we asked for ``pdf(g, -1)`` for example,
we would get an unhelpful abort, but with a catch:
[pre
Message from thrown exception was:
 Error in function boost::math::pdf(const exponential_distribution<double>&, double):
 Number of failures argument is -1, but must be >= 0 !
] [/pre]
*/
//] [/ geometric_eg1_2]
  }
  return 0;
}  // int main()


/*
Output is:

  Geometric distribution example

  success fraction of a six-sided dice is 0.1667
  0.1667
  0.1667
  0.1389
  pdf(g6, 0) + pdf(g6, 1) = 0.3056
  cdf(g6, 1) = 0.3056
  cdf(g6, 12) = 0.9065
  quantile(g6, 0.99) = 24.26
  0.1389
  0.3056
  quantile(complement(g6, 1 - p))  24.26
  0.5000346561579232
  0.5
   quantile(g, 0.95) = 147.28
  To be 95% confident of finding we customer with a fault, need to survey 148 customers.
  To be 99% confident of finding we customer with a fault, need to survey 227 customers.
  To be 80% confident of finding we customer with a fault, need to survey 79 customers.
  Probability of score on 1st try = 0.75
  Probability of score on 5th try = 0.0029
  Probability of score on 5th try = 5.9e-006
  Probability of score on 5th try = 0.031
  Probability is failure is 0.01
  geometric::find_lower_bound_on_p(100, 0.025) = 0.00025
  geometric::find_upper_bound_on_p(100, 0.025) = 0.037
  geometric::find_lower_bound_on_p(100, 0.05) = 0.00051
  geometric::find_upper_bound_on_p(100, 0.05) = 0.03
  geometric::find_lower_bound_on_p(100, 0.005) = 5e-005
  geometric::find_upper_bound_on_p(100, 0.005) = 0.052

*/










