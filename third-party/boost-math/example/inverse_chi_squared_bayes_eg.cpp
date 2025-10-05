// inverse_chi_squared_bayes_eg.cpp

// Copyright Thomas Mang 2011.
// Copyright Paul A. Bristow 2011.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file is written to be included from a Quickbook .qbk document.
// It can still be compiled by the C++ compiler, and run.
// Any output can also be added here as comment or included or pasted in elsewhere.
// Caution: this file contains Quickbook markup as well as code
// and comments: don't change any of the special comment markups!

#include <iostream>
//  using std::cout; using std::endl;
 
//#define  define possible error-handling macros here?

#include "boost/math/distributions.hpp"
// using ::boost::math::inverse_chi_squared;

int main()
{
  using std::cout; using std::endl;

  using ::boost::math::inverse_chi_squared;
  using ::boost::math::inverse_gamma;
  using ::boost::math::quantile;
  using ::boost::math::cdf;
  
  cout << "Inverse_chi_squared_distribution Bayes example: " << endl <<endl;

  cout.precision(3);
// Examples of using the inverse_chi_squared distribution.

//[inverse_chi_squared_bayes_eg_1
/*`
The scaled-inversed-chi-squared distribution is the conjugate prior distribution
for the variance ([sigma][super 2]) parameter of a normal distribution
with known expectation ([mu]).
As such it has widespread application in Bayesian statistics:

In [@http://en.wikipedia.org/wiki/Bayesian_inference Bayesian inference],
the strength of belief into certain parameter values is
itself described through a distribution. Parameters
hence become themselves modelled and interpreted as random variables.

In this worked example, we perform such a Bayesian analysis by using
the scaled-inverse-chi-squared distribution as prior and posterior distribution
for the variance parameter of a normal distribution.

For more general information on Bayesian type of analyses,
see:

* Andrew Gelman, John B. Carlin, Hal E. Stern, Donald B. Rubin, Bayesian Data Analysis,
2003, ISBN 978-1439840955.

* Jim Albert, Bayesian Computation with R, Springer, 2009, ISBN 978-0387922973.

(As the scaled-inversed-chi-squared is another parameterization of the inverse-gamma distribution,
this example could also have used the inverse-gamma distribution).

Consider precision machines which produce balls for a high-quality ball bearing.
Ideally each ball should have a diameter of precisely 3000 [mu]m (3 mm).
Assume that machines generally produce balls of that size on average (mean),
but individual balls can vary slightly in either direction
following (approximately) a normal distribution. Depending on various production conditions
(e.g. raw material used for balls, workplace temperature and humidity, maintenance frequency and quality)
some machines produce balls tighter distributed around the target of 3000 [mu]m,
while others produce balls with a wider distribution. 
Therefore the variance parameter of the normal distribution of the ball sizes varies
from machine to machine. An extensive survey by the precision machinery manufacturer, however,
has shown that most machines operate with a variance between 15 and 50,
and near 25 [mu]m[super 2] on average.

Using this information, we want to model the variance of the machines.
The variance is strictly positive, and therefore we look for a statistical distribution
with support in the positive domain of the real numbers.
Given the expectation of the normal distribution of the balls is known (3000 [mu]m),
for reasons of conjugacy, it is customary practice in Bayesian statistics
to model the variance to be scaled-inverse-chi-squared distributed. 

In a first step, we will try to use the survey information to model
the general knowledge about the variance parameter of machines measured by the manufacturer. 
This will provide us with a generic prior distribution that is applicable
if nothing more specific is known about a particular machine.

In a second step, we will then combine the prior-distribution information in a Bayesian analysis
with data on a specific single machine to derive a posterior distribution for that machine.

[h5 Step one: Using the survey information.]

Using the survey results, we try to find the parameter set
of a scaled-inverse-chi-squared distribution 
so that the properties of this distribution match the results. 
Using the mathematical properties of the scaled-inverse-chi-squared distribution 
as guideline, we see that that both the mean and mode of the scaled-inverse-chi-squared distribution
are approximately given by the scale parameter (s) of the distribution. As the survey machines operated at a
variance of 25 [mu]m[super 2] on average, we hence set the scale parameter (s[sub prior]) of our prior distribution 
equal to this value. Using some trial-and-error and calls to the global quantile function, we also find that a
value of 20 for the degrees-of-freedom ([nu][sub prior]) parameter is adequate so that
most of the prior distribution mass is located between 15 and 50 (see figure below).

We first construct our prior distribution using these values, and then list out a few quantiles:

*/
  double priorDF = 20.0;
  double priorScale = 25.0; 

  inverse_chi_squared prior(priorDF, priorScale);
  // Using an inverse_gamma distribution instead, we could equivalently write
  // inverse_gamma prior(priorDF / 2.0, priorScale * priorDF / 2.0);
  
  cout << "Prior distribution:" << endl << endl;
  cout << "  2.5% quantile: " << quantile(prior, 0.025) << endl;
  cout << "  50% quantile: " << quantile(prior, 0.5) << endl;
  cout << "  97.5% quantile: " << quantile(prior, 0.975) << endl << endl;

 //] [/inverse_chi_squared_bayes_eg_1]

//[inverse_chi_squared_bayes_eg_output_1
/*`This produces this output:

    Prior distribution:
  
    2.5% quantile: 14.6
    50% quantile: 25.9
    97.5% quantile: 52.1

*/
//] [/inverse_chi_squared_bayes_eg_output_1]

//[inverse_chi_squared_bayes_eg_2
/*`
Based on this distribution, we can now calculate the probability of having a machine
working with an unusual work precision (variance) at <= 15 or > 50.
For this task, we use calls to the `boost::math::` functions `cdf` and `complement`,
respectively, and find a probability of about 0.031 (3.1%) for each case.
*/
  
  cout << "  probability variance <= 15: " << boost::math::cdf(prior, 15.0) << endl;
  cout << "  probability variance <= 25: " << boost::math::cdf(prior, 25.0) << endl;
  cout << "  probability variance > 50: " 
    << boost::math::cdf(boost::math::complement(prior, 50.0))
  << endl << endl;
 //] [/inverse_chi_squared_bayes_eg_2]

//[inverse_chi_squared_bayes_eg_output_2
/*`This produces this output:

    probability variance <= 15: 0.031
    probability variance <= 25: 0.458
    probability variance > 50: 0.0318

*/
//] [/inverse_chi_squared_bayes_eg_output_2]
  
//[inverse_chi_squared_bayes_eg_3
/*`Therefore, only 3.1% of all precision machines produce balls with a variance of 15 or less
(particularly precise machines),
but also only 3.2% of all machines produce balls
with a variance of as high as 50 or more (particularly imprecise machines). Moreover, slightly more than
one-half (1 - 0.458 = 54.2%) of the machines work at a variance greater than 25. 

Notice here the distinction between a
[@http://en.wikipedia.org/wiki/Bayesian_inference Bayesian] analysis and a
[@http://en.wikipedia.org/wiki/Frequentist_inference frequentist] analysis:
because we model the variance as random variable itself,
we can calculate and straightforwardly interpret probabilities for given parameter values directly,
while such an approach is not possible (and interpretationally a strict ['must-not]) in the frequentist
world.

[h5 Step 2: Investigate a single machine]

In the second step, we investigate a single machine,
which is suspected to suffer from a major fault
as the produced balls show fairly high size variability.
Based on the prior distribution of generic machinery performance (derived above)
and data on balls produced by the suspect machine, we calculate the posterior distribution for that 
machine and use its properties for guidance regarding continued machine operation or suspension.

It can be shown that if the prior distribution
was chosen to be scaled-inverse-chi-square distributed,
then the posterior distribution is also scaled-inverse-chi-squared-distributed 
(prior and posterior distributions are hence conjugate).
For more details regarding conjugacy and formula to derive the parameters set
for the posterior distribution see
[@http://en.wikipedia.org/wiki/Conjugate_prior Conjugate prior].


Given the prior distribution parameters and sample data (of size n), the posterior distribution parameters 
are given by the two expressions:

__spaces [nu][sub posterior] = [nu][sub prior] + n

which gives the posteriorDF below, and

__spaces s[sub posterior] = ([nu][sub prior]s[sub prior] + [Sigma][super n][sub i=1](x[sub i] - [mu])[super 2]) / ([nu][sub prior] + n)

which after some rearrangement gives the formula for the posteriorScale below.

Machine-specific data consist of 100 balls which were accurately measured
and show the expected mean of 3000 [mu]m and a sample variance of 55 (calculated for a sample mean defined to be 3000 exactly).
From these data, the prior parameterization, and noting that the term 
[Sigma][super n][sub i=1](x[sub i] - [mu])[super 2] equals the sample variance multiplied by n - 1,
it follows that the posterior distribution of the variance parameter
is scaled-inverse-chi-squared distribution with degrees-of-freedom ([nu][sub posterior]) = 120 and 
scale (s[sub posterior]) = 49.54.
*/

  int ballsSampleSize = 100;
  cout <<"balls sample size: " << ballsSampleSize << endl;
  double ballsSampleVariance = 55.0;
  cout <<"balls sample variance: " << ballsSampleVariance << endl;

  double posteriorDF = priorDF + ballsSampleSize;
  cout << "prior degrees-of-freedom: " << priorDF << endl;
  cout << "posterior degrees-of-freedom: " << posteriorDF << endl;
  
  double posteriorScale = 
    (priorDF * priorScale + (ballsSampleVariance * (ballsSampleSize - 1))) / posteriorDF;
  cout << "prior scale: " << priorScale  << endl;
  cout << "posterior scale: " << posteriorScale << endl;

/*`An interesting feature here is that one needs only to know a summary statistics of the sample
to parameterize the posterior distribution: the 100 individual ball measurements are irrelevant,
just knowledge of the sample variance and number of measurements is sufficient.
*/

//] [/inverse_chi_squared_bayes_eg_3]

//[inverse_chi_squared_bayes_eg_output_3
/*`That produces this output:


  balls sample size: 100
  balls sample variance: 55
  prior degrees-of-freedom: 20
  posterior degrees-of-freedom: 120
  prior scale: 25
  posterior scale: 49.5
   
  */
//] [/inverse_chi_squared_bayes_eg_output_3]

//[inverse_chi_squared_bayes_eg_4
/*`To compare the generic machinery performance with our suspect machine,
we calculate again the same quantiles and probabilities as above,
and find a distribution clearly shifted to greater values (see figure).

[graph prior_posterior_plot]

*/

 inverse_chi_squared posterior(posteriorDF, posteriorScale);

  cout << "Posterior distribution:" << endl << endl;
  cout << "  2.5% quantile: " << boost::math::quantile(posterior, 0.025) << endl;
  cout << "  50% quantile: " << boost::math::quantile(posterior, 0.5) << endl;
  cout << "  97.5% quantile: " << boost::math::quantile(posterior, 0.975) << endl << endl;

  cout << "  probability variance <= 15: " << boost::math::cdf(posterior, 15.0) << endl;
  cout << "  probability variance <= 25: " << boost::math::cdf(posterior, 25.0) << endl;
  cout << "  probability variance > 50: " 
    << boost::math::cdf(boost::math::complement(posterior, 50.0)) << endl;

//] [/inverse_chi_squared_bayes_eg_4]

//[inverse_chi_squared_bayes_eg_output_4
/*`This produces this output:

 Posterior distribution:
  
    2.5% quantile: 39.1
    50% quantile: 49.8
    97.5% quantile: 64.9
  
    probability variance <= 15: 2.97e-031
    probability variance <= 25: 8.85e-010
    probability variance > 50: 0.489
  
*/
//] [/inverse_chi_squared_bayes_eg_output_4]

//[inverse_chi_squared_bayes_eg_5
/*`Indeed, the probability that the machine works at a low variance (<= 15) is almost zero,
and even the probability of working at average or better performance is negligibly small
(less than one-millionth of a permille). 
On the other hand, with an almost near-half probability (49%), the machine operates in the
extreme high variance range of > 50 characteristic for poorly performing machines.

Based on this information the operation of the machine is taken out of use and serviced.

In summary, the Bayesian analysis allowed us to make exact probabilistic statements about a
parameter of interest, and hence provided us results with straightforward interpretation. 

*/
//] [/inverse_chi_squared_bayes_eg_5]

} // int main()

//[inverse_chi_squared_bayes_eg_output
/*`
[pre
 Inverse_chi_squared_distribution Bayes example: 
  
   Prior distribution:
  
    2.5% quantile: 14.6
    50% quantile: 25.9
    97.5% quantile: 52.1
  
    probability variance <= 15: 0.031
    probability variance <= 25: 0.458
    probability variance > 50: 0.0318
  
  balls sample size: 100
  balls sample variance: 55
  prior degrees-of-freedom: 20
  posterior degrees-of-freedom: 120
  prior scale: 25
  posterior scale: 49.5
  Posterior distribution:
  
    2.5% quantile: 39.1
    50% quantile: 49.8
    97.5% quantile: 64.9
  
    probability variance <= 15: 2.97e-031
    probability variance <= 25: 8.85e-010
    probability variance > 50: 0.489

] [/pre]
*/
//] [/inverse_chi_squared_bayes_eg_output]
