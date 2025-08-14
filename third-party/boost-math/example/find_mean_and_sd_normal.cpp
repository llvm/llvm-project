// find_mean_and_sd_normal.cpp

// Copyright Paul A. Bristow 2007, 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Example of finding mean or sd for normal distribution.

// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

//[normal_std
/*`
First we need some includes to access the normal distribution,
the algorithms to find location and scale
(and some std output of course).
*/

#include <boost/math/distributions/normal.hpp> // for normal_distribution
  using boost::math::normal; // typedef provides default type is double.
#include <boost/math/distributions/cauchy.hpp> // for cauchy_distribution
  using boost::math::cauchy; // typedef provides default type is double.
#include <boost/math/distributions/find_location.hpp>
  using boost::math::find_location;
#include <boost/math/distributions/find_scale.hpp>
  using boost::math::find_scale;
  using boost::math::complement;
  using boost::math::policies::policy;

#include <iostream>
  using std::cout; using std::endl; using std::left; using std::showpoint; using std::noshowpoint;
#include <iomanip>
  using std::setw; using std::setprecision;
#include <limits>
  using std::numeric_limits;
#include <stdexcept>
  
//] [/normal_std Quickbook]

int main()
{
  cout << "Find_location (mean) and find_scale (standard deviation) examples." << endl;
  try
  {

//[normal_find_location_and_scale_eg

/*`
[h4 Using find_location and find_scale to meet dispensing and measurement specifications]

Consider an example from K Krishnamoorthy,
Handbook of Statistical Distributions with Applications,
ISBN 1-58488-635-8, (2006) p 126, example 10.3.7.

"A machine is set to pack 3 kg of ground beef per pack.
Over a long period of time it is found that the average packed was 3 kg
with a standard deviation of 0.1 kg.
Assume the packing is normally distributed."

We start by constructing a normal distribution with the given parameters:
*/

double mean = 3.; // kg
double standard_deviation = 0.1; // kg
normal packs(mean, standard_deviation);
/*`We can then find the fraction (or %) of packages that weigh more than 3.1 kg.
*/

double max_weight = 3.1; // kg
cout << "Percentage of packs > " << max_weight << " is "
<< cdf(complement(packs, max_weight)) * 100. << endl; // P(X > 3.1)

/*`We might want to ensure that 95% of packs are over a minimum weight specification,
then we want the value of the mean such that P(X < 2.9) = 0.05.

Using the mean of 3 kg, we can estimate
the fraction of packs that fail to meet the specification of 2.9 kg.
*/

double minimum_weight = 2.9;
cout <<"Fraction of packs <= " << minimum_weight << " with a mean of " << mean
  << " is " << cdf(complement(packs, minimum_weight)) << endl;
// fraction of packs <= 2.9 with a mean of 3 is 0.841345

/*`This is 0.84 - more than the target fraction of 0.95.
If we want 95% to be over the minimum weight,
what should we set the mean weight to be?

Using the KK StatCalc program supplied with the book and the method given on page 126 gives 3.06449.

We can confirm this by constructing a new distribution which we call 'xpacks'
with a safety margin mean of 3.06449 thus:
*/
double over_mean = 3.06449;
normal xpacks(over_mean, standard_deviation);
cout << "Fraction of packs >= " << minimum_weight
<< " with a mean of " << xpacks.mean()
  << " is " << cdf(complement(xpacks, minimum_weight)) << endl;
// fraction of packs >= 2.9 with a mean of 3.06449 is 0.950005

/*`Using this Math Toolkit, we can calculate the required mean directly thus:
*/
double under_fraction = 0.05;  // so 95% are above the minimum weight mean - sd = 2.9
double low_limit = standard_deviation;
double offset = mean - low_limit - quantile(packs, under_fraction);
double nominal_mean = mean + offset;
// mean + (mean - low_limit - quantile(packs, under_fraction));

normal nominal_packs(nominal_mean, standard_deviation);
cout << "Setting the packer to " << nominal_mean << " will mean that "
  << "fraction of packs >= " << minimum_weight
  << " is " << cdf(complement(nominal_packs, minimum_weight)) << endl;
// Setting the packer to 3.06449 will mean that fraction of packs >= 2.9 is 0.95

/*`
This calculation is generalized as the free function called `find_location`,
see __algorithms.

To use this we will need to
*/

#include <boost/math/distributions/find_location.hpp>
  using boost::math::find_location;
/*`and then use find_location function to find safe_mean,
& construct a new normal distribution called 'goodpacks'.
*/
double safe_mean = find_location<normal>(minimum_weight, under_fraction, standard_deviation);
normal good_packs(safe_mean, standard_deviation);
/*`with the same confirmation as before:
*/
cout << "Setting the packer to " << nominal_mean << " will mean that "
  << "fraction of packs >= " << minimum_weight
  << " is " << cdf(complement(good_packs, minimum_weight)) << endl;
// Setting the packer to 3.06449 will mean that fraction of packs >= 2.9 is 0.95

/*`
[h4 Using Cauchy-Lorentz instead of normal distribution]

After examining the weight distribution of a large number of packs, we might decide that,
after all, the assumption of a normal distribution is not really justified.
We might find that the fit is better to a __cauchy_distrib.
This distribution has wider 'wings', so that whereas most of the values
are closer to the mean than the normal, there are also more values than 'normal'
that lie further from the mean than the normal.

This might happen because a larger than normal lump of meat is either included or excluded.

We first create a __cauchy_distrib with the original mean and standard deviation,
and estimate the fraction that lie below our minimum weight specification.
*/

cauchy cpacks(mean, standard_deviation);
cout << "Cauchy Setting the packer to " << mean << " will mean that "
  << "fraction of packs >= " << minimum_weight
  << " is " << cdf(complement(cpacks, minimum_weight)) << endl;
// Cauchy Setting the packer to 3 will mean that fraction of packs >= 2.9 is 0.75

/*`Note that far fewer of the packs meet the specification, only 75% instead of 95%.
Now we can repeat the find_location, using the cauchy distribution as template parameter,
in place of the normal used above.
*/

double lc = find_location<cauchy>(minimum_weight, under_fraction, standard_deviation);
cout << "find_location<cauchy>(minimum_weight, over fraction, standard_deviation); " << lc << endl;
// find_location<cauchy>(minimum_weight, over fraction, packs.standard_deviation()); 3.53138
/*`Note that the safe_mean setting needs to be much higher, 3.53138 instead of 3.06449,
so we will make rather less profit.

And again confirm that the fraction meeting specification is as expected.
*/
cauchy goodcpacks(lc, standard_deviation);
cout << "Cauchy Setting the packer to " << lc << " will mean that "
  << "fraction of packs >= " << minimum_weight
  << " is " << cdf(complement(goodcpacks, minimum_weight)) << endl;
// Cauchy Setting the packer to 3.53138 will mean that fraction of packs >= 2.9 is 0.95

/*`Finally we could estimate the effect of a much tighter specification,
that 99% of packs met the specification.
*/

cout << "Cauchy Setting the packer to "
  << find_location<cauchy>(minimum_weight, 0.99, standard_deviation)
  << " will mean that "
  << "fraction of packs >= " << minimum_weight
  << " is " << cdf(complement(goodcpacks, minimum_weight)) << endl;

/*`Setting the packer to 3.13263 will mean that fraction of packs >= 2.9 is 0.99,
but will more than double the mean loss from 0.0644 to 0.133 kg per pack.

Of course, this calculation is not limited to packs of meat, it applies to dispensing anything,
and it also applies to a 'virtual' material like any measurement.

The only caveat is that the calculation assumes that the standard deviation (scale) is known with
a reasonably low uncertainty, something that is not so easy to ensure in practice.
And that the distribution is well defined, __normal_distrib or __cauchy_distrib, or some other.

If one is simply dispensing a very large number of packs,
then it may be feasible to measure the weight of hundreds or thousands of packs.
With a healthy 'degrees of freedom', the confidence intervals for the standard deviation
are not too wide, typically about + and - 10% for hundreds of observations.

For other applications, where it is more difficult or expensive to make many observations,
the confidence intervals are depressingly wide.

See [link math_toolkit.stat_tut.weg.cs_eg.chi_sq_intervals Confidence Intervals on the standard deviation]
for a worked example
[@../../example/chi_square_std_dev_test.cpp chi_square_std_dev_test.cpp]
of estimating these intervals.


[h4 Changing the scale or standard deviation]

Alternatively, we could invest in a better (more precise) packer
(or measuring device) with a lower standard deviation, or scale.

This might cost more, but would reduce the amount we have to 'give away'
in order to meet the specification.

To estimate how much better (how much smaller standard deviation) it would have to be,
we need to get the 5% quantile to be located at the under_weight limit, 2.9
*/
double p = 0.05; // wanted p th quantile.
cout << "Quantile of " << p << " = " << quantile(packs, p)
  << ", mean = " << packs.mean() << ", sd = " << packs.standard_deviation() << endl;
/*`
Quantile of 0.05 = 2.83551, mean = 3, sd = 0.1

With the current packer (mean = 3, sd = 0.1), the 5% quantile is at 2.8551 kg,
a little below our target of 2.9 kg.
So we know that the standard deviation is going to have to be smaller.

Let's start by guessing that it (now 0.1) needs to be halved, to a standard deviation of 0.05 kg.
*/
normal pack05(mean, 0.05);
cout << "Quantile of " << p << " = " << quantile(pack05, p)
  << ", mean = " << pack05.mean() << ", sd = " << pack05.standard_deviation() << endl;
// Quantile of 0.05 = 2.91776, mean = 3, sd = 0.05

cout <<"Fraction of packs >= " << minimum_weight << " with a mean of " << mean
  << " and standard deviation of " << pack05.standard_deviation()
  << " is " << cdf(complement(pack05, minimum_weight)) << endl;
// Fraction of packs >= 2.9 with a mean of 3 and standard deviation of 0.05 is 0.97725
/*`
So 0.05 was quite a good guess, but we are a little over the 2.9 target,
so the standard deviation could be a tiny bit more. So we could do some
more guessing to get closer, say by increasing standard deviation to 0.06 kg,
constructing another new distribution called pack06.
*/
normal pack06(mean, 0.06);
cout << "Quantile of " << p << " = " << quantile(pack06, p)
  << ", mean = " << pack06.mean() << ", sd = " << pack06.standard_deviation() << endl;
// Quantile of 0.05 = 2.90131, mean = 3, sd = 0.06

cout <<"Fraction of packs >= " << minimum_weight << " with a mean of " << mean
  << " and standard deviation of " << pack06.standard_deviation()
  << " is " << cdf(complement(pack06, minimum_weight)) << endl;
// Fraction of packs >= 2.9 with a mean of 3 and standard deviation of 0.06 is 0.95221
/*`
Now we are getting really close, but to do the job properly,
we might need to use root finding method, for example the tools provided,
and used elsewhere, in the Math Toolkit, see __root_finding_without_derivatives

But in this (normal) distribution case, we can and should be even smarter
and make a direct calculation.
*/

/*`Our required limit is minimum_weight = 2.9 kg, often called the random variate z.
For a standard normal distribution, then probability p = N((minimum_weight - mean) / sd).

We want to find the standard deviation that would be required to meet this limit,
so that the p th quantile is located at z (minimum_weight).
In this case, the 0.05 (5%) quantile is at 2.9 kg pack weight, when the mean is 3 kg,
ensuring that 0.95 (95%) of packs are above the minimum weight.

Rearranging, we can directly calculate the required standard deviation:
*/
normal N01; // standard normal distribution with mean zero and unit standard deviation.
p = 0.05;
double qp = quantile(N01, p);
double sd95 = (minimum_weight - mean) / qp;

cout << "For the "<< p << "th quantile to be located at "
  << minimum_weight << ", would need a standard deviation of " << sd95 << endl;
// For the 0.05th quantile to be located at 2.9, would need a standard deviation of 0.0607957

/*`We can now construct a new (normal) distribution pack95 for the 'better' packer,
and check that our distribution will meet the specification.
*/

normal pack95(mean, sd95);
cout <<"Fraction of packs >= " << minimum_weight << " with a mean of " << mean
  << " and standard deviation of " << pack95.standard_deviation()
  << " is " << cdf(complement(pack95, minimum_weight)) << endl;
// Fraction of packs >= 2.9 with a mean of 3 and standard deviation of 0.0607957 is 0.95

/*`This calculation is generalized in the free function find_scale,
as shown below, giving the same standard deviation.
*/
double ss = find_scale<normal>(minimum_weight, under_fraction, packs.mean());
cout << "find_scale<normal>(minimum_weight, under_fraction, packs.mean()); " << ss << endl;
// find_scale<normal>(minimum_weight, under_fraction, packs.mean()); 0.0607957

/*`If we had defined an over_fraction, or percentage that must pass specification
*/
double over_fraction = 0.95;
/*`And (wrongly) written

 double sso = find_scale<normal>(minimum_weight, over_fraction, packs.mean());

With the default policy, we would get a message like

[pre
Message from thrown exception was:
   Error in function boost::math::find_scale<Dist, Policy>(double, double, double, Policy):
   Computed scale (-0.060795683191176959) is <= 0! Was the complement intended?
]

But this would return a *negative* standard deviation - obviously impossible.
The probability should be 1 - over_fraction, not over_fraction, thus:
*/

double ss1o = find_scale<normal>(minimum_weight, 1 - over_fraction, packs.mean());
cout << "find_scale<normal>(minimum_weight, under_fraction, packs.mean()); " << ss1o << endl;
// find_scale<normal>(minimum_weight, under_fraction, packs.mean()); 0.0607957

/*`But notice that using '1 - over_fraction' - will lead to a
loss of accuracy, especially if over_fraction was close to unity. (See __why_complements).
In this (very common) case, we should instead use the __complements,
giving the most accurate result.
*/

double ssc = find_scale<normal>(complement(minimum_weight, over_fraction, packs.mean()));
cout << "find_scale<normal>(complement(minimum_weight, over_fraction, packs.mean())); " << ssc << endl;
// find_scale<normal>(complement(minimum_weight, over_fraction, packs.mean())); 0.0607957

/*`Note that our guess of 0.06 was close to the accurate value of 0.060795683191176959.

We can again confirm our prediction thus:
*/

normal pack95c(mean, ssc);
cout <<"Fraction of packs >= " << minimum_weight << " with a mean of " << mean
  << " and standard deviation of " << pack95c.standard_deviation()
  << " is " << cdf(complement(pack95c, minimum_weight)) << endl;
// Fraction of packs >= 2.9 with a mean of 3 and standard deviation of 0.0607957 is 0.95

/*`Notice that these two deceptively simple questions:

* Do we over-fill to make sure we meet a minimum specification (or under-fill to avoid an overdose)?

and/or

* Do we measure better?

are actually extremely common.

The weight of beef might be replaced by a measurement of more or less anything,
from drug tablet content, Apollo landing rocket firing, X-ray treatment doses...

The scale can be variation in dispensing or uncertainty in measurement.
*/
//] [/normal_find_location_and_scale_eg Quickbook end]

  }
  catch(const std::exception& e)
  { // Always useful to include try & catch blocks because default policies
    // are to throw exceptions on arguments that cause errors like underflow, overflow.
    // Lacking try & catch blocks, the program will abort without a message below,
    // which may give some helpful clues as to the cause of the exception.
    cout <<
      "\n""Message from thrown exception was:\n   " << e.what() << endl;
  }
  return 0;
}  // int main()


/*

Output is:

//[normal_find_location_and_scale_output

Find_location (mean) and find_scale (standard deviation) examples.
Percentage of packs > 3.1 is 15.8655
Fraction of packs <= 2.9 with a mean of 3 is 0.841345
Fraction of packs >= 2.9 with a mean of 3.06449 is 0.950005
Setting the packer to 3.06449 will mean that fraction of packs >= 2.9 is 0.95
Setting the packer to 3.06449 will mean that fraction of packs >= 2.9 is 0.95
Cauchy Setting the packer to 3 will mean that fraction of packs >= 2.9 is 0.75
find_location<cauchy>(minimum_weight, over fraction, standard_deviation); 3.53138
Cauchy Setting the packer to 3.53138 will mean that fraction of packs >= 2.9 is 0.95
Cauchy Setting the packer to -0.282052 will mean that fraction of packs >= 2.9 is 0.95
Quantile of 0.05 = 2.83551, mean = 3, sd = 0.1
Quantile of 0.05 = 2.91776, mean = 3, sd = 0.05
Fraction of packs >= 2.9 with a mean of 3 and standard deviation of 0.05 is 0.97725
Quantile of 0.05 = 2.90131, mean = 3, sd = 0.06
Fraction of packs >= 2.9 with a mean of 3 and standard deviation of 0.06 is 0.95221
For the 0.05th quantile to be located at 2.9, would need a standard deviation of 0.0607957
Fraction of packs >= 2.9 with a mean of 3 and standard deviation of 0.0607957 is 0.95
find_scale<normal>(minimum_weight, under_fraction, packs.mean()); 0.0607957
find_scale<normal>(minimum_weight, under_fraction, packs.mean()); 0.0607957
find_scale<normal>(complement(minimum_weight, over_fraction, packs.mean())); 0.0607957
Fraction of packs >= 2.9 with a mean of 3 and standard deviation of 0.0607957 is 0.95

//] [/normal_find_location_and_scale_eg_output]

*/



