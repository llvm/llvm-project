// find_scale.cpp

// Copyright Paul A. Bristow 2007, 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Example of finding scale (standard deviation) for normal (Gaussian).

// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

//[find_scale1
/*`
First we need some includes to access the __normal_distrib,
the algorithms to find scale (and some std output of course).
*/

#include <boost/math/distributions/normal.hpp> // for normal_distribution
  using boost::math::normal; // typedef provides default type is double.
#include <boost/math/distributions/find_scale.hpp>
  using boost::math::find_scale;
  using boost::math::complement; // Needed if you want to use the complement version.
  using boost::math::policies::policy; // Needed to specify the error handling policy.

#include <iostream>
  using std::cout; using std::endl;
#include <iomanip>
  using std::setw; using std::setprecision;
#include <limits>
  using std::numeric_limits;
//] [/find_scale1]

int main()
{
  cout << "Example: Find scale (standard deviation)." << endl;
  try
  {
//[find_scale2
/*`
For this example, we will use the standard __normal_distrib,
with location (mean) zero and standard deviation (scale) unity.
Conveniently, this is also the default for this implementation's constructor.
*/
  normal N01;  // Default 'standard' normal distribution with zero mean
  double sd = 1.; // and standard deviation is 1.
/*`Suppose we want to find a different normal distribution with standard deviation
so that only fraction p (here 0.001 or 0.1%) are below a certain chosen limit
(here -2. standard deviations).
*/
  double z = -2.; // z to give prob p
  double p = 0.001; // only 0.1% below z = -2

  cout << "Normal distribution with mean = " << N01.location()  // aka N01.mean()
    << ", standard deviation " << N01.scale() // aka N01.standard_deviation()
    << ", has " << "fraction <= " << z
    << ", p = "  << cdf(N01, z) << endl;
  cout << "Normal distribution with mean = " << N01.location()
    << ", standard deviation " << N01.scale()
    << ", has " << "fraction > " << z
    << ", p = "  << cdf(complement(N01, z)) << endl; // Note: uses complement.
/*`
[pre
Normal distribution with mean = 0 has fraction <= -2, p = 0.0227501
Normal distribution with mean = 0 has fraction > -2, p = 0.97725
]
Noting that p = 0.02 instead of our target of 0.001,
we can now use `find_scale` to give a new standard deviation.
*/
   double l = N01.location();
   double s = find_scale<normal>(z, p, l);
   cout << "scale (standard deviation) = " << s << endl;
/*`
that outputs:
[pre
scale (standard deviation) = 0.647201
]
showing that we need to reduce the standard deviation from 1. to 0.65.

Then we can check that we have achieved our objective
by constructing a new distribution
with the new standard deviation (but same zero mean):
*/
  normal np001pc(N01.location(), s);
/*`
And re-calculating the fraction below (and above) our chosen limit.
*/
  cout << "Normal distribution with mean = " << l
    << " has " << "fraction <= " << z
    << ", p = "  << cdf(np001pc, z) << endl;
  cout << "Normal distribution with mean = " << l
    << " has " << "fraction > " << z
    << ", p = "  << cdf(complement(np001pc, z)) << endl;
/*`
[pre
Normal distribution with mean = 0 has fraction <= -2, p = 0.001
Normal distribution with mean = 0 has fraction > -2, p = 0.999
]

[h4 Controlling how Errors from find_scale are handled]
We can also control the policy for handling various errors.
For example, we can define a new (possibly unwise)
policy to ignore domain errors ('bad' arguments).

Unless we are using the boost::math namespace, we will need:
*/
  using boost::math::policies::policy;
  using boost::math::policies::domain_error;
  using boost::math::policies::ignore_error;

/*`
Using a typedef is convenient, especially if it is re-used,
although it is not required, as the various examples below show.
*/
  typedef policy<domain_error<ignore_error> > ignore_domain_policy;
  // find_scale with new policy, using typedef.
  l = find_scale<normal>(z, p, l, ignore_domain_policy());
  // Default policy policy<>, needs using boost::math::policies::policy;

  l = find_scale<normal>(z, p, l, policy<>());
  // Default policy, fully specified.
  l = find_scale<normal>(z, p, l, boost::math::policies::policy<>());
  // New policy, without typedef.
  l = find_scale<normal>(z, p, l, policy<domain_error<ignore_error> >());
/*`
If we want to express a probability, say 0.999, that is a complement, `1 - p`
we should not even think of writing `find_scale<normal>(z, 1 - p, l)`,
but use the __complements version (see __why_complements).
*/
  z = -2.;
  double q = 0.999; // = 1 - p; // complement of 0.001.
  sd = find_scale<normal>(complement(z, q, l));

  normal np95pc(l, sd); // Same standard_deviation (scale) but with mean(scale) shifted
  cout << "Normal distribution with mean = " << l << " has "
    << "fraction <= " << z << " = "  << cdf(np95pc, z) << endl;
  cout << "Normal distribution with mean = " << l << " has "
    << "fraction > " << z << " = "  << cdf(complement(np95pc, z)) << endl;

/*`
Sadly, it is all too easy to get probabilities the wrong way round,
when you may get a warning like this:
[pre
Message from thrown exception was:
   Error in function boost::math::find_scale<Dist, Policy>(complement(double, double, double, Policy)):
   Computed scale (-0.48043523852179076) is <= 0! Was the complement intended?
]
The default error handling policy is to throw an exception with this message,
but if you chose a policy to ignore the error,
the (impossible) negative scale is quietly returned.
*/
//] [/find_scale2]
  }
  catch(const std::exception& e)
  { // Always useful to include try & catch blocks because default policies
    // are to throw exceptions on arguments that cause errors like underflow, overflow.
    // Lacking try & catch blocks, the program will abort without a message below,
    // which may give some helpful clues as to the cause of the exception.
    std::cout <<
      "\n""Message from thrown exception was:\n   " << e.what() << std::endl;
  }
  return 0;
}  // int main()

//[find_scale_example_output
/*`
[pre
Example: Find scale (standard deviation).
Normal distribution with mean = 0, standard deviation 1, has fraction <= -2, p = 0.0227501
Normal distribution with mean = 0, standard deviation 1, has fraction > -2, p = 0.97725
scale (standard deviation) = 0.647201
Normal distribution with mean = 0 has fraction <= -2, p = 0.001
Normal distribution with mean = 0 has fraction > -2, p = 0.999
Normal distribution with mean = 0.946339 has fraction <= -2 = 0.001
Normal distribution with mean = 0.946339 has fraction > -2 = 0.999
]
*/
//] [/find_scale_example_output]
