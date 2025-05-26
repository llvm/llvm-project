// find_location.cpp

// Copyright Paul A. Bristow 2008, 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Example of finding location (mean)
// for normal (Gaussian) & Cauchy distribution.

// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

//#ifdef _MSC_VER
//#  pragma warning(disable: 4180) // qualifier has no effect (in Fusion).
//#endif

//[find_location1
/*`
First we need some includes to access the normal distribution,
the algorithms to find location (and some std output of course).
*/

#include <boost/math/distributions/normal.hpp> // for normal_distribution
  using boost::math::normal; // typedef provides default type is double.
#include <boost/math/distributions/cauchy.hpp> // for cauchy_distribution
  using boost::math::cauchy; // typedef provides default type is double.
#include <boost/math/distributions/find_location.hpp>
  using boost::math::find_location; // for mean
#include <boost/math/distributions/find_scale.hpp>
  using boost::math::find_scale; // for standard deviation
  using boost::math::complement; // Needed if you want to use the complement version.
  using boost::math::policies::policy;

#include <iostream>
  using std::cout; using std::endl;
#include <iomanip>
  using std::setw; using std::setprecision;
#include <limits>
  using std::numeric_limits;

//] [/find_location1]

int main()
{
  cout << "Example: Find location (or mean)." << endl;
  try
  {
//[find_location2
/*`
For this example, we will use the standard normal distribution,
with mean (location) zero and standard deviation (scale) unity.
This is also the default for this implementation.
*/
  normal N01;  // Default 'standard' normal distribution with zero mean and
  double sd = 1.; // normal default standard deviation is 1.
/*`Suppose we want to find a different normal distribution whose mean is shifted
so that only fraction p (here 0.001 or 0.1%) are below a certain chosen limit
(here -2, two standard deviations).
*/
  double z = -2.; // z to give prob p
  double p = 0.001; // only 0.1% below z

  cout << "Normal distribution with mean = " << N01.location()
    << ", standard deviation " << N01.scale()
    << ", has " << "fraction <= " << z
    << ", p = "  << cdf(N01, z) << endl;
  cout << "Normal distribution with mean = " << N01.location()
    << ", standard deviation " << N01.scale()
    << ", has " << "fraction > " << z
    << ", p = "  << cdf(complement(N01, z)) << endl; // Note: uses complement.
/*`
[pre
Normal distribution with mean = 0, standard deviation 1, has fraction <= -2, p = 0.0227501
Normal distribution with mean = 0, standard deviation 1, has fraction > -2, p = 0.97725
]
We can now use ''find_location'' to give a new offset mean.
*/
   double l = find_location<normal>(z, p, sd);
   cout << "offset location (mean) = " << l << endl;
/*`
that outputs:
[pre
offset location (mean) = 1.09023
]
showing that we need to shift the mean just over one standard deviation from its previous value of zero.

Then we can check that we have achieved our objective
by constructing a new distribution
with the offset mean (but same standard deviation):
*/
  normal np001pc(l, sd); // Same standard_deviation (scale) but with mean (location) shifted.
/*`
And re-calculating the fraction below our chosen limit.
*/
cout << "Normal distribution with mean = " << l
    << " has " << "fraction <= " << z
    << ", p = "  << cdf(np001pc, z) << endl;
  cout << "Normal distribution with mean = " << l
    << " has " << "fraction > " << z
    << ", p = "  << cdf(complement(np001pc, z)) << endl;
/*`
[pre
Normal distribution with mean = 1.09023 has fraction <= -2, p = 0.001
Normal distribution with mean = 1.09023 has fraction > -2, p = 0.999
]

[h4 Controlling Error Handling from find_location]
We can also control the policy for handling various errors.
For example, we can define a new (possibly unwise)
policy to ignore domain errors ('bad' arguments).

Unless we are using the boost::math namespace, we will need:
*/
  using boost::math::policies::policy;
  using boost::math::policies::domain_error;
  using boost::math::policies::ignore_error;

/*`
Using a typedef is often convenient, especially if it is re-used,
although it is not required, as the various examples below show.
*/
  typedef policy<domain_error<ignore_error> > ignore_domain_policy;
  // find_location with new policy, using typedef.
  l = find_location<normal>(z, p, sd, ignore_domain_policy());
  // Default policy policy<>, needs "using boost::math::policies::policy;"
  l = find_location<normal>(z, p, sd, policy<>());
  // Default policy, fully specified.
  l = find_location<normal>(z, p, sd, boost::math::policies::policy<>());
  // A new policy, ignoring domain errors, without using a typedef.
  l = find_location<normal>(z, p, sd, policy<domain_error<ignore_error> >());
/*`
If we want to use a probability that is the __complements of our probability,
we should not even think of writing `find_location<normal>(z, 1 - p, sd)`,
but use the complement version, see __why_complements.
*/
  z = 2.;
  double q = 0.95; // = 1 - p; // complement.
  l = find_location<normal>(complement(z, q, sd));

  normal np95pc(l, sd); // Same standard_deviation (scale) but with mean(location) shifted
  cout << "Normal distribution with mean = " << l << " has "
    << "fraction <= " << z << " = "  << cdf(np95pc, z) << endl;
  cout << "Normal distribution with mean = " << l << " has "
    << "fraction > " << z << " = "  << cdf(complement(np95pc, z)) << endl;
  //] [/find_location2]
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

//[find_location_example_output
/*`
[pre
Example: Find location (mean).
Normal distribution with mean = 0, standard deviation 1, has fraction <= -2, p = 0.0227501
Normal distribution with mean = 0, standard deviation 1, has fraction > -2, p = 0.97725
offset location (mean) = 1.09023
Normal distribution with mean = 1.09023 has fraction <= -2, p = 0.001
Normal distribution with mean = 1.09023 has fraction > -2, p = 0.999
Normal distribution with mean = 0.355146 has fraction <= 2 = 0.95
Normal distribution with mean = 0.355146 has fraction > 2 = 0.05
]
*/
//] [/find_location_example_output]
