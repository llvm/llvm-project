// arcsine_example.cpp

// Copyright John Maddock 2014.
// Copyright  Paul A. Bristow 2014.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Example for the arcsine Distribution.

// Note: Contains Quickbook snippets in comments.

//[arcsine_snip_1
#include <boost/math/distributions/arcsine.hpp> // For arcsine_distribution.
//] [/arcsine_snip_1]

#include <iostream>
#include <exception>
#include <boost/math/tools/assert.hpp>

int main()
{
  std::cout << "Examples of Arcsine distribution." << std::endl;
  std::cout.precision(3);  // Avoid uninformative decimal digits.

  using boost::math::arcsine;

  arcsine as; // Construct a default `double` standard [0, 1] arcsine distribution.

//[arcsine_snip_2
  std::cout << pdf(as, 1. / 2) << std::endl; // 0.637
  // pdf has a minimum at x = 0.5
//]  [/arcsine_snip_2]

//[arcsine_snip_3
  std::cout << pdf(as, 1. / 4) << std::endl; // 0.735
//]  [/arcsine_snip_3]


//[arcsine_snip_4
  std::cout << cdf(as, 0.05) << std::endl; // 0.144
//] [/arcsine_snip_4]

//[arcsine_snip_5
  std::cout << 2 * cdf(as, 1 - 0.975) << std::endl; // 0.202
//] [/arcsine_snip_5]


//[arcsine_snip_6
  std::cout << 2 * cdf(complement(as, 0.975)) << std::endl; // 0.202
//] [/arcsine_snip_6]

//[arcsine_snip_7
  std::cout << quantile(as, 1 - 0.2 / 2) << std::endl; //  0.976

  std::cout << quantile(complement(as, 0.2 / 2)) << std::endl; // 0.976
//] [/arcsine_snip_7]

{
//[arcsine_snip_8
  using boost::math::arcsine_distribution;

  arcsine_distribution<> as(2, 5); // Constructs a double arcsine distribution.
  BOOST_MATH_ASSERT(as.x_min() == 2.);  // as.x_min() returns 2.
  BOOST_MATH_ASSERT(as.x_max() == 5.);   // as.x_max()  returns 5.
//] [/arcsine_snip_8]
}
    return 0;

} // int main()

/*
[arcsine_output

Example of Arcsine distribution
0.637
0.735
0.144
0.202
0.202
0.976
0.976

] [/arcsine_output]
*/


