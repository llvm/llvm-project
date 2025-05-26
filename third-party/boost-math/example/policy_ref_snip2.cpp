//  Copyright John Maddock 2007.
//  Copyright Paul A. Bristow 2010
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

#include <iostream>
using std::cout;  using std::endl;

//[policy_ref_snip2

#include <boost/math/distributions/normal.hpp>
using boost::math::normal_distribution;

using namespace boost::math::policies;

// Define a specific policy:
typedef policy<
      overflow_error<ignore_error>
      > my_policy;
      
// Define the distribution, using my_policy:
typedef normal_distribution<double, my_policy> my_norm;

// Construct a my_norm distribution, using default mean and standard deviation,
// and get a 0.05 or 5% quantile:
double q = quantile(my_norm(), 0.05); // = -1.64485

//] //[/policy_ref_snip2]

int main()
{
  my_norm n; // Construct a my_norm distribution,
  // using default mean zero and standard deviation unity.
  double q = quantile(n, 0.05); // and get a quantile.
  cout << "quantile(my_norm(), 0.05) = " << q << endl;
}

/*

Output:

  quantile(my_norm(), 0.05) = -1.64485
*/
