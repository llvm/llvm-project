//  Copyright John Maddock 2007.
//  Copyright Paul A. Bristow 2007, 2010.

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
# pragma warning (disable : 4305) // 'initializing' : truncation from 'long double' to 'const eval_type'
# pragma warning (disable : 4244) //  conversion from 'long double' to 'const eval_type'
#endif

#include <iostream>
using std::cout; using std::endl;

//[policy_eg_3

#include <boost/math/distributions/binomial.hpp>
using boost::math::binomial_distribution;

// Begin by defining a policy type, that gives the behaviour we want:

//using namespace boost::math::policies; or explicitly
using boost::math::policies::policy;

using boost::math::policies::promote_float;
using boost::math::policies::discrete_quantile;
using boost::math::policies::integer_round_nearest;

typedef policy<
   promote_float<false>, // Do not promote to double.
   discrete_quantile<integer_round_nearest> // Round result to nearest integer.
> mypolicy;
//
// Then define a new distribution that uses it:
typedef boost::math::binomial_distribution<float, mypolicy> mybinom;

//  And now use it to get the quantile:

int main()
{
   cout << "quantile(mybinom(200, 0.25), 0.05) is: " <<
      quantile(mybinom(200, 0.25), 0.05) << endl;
}

//]

/*

Output:

  quantile(mybinom(200, 0.25), 0.05) is: 40

*/

