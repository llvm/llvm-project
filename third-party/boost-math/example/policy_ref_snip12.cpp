//  Copyright John Maddock 2007.
//  Copyright Paul A. Bristow 2010
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

// Define tgamma function with a no overflow policy
// into a specific namespace-scope.

#include <iostream>
using std::cout; using std::endl;

//[policy_ref_snip12

#include <boost/math/special_functions/gamma.hpp>
//using boost::math::tgamma;
// Need not declare using boost::math::tgamma here,
// because will define tgamma in myspace using macro below.

namespace myspace
{
  using namespace boost::math::policies;

  // Define a policy that does not throw on overflow:
  typedef policy<overflow_error<errno_on_error> > my_policy;

  // Define the special functions in this scope to use the policy:   
  BOOST_MATH_DECLARE_SPECIAL_FUNCTIONS(my_policy)
}

// Now we can use myspace::tgamma etc.
// They will automatically use "my_policy":
//
double t = myspace::tgamma(30.0); // Will *not* throw on overflow,
// despite the large value of factorial 30 = 265252859812191058636308480000000
// unlike default policy boost::math::tgamma;

//]

int main()
{
   cout << "myspace::tgamma(30.0) = " << t << endl;
}

/*

Output:

myspace::tgamma(30.0) = 8.84176e+030

*/

