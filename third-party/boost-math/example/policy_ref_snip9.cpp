//  Copyright John Maddock 2007.
//  Copyright Paul A. Bristow 2010

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

#include <iostream>
using std::cout; using std::endl;

//[policy_ref_snip9

#include <boost/math/special_functions/gamma.hpp>
using boost::math::tgamma;
using boost::math::policies::policy;
using boost::math::policies::digits10;

typedef policy<digits10<5> > my_pol_5; // Define a new, non-default, policy
// to calculate tgamma to accuracy of approximately 5 decimal digits.
//]

int main()
{
  cout.precision(5); // To only show 5 (hopefully) accurate decimal digits.
  double t = tgamma(12, my_pol_5()); // Apply the 5 decimal digits accuracy policy to use of tgamma.
  cout << "tgamma(12, my_pol_5() = " << t << endl;
}

/*

Output:
     tgamma(12, my_pol_5() = 3.9917e+007
*/
