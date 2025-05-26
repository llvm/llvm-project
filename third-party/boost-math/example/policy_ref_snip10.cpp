//  Copyright John Maddock 2007.
//  Copyright Paul A. Bristow 2010
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

// Setting precision in a single function call using make_policy.

#include <iostream>
using std::cout; using std::endl;

//[policy_ref_snip10

#include <boost/math/special_functions/gamma.hpp>
using boost::math::tgamma;

using namespace boost::math::policies;

double t = tgamma(12, policy<digits10<5> >());  // Concise make_policy.

//] //[/policy_ref_snip10]



int main()
{
   cout << "tgamma(12, policy<digits10<5> >())  = "<< t << endl;
}

/*

Output:

 tgamma(12, policy<digits10<5> >())  = 3.99168e+007

*/

