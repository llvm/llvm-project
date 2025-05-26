//  Copyright John Maddock 2007.
//  Copyright Paul A. Bristow 2010
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

double some_value = 2.;

//[policy_ref_snip3

#include <boost/math/special_functions/gamma.hpp>

using namespace boost::math::policies;
using boost::math::tgamma;

// Define a new policy *not* internally promoting RealType to double:
typedef policy<
      promote_double<false> 
      > my_policy;
      
// Call the function, applying the new policy:
double t1 = tgamma(some_value, my_policy());

// Alternatively we could use helper function make_policy,
// and concisely define everything at the call site:
double t2 = tgamma(some_value, make_policy(promote_double<false>()));

//] //[\policy_ref_snip3]

#include <iostream>
using std::cout;  using std::endl;

int main()
{
   cout << "tgamma(some_value, my_policy()) = " << t1 
     << ", tgamma(some_value, make_policy(promote_double<false>()) = " << t2 << endl;
}
