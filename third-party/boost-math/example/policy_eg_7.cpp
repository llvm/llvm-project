//  Copyright John Maddock 2007.
//  Copyright Paul A. Bristow 2010
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

#include <iostream>
using std::cout;  using std::endl;
#include <cerrno> // for ::errno

//[policy_eg_7

#include <boost/math/distributions.hpp> // All distributions.
// using boost::math::normal; // Would create an ambiguity between
// boost::math::normal_distribution<RealType> boost::math::normal and
// 'anonymous-namespace'::normal'.

namespace
{ // anonymous or unnamed (rather than named as in policy_eg_6.cpp).

  using namespace boost::math::policies;
   // using boost::math::policies::errno_on_error; // etc.
  typedef policy<
     // return infinity and set errno rather than throw:
     overflow_error<errno_on_error>,
     // Don't promote double -> long double internally:
     promote_double<false>,
     // Return the closest integer result for discrete quantiles:
     discrete_quantile<integer_round_nearest>
  > my_policy;

  BOOST_MATH_DECLARE_DISTRIBUTIONS(double, my_policy)

} // close namespace my_namespace

int main()
{
   // Construct distribution with something we know will overflow.
   normal norm(10, 2); // using 'anonymous-namespace'::normal
   errno = 0;
   cout << "Result of quantile(norm, 0) is: " 
      << quantile(norm, 0) << endl;
   cout << "errno = " << errno << endl;
   errno = 0;
   cout << "Result of quantile(norm, 1) is: " 
      << quantile(norm, 1) << endl;
   cout << "errno = " << errno << endl;
   //
   // Now try a discrete distribution:
   binomial binom(20, 0.25);
   cout << "Result of quantile(binom, 0.05) is: " 
      << quantile(binom, 0.05) << endl;
   cout << "Result of quantile(complement(binom, 0.05)) is: " 
      << quantile(complement(binom, 0.05)) << endl;
}

//] //[/policy_eg_7]

/*

Output:

  Result of quantile(norm, 0) is: -1.#INF
  errno = 34
  Result of quantile(norm, 1) is: 1.#INF
  errno = 34
  Result of quantile(binom, 0.05) is: 1
  Result of quantile(complement(binom, 0.05)) is: 8

*/
