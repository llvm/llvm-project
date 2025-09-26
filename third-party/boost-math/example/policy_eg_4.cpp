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

//[policy_eg_4

/*`
Suppose we want `C::foo()` to behave in a C-compatible way and set
`::errno` on error rather than throwing any exceptions.

We'll begin by including the needed header for our function:
*/

#include <boost/math/special_functions.hpp>
//using boost::math::tgamma; // Not needed because using C::tgamma.

/*`
Open up the "C" namespace that we'll use for our functions, and
define the policy type we want: in this case a C-style one that sets
::errno and returns a standard value, rather than throwing exceptions.

Any policies we don't specify here will inherit the defaults.
*/

namespace C
{ // To hold our C-style policy.
  //using namespace boost::math::policies; or explicitly:
  using boost::math::policies::policy;

  using boost::math::policies::domain_error;
  using boost::math::policies::pole_error;
  using boost::math::policies::overflow_error;
  using boost::math::policies::evaluation_error;
  using boost::math::policies::errno_on_error;

  typedef policy<
     domain_error<errno_on_error>,
     pole_error<errno_on_error>,
     overflow_error<errno_on_error>,
     evaluation_error<errno_on_error>
  > c_policy;

/*`
All we need do now is invoke the BOOST_MATH_DECLARE_SPECIAL_FUNCTIONS
macro passing our policy type c_policy as the single argument:
*/

BOOST_MATH_DECLARE_SPECIAL_FUNCTIONS(c_policy)

} // close namespace C

/*`
We now have a set of forwarding functions defined in namespace C
that all look something like this:

``
template <class RealType>
inline typename boost::math::tools::promote_args<RT>::type
   tgamma(RT z)
{
   return boost::math::tgamma(z, c_policy());
}
``

So that when we call `C::tgamma(z)`, we really end up calling
`boost::math::tgamma(z, C::c_policy())`:
*/

int main()
{
   errno = 0;
   cout << "Result of tgamma(30000) is: "
      << C::tgamma(30000) << endl; // Note using C::tgamma
   cout << "errno = " << errno << endl; // errno = 34
   cout << "Result of tgamma(-10) is: "
      << C::tgamma(-10) << endl;
   cout << "errno = " << errno << endl; // errno = 33, overwriting previous value of 34.
}

/*`

Which outputs:

[pre
Result of C::tgamma(30000) is: 1.#INF
errno = 34
Result of C::tgamma(-10) is: 1.#QNAN
errno = 33
]

This mechanism is particularly useful when we want to define a project-wide policy,
and don't want to modify the Boost source,
or to set project wide build macros (possibly fragile and easy to forget).

*/
//] //[/policy_eg_4]

