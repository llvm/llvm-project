//  Copyright John Maddock 2007.
//  Copyright Paul a. Bristow 2010
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

#ifdef _MSC_VER
# pragma warning (disable : 4100) // unreferenced formal parameters
#endif

#include <iostream>
using std::cout;  using std::endl; using std::cerr;

//[policy_eg_8

/*`
Suppose we want our own user-defined error handlers rather than the
any of the default ones supplied by the library to be used.
If we set the policy for a specific type of error to `user_error`
then the library will call a user-supplied error handler.
These are forward declared, but not defined in
boost/math/policies/error_handling.hpp like this:

   namespace boost{ namespace math{ namespace policies{

   template <class T>
   T user_domain_error(const char* function, const char* message, const T& val);
   template <class T>
   T user_pole_error(const char* function, const char* message, const T& val);
   template <class T>
   T user_overflow_error(const char* function, const char* message, const T& val);
   template <class T>
   T user_underflow_error(const char* function, const char* message, const T& val);
   template <class T>
   T user_denorm_error(const char* function, const char* message, const T& val);
   template <class T>
   T user_evaluation_error(const char* function, const char* message, const T& val);
   template <class T, class TargetType>
   TargetType user_rounding_error(const char* function, const char* message, const T& val, const TargetType& t);
   template <class T>
   T user_indeterminate_result_error(const char* function, const char* message, const T& val);

   }}} // namespaces

So out first job is to include the header we want to use, and then
provide definitions for our user-defined error handlers that we want to use.
We only provide our special domain and pole error handlers;
other errors like overflow and underflow use the default.
*/

#include <boost/math/special_functions.hpp>

namespace boost{ namespace math
{
  namespace policies
  {
    template <class T>
    T user_domain_error(const char* function, const char* message, const T& val)
    { // Ignoring function, message and val for this example, perhaps unhelpfully.
       cerr << "Domain Error!" << endl;
       return std::numeric_limits<T>::quiet_NaN();
    }

    template <class T>
    T user_pole_error(const char* function, const char* message, const T& val)
    { // Ignoring function, message and val for this example, perhaps unhelpfully.
       cerr << "Pole Error!" << endl;
       return std::numeric_limits<T>::quiet_NaN();
    }
  } // namespace policies
}} // namespace boost{ namespace math


/*`
Now we'll need to define a suitable policy that will call these handlers,
and define some forwarding functions that make use of the policy:
*/

namespace mymath{

using namespace boost::math::policies;

typedef policy<
   domain_error<user_error>,
   pole_error<user_error>
> user_error_policy;

BOOST_MATH_DECLARE_SPECIAL_FUNCTIONS(user_error_policy)

} // close unnamed namespace

/*`
We now have a set of forwarding functions defined in namespace mymath
that all look something like this:

``
template <class RealType>
inline typename boost::math::tools::promote_args<RT>::type
   tgamma(RT z)
{
   return boost::math::tgamma(z, user_error_policy());
}
``

So that when we call `mymath::tgamma(z)` we really end up calling
`boost::math::tgamma(z, user_error_policy())`, and any
errors will get directed to our own error handlers.
*/

int main()
{
   cout << "Result of erf_inv(-10) is: "
      << mymath::erf_inv(-10) << endl;
   cout << "Result of tgamma(-10) is: "
      << mymath::tgamma(-10) << endl;
}

/*`

Which outputs:

[pre
  Domain Error!
  Pole Error!
  Result of erf_inv(-10) is: 1.#QNAN
  Result of tgamma(-10) is: 1.#QNAN
]
*/

//] // //[/policy_eg_8]
