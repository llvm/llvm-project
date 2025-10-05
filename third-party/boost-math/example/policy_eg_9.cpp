//  Copyright John Maddock 2007.
//  Copyright Paul A. Bristow 2010
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

#include <iostream>
#include <boost/format.hpp>
using std::cout; using std::endl; using std::cerr;

//[policy_eg_9

/*`
The previous example was all well and good, but the custom error handlers
didn't really do much of any use.  In this example we'll implement all
the custom handlers and show how the information provided to them can be
used to generate nice formatted error messages.

Each error handler has the general form:

   template <class T>
   T user_``['error_type]``(
      const char* function,
      const char* message,
      const T& val);

and accepts three arguments:

[variablelist
[[const char* function]
   [The name of the function that raised the error, this string
   contains one or more %1% format specifiers that should be
   replaced by the name of real type T, like float or double.]]
[[const char* message]
   [A message associated with the error, normally this
   contains a %1% format specifier that should be replaced with
   the value of ['value]: however note that overflow and underflow messages
   do not contain this %1% specifier (since the value of ['value] is
   immaterial in these cases).]]
[[const T& value]
   [The value that caused the error: either an argument to the function
   if this is a domain or pole error, the tentative result
   if this is a denorm or evaluation error, or zero or infinity for
   underflow or overflow errors.]]
]

As before we'll include the headers we need first:

*/

#include <boost/math/special_functions.hpp>

/*`
Next we'll implement our own error handlers for each type of error,
starting with domain errors:
*/

namespace boost{ namespace math{
namespace policies
{

template <class T>
T user_domain_error(const char* function, const char* message, const T& val)
{
   /*`
   We'll begin with a bit of defensive programming in case function or message are empty:
   */
   if(function == 0)
       function = "Unknown function with arguments of type %1%";
   if(message == 0)
       message = "Cause unknown with bad argument %1%";
   /*`
   Next we'll format the name of the function with the name of type T, perhaps double:
   */
   std::string msg("Error in function ");
   msg += (boost::format(function) % typeid(T).name()).str();
   /*`
   Then likewise format the error message with the value of parameter /val/,
   making sure we output all the potentially significant digits of /val/:
   */
   msg += ": \n";
   int prec = 2 + (std::numeric_limits<T>::digits * 30103UL) / 100000UL;
   // int prec = std::numeric_limits<T>::max_digits10; //  For C++0X Standard Library
   msg += (boost::format(message) % boost::io::group(std::setprecision(prec), val)).str();
   /*`
   Now we just have to do something with the message, we could throw an
   exception, but for the purposes of this example we'll just dump the message
   to std::cerr:
   */
   std::cerr << msg << std::endl;
   /*`
   Finally the only sensible value we can return from a domain error is a NaN:
   */
   return std::numeric_limits<T>::quiet_NaN();
}

/*`
Pole errors are essentially a special case of domain errors,
so in this example we'll just return the result of a domain error:
*/

template <class T>
T user_pole_error(const char* function, const char* message, const T& val)
{
   return user_domain_error(function, message, val);
}

/*`
Overflow errors are very similar to domain errors, except that there's
no %1% format specifier in the /message/ parameter:
*/
template <class T>
T user_overflow_error(const char* function, const char* message, const T& val)
{
   if(function == 0)
       function = "Unknown function with arguments of type %1%";
   if(message == 0)
       message = "Result of function is too large to represent";

   std::string msg("Error in function ");
   msg += (boost::format(function) % typeid(T).name()).str();

   msg += ": \n";
   msg += message;

   std::cerr << msg << std::endl;

   // Value passed to the function is an infinity, just return it:
   return val;
}

/*`
Underflow errors are much the same as overflow:
*/

template <class T>
T user_underflow_error(const char* function, const char* message, const T& val)
{
   if(function == 0)
       function = "Unknown function with arguments of type %1%";
   if(message == 0)
       message = "Result of function is too small to represent";

   std::string msg("Error in function ");
   msg += (boost::format(function) % typeid(T).name()).str();

   msg += ": \n";
   msg += message;

   std::cerr << msg << std::endl;

   // Value passed to the function is zero, just return it:
   return val;
}

/*`
Denormalised results are much the same as underflow:
*/

template <class T>
T user_denorm_error(const char* function, const char* message, const T& val)
{
   if(function == 0)
       function = "Unknown function with arguments of type %1%";
   if(message == 0)
       message = "Result of function is denormalised";

   std::string msg("Error in function ");
   msg += (boost::format(function) % typeid(T).name()).str();

   msg += ": \n";
   msg += message;

   std::cerr << msg << std::endl;

   // Value passed to the function is denormalised, just return it:
   return val;
}

/*`
Which leaves us with evaluation errors: these occur when an internal
error occurs that prevents the function being fully evaluated.
The parameter /val/ contains the closest approximation to the result
found so far:
*/

template <class T>
T user_evaluation_error(const char* function, const char* message, const T& val)
{
   if(function == 0)
       function = "Unknown function with arguments of type %1%";
   if(message == 0)
       message = "An internal evaluation error occurred with "
                  "the best value calculated so far of %1%";

   std::string msg("Error in function ");
   msg += (boost::format(function) % typeid(T).name()).str();

   msg += ": \n";
   int prec = 2 + (std::numeric_limits<T>::digits * 30103UL) / 100000UL;
   // int prec = std::numeric_limits<T>::max_digits10; // For C++0X Standard Library
   msg += (boost::format(message) % boost::io::group(std::setprecision(prec), val)).str();

   std::cerr << msg << std::endl;

   // What do we return here?  This is generally a fatal error, that should never occur,
   // so we just return a NaN for the purposes of the example:
   return std::numeric_limits<T>::quiet_NaN();
}

} // policies
}} // boost::math


/*`
Now we'll need to define a suitable policy that will call these handlers,
and define some forwarding functions that make use of the policy:
*/

namespace mymath
{ // unnamed.

using namespace boost::math::policies;

typedef policy<
   domain_error<user_error>,
   pole_error<user_error>,
   overflow_error<user_error>,
   underflow_error<user_error>,
   denorm_error<user_error>,
   evaluation_error<user_error>
> user_error_policy;

BOOST_MATH_DECLARE_SPECIAL_FUNCTIONS(user_error_policy)

} // unnamed namespace

/*`
We now have a set of forwarding functions, defined in namespace mymath,
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
errors will get directed to our own error handlers:
*/

int main()
{
   // Raise a domain error:
   cout << "Result of erf_inv(-10) is: "
      << mymath::erf_inv(-10) << std::endl << endl;
   // Raise a pole error:
   cout << "Result of tgamma(-10) is: "
      << mymath::tgamma(-10) << std::endl << endl;
   // Raise an overflow error:
   cout << "Result of tgamma(3000) is: "
      << mymath::tgamma(3000) << std::endl << endl;
   // Raise an underflow error:
   cout << "Result of tgamma(-190.5) is: "
      << mymath::tgamma(-190.5) << std::endl << endl;
   // Unfortunately we can't predictably raise a denormalised
   // result, nor can we raise an evaluation error in this example
   // since these should never really occur!
} // int main()

/*`

Which outputs:

[pre
Error in function boost::math::erf_inv<double>(double, double):
Argument outside range \[-1, 1\] in inverse erf function (got p=-10).
Result of erf_inv(-10) is: 1.#QNAN

Error in function boost::math::tgamma<long double>(long double):
Evaluation of tgamma at a negative integer -10.
Result of tgamma(-10) is: 1.#QNAN

Error in function boost::math::tgamma<long double>(long double):
Result of tgamma is too large to represent.
Error in function boost::math::tgamma<double>(double):
Result of function is too large to represent
Result of tgamma(3000) is: 1.#INF

Error in function boost::math::tgamma<long double>(long double):
Result of tgamma is too large to represent.
Error in function boost::math::tgamma<long double>(long double):
Result of tgamma is too small to represent.
Result of tgamma(-190.5) is: 0
]

Notice how some of the calls result in an error handler being called more
than once, or for more than one handler to be called: this is an artefact
of the fact that many functions are implemented in terms of one or more
sub-routines each of which may have it's own error handling.  For example
`tgamma(-190.5)` is implemented in terms of `tgamma(190.5)` - which overflows -
the reflection formula for `tgamma` then notices that it is dividing by
infinity and so underflows.
*/

//] //[/policy_eg_9]
