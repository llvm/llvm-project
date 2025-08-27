// example_error_handling.cpp

// Copyright Paul A. Bristow 2007, 2010.
// Copyright John Maddock 2007.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Note that this file contains quickbook markup as well as code
// and comments, don't change any of the special comment markups!

// Optional macro definitions described in text below:
//   #define BOOST_MATH_DOMAIN_ERROR_POLICY ignore_error
//   #define BOOST_MATH_DOMAIN_ERROR_POLICY errno_on_error
//   #define BOOST_MATH_DOMAIN_ERROR_POLICY is set to: throw_on_error

//[error_handling_example
/*`
The following example demonstrates the effect of
setting the macro BOOST_MATH_DOMAIN_ERROR_POLICY
when an invalid argument is encountered.  For the
purposes of this example, we'll pass a negative
degrees of freedom parameter to the student's t
distribution.

Since we know that this is a single file program we could
just add:

   #define BOOST_MATH_DOMAIN_ERROR_POLICY ignore_error

to the top of the source file to change the default policy
to one that simply returns a NaN when a domain error occurs.
Alternatively we could use:

   #define BOOST_MATH_DOMAIN_ERROR_POLICY errno_on_error

To ensure the `::errno` is set when a domain error occurs
as well as returning a NaN.

This is safe provided the program consists of a single
translation unit /and/ we place the define /before/ any
#includes.  Note that should we add the define after the includes
then it will have no effect!  A warning such as:

[pre warning C4005: 'BOOST_MATH_OVERFLOW_ERROR_POLICY' : macro redefinition]

is a certain sign that it will /not/ have the desired effect.

We'll begin our sample program with the needed includes:
*/


   #define BOOST_MATH_DOMAIN_ERROR_POLICY ignore_error

// Boost
#include <boost/math/distributions/students_t.hpp>
   using boost::math::students_t;  // Probability of students_t(df, t).

// std
#include <iostream>
   using std::cout;
   using std::endl;

#include <stdexcept>


#include <cstddef>
   // using ::errno

/*`
Next we'll define the program's main() to call the student's t
distribution with an invalid degrees of freedom parameter,
the program is set up to handle either an exception or a NaN:
*/

int main()
{
   cout << "Example error handling using Student's t function. " << endl;
   cout << "BOOST_MATH_DOMAIN_ERROR_POLICY is set to: "
      << BOOST_MATH_STRINGIZE(BOOST_MATH_DOMAIN_ERROR_POLICY) << endl;

   double degrees_of_freedom = -1; // A bad argument!
   double t = 10;

   try
   {
      errno = 0; // Clear/reset.
      students_t dist(degrees_of_freedom); // exception is thrown here if enabled.
      double p = cdf(dist, t);
      // Test for error reported by other means:
      if((boost::math::isnan)(p))
      {
         cout << "cdf returned a NaN!" << endl;
         if (errno != 0)
         { // So errno has been set.
           cout << "errno is set to: " << errno << endl;
         }
      }
      else
         cout << "Probability of Student's t is " << p << endl;
   }
   catch(const std::exception& e)
   {
      std::cout <<
         "\n""Message from thrown exception was:\n   " << e.what() << std::endl;
   }
   return 0;
} // int main()

/*`

Here's what the program output looks like with a default build
(one that *does throw exceptions*):

[pre
Example error handling using Student's t function.
BOOST_MATH_DOMAIN_ERROR_POLICY is set to: throw_on_error

Message from thrown exception was:
   Error in function boost::math::students_t_distribution<double>::students_t_distribution:
   Degrees of freedom argument is -1, but must be > 0 !
]

Alternatively let's build with:

   #define BOOST_MATH_DOMAIN_ERROR_POLICY ignore_error

Now the program output is:

[pre
Example error handling using Student's t function.
BOOST_MATH_DOMAIN_ERROR_POLICY is set to: ignore_error
cdf returned a NaN!
]

And finally let's build with:

   #define BOOST_MATH_DOMAIN_ERROR_POLICY errno_on_error

Which gives the output show errno:

[pre
Example error handling using Student's t function.
BOOST_MATH_DOMAIN_ERROR_POLICY is set to: errno_on_error
cdf returned a NaN!
errno is set to: 33
]

*/

//] [error_handling_eg end quickbook markup]
