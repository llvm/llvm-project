// example_policy_handling.cpp

// Copyright Paul A. Bristow 2007, 2010.
// Copyright John Maddock 2007.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// See error_handling_example.cpp for use of
// macro definition to change policy for
// domain_error - negative degrees of freedom argument
// for student's t distribution CDF,
// and catching the exception.

// See error_handling_policies.cpp for more examples.

// Boost
#include <boost/math/distributions/students_t.hpp>
using boost::math::students_t_distribution;  // Probability of students_t(df, t).
using boost::math::students_t;  // Probability of students_t(df, t) convenience typedef for double.

using boost::math::policies::policy;
using boost::math::policies::domain_error;
using boost::math::policies::ignore_error;

// std
#include <iostream>
   using std::cout;
   using std::endl;

#include <stdexcept>
   

// Define a (bad?) policy to ignore domain errors ('bad' arguments):
typedef policy<
      domain_error<ignore_error>
      > my_policy;

// Define my_students_t distribution with this different domain error policy:
typedef students_t_distribution<double, my_policy> my_students_t;

int main()
{  // Example of error handling of bad argument(s) to a distribution.
  cout << "Example error handling using Student's t function. " << endl;

  double degrees_of_freedom = -1; double t = -1.; // Two 'bad' arguments!

  try
  {
    cout << "Probability of ignore_error Student's t is "
      << cdf(my_students_t(degrees_of_freedom), t) << endl;
    cout << "Probability of default error policy Student's t is " << endl;
    // By contrast the students_t distribution default domain error policy is to throw,
    cout << cdf(students_t(-1), -1) << endl;  // so this will throw.
/*`
    Message from thrown exception was:
   Error in function boost::math::students_t_distribution<double>::students_t_distribution:
   Degrees of freedom argument is -1, but must be > 0 !
*/

    // We could also define a 'custom' distribution
    // with an "ignore overflow error policy" in a single statement:
    using boost::math::policies::overflow_error;
    students_t_distribution<double, policy<overflow_error<ignore_error> > > students_t_no_throw(-1);

  }
  catch(const std::exception& e)
  {
    std::cout <<
      "\n""Message from thrown exception was:\n   " << e.what() << std::endl;
  }

  return 0;
} // int main()

/*

Output:

   error_policy_example.cpp
  Generating code
  Finished generating code
  error_policy_example.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Release\error_policy_example.exe
  Example error handling using Student's t function. 
  Probability of ignore_error Student's t is 1.#QNAN
  Probability of default error policy Student's t is 
  
  Message from thrown exception was:
     Error in function boost::math::students_t_distribution<double>::students_t_distribution: Degrees of freedom argument is -1, but must be > 0 !

*/
