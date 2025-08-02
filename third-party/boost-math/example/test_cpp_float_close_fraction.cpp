//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Copyright Paul A. Bristow 2013
// Copyright Christopher Kormanyos 2013.
// Copyright John Maddock 2013.

#ifdef _MSC_VER
#  pragma warning (disable : 4512)
#  pragma warning (disable : 4996)
#endif

#define BOOST_TEST_MAIN
#define BOOST_LIB_DIAGNOSTIC "on"// Show library file details.

#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp> // Extra test tool for FP comparison.

#include <iostream>
#include <limits>

//[expression_template_1

#include <boost/multiprecision/cpp_dec_float.hpp>

/*`To define a 50 decimal digit type using `cpp_dec_float`,
you must pass two template parameters to `boost::multiprecision::number`.

It may be more legible to use a two-staged type definition such as this:

``
typedef boost::multiprecision::cpp_dec_float<50> mp_backend;
typedef boost::multiprecision::number<mp_backend, boost::multiprecision::et_off> cpp_dec_float_50_noet;
``

Here, we first define `mp_backend` as `cpp_dec_float` with 50 digits.
The second step passes this backend to `boost::multiprecision::number`
with `boost::multiprecision::et_off`, an enumerated type.

  typedef boost::multiprecision::number<boost::multiprecision::cpp_dec_float<50>, boost::multiprecision::et_off>
  cpp_dec_float_50_noet;

You can reduce typing with a `using` directive `using namespace boost::multiprecision;`
if desired, as shown below.
*/

using namespace boost::multiprecision;


/*`Now `cpp_dec_float_50_noet` or `cpp_dec_float_50_et`
can be used as a direct replacement for built-in types like `double` etc.
*/

BOOST_AUTO_TEST_CASE(cpp_float_test_check_close_noet)
{ // No expression templates/
  typedef number<cpp_dec_float<50>, et_off> cpp_dec_float_50_noet;

  std::cout.precision(std::numeric_limits<cpp_dec_float_50_noet>::digits10); // All significant digits.
  std::cout << std::showpoint << std::endl; // Show trailing zeros.

  cpp_dec_float_50_noet a ("1.0");
  cpp_dec_float_50_noet b ("1.0");
  b += std::numeric_limits<cpp_dec_float_50_noet>::epsilon(); // Increment least significant decimal digit.

  cpp_dec_float_50_noet eps = std::numeric_limits<cpp_dec_float_50_noet>::epsilon();

  std::cout <<"a = " << a << ",\nb = " << b << ",\neps = " << eps << std::endl;

  BOOST_CHECK_CLOSE(a, b, eps * 100); // Expected to pass (because tolerance is as percent).
  BOOST_CHECK_CLOSE_FRACTION(a, b, eps); // Expected to pass (because tolerance is as fraction).



} // BOOST_AUTO_TEST_CASE(cpp_float_test_check_close)

BOOST_AUTO_TEST_CASE(cpp_float_test_check_close_et)
{ // Using expression templates.
  typedef number<cpp_dec_float<50>, et_on> cpp_dec_float_50_et;

  std::cout.precision(std::numeric_limits<cpp_dec_float_50_et>::digits10); // All significant digits.
  std::cout << std::showpoint << std::endl; // Show trailing zeros.

  cpp_dec_float_50_et a("1.0");
  cpp_dec_float_50_et b("1.0");
  b += std::numeric_limits<cpp_dec_float_50_et>::epsilon(); // Increment least significant decimal digit.

  cpp_dec_float_50_et eps = std::numeric_limits<cpp_dec_float_50_et>::epsilon();

  std::cout << "a = " << a << ",\nb = " << b << ",\neps = " << eps << std::endl;

  BOOST_CHECK_CLOSE(a, b, eps * 100); // Expected to pass (because tolerance is as percent).
  BOOST_CHECK_CLOSE_FRACTION(a, b, eps); // Expected to pass (because tolerance is as fraction).

  /*`Using `cpp_dec_float_50` with the default expression template use switched on,
  the compiler error message for `BOOST_CHECK_CLOSE_FRACTION(a, b, eps); would be:
  */
  // failure floating_point_comparison.hpp(59): error C2440: 'static_cast' :
  // cannot convert from 'int' to 'boost::multiprecision::detail::expression<tag,Arg1,Arg2,Arg3,Arg4>'
//] [/expression_template_1]

} // BOOST_AUTO_TEST_CASE(cpp_float_test_check_close)

/*

Output:

  Description: Autorun "J:\Cpp\big_number\Debug\test_cpp_float_close_fraction.exe"
  Running 1 test case...

  a = 1.0000000000000000000000000000000000000000000000000,
  b = 1.0000000000000000000000000000000000000000000000001,
  eps = 1.0000000000000000000000000000000000000000000000000e-49

  *** No errors detected


*/

