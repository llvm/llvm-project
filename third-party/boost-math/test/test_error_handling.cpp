// Copyright Paul A. Bristow 2006-7.
// Copyright John Maddock 2006-7.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test error handling mechanism produces the expected error messages.
// for example Error in function boost::math::test_function<float>(float, float, float): Domain Error evaluating function at 0

// Define some custom dummy error handlers that do nothing but throw,
// in order to check that they are otherwise undefined.
// The user MUST define them before they can be used.
//
struct user_defined_error{};

namespace boost{ namespace math{ namespace policies{

#ifndef BOOST_NO_EXCEPTIONS

template <class T>
T user_domain_error(const char* , const char* , const T& )
{
   throw user_defined_error();
}

template <class T>
T user_pole_error(const char* , const char* , const T& )
{
   throw user_defined_error();
}

template <class T>
T user_overflow_error(const char* , const char* , const T& )
{
   throw user_defined_error();
}

template <class T>
T user_underflow_error(const char* , const char* , const T& )
{
   throw user_defined_error();
}

template <class T>
T user_denorm_error(const char* , const char* , const T& )
{
   throw user_defined_error();
}

template <class T>
T user_evaluation_error(const char* , const char* , const T& )
{
   throw user_defined_error();
}

template <class T>
T user_indeterminate_result_error(const char* , const char* , const T& )
{
   throw user_defined_error();
}
#endif
}}} // namespaces

#include <boost/math/tools/test.hpp>
#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/tools/polynomial.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // for test_main
#include <cerrno> // for errno
#include <iostream>
#include <iomanip>
//
// Define some policies:
//
using namespace boost::math::policies;
policy<
   domain_error<throw_on_error>,
   pole_error<throw_on_error>,
   overflow_error<throw_on_error>,
   underflow_error<throw_on_error>,
   denorm_error<throw_on_error>,
   evaluation_error<throw_on_error>,
   indeterminate_result_error<throw_on_error> > throw_policy;
policy<
   domain_error<errno_on_error>,
   pole_error<errno_on_error>,
   overflow_error<errno_on_error>,
   underflow_error<errno_on_error>,
   denorm_error<errno_on_error>,
   evaluation_error<errno_on_error>,
   indeterminate_result_error<errno_on_error> > errno_policy;
policy<
   domain_error<ignore_error>,
   pole_error<ignore_error>,
   overflow_error<ignore_error>,
   underflow_error<ignore_error>,
   denorm_error<ignore_error>,
   evaluation_error<ignore_error>,
   indeterminate_result_error<ignore_error> > ignore_policy;
policy<
   domain_error<user_error>,
   pole_error<user_error>,
   overflow_error<user_error>,
   underflow_error<user_error>,
   denorm_error<user_error>,
   evaluation_error<user_error>,
   indeterminate_result_error<user_error> > user_policy;
policy<> default_policy;

#define TEST_EXCEPTION(expression, exception, msg)\
   BOOST_MATH_CHECK_THROW(expression, exception);\
   try{ expression; }catch(const exception& e){ std::cout << e.what() << std::endl; BOOST_CHECK_EQUAL(std::string(e.what()), std::string(msg)); }

template <class T>
std::string format_message_string(const char* str)
{
   std::string type_name = boost::math::policies::detail::name_of<T>();
   std::string result(str);
   if(type_name != "float")
   {
      std::string::size_type pos = 0;
      while((pos = result.find("float", pos)) != std::string::npos)
      {
         result.replace(pos, 5, type_name);
         pos += type_name.size();
      }
   }
   return result;
}

template <class T>
void test_error(T)
{
   const char* func = "boost::math::test_function<%1%>(%1%, %1%, %1%)";
   const char* msg1 = "Error while handling value %1%";
   const char* msg2 = "Error message goes here...";

   // Check that exception is thrown, catch and show the message, for example:
   // Error in function boost::math::test_function<float>(float, float, float): Error while handling value 0
#ifndef BOOST_NO_EXCEPTIONS
   TEST_EXCEPTION(boost::math::policies::raise_domain_error(func, msg1, T(0.0), throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error while handling value 0"));
   TEST_EXCEPTION(boost::math::policies::raise_domain_error(func, 0, T(0.0), throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Domain Error evaluating function at 0"));
   TEST_EXCEPTION(boost::math::policies::raise_pole_error(func, msg1, T(0.0), throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error while handling value 0"));
   TEST_EXCEPTION(boost::math::policies::raise_pole_error(func, 0, T(0.0), throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Evaluation of function at pole 0"));
   TEST_EXCEPTION(boost::math::policies::raise_overflow_error<T>(func, msg2, throw_policy), std::overflow_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error message goes here..."));
   TEST_EXCEPTION(boost::math::policies::raise_overflow_error<T>(func, 0, throw_policy), std::overflow_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Overflow Error"));
   TEST_EXCEPTION(boost::math::policies::raise_underflow_error<T>(func, msg2, throw_policy), std::underflow_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error message goes here..."));
   TEST_EXCEPTION(boost::math::policies::raise_underflow_error<T>(func, 0, throw_policy), std::underflow_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Underflow Error"));
   TEST_EXCEPTION(boost::math::policies::raise_denorm_error<T>(func, msg2, T(0), throw_policy), std::underflow_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error message goes here..."));
   TEST_EXCEPTION(boost::math::policies::raise_denorm_error<T>(func, 0, T(0), throw_policy), std::underflow_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Denorm Error"));
   TEST_EXCEPTION(boost::math::policies::raise_evaluation_error(func, msg1, T(1.25), throw_policy), boost::math::evaluation_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error while handling value 1.25"));
   TEST_EXCEPTION(boost::math::policies::raise_evaluation_error(func, 0, T(1.25), throw_policy), boost::math::evaluation_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Internal Evaluation Error, best value so far was 1.25"));
   TEST_EXCEPTION(boost::math::policies::raise_indeterminate_result_error(func, msg1, T(1.25), T(12.34), throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error while handling value 1.25"));
   TEST_EXCEPTION(boost::math::policies::raise_indeterminate_result_error(func, 0, T(1.25), T(12.34), throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Indeterminate result with value 1.25"));
   //
   // Now try user error handlers: these should all throw user_error():
   // - because by design these are undefined and must be defined by the user ;-)
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_domain_error(func, msg1, T(0.0), user_policy), user_defined_error);
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_pole_error(func, msg1, T(0.0), user_policy), user_defined_error);
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_overflow_error<T>(func, msg2, user_policy), user_defined_error);
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_underflow_error<T>(func, msg2, user_policy), user_defined_error);
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_denorm_error<T>(func, msg2, T(0), user_policy), user_defined_error);
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_evaluation_error(func, msg1, T(0.0), user_policy), user_defined_error);
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_indeterminate_result_error(func, msg1, T(0.0), T(0.0), user_policy), user_defined_error);
#endif

   // Test with ignore_error
   BOOST_CHECK((boost::math::isnan)(boost::math::policies::raise_domain_error(func, msg1, T(0.0), ignore_policy)) || !std::numeric_limits<T>::has_quiet_NaN);
   BOOST_CHECK((boost::math::isnan)(boost::math::policies::raise_pole_error(func, msg1, T(0.0), ignore_policy)) || !std::numeric_limits<T>::has_quiet_NaN);
   BOOST_CHECK_EQUAL(boost::math::policies::raise_overflow_error<T>(func, msg2, ignore_policy), std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : boost::math::tools::max_value<T>());
   BOOST_CHECK_EQUAL(boost::math::policies::raise_underflow_error<T>(func, msg2, ignore_policy), T(0));
   BOOST_CHECK_EQUAL(boost::math::policies::raise_denorm_error<T>(func, msg2, T(1.25), ignore_policy), T(1.25));
   BOOST_CHECK_EQUAL(boost::math::policies::raise_evaluation_error(func, msg1, T(1.25), ignore_policy), T(1.25));
   BOOST_CHECK_EQUAL(boost::math::policies::raise_indeterminate_result_error(func, 0, T(0.0), T(12.34), ignore_policy), T(12.34));

   // Test with errno_on_error
   errno = 0;
   BOOST_CHECK((boost::math::isnan)(boost::math::policies::raise_domain_error(func, msg1, T(0.0), errno_policy)) || !std::numeric_limits<T>::has_quiet_NaN);
   BOOST_CHECK(errno == EDOM);
   errno = 0;
   BOOST_CHECK((boost::math::isnan)(boost::math::policies::raise_pole_error(func, msg1, T(0.0), errno_policy)) || !std::numeric_limits<T>::has_quiet_NaN);
   BOOST_CHECK(errno == EDOM);
   errno = 0;
   BOOST_CHECK_EQUAL(boost::math::policies::raise_overflow_error<T>(func, msg2, errno_policy), std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : boost::math::tools::max_value<T>());
   BOOST_CHECK_EQUAL(errno, ERANGE);
   errno = 0;
   BOOST_CHECK_EQUAL(boost::math::policies::raise_underflow_error<T>(func, msg2, errno_policy), T(0));
   BOOST_CHECK_EQUAL(errno, ERANGE);
   errno = 0;
   BOOST_CHECK_EQUAL(boost::math::policies::raise_denorm_error<T>(func, msg2, T(1.25), errno_policy), T(1.25));
   BOOST_CHECK_EQUAL(errno, ERANGE);
   errno = 0;
   BOOST_CHECK_EQUAL(boost::math::policies::raise_evaluation_error(func, msg1, T(1.25), errno_policy), T(1.25));
   BOOST_CHECK(errno == EDOM);
   errno = 0;
   BOOST_CHECK(boost::math::policies::raise_indeterminate_result_error(func, 0, T(0.0), T(12.34), errno_policy) == T(12.34));
   BOOST_CHECK_EQUAL(errno, EDOM);
}

template <class T>
void test_complex_error(T)
{
   //
   // Error handling that can be applied to non-scalar types such as std::complex
   //
   const char* func = "boost::math::test_function<%1%>(%1%, %1%, %1%)";
   const char* msg1 = "Error while handling value %1%";
   const char* msg2 = "Error message goes here...";

   // Check that exception is thrown, catch and show the message, for example:
   // Error in function boost::math::test_function<float>(float, float, float): Error while handling value 0
#ifndef BOOST_NO_EXCEPTIONS
   TEST_EXCEPTION(boost::math::policies::raise_domain_error(func, msg1, T(0.0), throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error while handling value (0,0)"));
   TEST_EXCEPTION(boost::math::policies::raise_domain_error(func, 0, T(0.0), throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Domain Error evaluating function at (0,0)"));
   TEST_EXCEPTION(boost::math::policies::raise_pole_error(func, msg1, T(0.0), throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error while handling value (0,0)"));
   TEST_EXCEPTION(boost::math::policies::raise_pole_error(func, 0, T(0.0), throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Evaluation of function at pole (0,0)"));
   //TEST_EXCEPTION(boost::math::policies::raise_overflow_error<T>(func, msg2, throw_policy), std::overflow_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error message goes here..."));
   //TEST_EXCEPTION(boost::math::policies::raise_overflow_error<T>(func, 0, throw_policy), std::overflow_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Overflow Error"));
   TEST_EXCEPTION(boost::math::policies::raise_underflow_error<T>(func, msg2, throw_policy), std::underflow_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error message goes here..."));
   TEST_EXCEPTION(boost::math::policies::raise_underflow_error<T>(func, 0, throw_policy), std::underflow_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Underflow Error"));
   TEST_EXCEPTION(boost::math::policies::raise_denorm_error<T>(func, msg2, T(0), throw_policy), std::underflow_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error message goes here..."));
   TEST_EXCEPTION(boost::math::policies::raise_denorm_error<T>(func, 0, T(0), throw_policy), std::underflow_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Denorm Error"));
   TEST_EXCEPTION(boost::math::policies::raise_evaluation_error(func, msg1, T(1.25), throw_policy), boost::math::evaluation_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error while handling value (1.25,0)"));
   TEST_EXCEPTION(boost::math::policies::raise_evaluation_error(func, 0, T(1.25), throw_policy), boost::math::evaluation_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Internal Evaluation Error, best value so far was (1.25,0)"));
   TEST_EXCEPTION(boost::math::policies::raise_indeterminate_result_error(func, msg1, T(1.25), T(12.34), throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error while handling value (1.25,0)"));
   TEST_EXCEPTION(boost::math::policies::raise_indeterminate_result_error(func, 0, T(1.25), T(12.34), throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Indeterminate result with value (1.25,0)"));
   //
   // Now try user error handlers: these should all throw user_error():
   // - because by design these are undefined and must be defined by the user ;-)
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_domain_error(func, msg1, T(0.0), user_policy), user_defined_error);
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_pole_error(func, msg1, T(0.0), user_policy), user_defined_error);
   //BOOST_MATH_CHECK_THROW(boost::math::policies::raise_overflow_error<T>(func, msg2, user_policy), user_defined_error);
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_underflow_error<T>(func, msg2, user_policy), user_defined_error);
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_denorm_error<T>(func, msg2, T(0), user_policy), user_defined_error);
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_evaluation_error(func, msg1, T(0.0), user_policy), user_defined_error);
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_indeterminate_result_error(func, msg1, T(0.0), T(0.0), user_policy), user_defined_error);
#endif

   // Test with ignore_error
   BOOST_CHECK((boost::math::isnan)(boost::math::policies::raise_domain_error(func, msg1, T(0.0), ignore_policy)) || !std::numeric_limits<T>::has_quiet_NaN);
   BOOST_CHECK((boost::math::isnan)(boost::math::policies::raise_pole_error(func, msg1, T(0.0), ignore_policy)) || !std::numeric_limits<T>::has_quiet_NaN);
   //BOOST_CHECK_EQUAL(boost::math::policies::raise_overflow_error<T>(func, msg2, ignore_policy), std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : boost::math::tools::max_value<T>());
   BOOST_CHECK_EQUAL(boost::math::policies::raise_underflow_error<T>(func, msg2, ignore_policy), T(0));
   BOOST_CHECK_EQUAL(boost::math::policies::raise_denorm_error<T>(func, msg2, T(1.25), ignore_policy), T(1.25));
   BOOST_CHECK_EQUAL(boost::math::policies::raise_evaluation_error(func, msg1, T(1.25), ignore_policy), T(1.25));
   BOOST_CHECK_EQUAL(boost::math::policies::raise_indeterminate_result_error(func, 0, T(0.0), T(12.34), ignore_policy), T(12.34));

   // Test with errno_on_error
   errno = 0;
   BOOST_CHECK((boost::math::isnan)(boost::math::policies::raise_domain_error(func, msg1, T(0.0), errno_policy)) || !std::numeric_limits<T>::has_quiet_NaN);
   BOOST_CHECK(errno == EDOM);
   errno = 0;
   BOOST_CHECK((boost::math::isnan)(boost::math::policies::raise_pole_error(func, msg1, T(0.0), errno_policy)) || !std::numeric_limits<T>::has_quiet_NaN);
   BOOST_CHECK(errno == EDOM);
   errno = 0;
   //BOOST_CHECK_EQUAL(boost::math::policies::raise_overflow_error<T>(func, msg2, errno_policy), std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : boost::math::tools::max_value<T>());
   //BOOST_CHECK_EQUAL(errno, ERANGE);
   errno = 0;
   BOOST_CHECK_EQUAL(boost::math::policies::raise_underflow_error<T>(func, msg2, errno_policy), T(0));
   BOOST_CHECK_EQUAL(errno, ERANGE);
   errno = 0;
   BOOST_CHECK_EQUAL(boost::math::policies::raise_denorm_error<T>(func, msg2, T(1.25), errno_policy), T(1.25));
   BOOST_CHECK_EQUAL(errno, ERANGE);
   errno = 0;
   BOOST_CHECK_EQUAL(boost::math::policies::raise_evaluation_error(func, msg1, T(1.25), errno_policy), T(1.25));
   BOOST_CHECK(errno == EDOM);
   errno = 0;
   BOOST_CHECK(boost::math::policies::raise_indeterminate_result_error(func, 0, T(0.0), T(12.34), errno_policy) == T(12.34));
   BOOST_CHECK_EQUAL(errno, EDOM);
}

template <class T>
void test_polynomial_error(T)
{
   //
   // Error handling that can be applied to non-scalar types such as std::complex
   //
   const char* func = "boost::math::test_function<%1%>(%1%, %1%, %1%)";
   const char* msg1 = "Error while handling value %1%";

   static const typename T::value_type data[] = { 1, 2, 3 };
   static const T val(data, 2);

   // Check that exception is thrown, catch and show the message, for example:
   // Error in function boost::math::test_function<float>(float, float, float): Error while handling value 0
#ifndef BOOST_NO_EXCEPTIONS
   TEST_EXCEPTION(boost::math::policies::raise_domain_error(func, msg1, val, throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error while handling value { 1, 2, 3 }"));
   TEST_EXCEPTION(boost::math::policies::raise_domain_error(func, 0, val, throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Domain Error evaluating function at { 1, 2, 3 }"));
   TEST_EXCEPTION(boost::math::policies::raise_pole_error(func, msg1, val, throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error while handling value { 1, 2, 3 }"));
   TEST_EXCEPTION(boost::math::policies::raise_pole_error(func, 0, val, throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Evaluation of function at pole { 1, 2, 3 }"));
   TEST_EXCEPTION(boost::math::policies::raise_evaluation_error(func, msg1, val, throw_policy), boost::math::evaluation_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error while handling value { 1, 2, 3 }"));
   TEST_EXCEPTION(boost::math::policies::raise_evaluation_error(func, 0, val, throw_policy), boost::math::evaluation_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Internal Evaluation Error, best value so far was { 1, 2, 3 }"));
   TEST_EXCEPTION(boost::math::policies::raise_indeterminate_result_error(func, msg1, val, T(12.34), throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Error while handling value { 1, 2, 3 }"));
   TEST_EXCEPTION(boost::math::policies::raise_indeterminate_result_error(func, 0, val, T(12.34), throw_policy), std::domain_error, format_message_string<T>("Error in function boost::math::test_function<float>(float, float, float): Indeterminate result with value { 1, 2, 3 }"));
   //
   // Now try user error handlers: these should all throw user_error():
   // - because by design these are undefined and must be defined by the user ;-)
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_domain_error(func, msg1, T(0.0), user_policy), user_defined_error);
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_pole_error(func, msg1, T(0.0), user_policy), user_defined_error);
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_evaluation_error(func, msg1, T(0.0), user_policy), user_defined_error);
   BOOST_MATH_CHECK_THROW(boost::math::policies::raise_indeterminate_result_error(func, msg1, T(0.0), T(0.0), user_policy), user_defined_error);
#endif

   // Test with ignore_error
   BOOST_CHECK((boost::math::isnan)(boost::math::policies::raise_domain_error(func, msg1, T(0.0), ignore_policy)) || !std::numeric_limits<T>::has_quiet_NaN);
   BOOST_CHECK((boost::math::isnan)(boost::math::policies::raise_pole_error(func, msg1, T(0.0), ignore_policy)) || !std::numeric_limits<T>::has_quiet_NaN);
   BOOST_CHECK_EQUAL(boost::math::policies::raise_evaluation_error(func, msg1, T(1.25), ignore_policy), T(1.25));
   BOOST_CHECK_EQUAL(boost::math::policies::raise_indeterminate_result_error(func, 0, T(0.0), T(12.34), ignore_policy), T(12.34));

   // Test with errno_on_error
   errno = 0;
   BOOST_CHECK((boost::math::isnan)(boost::math::policies::raise_domain_error(func, msg1, T(0.0), errno_policy)) || !std::numeric_limits<T>::has_quiet_NaN);
   BOOST_CHECK(errno == EDOM);
   errno = 0;
   BOOST_CHECK((boost::math::isnan)(boost::math::policies::raise_pole_error(func, msg1, T(0.0), errno_policy)) || !std::numeric_limits<T>::has_quiet_NaN);
   BOOST_CHECK(errno == EDOM);
   errno = 0;
   BOOST_CHECK_EQUAL(boost::math::policies::raise_evaluation_error(func, msg1, T(1.25), errno_policy), T(1.25));
   BOOST_CHECK(errno == EDOM);
   errno = 0;
   BOOST_CHECK(boost::math::policies::raise_indeterminate_result_error(func, 0, T(0.0), T(12.34), errno_policy) == T(12.34));
   BOOST_CHECK_EQUAL(errno, EDOM);
}

BOOST_AUTO_TEST_CASE( test_main )
{
   // Test error handling.
   // (Parameter value, arbitrarily zero, only communicates the floating point type FPT).
   test_error(0.0F); // Test float.
   test_error(0.0); // Test double.
   test_error(0.0L); // Test long double.
   test_error(boost::math::concepts::real_concept(0.0L)); // Test concepts.

   // try complex numbers too:
   test_complex_error(std::complex<float>(0));
   test_complex_error(std::complex<double>(0));
   test_complex_error(std::complex<long double>(0));

   test_polynomial_error(boost::math::tools::polynomial<float>());

} // BOOST_AUTO_TEST_CASE( test_main )

/*

Autorun "i:\boost-06-05-03-1300\libs\math\test\Math_test\debug\test_error_handling.exe"
Running 1 test case...
Error in function boost::math::test_function<float>(float, float, float): Error while handling value 0
Error in function boost::math::test_function<float>(float, float, float): Domain Error evaluating function at 0
Error in function boost::math::test_function<float>(float, float, float): Error while handling value 0
Error in function boost::math::test_function<float>(float, float, float): Evaluation of function at pole 0
Error in function boost::math::test_function<float>(float, float, float): Error message goes here...
Error in function boost::math::test_function<float>(float, float, float): Overflow Error
Error in function boost::math::test_function<float>(float, float, float): Error message goes here...
Error in function boost::math::test_function<float>(float, float, float): Underflow Error
Error in function boost::math::test_function<float>(float, float, float): Error message goes here...
Error in function boost::math::test_function<float>(float, float, float): Denorm Error
Error in function boost::math::test_function<float>(float, float, float): Error while handling value 1.25
Error in function boost::math::test_function<float>(float, float, float): Internal Evaluation Error, best value so far was 1.25
Error in function boost::math::test_function<double>(double, double, double): Error while handling value 0
Error in function boost::math::test_function<double>(double, double, double): Domain Error evaluating function at 0
Error in function boost::math::test_function<double>(double, double, double): Error while handling value 0
Error in function boost::math::test_function<double>(double, double, double): Evaluation of function at pole 0
Error in function boost::math::test_function<double>(double, double, double): Error message goes here...
Error in function boost::math::test_function<double>(double, double, double): Overflow Error
Error in function boost::math::test_function<double>(double, double, double): Error message goes here...
Error in function boost::math::test_function<double>(double, double, double): Underflow Error
Error in function boost::math::test_function<double>(double, double, double): Error message goes here...
Error in function boost::math::test_function<double>(double, double, double): Denorm Error
Error in function boost::math::test_function<double>(double, double, double): Error while handling value 1.25
Error in function boost::math::test_function<double>(double, double, double): Internal Evaluation Error, best value so far was 1.25
Error in function boost::math::test_function<long double>(long double, long double, long double): Error while handling value 0
Error in function boost::math::test_function<long double>(long double, long double, long double): Domain Error evaluating function at 0
Error in function boost::math::test_function<long double>(long double, long double, long double): Error while handling value 0
Error in function boost::math::test_function<long double>(long double, long double, long double): Evaluation of function at pole 0
Error in function boost::math::test_function<long double>(long double, long double, long double): Error message goes here...
Error in function boost::math::test_function<long double>(long double, long double, long double): Overflow Error
Error in function boost::math::test_function<long double>(long double, long double, long double): Error message goes here...
Error in function boost::math::test_function<long double>(long double, long double, long double): Underflow Error
Error in function boost::math::test_function<long double>(long double, long double, long double): Error message goes here...
Error in function boost::math::test_function<long double>(long double, long double, long double): Denorm Error
Error in function boost::math::test_function<long double>(long double, long double, long double): Error while handling value 1.25
Error in function boost::math::test_function<long double>(long double, long double, long double): Internal Evaluation Error, best value so far was 1.25
Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Error while handling value 0
Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Domain Error evaluating function at 0
Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Error while handling value 0
Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Evaluation of function at pole 0
Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Error message goes here...
Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Overflow Error
Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Error message goes here...
Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Underflow Error
Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Error message goes here...
Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Denorm Error
Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Error while handling value 1.25
Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Internal Evaluation Error, best value so far was 1.25
*** No errors detected

VS 2010
------ Rebuild All started: Project: test_error_handling, Configuration: Release Win32 ------
  test_error_handling.cpp
  Generating code
  Finished generating code
  test_error_handling.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Release\test_error_handling.exe
  Running 1 test case...
  Error in function boost::math::test_function<float>(float, float, float): Error while handling value 0
  Error in function boost::math::test_function<float>(float, float, float): Domain Error evaluating function at 0
  Error in function boost::math::test_function<float>(float, float, float): Error while handling value 0
  Error in function boost::math::test_function<float>(float, float, float): Evaluation of function at pole 0
  Error in function boost::math::test_function<float>(float, float, float): Error message goes here...
  Error in function boost::math::test_function<float>(float, float, float): Overflow Error
  Error in function boost::math::test_function<float>(float, float, float): Error message goes here...
  Error in function boost::math::test_function<float>(float, float, float): Underflow Error
  Error in function boost::math::test_function<float>(float, float, float): Error message goes here...
  Error in function boost::math::test_function<float>(float, float, float): Denorm Error
  Error in function boost::math::test_function<float>(float, float, float): Error while handling value 1.25
  Error in function boost::math::test_function<float>(float, float, float): Internal Evaluation Error, best value so far was 1.25
  Error in function boost::math::test_function<float>(float, float, float): Error while handling value 1.25
  Error in function boost::math::test_function<float>(float, float, float): Indeterminate result with value 1.25
  Error in function boost::math::test_function<double>(double, double, double): Error while handling value 0
  Error in function boost::math::test_function<double>(double, double, double): Domain Error evaluating function at 0
  Error in function boost::math::test_function<double>(double, double, double): Error while handling value 0
  Error in function boost::math::test_function<double>(double, double, double): Evaluation of function at pole 0
  Error in function boost::math::test_function<double>(double, double, double): Error message goes here...
  Error in function boost::math::test_function<double>(double, double, double): Overflow Error
  Error in function boost::math::test_function<double>(double, double, double): Error message goes here...
  Error in function boost::math::test_function<double>(double, double, double): Underflow Error
  Error in function boost::math::test_function<double>(double, double, double): Error message goes here...
  Error in function boost::math::test_function<double>(double, double, double): Denorm Error
  Error in function boost::math::test_function<double>(double, double, double): Error while handling value 1.25
  Error in function boost::math::test_function<double>(double, double, double): Internal Evaluation Error, best value so far was 1.25
  Error in function boost::math::test_function<double>(double, double, double): Error while handling value 1.25
  Error in function boost::math::test_function<double>(double, double, double): Indeterminate result with value 1.25
  Error in function boost::math::test_function<long double>(long double, long double, long double): Error while handling value 0
  Error in function boost::math::test_function<long double>(long double, long double, long double): Domain Error evaluating function at 0
  Error in function boost::math::test_function<long double>(long double, long double, long double): Error while handling value 0
  Error in function boost::math::test_function<long double>(long double, long double, long double): Evaluation of function at pole 0
  Error in function boost::math::test_function<long double>(long double, long double, long double): Error message goes here...
  Error in function boost::math::test_function<long double>(long double, long double, long double): Overflow Error
  Error in function boost::math::test_function<long double>(long double, long double, long double): Error message goes here...
  Error in function boost::math::test_function<long double>(long double, long double, long double): Underflow Error
  Error in function boost::math::test_function<long double>(long double, long double, long double): Error message goes here...
  Error in function boost::math::test_function<long double>(long double, long double, long double): Denorm Error
  Error in function boost::math::test_function<long double>(long double, long double, long double): Error while handling value 1.25
  Error in function boost::math::test_function<long double>(long double, long double, long double): Internal Evaluation Error, best value so far was 1.25
  Error in function boost::math::test_function<long double>(long double, long double, long double): Error while handling value 1.25
  Error in function boost::math::test_function<long double>(long double, long double, long double): Indeterminate result with value 1.25
  Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Error while handling value 0
  Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Domain Error evaluating function at 0
  Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Error while handling value 0
  Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Evaluation of function at pole 0
  Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Error message goes here...
  Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Overflow Error
  Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Error message goes here...
  Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Underflow Error
  Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Error message goes here...
  Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Denorm Error
  Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Error while handling value 1.25
  Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Internal Evaluation Error, best value so far was 1.25
  Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Error while handling value 1.25
  
  *** No errors detected
  Error in function boost::math::test_function<class boost::math::concepts::real_concept>(class boost::math::concepts::real_concept, class boost::math::concepts::real_concept, class boost::math::concepts::real_concept): Indeterminate result with value 1.25
========== Rebuild All: 1 succeeded, 0 failed, 0 skipped ==========


*/

