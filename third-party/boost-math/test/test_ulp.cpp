//  (C) Copyright John Maddock 2025.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/special_functions/ulp.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <boost/multiprecision/cpp_dec_float.hpp>


template <class T>
void test()
{
   //
   // Really just test our error handling:
   //
   BOOST_MATH_IF_CONSTEXPR(std::numeric_limits<T>::has_quiet_NaN)
   {
      BOOST_CHECK_THROW(boost::math::ulp(std::numeric_limits<T>::quiet_NaN()), std::domain_error);
   }
   BOOST_MATH_IF_CONSTEXPR(std::numeric_limits<T>::has_infinity)
   {
      BOOST_CHECK_THROW(boost::math::ulp(std::numeric_limits<T>::infinity()), std::overflow_error);
   }
   BOOST_CHECK_THROW(boost::math::ulp(boost::math::tools::max_value<T>()), std::overflow_error);

   BOOST_CHECK(boost::math::ulp(static_cast<T>(0)) != 0);
}

BOOST_AUTO_TEST_CASE( test_main )
{
   test<float>();
   test<double>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test<long double>();
#endif
   test<boost::multiprecision::cpp_dec_float_50>();
}

