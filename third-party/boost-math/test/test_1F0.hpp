// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/tools/stats.hpp>
#include <boost/math/tools/test.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#include <boost/math/special_functions/hypergeometric_1F0.hpp>

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif


template <class T>
void test_spots(T)
{
   using std::pow;
   //
   // basic sanity checks, tolerance is 10 epsilon expressed as a percentage:
   //
   T tolerance = boost::math::tools::epsilon<T>() * 1000;

   BOOST_CHECK_CLOSE(boost::math::hypergeometric_1F0(T(-3), T(2)), T(-1), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_1F0(T(-3), T(4)), T(-27), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_1F0(T(-3), T(0.5)), T(0.125), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_1F0(T(3), T(0.5)), T(8), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_1F0(T(3), T(2)), T(-1), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_1F0(T(3), T(4)), T(T(-1) / 27), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_1F0(T(3), T(-0.5)), pow(T(1.5), -3), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_1F0(T(3), T(-2)), T(1 / T(27)), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_1F0(T(3), T(-4)), T(T(1) / 125), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_1F0(T(-3), T(-0.5)), pow(T(1.5), 3), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_1F0(T(-3), T(-2)), T(27), tolerance);
   BOOST_CHECK_CLOSE(boost::math::hypergeometric_1F0(T(-3), T(-4)), T(125), tolerance);

   BOOST_CHECK_THROW(boost::math::hypergeometric_1F0(T(3), T(1)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::hypergeometric_1F0(T(-3), T(1)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::hypergeometric_1F0(T(3.25), T(1)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::hypergeometric_1F0(T(-3.25), T(1)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::hypergeometric_1F0(T(3.25), T(2)), std::domain_error);
   BOOST_CHECK_THROW(boost::math::hypergeometric_1F0(T(-3.25), T(2)), std::domain_error);
}

