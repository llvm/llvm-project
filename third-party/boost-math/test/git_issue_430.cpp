// Copyright Matt Borland 2023
// Copyright John Maddock 2023
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// See: https://godbolt.org/z/Ev4ManrsW

#include <boost/math/special_functions/round.hpp>
#include <boost/math/special_functions/trunc.hpp>
#include <boost/math/special_functions/next.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <cmath>

template <typename Real>
void test_llround_near_boundary()
{
   using std::ldexp;
   Real boundary = ldexp(static_cast<Real>(1), std::numeric_limits<long long>::digits);

   Real value;
   int i;

    for (value = boundary, i = 0; i < 100; value = boost::math::float_next(value), ++i)
    {
        BOOST_CHECK_THROW(boost::math::llround(value), boost::math::rounding_error);
    }
    for (value = boost::math::float_prior(boundary), i = 0; i < 1000; value = boost::math::float_prior(value), ++i)
    {
        BOOST_CHECK_EQUAL(static_cast<Real>(boost::math::llround(value)), boost::math::round(value));
    }
    for (value = boost::math::float_prior(-boundary), i = 0; i < 100; value = boost::math::float_prior(value), ++i)
    {
        BOOST_CHECK_THROW(boost::math::llround(value), boost::math::rounding_error);
    }
    for (value = -boundary, i = 0; i < 1000; value = boost::math::float_next(value), ++i)
    {
        BOOST_CHECK_EQUAL(static_cast<Real>(boost::math::llround(value)), boost::math::round(value));
    }
}

template <typename Real>
void test_lround_near_boundary()
{
   using std::ldexp;
   Real boundary = ldexp(static_cast<Real>(1), std::numeric_limits<long>::digits);

   Real value;
   int i;

    for (value = boundary, i = 0; i < 100; value = boost::math::float_next(value), ++i)
    {
        BOOST_CHECK_THROW(boost::math::lround(value), boost::math::rounding_error);
    }
    for (value = boost::math::float_prior(boundary), i = 0; i < 1000; value = boost::math::float_prior(value), ++i)
    {
        BOOST_CHECK_EQUAL(static_cast<Real>(boost::math::lround(value)), boost::math::round(value));
    }
    for (value = boost::math::float_prior(-boundary), i = 0; i < 100; value = boost::math::float_prior(value), ++i)
    {
        BOOST_CHECK_THROW(boost::math::lround(value), boost::math::rounding_error);
    }
    for (value = -boundary, i = 0; i < 1000; value = boost::math::float_next(value), ++i)
    {
        BOOST_CHECK_EQUAL(static_cast<Real>(boost::math::lround(value)), boost::math::round(value));
    }
}

template <typename Real>
void test_iround_near_boundary()
{
   using std::ldexp;
   Real boundary = ldexp(static_cast<Real>(1), std::numeric_limits<int>::digits);

   Real value;
   int i;

    for (value = boundary, i = 0; i < 100; value = boost::math::float_next(value), ++i)
    {
        BOOST_CHECK_THROW(boost::math::iround(value), boost::math::rounding_error);
    }
    for (value = boost::math::float_prior(boundary), i = 0; i < 1000; value = boost::math::float_prior(value), ++i)
    {
        BOOST_CHECK_EQUAL(static_cast<Real>(boost::math::iround(value)), boost::math::round(value));
    }
    for (value = boost::math::float_prior(-boundary), i = 0; i < 100; value = boost::math::float_prior(value), ++i)
    {
        BOOST_CHECK_THROW(boost::math::iround(value), boost::math::rounding_error);
    }
    for (value = -boundary, i = 0; i < 1000; value = boost::math::float_next(value), ++i)
    {
        BOOST_CHECK_EQUAL(static_cast<Real>(boost::math::iround(value)), boost::math::round(value));
    }  
}

template <typename Real>
void test_lltrunc_near_boundary()
{
   using std::ldexp;
   Real boundary = ldexp(static_cast<Real>(1), std::numeric_limits<long long>::digits);

   Real value;
   int i;    

    for (value = boundary, i = 0; i < 100; value = boost::math::float_next(value), ++i)
    {
        BOOST_CHECK_THROW(boost::math::lltrunc(value), boost::math::rounding_error);
    }
    for (value = boost::math::float_prior(boundary), i = 0; i < 1000; value = boost::math::float_prior(value), ++i)
    {
        BOOST_CHECK_EQUAL(static_cast<Real>(boost::math::lltrunc(value)), boost::math::lltrunc(value));
    }
    for (value = boost::math::float_prior(-boundary), i = 0; i < 100; value = boost::math::float_prior(value), ++i)
    {
        BOOST_CHECK_THROW(boost::math::lltrunc(value), boost::math::rounding_error);
    }
    for (value = -boundary, i = 0; i < 1000; value = boost::math::float_next(value), ++i)
    {
        BOOST_CHECK_EQUAL(static_cast<Real>(boost::math::lltrunc(value)), boost::math::lltrunc(value));
    } 
}

template <typename Real>
void test_ltrunc_near_boundary()
{
   using std::ldexp;
   Real boundary = ldexp(static_cast<Real>(1), std::numeric_limits<long>::digits);

   Real value;
   int i;    

    for (value = boundary, i = 0; i < 100; value = boost::math::float_next(value), ++i)
    {
        BOOST_CHECK_THROW(boost::math::ltrunc(value), boost::math::rounding_error);
    }
    for (value = boost::math::float_prior(boundary), i = 0; i < 1000; value = boost::math::float_prior(value), ++i)
    {
        BOOST_CHECK_EQUAL(static_cast<Real>(boost::math::ltrunc(value)), boost::math::ltrunc(value));
    }
    for (value = boost::math::float_prior(-boundary), i = 0; i < 100; value = boost::math::float_prior(value), ++i)
    {
        BOOST_CHECK_THROW(boost::math::ltrunc(value), boost::math::rounding_error);
    }
    for (value = -boundary, i = 0; i < 1000; value = boost::math::float_next(value), ++i)
    {
        BOOST_CHECK_EQUAL(static_cast<Real>(boost::math::ltrunc(value)), boost::math::ltrunc(value));
    } 
}

template <typename Real>
void test_itrunc_near_boundary()
{
   using std::ldexp;
   Real boundary = ldexp(static_cast<Real>(1), std::numeric_limits<int>::digits);

   Real value;
   int i;    

    for (value = boundary, i = 0; i < 100; value = boost::math::float_next(value), ++i)
    {
        BOOST_CHECK_THROW(boost::math::itrunc(value), boost::math::rounding_error);
    }
    for (value = boost::math::float_prior(boundary), i = 0; i < 1000; value = boost::math::float_prior(value), ++i)
    {
        BOOST_CHECK_EQUAL(static_cast<Real>(boost::math::itrunc(value)), boost::math::itrunc(value));
    }
    for (value = boost::math::float_prior(-boundary), i = 0; i < 100; value = boost::math::float_prior(value), ++i)
    {
        BOOST_CHECK_THROW(boost::math::itrunc(value), boost::math::rounding_error);
    }
    for (value = -boundary, i = 0; i < 1000; value = boost::math::float_next(value), ++i)
    {
        BOOST_CHECK_EQUAL(static_cast<Real>(boost::math::itrunc(value)), boost::math::itrunc(value));
    } 
}


BOOST_AUTO_TEST_CASE( test_main )
{
    // Round
    test_llround_near_boundary<float>();
    test_llround_near_boundary<double>();

    test_lround_near_boundary<float>();

    test_iround_near_boundary<float>();

    // Trunc
    test_lltrunc_near_boundary<float>();
    test_lltrunc_near_boundary<double>();

    test_ltrunc_near_boundary<float>();

    test_itrunc_near_boundary<float>();
}
