//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <boost/math/ccmath/sqrt.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/math/tools/assert.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>

template <typename Real>
void test_mp_sqrt()
{
    constexpr Real tol = 2*std::numeric_limits<Real>::epsilon();

    // Sqrt(2)
    constexpr Real test_val = boost::math::ccmath::sqrt(Real(2));
    constexpr Real sqrt2 = Real(1.4142135623730950488016887242096980785696718753769480731766797379Q);
    constexpr Real abs_test_error = (test_val - sqrt2) > 0 ? (test_val - sqrt2) : (sqrt2 - test_val);
    static_assert(abs_test_error < tol, "Out of tolerance");

    // inf
    constexpr Real test_inf = boost::math::ccmath::sqrt(std::numeric_limits<Real>::infinity());
    static_assert(test_inf == std::numeric_limits<Real>::infinity(), "Not infinity");

    // NAN
    constexpr Real test_nan = boost::math::ccmath::sqrt(std::numeric_limits<Real>::quiet_NaN());
    static_assert(test_nan, "Not a NAN");

    // 100'000'000
    constexpr Real test_100m = boost::math::ccmath::sqrt(100000000);
    static_assert(test_100m == 10000, "Incorrect");
}

#endif

template <typename Real>
void test_float_sqrt()
{
    using std::abs;
    
    constexpr Real tol = 2*std::numeric_limits<Real>::epsilon();
    
    constexpr Real test_val = boost::math::ccmath::sqrt(Real(2));
    constexpr Real sqrt2 = Real(1.4142135623730950488016887242096980785696718753769480731766797379L);
    constexpr Real abs_test_error = (test_val - sqrt2) > 0 ? (test_val - sqrt2) : (sqrt2 - test_val);
    static_assert(abs_test_error < tol, "Out of tolerance");

    Real known_val = std::sqrt(Real(2));
    BOOST_TEST(abs(test_val - known_val) < tol);

    // 1000 eps
    constexpr Real test_1000 = boost::math::ccmath::sqrt(1000*std::numeric_limits<Real>::epsilon());
    Real known_1000 = std::sqrt(1000*std::numeric_limits<Real>::epsilon());
    BOOST_TEST(abs(test_1000 - known_1000) < tol);

    // inf
    constexpr Real test_inf = boost::math::ccmath::sqrt(std::numeric_limits<Real>::infinity());
    static_assert(test_inf == std::numeric_limits<Real>::infinity(), "Not infinity");

    // neg inf
    constexpr Real neg_inf = boost::math::ccmath::sqrt(-std::numeric_limits<Real>::infinity());
    static_assert(boost::math::ccmath::isnan(neg_inf));
    Real stl_neg_inf = std::sqrt(-std::numeric_limits<Real>::infinity());
    BOOST_MATH_ASSERT(boost::math::fpclassify(neg_inf) == boost::math::fpclassify(stl_neg_inf));

    // NAN
    constexpr Real test_nan = boost::math::ccmath::sqrt(std::numeric_limits<Real>::quiet_NaN());
    static_assert(test_nan, "Not a NAN");

    // 100'000'000
    constexpr Real test_100m = boost::math::ccmath::sqrt(100000000);
    static_assert(test_100m == 10000, "Incorrect");

    // MAX / 2
    // Only tests float since double and long double will exceed maximum template depth
    if constexpr (std::is_same_v<float, Real>)
    {
        constexpr Real test_max = boost::math::ccmath::sqrt((std::numeric_limits<Real>::max)() / 2);
        Real known_max = std::sqrt((std::numeric_limits<Real>::max)() / 2);
        BOOST_TEST(abs(test_max - known_max) < tol);
    }
}

template <typename Z>
void test_int_sqrt()
{
    using std::abs;

    constexpr double tol = 2*std::numeric_limits<double>::epsilon();

    constexpr double test_val = boost::math::ccmath::sqrt(Z(2));
    constexpr double dummy = 1;
    static_assert(test_val > dummy, "Not constexpr");

    double known_val = std::sqrt(2.0);

    BOOST_TEST(abs(test_val - known_val) < tol);
}

// Only test on platforms that provide BOOST_MATH_IS_CONSTANT_EVALUATED
#ifndef BOOST_MATH_NO_CONSTEXPR_DETECTION
int main()
{
    test_float_sqrt<float>();
    test_float_sqrt<double>();
    
    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_float_sqrt<long double>();
    #endif

    #if defined(BOOST_MATH_TEST_FLOAT128) && !defined(BOOST_MATH_USING_BUILTIN_CONSTANT_P)  && !defined(__STRICT_ANSI__)
    test_mp_sqrt<boost::multiprecision::float128>();
    #endif

    test_int_sqrt<int>();
    test_int_sqrt<unsigned>();
    test_int_sqrt<long>();
    test_int_sqrt<std::int32_t>();
    test_int_sqrt<std::int64_t>();
    test_int_sqrt<std::uint32_t>();

    return boost::report_errors();
}
#else
int main()
{
    return 0;
}
#endif
