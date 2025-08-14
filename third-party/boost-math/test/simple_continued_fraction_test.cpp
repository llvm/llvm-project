/*
 * Copyright Nick Thompson, 2020
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <boost/math/tools/simple_continued_fraction.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/core/demangle.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif
#include <boost/multiprecision/cpp_bin_float.hpp>

using boost::math::tools::simple_continued_fraction;
using boost::multiprecision::cpp_bin_float_100;
using boost::math::constants::pi;

template<class Real>
void test_integral()
{
    for (int64_t i = -20; i < 20; ++i) {
        Real ii = i;
        auto cfrac = simple_continued_fraction<Real>(ii);
        auto const & a = cfrac.partial_denominators();
        CHECK_EQUAL(size_t(1), a.size());
        CHECK_EQUAL(i, a.front());
    }
}

template<class Real>
void test_halves()
{
    for (int64_t i = -20; i < 20; ++i) {
        Real x = i + Real(1)/Real(2);
        auto cfrac = simple_continued_fraction<Real>(x);
        auto const & a = cfrac.partial_denominators();
        CHECK_EQUAL(size_t(2), a.size());
        CHECK_EQUAL(i, a.front());
        CHECK_EQUAL(int64_t(2), a.back());
    }

    // We'll also test quarters; why not?
    for (int64_t i = -20; i < 20; ++i) {
        Real x = i + Real(1)/Real(4);
        auto cfrac = simple_continued_fraction<Real>(x);
        auto const & a = cfrac.partial_denominators();
        CHECK_EQUAL(size_t(2), a.size());
        CHECK_EQUAL(i, a.front());
        CHECK_EQUAL(int64_t(4), a.back());
    }

    for (int64_t i = -20; i < 20; ++i) {
        Real x = i + Real(1)/Real(8);
        auto cfrac = simple_continued_fraction<Real>(x);
        auto const & a = cfrac.partial_denominators();
        CHECK_EQUAL(size_t(2), a.size());
        CHECK_EQUAL(i, a.front());
        CHECK_EQUAL(int64_t(8), a.back());
    }

    for (int64_t i = -20; i < 20; ++i) {
        Real x = i + Real(3)/Real(4);
        auto cfrac = simple_continued_fraction<Real>(x);
        auto const & a = cfrac.partial_denominators();
        CHECK_EQUAL(size_t(3), a.size());
        CHECK_EQUAL(i, a.front());
        CHECK_EQUAL(int64_t(1), a[1]);
        CHECK_EQUAL(int64_t(3), a.back());
    }

    for (int64_t i = -20; i < 20; ++i) {
        Real x = i + Real(7)/Real(8);
        auto cfrac = simple_continued_fraction<Real>(x);
        auto const & a = cfrac.partial_denominators();
        CHECK_EQUAL(size_t(3), a.size());
        CHECK_EQUAL(i, a.front());
        CHECK_EQUAL(int64_t(1), a[1]);
        CHECK_EQUAL(int64_t(7), a.back());
    }
}

template<typename Real>
void test_simple()
{
    std::cout << "Testing rational numbers on type " << boost::core::demangle(typeid(Real).name()) << "\n";
    {
        Real x = Real(649)/200;
        // ContinuedFraction[649/200] = [3; 4, 12, 4]
        auto cfrac = simple_continued_fraction(x);
        auto const & a = cfrac.partial_denominators();
        CHECK_EQUAL(size_t(4), a.size());
        CHECK_EQUAL(int64_t(3), a[0]);
        CHECK_EQUAL(int64_t(4), a[1]);
        CHECK_EQUAL(int64_t(12), a[2]);
        CHECK_EQUAL(int64_t(4), a[3]);
    }

    {
        Real x = Real(415)/Real(93);
        // [4; 2, 6, 7]:
        auto cfrac = simple_continued_fraction(x);
        auto const & a = cfrac.partial_denominators();
        CHECK_EQUAL(size_t(4), a.size());
        CHECK_EQUAL(int64_t(4), a[0]);
        CHECK_EQUAL(int64_t(2), a[1]);
        CHECK_EQUAL(int64_t(6), a[2]);
        CHECK_EQUAL(int64_t(7), a[3]);
    }

}

template<typename Real>
void test_khinchin()
{
    // These are simply sanity checks; the convergence is too slow otherwise:
    auto cfrac = simple_continued_fraction(pi<Real>());
    auto K0 = cfrac.khinchin_geometric_mean();
    CHECK_MOLLIFIED_CLOSE(Real(2.6854520010), K0, 0.1);
    auto Km1 = cfrac.khinchin_harmonic_mean();
    CHECK_MOLLIFIED_CLOSE(Real(1.74540566240), Km1, 0.1);
    
    using std::sqrt;
    auto rt_cfrac = simple_continued_fraction(sqrt(static_cast<Real>(2)));
    K0 = rt_cfrac.khinchin_geometric_mean();
    CHECK_ULP_CLOSE(Real(2), K0, 10);
    Km1 = rt_cfrac.khinchin_harmonic_mean();
    CHECK_ULP_CLOSE(Real(2), Km1, 10);
}


int main()
{
    test_integral<float>();
    test_integral<double>();
    test_integral<long double>();
    test_integral<cpp_bin_float_100>();

    test_halves<float>();
    test_halves<double>();
    test_halves<long double>();
    test_halves<cpp_bin_float_100>();

    test_simple<float>();
    test_simple<double>();
    test_simple<long double>();
    test_simple<cpp_bin_float_100>();
    
    test_khinchin<cpp_bin_float_100>();
    
    #ifdef BOOST_HAS_FLOAT128
    test_integral<float128>();
    test_halves<float128>();
    test_simple<float128>();
    test_khinchin<float128>();
    #endif
    return boost::math::test::report_errors();
}
