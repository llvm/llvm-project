/*
 * Copyright Nick Thompson, 2020
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <boost/math/tools/luroth_expansion.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif

using boost::math::tools::luroth_expansion;
using boost::multiprecision::cpp_bin_float_100;
using boost::math::constants::pi;

template<class Real>
void test_integral()
{
    for (int64_t i = -20; i < 20; ++i) {
        Real ii = i;
        auto luroth = luroth_expansion<Real>(ii);
        auto const & a = luroth.digits();
        CHECK_EQUAL(size_t(1), a.size());
        CHECK_EQUAL(i, a.front());
    }
}


template<class Real>
void test_halves()
{
    // x = n + 1/k => lur(x) = ((n; k - 1))
    // Note that this is a bit different that Kalpazidou (examine the half-open interval of definition carefully).
    // One way to examine this definition is correct for rationals (it never happens for irrationals)
    // is to consider i + 1/3. If you follow Kalpazidou, then you get ((i, 3, 0)); a zero digit!
    // That's bad since it destroys uniqueness and also breaks the computation of the geometric mean.
    for (int64_t i = -20; i < 20; ++i) {
        Real x = i + Real(1)/Real(2);
        auto luroth = luroth_expansion<Real>(x);
        auto const & a = luroth.digits();
        CHECK_EQUAL(size_t(2), a.size());
        CHECK_EQUAL(i, a.front());
        CHECK_EQUAL(int64_t(1), a.back());
    }

    for (int64_t i = -20; i < 20; ++i) {
        Real x = i + Real(1)/Real(4);
        auto luroth = luroth_expansion<Real>(x);
        auto const & a = luroth.digits();
        CHECK_EQUAL(size_t(2), a.size());
        CHECK_EQUAL(i, a.front());
        CHECK_EQUAL(int64_t(3), a.back());
    }

    for (int64_t i = -20; i < 20; ++i) {
        Real x = i + Real(1)/Real(8);
        auto luroth = luroth_expansion<Real>(x);
        auto const & a = luroth.digits();
        CHECK_EQUAL(size_t(2), a.size());
        CHECK_EQUAL(i, a.front());
        CHECK_EQUAL(int64_t(7), a.back());
    }
    // 1/3 is a pain because it's not representable:
    Real x = Real(1)/Real(3);
    auto luroth = luroth_expansion<Real>(x);
    auto const & a = luroth.digits();
    CHECK_EQUAL(size_t(2), a.size());
    CHECK_EQUAL(int64_t(0), a.front());
    CHECK_EQUAL(int64_t(2), a.back());
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

    #ifdef BOOST_HAS_FLOAT128
    test_integral<float128>();
    test_halves<float128>();
    #endif
    return boost::math::test::report_errors();
}
