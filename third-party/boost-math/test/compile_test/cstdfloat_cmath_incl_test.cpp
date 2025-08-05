//  Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header
// #includes all the files that it needs to.
//
#include <boost/math/cstdfloat/cstdfloat_cmath.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
    #ifdef BOOST_FLOAT128_C
    boost::float128_t f128 = 0;

    check_result<boost::float128_t>(std::ldexp(f128, 0));
    check_result<boost::float128_t>(std::frexp(f128, 0));
    check_result<boost::float128_t>(std::fabs(f128));
    check_result<boost::float128_t>(std::abs(f128));
    check_result<boost::float128_t>(std::floor(f128));
    check_result<boost::float128_t>(std::ceil(f128));
    check_result<boost::float128_t>(std::sqrt(f128));
    check_result<boost::float128_t>(std::trunc(f128));
    check_result<boost::float128_t>(std::exp(f128));
    check_result<boost::float128_t>(std::expm1(f128));
    check_result<boost::float128_t>(std::pow(f128, 0));
    check_result<boost::float128_t>(std::log(f128));
    check_result<boost::float128_t>(std::log10(f128));
    check_result<boost::float128_t>(std::sin(f128));
    check_result<boost::float128_t>(std::cos(f128));
    check_result<boost::float128_t>(std::tan(f128));
    check_result<boost::float128_t>(std::asin(f128));
    check_result<boost::float128_t>(std::acos(f128));
    check_result<boost::float128_t>(std::atan(f128));
    check_result<boost::float128_t>(std::sinh(f128));
    check_result<boost::float128_t>(std::cosh(f128));
    check_result<boost::float128_t>(std::tanh(f128));
    check_result<boost::float128_t>(std::asinh(f128));
    check_result<boost::float128_t>(std::acosh(f128));
    check_result<boost::float128_t>(std::atanh(f128));
    check_result<boost::float128_t>(std::fmod(f128, f128));
    check_result<boost::float128_t>(std::atan2(f128, f128));
    check_result<boost::float128_t>(std::lgamma(f128));
    check_result<boost::float128_t>(std::tgamma(f128));
    check_result<boost::float128_t>(std::remainder(f128, f128));
    check_result<boost::float128_t>(std::remquo(f128, f128, 0));
    check_result<boost::float128_t>(std::fma(f128, f128, f128));
    check_result<boost::float128_t>(std::fmax(f128, f128));
    check_result<boost::float128_t>(std::fmin(f128, f128));
    check_result<boost::float128_t>(std::fdim(f128, f128));
#if __LDBL_MANT_DIG__ == 113
    check_result<boost::float128_t>(std::nanl(""));
#else
    check_result<boost::float128_t>(std::nanq(""));
#endif
    check_result<boost::float128_t>(std::exp2(f128));
    check_result<boost::float128_t>(std::log2(f128));
    check_result<boost::float128_t>(std::log1p(f128));
    check_result<boost::float128_t>(std::cbrt(f128));
    check_result<boost::float128_t>(std::hypot(f128, f128));
    check_result<boost::float128_t>(std::erf(f128));
    check_result<boost::float128_t>(std::erfc(f128));
    check_result<long long>(std::llround(f128));
    check_result<long>(std::lround(f128));
    check_result<boost::float128_t>(std::round(f128));
    check_result<boost::float128_t>(std::nearbyint(f128));
    check_result<long long>(std::llrint(f128));
    check_result<long>(std::lrint(f128));
    check_result<boost::float128_t>(std::rint(f128));
    check_result<boost::float128_t>(std::modf(f128, nullptr));
    check_result<boost::float128_t>(std::scalbln(f128, 0));
    check_result<boost::float128_t>(std::scalbn(f128, 0));
    check_result<int>(std::ilogb(f128));
    check_result<boost::float128_t>(std::logb(f128));
    check_result<boost::float128_t>(std::nextafter(f128, f128));
    check_result<boost::float128_t>(std::nexttoward(f128, f128));
    check_result<boost::float128_t>(std::copysign(f128, f128));
    check_result<bool>(std::signbit(f128));
    check_result<int>(std::fpclassify(f128));
    check_result<bool>(std::isfinite(f128));
    check_result<bool>(std::isinf(f128));
    check_result<bool>(std::isnan(f128));
    check_result<bool>(std::isnormal(f128));
    check_result<bool>(std::isgreater(f128, f128));
    check_result<bool>(std::isgreaterequal(f128, f128));
    check_result<bool>(std::isless(f128, f128));
    check_result<bool>(std::islessequal(f128, f128));
    check_result<bool>(std::islessgreater(f128, f128));
    check_result<bool>(std::isunordered(f128, f128));
    #endif // boost::float128_t
}
