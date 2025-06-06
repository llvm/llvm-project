//  Copyright John Maddock 2008.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Basic sanity check that header <boost/math/tr1.hpp>
// #includes all the files that it needs to.
//
#ifndef BOOST_MATH_STANDALONE
#include <boost/math/tr1.hpp>
//
// Note this header includes no other headers, this is
// important if this test is to be meaningful:
//
#include "test_compile_result.hpp"

void compile_and_link_test()
{
   unsigned ui = 0;

   check_result<float>(boost::math::tr1::assoc_laguerre(ui, ui, f));
   check_result<float>(boost::math::tr1::assoc_laguerref(ui, ui, f));
   check_result<double>(boost::math::tr1::assoc_laguerre(ui, ui, d));
   check_result<long double>(boost::math::tr1::assoc_laguerre(ui, ui, l));
   check_result<long double>(boost::math::tr1::assoc_laguerrel(ui, ui, l));
   check_result<double>(boost::math::tr1::assoc_laguerre(ui, ui, i));
   check_result<double>(boost::math::tr1::assoc_laguerre(ui, ui, ui));

   check_result<float>(boost::math::tr1::assoc_legendre(ui, ui, f));
   check_result<float>(boost::math::tr1::assoc_legendref(ui, ui, f));
   check_result<double>(boost::math::tr1::assoc_legendre(ui, ui, d));
   check_result<long double>(boost::math::tr1::assoc_legendre(ui, ui, l));
   check_result<long double>(boost::math::tr1::assoc_legendrel(ui, ui, l));
   check_result<double>(boost::math::tr1::assoc_legendre(ui, ui, i));
   check_result<double>(boost::math::tr1::assoc_legendre(ui, ui, ui));

   check_result<float>(boost::math::tr1::beta(f, f));
   check_result<float>(boost::math::tr1::betaf(f, f));
   check_result<double>(boost::math::tr1::beta(d, d));
   check_result<long double>(boost::math::tr1::beta(l, l));
   check_result<long double>(boost::math::tr1::betal(l, l));
   check_result<double>(boost::math::tr1::beta(ui, ui));
   check_result<double>(boost::math::tr1::beta(i, ui));
   check_result<double>(boost::math::tr1::beta(f, d));
   check_result<long double>(boost::math::tr1::beta(l, d));

   check_result<float>(boost::math::tr1::comp_ellint_1(f));
   check_result<float>(boost::math::tr1::comp_ellint_1f(f));
   check_result<double>(boost::math::tr1::comp_ellint_1(d));
   check_result<long double>(boost::math::tr1::comp_ellint_1(l));
   check_result<long double>(boost::math::tr1::comp_ellint_1l(l));
   check_result<double>(boost::math::tr1::comp_ellint_1(ui));
   check_result<double>(boost::math::tr1::comp_ellint_1(i));

   check_result<float>(boost::math::tr1::comp_ellint_2(f));
   check_result<float>(boost::math::tr1::comp_ellint_2f(f));
   check_result<double>(boost::math::tr1::comp_ellint_2(d));
   check_result<long double>(boost::math::tr1::comp_ellint_2(l));
   check_result<long double>(boost::math::tr1::comp_ellint_2l(l));
   check_result<double>(boost::math::tr1::comp_ellint_2(ui));
   check_result<double>(boost::math::tr1::comp_ellint_2(i));

   check_result<float>(boost::math::tr1::comp_ellint_3(f, f));
   check_result<float>(boost::math::tr1::comp_ellint_3f(f, f));
   check_result<double>(boost::math::tr1::comp_ellint_3(d, d));
   check_result<long double>(boost::math::tr1::comp_ellint_3(l, l));
   check_result<long double>(boost::math::tr1::comp_ellint_3l(l, l));
   check_result<double>(boost::math::tr1::comp_ellint_3(ui, ui));
   check_result<double>(boost::math::tr1::comp_ellint_3(i, ui));
   check_result<double>(boost::math::tr1::comp_ellint_3(f, d));
   check_result<long double>(boost::math::tr1::comp_ellint_3(l, d));

   check_result<float>(boost::math::tr1::cyl_bessel_i(f, f));
   check_result<float>(boost::math::tr1::cyl_bessel_if(f, f));
   check_result<double>(boost::math::tr1::cyl_bessel_i(d, d));
   check_result<long double>(boost::math::tr1::cyl_bessel_i(l, l));
   check_result<long double>(boost::math::tr1::cyl_bessel_il(l, l));
   check_result<double>(boost::math::tr1::cyl_bessel_i(ui, ui));
   check_result<double>(boost::math::tr1::cyl_bessel_i(i, ui));
   check_result<double>(boost::math::tr1::cyl_bessel_i(f, d));
   check_result<long double>(boost::math::tr1::cyl_bessel_i(l, d));

   check_result<float>(boost::math::tr1::cyl_bessel_j(f, f));
   check_result<float>(boost::math::tr1::cyl_bessel_jf(f, f));
   check_result<double>(boost::math::tr1::cyl_bessel_j(d, d));
   check_result<long double>(boost::math::tr1::cyl_bessel_j(l, l));
   check_result<long double>(boost::math::tr1::cyl_bessel_jl(l, l));
   check_result<double>(boost::math::tr1::cyl_bessel_j(ui, ui));
   check_result<double>(boost::math::tr1::cyl_bessel_j(i, ui));
   check_result<double>(boost::math::tr1::cyl_bessel_j(f, d));
   check_result<long double>(boost::math::tr1::cyl_bessel_j(l, d));

   check_result<float>(boost::math::tr1::cyl_bessel_k(f, f));
   check_result<float>(boost::math::tr1::cyl_bessel_kf(f, f));
   check_result<double>(boost::math::tr1::cyl_bessel_k(d, d));
   check_result<long double>(boost::math::tr1::cyl_bessel_k(l, l));
   check_result<long double>(boost::math::tr1::cyl_bessel_kl(l, l));
   check_result<double>(boost::math::tr1::cyl_bessel_k(ui, ui));
   check_result<double>(boost::math::tr1::cyl_bessel_k(i, ui));
   check_result<double>(boost::math::tr1::cyl_bessel_k(f, d));
   check_result<long double>(boost::math::tr1::cyl_bessel_k(l, d));

   check_result<float>(boost::math::tr1::cyl_neumann(f, f));
   check_result<float>(boost::math::tr1::cyl_neumannf(f, f));
   check_result<double>(boost::math::tr1::cyl_neumann(d, d));
   check_result<long double>(boost::math::tr1::cyl_neumann(l, l));
   check_result<long double>(boost::math::tr1::cyl_neumannl(l, l));
   check_result<double>(boost::math::tr1::cyl_neumann(ui, ui));
   check_result<double>(boost::math::tr1::cyl_neumann(i, ui));
   check_result<double>(boost::math::tr1::cyl_neumann(f, d));
   check_result<long double>(boost::math::tr1::cyl_neumann(l, d));

   check_result<float>(boost::math::tr1::ellint_1(f, f));
   check_result<float>(boost::math::tr1::ellint_1f(f, f));
   check_result<double>(boost::math::tr1::ellint_1(d, d));
   check_result<long double>(boost::math::tr1::ellint_1(l, l));
   check_result<long double>(boost::math::tr1::ellint_1l(l, l));
   check_result<double>(boost::math::tr1::ellint_1(ui, ui));
   check_result<double>(boost::math::tr1::ellint_1(i, ui));
   check_result<double>(boost::math::tr1::ellint_1(f, d));
   check_result<long double>(boost::math::tr1::ellint_1(l, d));

   check_result<float>(boost::math::tr1::ellint_2(f, f));
   check_result<float>(boost::math::tr1::ellint_2f(f, f));
   check_result<double>(boost::math::tr1::ellint_2(d, d));
   check_result<long double>(boost::math::tr1::ellint_2(l, l));
   check_result<long double>(boost::math::tr1::ellint_2l(l, l));
   check_result<double>(boost::math::tr1::ellint_2(ui, ui));
   check_result<double>(boost::math::tr1::ellint_2(i, ui));
   check_result<double>(boost::math::tr1::ellint_2(f, d));
   check_result<long double>(boost::math::tr1::ellint_2(l, d));

   check_result<float>(boost::math::tr1::ellint_3(f, f, f));
   check_result<float>(boost::math::tr1::ellint_3f(f, f, f));
   check_result<double>(boost::math::tr1::ellint_3(d, d, d));
   check_result<long double>(boost::math::tr1::ellint_3(l, l, l));
   check_result<long double>(boost::math::tr1::ellint_3l(l, l, l));
   check_result<double>(boost::math::tr1::ellint_3(ui, ui, i));
   check_result<double>(boost::math::tr1::ellint_3(i, ui, f));
   check_result<double>(boost::math::tr1::ellint_3(f, d, i));
   check_result<long double>(boost::math::tr1::ellint_3(l, d, f));

   check_result<float>(boost::math::tr1::expint(f));
   check_result<float>(boost::math::tr1::expintf(f));
   check_result<double>(boost::math::tr1::expint(d));
   check_result<long double>(boost::math::tr1::expint(l));
   check_result<long double>(boost::math::tr1::expintl(l));
   check_result<double>(boost::math::tr1::expint(ui));
   check_result<double>(boost::math::tr1::expint(i));

   check_result<float>(boost::math::tr1::hermite(ui, f));
   check_result<float>(boost::math::tr1::hermitef(ui, f));
   check_result<double>(boost::math::tr1::hermite(ui, d));
   check_result<long double>(boost::math::tr1::hermite(ui, l));
   check_result<long double>(boost::math::tr1::hermitel(ui, l));
   check_result<double>(boost::math::tr1::hermite(ui, i));
   check_result<double>(boost::math::tr1::hermite(ui, ui));

   check_result<float>(boost::math::tr1::laguerre(ui, f));
   check_result<float>(boost::math::tr1::laguerref(ui, f));
   check_result<double>(boost::math::tr1::laguerre(ui, d));
   check_result<long double>(boost::math::tr1::laguerre(ui, l));
   check_result<long double>(boost::math::tr1::laguerrel(ui, l));
   check_result<double>(boost::math::tr1::laguerre(ui, i));
   check_result<double>(boost::math::tr1::laguerre(ui, ui));

   check_result<float>(boost::math::tr1::legendre(ui, f));
   check_result<float>(boost::math::tr1::legendref(ui, f));
   check_result<double>(boost::math::tr1::legendre(ui, d));
   check_result<long double>(boost::math::tr1::legendre(ui, l));
   check_result<long double>(boost::math::tr1::legendrel(ui, l));
   check_result<double>(boost::math::tr1::legendre(ui, i));
   check_result<double>(boost::math::tr1::legendre(ui, ui));

   check_result<float>(boost::math::tr1::riemann_zeta(f));
   check_result<float>(boost::math::tr1::riemann_zetaf(f));
   check_result<double>(boost::math::tr1::riemann_zeta(d));
   check_result<long double>(boost::math::tr1::riemann_zeta(l));
   check_result<long double>(boost::math::tr1::riemann_zetal(l));
   check_result<double>(boost::math::tr1::riemann_zeta(ui));
   check_result<double>(boost::math::tr1::riemann_zeta(i));

   check_result<float>(boost::math::tr1::sph_bessel(ui, f));
   check_result<float>(boost::math::tr1::sph_besself(ui, f));
   check_result<double>(boost::math::tr1::sph_bessel(ui, d));
   check_result<long double>(boost::math::tr1::sph_bessel(ui, l));
   check_result<long double>(boost::math::tr1::sph_bessell(ui, l));
   check_result<double>(boost::math::tr1::sph_bessel(ui, i));
   check_result<double>(boost::math::tr1::sph_bessel(ui, ui));

   check_result<float>(boost::math::tr1::sph_legendre(ui, ui, f));
   check_result<float>(boost::math::tr1::sph_legendref(ui, ui, f));
   check_result<double>(boost::math::tr1::sph_legendre(ui, ui, d));
   check_result<long double>(boost::math::tr1::sph_legendre(ui, ui, l));
   check_result<long double>(boost::math::tr1::sph_legendrel(ui, ui, l));
   check_result<double>(boost::math::tr1::sph_legendre(ui, ui, i));
   check_result<double>(boost::math::tr1::sph_legendre(ui, ui, ui));

   check_result<float>(boost::math::tr1::sph_neumann(ui, f));
   check_result<float>(boost::math::tr1::sph_neumannf(ui, f));
   check_result<double>(boost::math::tr1::sph_neumann(ui, d));
   check_result<long double>(boost::math::tr1::sph_neumann(ui, l));
   check_result<long double>(boost::math::tr1::sph_neumannl(ui, l));
   check_result<double>(boost::math::tr1::sph_neumann(ui, i));
   check_result<double>(boost::math::tr1::sph_neumann(ui, ui));

   check_result<float>(boost::math::tr1::acosh(f));
   check_result<float>(boost::math::tr1::acoshf(f));
   check_result<double>(boost::math::tr1::acosh(d));
   check_result<long double>(boost::math::tr1::acosh(l));
   check_result<long double>(boost::math::tr1::acoshl(l));
   check_result<double>(boost::math::tr1::acosh(ui));
   check_result<double>(boost::math::tr1::acosh(i));

   check_result<float>(boost::math::tr1::asinh(f));
   check_result<float>(boost::math::tr1::asinhf(f));
   check_result<double>(boost::math::tr1::asinh(d));
   check_result<long double>(boost::math::tr1::asinh(l));
   check_result<long double>(boost::math::tr1::asinhl(l));
   check_result<double>(boost::math::tr1::asinh(ui));
   check_result<double>(boost::math::tr1::asinh(i));

   check_result<float>(boost::math::tr1::atanh(f));
   check_result<float>(boost::math::tr1::atanhf(f));
   check_result<double>(boost::math::tr1::atanh(d));
   check_result<long double>(boost::math::tr1::atanh(l));
   check_result<long double>(boost::math::tr1::atanhl(l));
   check_result<double>(boost::math::tr1::atanh(ui));
   check_result<double>(boost::math::tr1::atanh(i));

   check_result<float>(boost::math::tr1::cbrt(f));
   check_result<float>(boost::math::tr1::cbrtf(f));
   check_result<double>(boost::math::tr1::cbrt(d));
   check_result<long double>(boost::math::tr1::cbrt(l));
   check_result<long double>(boost::math::tr1::cbrtl(l));
   check_result<double>(boost::math::tr1::cbrt(ui));
   check_result<double>(boost::math::tr1::cbrt(i));

   check_result<float>(boost::math::tr1::copysign(f, f));
   check_result<float>(boost::math::tr1::copysignf(f, f));
   check_result<double>(boost::math::tr1::copysign(d, d));
   check_result<long double>(boost::math::tr1::copysign(l, l));
   check_result<long double>(boost::math::tr1::copysignl(l, l));
   check_result<double>(boost::math::tr1::copysign(d, i));
   check_result<double>(boost::math::tr1::copysign(ui, f));

   check_result<float>(boost::math::tr1::erf(f));
   check_result<float>(boost::math::tr1::erff(f));
   check_result<double>(boost::math::tr1::erf(d));
   check_result<long double>(boost::math::tr1::erf(l));
   check_result<long double>(boost::math::tr1::erfl(l));
   check_result<double>(boost::math::tr1::erf(ui));
   check_result<double>(boost::math::tr1::erf(i));

   check_result<float>(boost::math::tr1::erfc(f));
   check_result<float>(boost::math::tr1::erfcf(f));
   check_result<double>(boost::math::tr1::erfc(d));
   check_result<long double>(boost::math::tr1::erfc(l));
   check_result<long double>(boost::math::tr1::erfcl(l));
   check_result<double>(boost::math::tr1::erfc(ui));
   check_result<double>(boost::math::tr1::erfc(i));

   check_result<float>(boost::math::tr1::expm1(f));
   check_result<float>(boost::math::tr1::expm1f(f));
   check_result<double>(boost::math::tr1::expm1(d));
   check_result<long double>(boost::math::tr1::expm1(l));
   check_result<long double>(boost::math::tr1::expm1l(l));
   check_result<double>(boost::math::tr1::expm1(ui));
   check_result<double>(boost::math::tr1::expm1(i));

   check_result<float>(boost::math::tr1::fmin(f, f));
   check_result<float>(boost::math::tr1::fminf(f, f));
   check_result<double>(boost::math::tr1::fmin(d, d));
   check_result<long double>(boost::math::tr1::fmin(l, l));
   check_result<long double>(boost::math::tr1::fminl(l, l));
   check_result<double>(boost::math::tr1::fmin(d, i));
   check_result<double>(boost::math::tr1::fmin(ui, f));

   check_result<float>(boost::math::tr1::fmax(f, f));
   check_result<float>(boost::math::tr1::fmaxf(f, f));
   check_result<double>(boost::math::tr1::fmax(d, d));
   check_result<long double>(boost::math::tr1::fmax(l, l));
   check_result<long double>(boost::math::tr1::fmaxl(l, l));
   check_result<double>(boost::math::tr1::fmax(d, i));
   check_result<double>(boost::math::tr1::fmax(ui, f));

   check_result<float>(boost::math::tr1::hypot(f, f));
   check_result<float>(boost::math::tr1::hypotf(f, f));
   check_result<double>(boost::math::tr1::hypot(d, d));
   check_result<long double>(boost::math::tr1::hypot(l, l));
   check_result<long double>(boost::math::tr1::hypotl(l, l));
   check_result<double>(boost::math::tr1::hypot(d, i));
   check_result<double>(boost::math::tr1::hypot(ui, f));

   check_result<float>(boost::math::tr1::lgamma(f));
   check_result<float>(boost::math::tr1::lgammaf(f));
   check_result<double>(boost::math::tr1::lgamma(d));
   check_result<long double>(boost::math::tr1::lgamma(l));
   check_result<long double>(boost::math::tr1::lgammal(l));
   check_result<double>(boost::math::tr1::lgamma(ui));
   check_result<double>(boost::math::tr1::lgamma(i));

   check_result<long long>(boost::math::tr1::llround(f));
   check_result<long long>(boost::math::tr1::llroundf(f));
   check_result<long long>(boost::math::tr1::llround(d));
   check_result<long long>(boost::math::tr1::llround(l));
   check_result<long long>(boost::math::tr1::llroundl(l));
   check_result<long long>(boost::math::tr1::llround(ui));
   check_result<long long>(boost::math::tr1::llround(i));

   check_result<float>(boost::math::tr1::log1p(f));
   check_result<float>(boost::math::tr1::log1pf(f));
   check_result<double>(boost::math::tr1::log1p(d));
   check_result<long double>(boost::math::tr1::log1p(l));
   check_result<long double>(boost::math::tr1::log1pl(l));
   check_result<double>(boost::math::tr1::log1p(ui));
   check_result<double>(boost::math::tr1::log1p(i));

   check_result<long>(boost::math::tr1::lround(f));
   check_result<long>(boost::math::tr1::lroundf(f));
   check_result<long>(boost::math::tr1::lround(d));
   check_result<long>(boost::math::tr1::lround(l));
   check_result<long>(boost::math::tr1::lroundl(l));
   check_result<long>(boost::math::tr1::lround(ui));
   check_result<long>(boost::math::tr1::lround(i));

   check_result<float>(boost::math::tr1::round(f));
   check_result<float>(boost::math::tr1::roundf(f));
   check_result<double>(boost::math::tr1::round(d));
   check_result<long double>(boost::math::tr1::round(l));
   check_result<long double>(boost::math::tr1::roundl(l));
   check_result<double>(boost::math::tr1::round(ui));
   check_result<double>(boost::math::tr1::round(i));

   check_result<float>(boost::math::tr1::nextafter(f, f));
   check_result<float>(boost::math::tr1::nextafterf(f, f));
   check_result<double>(boost::math::tr1::nextafter(d, d));
   check_result<long double>(boost::math::tr1::nextafter(l, l));
   check_result<long double>(boost::math::tr1::nextafterl(l, l));
   check_result<double>(boost::math::tr1::nextafter(d, i));
   check_result<double>(boost::math::tr1::nextafter(ui, f));

   check_result<float>(boost::math::tr1::nexttoward(f, f));
   check_result<float>(boost::math::tr1::nexttowardf(f, f));
   check_result<double>(boost::math::tr1::nexttoward(d, d));
   check_result<long double>(boost::math::tr1::nexttoward(l, l));
   check_result<long double>(boost::math::tr1::nexttowardl(l, l));
   check_result<double>(boost::math::tr1::nexttoward(d, i));
   check_result<double>(boost::math::tr1::nexttoward(ui, f));

   check_result<float>(boost::math::tr1::tgamma(f));
   check_result<float>(boost::math::tr1::tgammaf(f));
   check_result<double>(boost::math::tr1::tgamma(d));
   check_result<long double>(boost::math::tr1::tgamma(l));
   check_result<long double>(boost::math::tr1::tgammal(l));
   check_result<double>(boost::math::tr1::tgamma(ui));
   check_result<double>(boost::math::tr1::tgamma(i));

   check_result<float>(boost::math::tr1::trunc(f));
   check_result<float>(boost::math::tr1::truncf(f));
   check_result<double>(boost::math::tr1::trunc(d));
   check_result<long double>(boost::math::tr1::trunc(l));
   check_result<long double>(boost::math::tr1::truncl(l));
   check_result<double>(boost::math::tr1::trunc(ui));
   check_result<double>(boost::math::tr1::trunc(i));

}
#endif
