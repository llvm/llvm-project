//  (C) Copyright John Maddock 2008.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/math/tools/config.hpp>
#include <math.h>  // ldexpf
#include <iostream>
#include <iomanip>

#ifdef TEST_STD
#include <cmath>
namespace tr1 = std::tr1;
#else
#include <boost/math/tr1.hpp>
namespace tr1 = boost::math::tr1;
#endif

#ifdef _MSC_VER
#  pragma warning (disable : 4100) // unreferenced formal parameter
// Can't just comment parameter out because reference or not depends on macro define.
#endif

void test_values(float, const char* name)
{
   std::cout << "Testing type " << name << std::endl;
#ifndef TEST_LD
   //
   // First the C99 math functions:
   //
   float eps = boost::math::tools::epsilon<float>();
   BOOST_CHECK_CLOSE(tr1::acoshf(std::cosh(0.5f)), 0.5f, 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::asinhf(std::sinh(0.5f)), 0.5f, 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::atanhf(std::tanh(0.5f)), 0.5f, 5000 * eps);

   BOOST_CHECK_CLOSE(tr1::cbrtf(1.5f * 1.5f * 1.5f), 1.5f, 5000 * eps);

   BOOST_CHECK(tr1::copysignf(1.0f, 1.0f) == 1.0f);
   BOOST_CHECK(tr1::copysignf(1.0f, -1.0f) == -1.0f);
   BOOST_CHECK(tr1::copysignf(-1.0f, 1.0f) == 1.0f);
   BOOST_CHECK(tr1::copysignf(-1.0f, -1.0f) == -1.0f);

   BOOST_CHECK_CLOSE(tr1::erfcf(static_cast<float>(0.125)), static_cast<float>(0.85968379519866618260697055347837660181302041685015L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfcf(static_cast<float>(0.5)), static_cast<float>(0.47950012218695346231725334610803547126354842424204L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfcf(static_cast<float>(1)), static_cast<float>(0.15729920705028513065877936491739074070393300203370L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfcf(static_cast<float>(5)), static_cast<float>(1.5374597944280348501883434853833788901180503147234e-12L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfcf(static_cast<float>(-0.125)), static_cast<float>(1.1403162048013338173930294465216233981869795831498L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfcf(static_cast<float>(-0.5)), static_cast<float>(1.5204998778130465376827466538919645287364515757580L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfcf(static_cast<float>(0)), static_cast<float>(1), eps * 1000);

   BOOST_CHECK_CLOSE(tr1::erff(static_cast<float>(0.125)), static_cast<float>(0.14031620480133381739302944652162339818697958314985L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erff(static_cast<float>(0.5)), static_cast<float>(0.52049987781304653768274665389196452873645157575796L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erff(static_cast<float>(1)), static_cast<float>(0.84270079294971486934122063508260925929606699796630L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erff(static_cast<float>(5)), static_cast<float>(0.9999999999984625402055719651498116565146166211099L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erff(static_cast<float>(-0.125)), static_cast<float>(-0.14031620480133381739302944652162339818697958314985L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erff(static_cast<float>(-0.5)), static_cast<float>(-0.52049987781304653768274665389196452873645157575796L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erff(static_cast<float>(0)), static_cast<float>(0), eps * 1000);

   BOOST_CHECK_CLOSE(tr1::log1pf(static_cast<float>(0.582029759883880615234375e0)), static_cast<float>(0.4587086807259736626531803258754840111707e0L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::expm1f(static_cast<float>(0.582029759883880615234375e0)), static_cast<float>(0.7896673415707786528734865994546559029663e0L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::log1pf(static_cast<float>(-0.2047410048544406890869140625e-1)), static_cast<float>(-0.2068660038044094868521052319477265955827e-1L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::expm1f(static_cast<float>(-0.2047410048544406890869140625e-1)), static_cast<float>(-0.2026592921724753704129022027337835687888e-1L), eps * 1000);

   BOOST_CHECK_EQUAL(tr1::fmaxf(0.1f, -0.1f), 0.1f);
   BOOST_CHECK_EQUAL(tr1::fminf(0.1f, -0.1f), -0.1f);

   BOOST_CHECK_CLOSE(tr1::hypotf(1.0f, 3.0f), std::sqrt(10.0f), eps * 500);

   BOOST_CHECK_CLOSE(tr1::lgammaf(static_cast<float>(3.5)), static_cast<float>(1.2009736023470742248160218814507129957702389154682L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgammaf(static_cast<float>(0.125)), static_cast<float>(2.0194183575537963453202905211670995899482809521344L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgammaf(static_cast<float>(-0.125)), static_cast<float>(2.1653002489051702517540619481440174064962195287626L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgammaf(static_cast<float>(-3.125)), static_cast<float>(0.1543111276840418242676072830970532952413339012367L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgammaf(static_cast<float>(-53249.0/1024)), static_cast<float>(-149.43323093420259741100038126078721302600128285894L), 5000 * eps);

   BOOST_CHECK_CLOSE(tr1::tgammaf(static_cast<float>(3.5)), static_cast<float>(3.3233509704478425511840640312646472177454052302295L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::tgammaf(static_cast<float>(0.125)), static_cast<float>(7.5339415987976119046992298412151336246104195881491L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::tgammaf(static_cast<float>(-0.125)), static_cast<float>(-8.7172188593831756100190140408231437691829605421405L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::tgammaf(static_cast<float>(-3.125)), static_cast<float>(1.1668538708507675587790157356605097019141636072094L), 5000 * eps);

#ifdef BOOST_HAS_LONG_LONG
   BOOST_CHECK(tr1::llroundf(2.5f) == 3LL);
   BOOST_CHECK(tr1::llroundf(2.25f) == 2LL);
#endif
   BOOST_CHECK(tr1::lroundf(2.5f) == 3L);
   BOOST_CHECK(tr1::lroundf(2.25f) == 2L);
   BOOST_CHECK(tr1::roundf(2.5f) == 3.0f);
   BOOST_CHECK(tr1::roundf(2.25f) == 2.0f);

   BOOST_CHECK(tr1::nextafterf(1.0f, 2.0f) > 1.0f);
   BOOST_CHECK(tr1::nextafterf(1.0f, -2.0f) < 1.0f);
   BOOST_CHECK(tr1::nextafterf(tr1::nextafterf(1.0f, 2.0f), -2.0f) == 1.0f);
   BOOST_CHECK(tr1::nextafterf(tr1::nextafterf(1.0f, -2.0f), 2.0f) == 1.0f);
   BOOST_CHECK(tr1::nextafterf(1.0f, 2.0f) > 1.0f);
   BOOST_CHECK(tr1::nextafterf(1.0f, -2.0f) < 1.0f);
   BOOST_CHECK(tr1::nextafterf(tr1::nextafterf(1.0f, 2.0f), -2.0f) == 1.0f);
   BOOST_CHECK(tr1::nextafterf(tr1::nextafterf(1.0f, -2.0f), 2.0f) == 1.0f);

   BOOST_CHECK(tr1::truncf(2.5f) == 2.0f);
   BOOST_CHECK(tr1::truncf(2.25f) == 2.0f);

   //
   // And again but without the "f" suffix on the function names:
   //
   BOOST_CHECK_CLOSE(tr1::acosh(std::cosh(0.5f)), 0.5f, 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::asinh(std::sinh(0.5f)), 0.5f, 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::atanh(std::tanh(0.5f)), 0.5f, 5000 * eps);

   BOOST_CHECK_CLOSE(tr1::cbrt(1.5f * 1.5f * 1.5f), 1.5f, 5000 * eps);

   BOOST_CHECK(tr1::copysign(1.0f, 1.0f) == 1.0f);
   BOOST_CHECK(tr1::copysign(1.0f, -1.0f) == -1.0f);
   BOOST_CHECK(tr1::copysign(-1.0f, 1.0f) == 1.0f);
   BOOST_CHECK(tr1::copysign(-1.0f, -1.0f) == -1.0f);

   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<float>(0.125)), static_cast<float>(0.85968379519866618260697055347837660181302041685015L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<float>(0.5)), static_cast<float>(0.47950012218695346231725334610803547126354842424204L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<float>(1)), static_cast<float>(0.15729920705028513065877936491739074070393300203370L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<float>(5)), static_cast<float>(1.5374597944280348501883434853833788901180503147234e-12L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<float>(-0.125)), static_cast<float>(1.1403162048013338173930294465216233981869795831498L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<float>(-0.5)), static_cast<float>(1.5204998778130465376827466538919645287364515757580L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<float>(0)), static_cast<float>(1), eps * 1000);

   BOOST_CHECK_CLOSE(tr1::erf(static_cast<float>(0.125)), static_cast<float>(0.14031620480133381739302944652162339818697958314985L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<float>(0.5)), static_cast<float>(0.52049987781304653768274665389196452873645157575796L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<float>(1)), static_cast<float>(0.84270079294971486934122063508260925929606699796630L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<float>(5)), static_cast<float>(0.9999999999984625402055719651498116565146166211099L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<float>(-0.125)), static_cast<float>(-0.14031620480133381739302944652162339818697958314985L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<float>(-0.5)), static_cast<float>(-0.52049987781304653768274665389196452873645157575796L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<float>(0)), static_cast<float>(0), eps * 1000);

   BOOST_CHECK_CLOSE(tr1::log1p(static_cast<float>(0.582029759883880615234375e0)), static_cast<float>(0.4587086807259736626531803258754840111707e0L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::expm1(static_cast<float>(0.582029759883880615234375e0)), static_cast<float>(0.7896673415707786528734865994546559029663e0L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::log1p(static_cast<float>(-0.2047410048544406890869140625e-1)), static_cast<float>(-0.2068660038044094868521052319477265955827e-1L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::expm1(static_cast<float>(-0.2047410048544406890869140625e-1)), static_cast<float>(-0.2026592921724753704129022027337835687888e-1L), eps * 1000);

   BOOST_CHECK_EQUAL(tr1::fmax(0.1f, -0.1f), 0.1f);
   BOOST_CHECK_EQUAL(tr1::fmin(0.1f, -0.1f), -0.1f);

   BOOST_CHECK_CLOSE(tr1::hypot(1.0f, 3.0f), std::sqrt(10.0f), eps * 500);

   BOOST_CHECK_CLOSE(tr1::lgamma(static_cast<float>(3.5)), static_cast<float>(1.2009736023470742248160218814507129957702389154682L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgamma(static_cast<float>(0.125)), static_cast<float>(2.0194183575537963453202905211670995899482809521344L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgamma(static_cast<float>(-0.125)), static_cast<float>(2.1653002489051702517540619481440174064962195287626L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgamma(static_cast<float>(-3.125)), static_cast<float>(0.1543111276840418242676072830970532952413339012367L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgamma(static_cast<float>(-53249.0/1024)), static_cast<float>(-149.43323093420259741100038126078721302600128285894L), 5000 * eps);

   BOOST_CHECK_CLOSE(tr1::tgamma(static_cast<float>(3.5)), static_cast<float>(3.3233509704478425511840640312646472177454052302295L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::tgamma(static_cast<float>(0.125)), static_cast<float>(7.5339415987976119046992298412151336246104195881491L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::tgamma(static_cast<float>(-0.125)), static_cast<float>(-8.7172188593831756100190140408231437691829605421405L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::tgamma(static_cast<float>(-3.125)), static_cast<float>(1.1668538708507675587790157356605097019141636072094L), 5000 * eps);

#ifdef BOOST_HAS_LONG_LONG
   BOOST_CHECK(tr1::llround(2.5f) == 3LL);
   BOOST_CHECK(tr1::llround(2.25f) == 2LL);
#endif
   BOOST_CHECK(tr1::lround(2.5f) == 3L);
   BOOST_CHECK(tr1::lround(2.25f) == 2L);
   BOOST_CHECK(tr1::round(2.5f) == 3.0f);
   BOOST_CHECK(tr1::round(2.25f) == 2.0f);

   BOOST_CHECK(tr1::nextafter(1.0f, 2.0f) > 1.0f);
   BOOST_CHECK(tr1::nextafter(1.0f, -2.0f) < 1.0f);
   BOOST_CHECK(tr1::nextafter(tr1::nextafter(1.0f, 2.0f), -2.0f) == 1.0f);
   BOOST_CHECK(tr1::nextafter(tr1::nextafter(1.0f, -2.0f), 2.0f) == 1.0f);
   BOOST_CHECK(tr1::nextafter(1.0f, 2.0f) > 1.0f);
   BOOST_CHECK(tr1::nextafter(1.0f, -2.0f) < 1.0f);
   BOOST_CHECK(tr1::nextafter(tr1::nextafter(1.0f, 2.0f), -2.0f) == 1.0f);
   BOOST_CHECK(tr1::nextafter(tr1::nextafter(1.0f, -2.0f), 2.0f) == 1.0f);

   BOOST_CHECK(tr1::trunc(2.5f) == 2.0f);
   BOOST_CHECK(tr1::trunc(2.25f) == 2.0f);

   //
   // Now for the TR1 math functions:
   //
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerref(4, 5, static_cast<float>(0.5L)), static_cast<float>(88.31510416666666666666666666666666666667L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerref(10, 0, static_cast<float>(2.5L)), static_cast<float>(-0.8802526766660982969576719576719576719577L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerref(10, 1, static_cast<float>(4.5L)), static_cast<float>(1.564311458042689732142857142857142857143L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerref(10, 6, static_cast<float>(8.5L)), static_cast<float>(20.51596541066649098875661375661375661376L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerref(10, 12, static_cast<float>(12.5L)), static_cast<float>(-199.5560968456234671241181657848324514991L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerref(50, 40, static_cast<float>(12.5L)), static_cast<float>(-4.996769495006119488583146995907246595400e16L), eps * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerref(1, static_cast<float>(0.5L)), static_cast<float>(0.5L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerref(4, static_cast<float>(0.5L)), static_cast<float>(-0.3307291666666666666666666666666666666667L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerref(7, static_cast<float>(0.5L)), static_cast<float>(-0.5183392237103174603174603174603174603175L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerref(20, static_cast<float>(0.5L)), static_cast<float>(0.3120174870800154148915399248893113634676L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerref(50, static_cast<float>(0.5L)), static_cast<float>(-0.3181388060269979064951118308575628226834L), eps * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerref(1, static_cast<float>(-0.5L)), static_cast<float>(1.5L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerref(4, static_cast<float>(-0.5L)), static_cast<float>(3.835937500000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerref(7, static_cast<float>(-0.5L)), static_cast<float>(7.950934709821428571428571428571428571429L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerref(20, static_cast<float>(-0.5L)), static_cast<float>(76.12915699869631476833699787070874048223L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerref(50, static_cast<float>(-0.5L)), static_cast<float>(2307.428631277506570629232863491518399720L), eps * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerref(1, static_cast<float>(4.5L)), static_cast<float>(-3.500000000000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerref(4, static_cast<float>(4.5L)), static_cast<float>(0.08593750000000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerref(7, static_cast<float>(4.5L)), static_cast<float>(-1.036928013392857142857142857142857142857L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerref(20, static_cast<float>(4.5L)), static_cast<float>(1.437239150257817378525582974722170737587L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerref(50, static_cast<float>(4.5L)), static_cast<float>(-0.7795068145562651416494321484050019245248L), eps * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendref(4, 2, static_cast<float>(0.5L)), static_cast<float>(4.218750000000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendref(7, 5, static_cast<float>(0.5L)), static_cast<float>(5696.789530152175143607977274672800795328L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendref(4, 2, static_cast<float>(-0.5L)), static_cast<float>(4.218750000000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendref(7, 5, static_cast<float>(-0.5L)), static_cast<float>(5696.789530152175143607977274672800795328L), eps * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::legendref(1, static_cast<float>(0.5L)), static_cast<float>(0.5L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::legendref(4, static_cast<float>(0.5L)), static_cast<float>(-0.2890625000000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::legendref(7, static_cast<float>(0.5L)), static_cast<float>(0.2231445312500000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::legendref(40, static_cast<float>(0.5L)), static_cast<float>(-0.09542943523261546936538467572384923220258L), eps * 100);

   float sv = eps / 1024;
   BOOST_CHECK_CLOSE(tr1::betaf(static_cast<float>(1), static_cast<float>(1)), static_cast<float>(1), eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::betaf(static_cast<float>(1), static_cast<float>(4)), static_cast<float>(0.25), eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::betaf(static_cast<float>(4), static_cast<float>(1)), static_cast<float>(0.25), eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::betaf(sv, static_cast<float>(4)), 1/sv, eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::betaf(static_cast<float>(4), sv), 1/sv, eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::betaf(static_cast<float>(4), static_cast<float>(20)), static_cast<float>(0.00002823263692828910220214568040654997176736L), eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::betaf(static_cast<float>(0.0125L), static_cast<float>(0.000023L)), static_cast<float>(43558.24045647538375006349016083320744662L), eps * 20 * 100);

   BOOST_CHECK_CLOSE(tr1::comp_ellint_1f(static_cast<float>(0)), static_cast<float>(1.5707963267948966192313216916397514420985846996876), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1f(static_cast<float>(0.125)), static_cast<float>(1.5769867712158131421244030532288080803822271060839), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1f(static_cast<float>(0.25)), static_cast<float>(1.5962422221317835101489690714979498795055744578951), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1f(static_cast<float>(300)/1024), static_cast<float>(1.6062331054696636704261124078746600894998873503208), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1f(static_cast<float>(400)/1024), static_cast<float>(1.6364782007562008756208066125715722889067992997614), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1f(static_cast<float>(-0.5)), static_cast<float>(1.6857503548125960428712036577990769895008008941411), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1f(static_cast<float>(-0.75)), static_cast<float>(1.9109897807518291965531482187613425592531451316788), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::comp_ellint_2f(static_cast<float>(-1)), static_cast<float>(1), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2f(static_cast<float>(0)), static_cast<float>(1.5707963267948966192313216916397514420985846996876), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2f(static_cast<float>(100) / 1024), static_cast<float>(1.5670445330545086723323795143598956428788609133377), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2f(static_cast<float>(200) / 1024), static_cast<float>(1.5557071588766556854463404816624361127847775545087), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2f(static_cast<float>(300) / 1024), static_cast<float>(1.5365278991162754883035625322482669608948678755743), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2f(static_cast<float>(400) / 1024), static_cast<float>(1.5090417763083482272165682786143770446401437564021), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2f(static_cast<float>(-0.5)), static_cast<float>(1.4674622093394271554597952669909161360253617523272), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2f(static_cast<float>(-600) / 1024), static_cast<float>(1.4257538571071297192428217218834579920545946473778), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2f(static_cast<float>(-800) / 1024), static_cast<float>(1.2927868476159125056958680222998765985004489572909), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2f(static_cast<float>(-900) / 1024), static_cast<float>(1.1966864890248739524112920627353824133420353430982), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::comp_ellint_3f(static_cast<float>(0.2), static_cast<float>(0)), static_cast<float>(1.586867847454166237308008033828114192951), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3f(static_cast<float>(0.4), static_cast<float>(0)), static_cast<float>(1.639999865864511206865258329748601457626), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3f(static_cast<float>(0), static_cast<float>(0)), static_cast<float>(1.57079632679489661923132169163975144209858469968755291048747), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3f(static_cast<float>(0), static_cast<float>(0.5)), static_cast<float>(2.221441469079183123507940495030346849307), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3f(static_cast<float>(0.3), static_cast<float>(-4)), static_cast<float>(0.712708870925620061597924858162260293305195624270730660081949), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3f(static_cast<float>(-0.5), static_cast<float>(-1e+05)), static_cast<float>(0.00496944596485066055800109163256108604615568144080386919012831), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3f(static_cast<float>(-0.75), static_cast<float>(-1e+10)), static_cast<float>(0.0000157080225184890546939710019277357161497407143903832703317801), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3f(static_cast<float>(-0.875), static_cast<float>(1) / 1024), static_cast<float>(2.18674503176462374414944618968850352696579451638002110619287), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3f(static_cast<float>(-0.875), static_cast<float>(1023)/1024), static_cast<float>(101.045289804941384100960063898569538919135722087486350366997), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::cyl_bessel_if(static_cast<float>(2.25), static_cast<float>(1)/(1024*1024)), static_cast<float>(2.34379212133481347189068464680335815256364262507955635911656e-15), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_if(static_cast<float>(5.5), static_cast<float>(3.125)), static_cast<float>(0.0583514045989371500460946536220735787163510569634133670181210), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_if(static_cast<float>(-5) + static_cast<float>(1)/1024, static_cast<float>(2.125)), static_cast<float>(0.0267920938009571023702933210070984416052633027166975342895062), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_if(static_cast<float>(-5.5), static_cast<float>(10)), static_cast<float>(597.577606961369169607937419869926705730305175364662688426534), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_if(static_cast<float>(-10486074)/(1024*1024), static_cast<float>(1)/1024), static_cast<float>(1.41474005665181350367684623930576333542989766867888186478185e35), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_if(static_cast<float>(-10486074)/(1024*1024), static_cast<float>(50)), static_cast<float>(1.07153277202900671531087024688681954238311679648319534644743e20), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_if(static_cast<float>(144794)/1024, static_cast<float>(100)), static_cast<float>(2066.27694757392660413922181531984160871678224178890247540320), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_if(static_cast<float>(-144794)/1024, static_cast<float>(100)), static_cast<float>(2066.27694672763190927440969155740243346136463461655104698748), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(2457)/1024, static_cast<float>(1)/1024), static_cast<float>(3.80739920118603335646474073457326714709615200130620574875292e-9), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(5.5), static_cast<float>(3217)/1024), static_cast<float>(0.0281933076257506091621579544064767140470089107926550720453038), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(-5.5), static_cast<float>(3217)/1024), static_cast<float>(-2.55820064470647911823175836997490971806135336759164272675969), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(-5.5), static_cast<float>(1e+04)), static_cast<float>(2.449843111985605522111159013846599118397e-03), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(5.5), static_cast<float>(1e+04)), static_cast<float>(0.00759343502722670361395585198154817047185480147294665270646578), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(5.5), static_cast<float>(1e+06)), static_cast<float>(-0.000747424248595630177396350688505919533097973148718960064663632), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(5.125), static_cast<float>(1e+06)), static_cast<float>(-0.000776600124835704280633640911329691642748783663198207360238214), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(5.875), static_cast<float>(1e+06)), static_cast<float>(-0.000466322721115193071631008581529503095819705088484386434589780), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(0.5), static_cast<float>(101)), static_cast<float>(0.0358874487875643822020496677692429287863419555699447066226409), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(-5.5), static_cast<float>(1e+04)), static_cast<float>(0.00244984311198560552211115901384659911839737686676766460822577), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(-5.5), static_cast<float>(1e+06)), static_cast<float>(0.000279243200433579511095229508894156656558211060453622750659554), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(-0.5), static_cast<float>(101)), static_cast<float>(0.0708184798097594268482290389188138201440114881159344944791454), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(-10486074) / (1024*1024), static_cast<float>(1)/1024), static_cast<float>(1.41474013160494695750009004222225969090304185981836460288562e35), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(-10486074) / (1024*1024), static_cast<float>(15)), static_cast<float>(-0.0902239288885423309568944543848111461724911781719692852541489), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(10486074) / (1024*1024), static_cast<float>(1e+02)), static_cast<float>(-0.0547064914615137807616774867984047583596945624129838091326863), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(10486074) / (1024*1024), static_cast<float>(2e+04)), static_cast<float>(-0.00556783614400875611650958980796060611309029233226596737701688), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jf(static_cast<float>(-10486074) / (1024*1024), static_cast<float>(1e+02)), static_cast<float>(-0.0547613660316806551338637153942604550779513947674222863858713), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kf(static_cast<float>(0.5), static_cast<float>(0.875)), static_cast<float>(0.558532231646608646115729767013630967055657943463362504577189), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kf(static_cast<float>(0.5), static_cast<float>(1.125)), static_cast<float>(0.383621010650189547146769320487006220295290256657827220786527), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kf(static_cast<float>(2.25), static_cast<float>(std::ldexp(1.0, -30))), static_cast<float>(5.62397392719283271332307799146649700147907612095185712015604e20), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kf(static_cast<float>(5.5), static_cast<float>(3217)/1024), static_cast<float>(1.30623288775012596319554857587765179889689223531159532808379), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kf(static_cast<float>(-5.5), static_cast<float>(10)), static_cast<float>(0.0000733045300798502164644836879577484533096239574909573072142667), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kf(static_cast<float>(-5.5), static_cast<float>(100)), static_cast<float>(5.41274555306792267322084448693957747924412508020839543293369e-45), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kf(static_cast<float>(10240)/1024, static_cast<float>(1)/1024), static_cast<float>(2.35522579263922076203415803966825431039900000000993410734978e38), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kf(static_cast<float>(10240)/1024, static_cast<float>(10)), static_cast<float>(0.00161425530039067002345725193091329085443750382929208307802221), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kf(static_cast<float>(144793)/1024, static_cast<float>(100)), static_cast<float>(1.39565245860302528069481472855619216759142225046370312329416e-6), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kf(static_cast<float>(144793)/1024, static_cast<float>(200)), static_cast<float>(9.11950412043225432171915100042647230802198254567007382956336e-68), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::cyl_neumannf(static_cast<float>(0.5), static_cast<float>(1) / (1024*1024)), static_cast<float>(-817.033790261762580469303126467917092806755460418223776544122), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannf(static_cast<float>(5.5), static_cast<float>(3.125)), static_cast<float>(-2.61489440328417468776474188539366752698192046890955453259866), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannf(static_cast<float>(-5.5), static_cast<float>(3.125)), static_cast<float>(-0.0274994493896489729948109971802244976377957234563871795364056), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannf(static_cast<float>(-5.5), static_cast<float>(1e+04)), static_cast<float>(-0.00759343502722670361395585198154817047185480147294665270646578), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannf(static_cast<float>(-10486074) / (1024*1024), static_cast<float>(1)/1024), static_cast<float>(-1.50382374389531766117868938966858995093408410498915220070230e38), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannf(static_cast<float>(-10486074) / (1024*1024), static_cast<float>(1e+02)), static_cast<float>(0.0583041891319026009955779707640455341990844522293730214223545), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannf(static_cast<float>(141.75), static_cast<float>(1e+02)), static_cast<float>(-5.38829231428696507293191118661269920130838607482708483122068e9), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannf(static_cast<float>(141.75), static_cast<float>(2e+04)), static_cast<float>(-0.00376577888677186194728129112270988602876597726657372330194186), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannf(static_cast<float>(-141.75), static_cast<float>(1e+02)), static_cast<float>(-3.81009803444766877495905954105669819951653361036342457919021e9), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(0), static_cast<float>(0)), static_cast<float>(0), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(0), static_cast<float>(-10)), static_cast<float>(-10), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(-1), static_cast<float>(-1)), static_cast<float>(-1.2261911708835170708130609674719067527242483502207), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(0.875), static_cast<float>(-4)), static_cast<float>(-5.3190556182262405182189463092940736859067548232647), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(-0.625), static_cast<float>(8)), static_cast<float>(9.0419973860310100524448893214394562615252527557062), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(0.875), static_cast<float>(1e-05)), static_cast<float>(0.000010000000000127604166668510945638036143355898993088), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(10)/1024, static_cast<float>(1e+05)), static_cast<float>(100002.38431454899771096037307519328741455615271038), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(1), static_cast<float>(1e-20)), static_cast<float>(1.0000000000000000000000000000000000000000166666667e-20), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(1e-20), static_cast<float>(1e-20)), static_cast<float>(1.000000000000000e-20), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(400)/1024, static_cast<float>(1e+20)), static_cast<float>(1.0418143796499216839719289963154558027005142709763e20), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(0.5), static_cast<float>(2)), static_cast<float>(2.1765877052210673672479877957388515321497888026770), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(0.5), static_cast<float>(4)), static_cast<float>(4.2543274975235836861894752787874633017836785640477), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(0.5), static_cast<float>(6)), static_cast<float>(6.4588766202317746302999080620490579800463614807916), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(0.5), static_cast<float>(10)), static_cast<float>(10.697409951222544858346795279378531495869386960090), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(0.5), static_cast<float>(-2)), static_cast<float>(-2.1765877052210673672479877957388515321497888026770), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(0.5), static_cast<float>(-4)), static_cast<float>(-4.2543274975235836861894752787874633017836785640477), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(0.5), static_cast<float>(-6)), static_cast<float>(-6.4588766202317746302999080620490579800463614807916), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1f(static_cast<float>(0.5), static_cast<float>(-10)), static_cast<float>(-10.697409951222544858346795279378531495869386960090), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::ellint_2f(static_cast<float>(0), static_cast<float>(0)), static_cast<float>(0), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2f(static_cast<float>(0), static_cast<float>(-10)), static_cast<float>(-10), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2f(static_cast<float>(-1), static_cast<float>(-1)), static_cast<float>(-0.84147098480789650665250232163029899962256306079837), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2f(static_cast<float>(900) / 1024, static_cast<float>(-4)), static_cast<float>(-3.1756145986492562317862928524528520686391383168377), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2f(static_cast<float>(-600) / 1024, static_cast<float>(8)), static_cast<float>(7.2473147180505693037677015377802777959345489333465), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2f(static_cast<float>(800) / 1024, static_cast<float>(1e-05)), static_cast<float>(9.999999999898274739584436515967055859383969942432E-6), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2f(static_cast<float>(100) / 1024, static_cast<float>(1e+05)), static_cast<float>(99761.153306972066658135668386691227343323331995888), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2f(static_cast<float>(-0.5), static_cast<float>(1e+10)), static_cast<float>(9.3421545766487137036576748555295222252286528414669e9), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2f(static_cast<float>(400) / 1024, ldexp(static_cast<float>(1), 66)), static_cast<float>(7.0886102721911705466476846969992069994308167515242e19), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0), static_cast<float>(1), static_cast<float>(-1)), static_cast<float>(-1.557407724654902230506974807458360173087), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0.4), static_cast<float>(0), static_cast<float>(-4)), static_cast<float>(-4.153623371196831087495427530365430979011), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(-0.6), static_cast<float>(0), static_cast<float>(8)), static_cast<float>(8.935930619078575123490612395578518914416), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0.25), static_cast<float>(0), static_cast<float>(0.5)), static_cast<float>(0.501246705365439492445236118603525029757890291780157969500480), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0), static_cast<float>(0), static_cast<float>(0.5)), static_cast<float>(0.5), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0), static_cast<float>(-2), static_cast<float>(0.5)), static_cast<float>(0.437501067017546278595664813509803743009132067629603474488486), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0), static_cast<float>(0.25), static_cast<float>(0.5)), static_cast<float>(0.510269830229213412212501938035914557628394166585442994564135), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0), static_cast<float>(0.75), static_cast<float>(0.5)), static_cast<float>(0.533293253875952645421201146925578536430596894471541312806165), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0), static_cast<float>(0.75), static_cast<float>(0.75)), static_cast<float>(0.871827580412760575085768367421866079353646112288567703061975), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0), static_cast<float>(1), static_cast<float>(0.25)), static_cast<float>(0.255341921221036266504482236490473678204201638800822621740476), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0), static_cast<float>(2), static_cast<float>(0.25)), static_cast<float>(0.261119051639220165094943572468224137699644963125853641716219), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0), static_cast<float>(1023)/1024, static_cast<float>(1.5)), static_cast<float>(13.2821612239764190363647953338544569682942329604483733197131), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0.5), static_cast<float>(0.5), static_cast<float>(-1)), static_cast<float>(-1.228014414316220642611298946293865487807), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0.5), static_cast<float>(0.5), static_cast<float>(1e+10)), static_cast<float>(1.536591003599172091573590441336982730551e+10), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0.75), static_cast<float>(-1e+05), static_cast<float>(10)), static_cast<float>(0.0347926099493147087821620459290460547131012904008557007934290), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0.875), static_cast<float>(-1e+10), static_cast<float>(10)), static_cast<float>(0.000109956202759561502329123384755016959364346382187364656768212), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0.875), static_cast<float>(-1e+10), static_cast<float>(1e+20)), static_cast<float>(1.00000626665567332602765201107198822183913978895904937646809e15), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0.875), static_cast<float>(-1e+10), static_cast<float>(1608)/1024), static_cast<float>(0.0000157080616044072676127333183571107873332593142625043567690379), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0.875), 1-static_cast<float>(1) / 1024, static_cast<float>(1e+20)), static_cast<float>(6.43274293944380717581167058274600202023334985100499739678963e21), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0.25), static_cast<float>(50), static_cast<float>(0.1)), static_cast<float>(0.124573770342749525407523258569507331686458866564082916835900), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3f(static_cast<float>(0.25), static_cast<float>(1.125), static_cast<float>(1)), static_cast<float>(1.77299767784815770192352979665283069318388205110727241629752), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::expintf(static_cast<float>(1)/1024), static_cast<float>(-6.35327933972759151358547423727042905862963067106751711596065L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expintf(static_cast<float>(0.125)), static_cast<float>(-1.37320852494298333781545045921206470808223543321810480716122L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expintf(static_cast<float>(0.5)), static_cast<float>(0.454219904863173579920523812662802365281405554352642045162818L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expintf(static_cast<float>(1)), static_cast<float>(1.89511781635593675546652093433163426901706058173270759164623L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expintf(static_cast<float>(50.5)), static_cast<float>(1.72763195602911805201155668940185673806099654090456049881069e20L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expintf(static_cast<float>(-1)/1024), static_cast<float>(-6.35523246483107180261445551935803221293763008553775821607264L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expintf(static_cast<float>(-0.125)), static_cast<float>(-1.62342564058416879145630692462440887363310605737209536579267L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expintf(static_cast<float>(-0.5)), static_cast<float>(-0.559773594776160811746795939315085235226846890316353515248293L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expintf(static_cast<float>(-1)), static_cast<float>(-0.219383934395520273677163775460121649031047293406908207577979L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expintf(static_cast<float>(-50.5)), static_cast<float>(-2.27237132932219350440719707268817831250090574830769670186618e-24L), eps * 5000);

   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitef(0, static_cast<float>(1)), static_cast<float>(1.L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitef(1, static_cast<float>(1)), static_cast<float>(2.L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitef(1, static_cast<float>(2)), static_cast<float>(4.L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitef(1, static_cast<float>(10)), static_cast<float>(20), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitef(1, static_cast<float>(100)), static_cast<float>(200), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitef(1, static_cast<float>(1e6)), static_cast<float>(2e6), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitef(10, static_cast<float>(30)), static_cast<float>(5.896624628001300E+17L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitef(10, static_cast<float>(1000)), static_cast<float>(1.023976960161280E+33L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitef(10, static_cast<float>(10)), static_cast<float>(8.093278209760000E+12L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitef(10, static_cast<float>(-10)), static_cast<float>(8.093278209760000E+12L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitef(3, static_cast<float>(-10)), static_cast<float>(-7.880000000000000E+3L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitef(3, static_cast<float>(-1000)), static_cast<float>(-7.999988000000000E+9L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitef(3, static_cast<float>(-1000000)), static_cast<float>(-7.999999999988000E+18L), 100 * eps);

   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(0.125)), static_cast<float>(-0.63277562349869525529352526763564627152686379131122L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(1023) / static_cast<float>(1024)), static_cast<float>(-1023.4228554489429786541032870895167448906103303056L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(1025) / static_cast<float>(1024)), static_cast<float>(1024.5772867695045940578681624248887776501597556226L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(0.5)), static_cast<float>(-1.46035450880958681288949915251529801246722933101258149054289L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(1.125)), static_cast<float>(8.5862412945105752999607544082693023591996301183069L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(2)), static_cast<float>(1.6449340668482264364724151666460251892189499012068L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(3.5)), static_cast<float>(1.1267338673170566464278124918549842722219969574036L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(4)), static_cast<float>(1.08232323371113819151600369654116790277475095191872690768298L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(4 + static_cast<float>(1) / 1024), static_cast<float>(1.08225596856391369799036835439238249195298434901488518878804L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(4.5)), static_cast<float>(1.05470751076145426402296728896028011727249383295625173068468L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(6.5)), static_cast<float>(1.01200589988852479610078491680478352908773213619144808841031L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(7.5)), static_cast<float>(1.00582672753652280770224164440459408011782510096320822989663L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(8.125)), static_cast<float>(1.0037305205308161603183307711439385250181080293472L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(16.125)), static_cast<float>(1.0000140128224754088474783648500235958510030511915L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(0)), static_cast<float>(-0.5L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(-0.125)), static_cast<float>(-0.39906966894504503550986928301421235400280637468895L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(-1)), static_cast<float>(-0.083333333333333333333333333333333333333333333333333L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(-2)), static_cast<float>(0L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(-2.5)), static_cast<float>(0.0085169287778503305423585670283444869362759902200745L), eps * 5000 * 3);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(-3)), static_cast<float>(0.0083333333333333333333333333333333333333333333333333L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(-4)), static_cast<float>(0), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(-20)), static_cast<float>(0), eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(-21)), static_cast<float>(-281.46014492753623188405797101449275362318840579710L), eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::riemann_zetaf(static_cast<float>(-30.125)), static_cast<float>(2.2762941726834511267740045451463455513839970804578e7L), eps * 5000 * 100);

   BOOST_CHECK_CLOSE(tr1::sph_besself(0, static_cast<float>(0.1433600485324859619140625e-1)), static_cast<float>(0.9999657468461303487880990241993035937654e0),  eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_besself(0, static_cast<float>(0.1760916970670223236083984375e-1)), static_cast<float>(0.9999483203249623334100130061926184665364e0),  eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_besself(2, static_cast<float>(0.1433600485324859619140625e-1)), static_cast<float>(0.1370120120703995134662099191103188366059e-4),  eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_besself(2, static_cast<float>(0.1760916970670223236083984375e-1)), static_cast<float>(0.2067173265753174063228459655801741280461e-4),  eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_besself(7, static_cast<float>(0.1252804412841796875e3)), static_cast<float>(0.7887555711993028736906736576314283291289e-2),  eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_besself(7, static_cast<float>(0.25554705810546875e3)), static_cast<float>(-0.1463292767579579943284849187188066532514e-2),  eps * 5000 * 100);

   BOOST_CHECK_CLOSE(tr1::sph_neumannf(0, static_cast<float>(0.408089816570281982421875e0)), static_cast<float>(-0.2249212131304610409189209411089291558038e1), eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_neumannf(0, static_cast<float>(0.6540834903717041015625e0)), static_cast<float>(-0.1213309779166084571756446746977955970241e1),   eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_neumannf(2, static_cast<float>(0.408089816570281982421875e0)), static_cast<float>(-0.4541702641837159203058389758895634766256e2),   eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_neumannf(2, static_cast<float>(0.6540834903717041015625e0)), static_cast<float>(-0.1156112621471167110574129561700037138981e2),   eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_neumannf(10, static_cast<float>(0.1097540378570556640625e1)), static_cast<float>(-0.2427889658115064857278886600528596240123e9),   eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_neumannf(10, static_cast<float>(0.30944411754608154296875e1)), static_cast<float>(-0.3394649246350136450439882104151313759251e4),   eps * 5000 * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::sph_legendref(3, 2, static_cast<float>(0.5)), static_cast<float>(0.2061460599687871330692286791802688341213L), eps * 5000);
   BOOST_CHECK_CLOSE_FRACTION(tr1::sph_legendref(40, 15, static_cast<float>(0.75)), static_cast<float>(-0.406036847302819452666908966769096223205057182668333862900509L), eps * 5000);

   //
   // Now all over again but without the "f" suffix on the function names this time:
   //
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(4, 5, static_cast<float>(0.5L)), static_cast<float>(88.31510416666666666666666666666666666667L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(10, 0, static_cast<float>(2.5L)), static_cast<float>(-0.8802526766660982969576719576719576719577L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(10, 1, static_cast<float>(4.5L)), static_cast<float>(1.564311458042689732142857142857142857143L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(10, 6, static_cast<float>(8.5L)), static_cast<float>(20.51596541066649098875661375661375661376L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(10, 12, static_cast<float>(12.5L)), static_cast<float>(-199.5560968456234671241181657848324514991L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(50, 40, static_cast<float>(12.5L)), static_cast<float>(-4.996769495006119488583146995907246595400e16L), eps * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(1, static_cast<float>(0.5L)), static_cast<float>(0.5L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(4, static_cast<float>(0.5L)), static_cast<float>(-0.3307291666666666666666666666666666666667L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(7, static_cast<float>(0.5L)), static_cast<float>(-0.5183392237103174603174603174603174603175L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(20, static_cast<float>(0.5L)), static_cast<float>(0.3120174870800154148915399248893113634676L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(50, static_cast<float>(0.5L)), static_cast<float>(-0.3181388060269979064951118308575628226834L), eps * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(1, static_cast<float>(-0.5L)), static_cast<float>(1.5L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(4, static_cast<float>(-0.5L)), static_cast<float>(3.835937500000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(7, static_cast<float>(-0.5L)), static_cast<float>(7.950934709821428571428571428571428571429L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(20, static_cast<float>(-0.5L)), static_cast<float>(76.12915699869631476833699787070874048223L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(50, static_cast<float>(-0.5L)), static_cast<float>(2307.428631277506570629232863491518399720L), eps * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(1, static_cast<float>(4.5L)), static_cast<float>(-3.500000000000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(4, static_cast<float>(4.5L)), static_cast<float>(0.08593750000000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(7, static_cast<float>(4.5L)), static_cast<float>(-1.036928013392857142857142857142857142857L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(20, static_cast<float>(4.5L)), static_cast<float>(1.437239150257817378525582974722170737587L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(50, static_cast<float>(4.5L)), static_cast<float>(-0.7795068145562651416494321484050019245248L), eps * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendre(4, 2, static_cast<float>(0.5L)), static_cast<float>(4.218750000000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendre(7, 5, static_cast<float>(0.5L)), static_cast<float>(5696.789530152175143607977274672800795328L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendre(4, 2, static_cast<float>(-0.5L)), static_cast<float>(4.218750000000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendre(7, 5, static_cast<float>(-0.5L)), static_cast<float>(5696.789530152175143607977274672800795328L), eps * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::legendre(1, static_cast<float>(0.5L)), static_cast<float>(0.5L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::legendre(4, static_cast<float>(0.5L)), static_cast<float>(-0.2890625000000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::legendre(7, static_cast<float>(0.5L)), static_cast<float>(0.2231445312500000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::legendre(40, static_cast<float>(0.5L)), static_cast<float>(-0.09542943523261546936538467572384923220258L), eps * 100);

   BOOST_CHECK_CLOSE(tr1::beta(static_cast<float>(1), static_cast<float>(1)), static_cast<float>(1), eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::beta(static_cast<float>(1), static_cast<float>(4)), static_cast<float>(0.25), eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::beta(static_cast<float>(4), static_cast<float>(1)), static_cast<float>(0.25), eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::beta(sv, static_cast<float>(4)), 1/sv, eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::beta(static_cast<float>(4), sv), 1/sv, eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::beta(static_cast<float>(4), static_cast<float>(20)), static_cast<float>(0.00002823263692828910220214568040654997176736L), eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::beta(static_cast<float>(0.0125L), static_cast<float>(0.000023L)), static_cast<float>(43558.24045647538375006349016083320744662L), eps * 20 * 100);

   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<float>(0)), static_cast<float>(1.5707963267948966192313216916397514420985846996876), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<float>(0.125)), static_cast<float>(1.5769867712158131421244030532288080803822271060839), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<float>(0.25)), static_cast<float>(1.5962422221317835101489690714979498795055744578951), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<float>(300)/1024), static_cast<float>(1.6062331054696636704261124078746600894998873503208), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<float>(400)/1024), static_cast<float>(1.6364782007562008756208066125715722889067992997614), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<float>(-0.5)), static_cast<float>(1.6857503548125960428712036577990769895008008941411), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<float>(-0.75)), static_cast<float>(1.9109897807518291965531482187613425592531451316788), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<float>(-1)), static_cast<float>(1), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<float>(0)), static_cast<float>(1.5707963267948966192313216916397514420985846996876), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<float>(100) / 1024), static_cast<float>(1.5670445330545086723323795143598956428788609133377), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<float>(200) / 1024), static_cast<float>(1.5557071588766556854463404816624361127847775545087), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<float>(300) / 1024), static_cast<float>(1.5365278991162754883035625322482669608948678755743), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<float>(400) / 1024), static_cast<float>(1.5090417763083482272165682786143770446401437564021), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<float>(-0.5)), static_cast<float>(1.4674622093394271554597952669909161360253617523272), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<float>(-600) / 1024), static_cast<float>(1.4257538571071297192428217218834579920545946473778), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<float>(-800) / 1024), static_cast<float>(1.2927868476159125056958680222998765985004489572909), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<float>(-900) / 1024), static_cast<float>(1.1966864890248739524112920627353824133420353430982), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<float>(0.2), static_cast<float>(0)), static_cast<float>(1.586867847454166237308008033828114192951), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<float>(0.4), static_cast<float>(0)), static_cast<float>(1.639999865864511206865258329748601457626), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<float>(0), static_cast<float>(0)), static_cast<float>(1.57079632679489661923132169163975144209858469968755291048747), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<float>(0), static_cast<float>(0.5)), static_cast<float>(2.221441469079183123507940495030346849307), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<float>(0.3), static_cast<float>(-4)), static_cast<float>(0.712708870925620061597924858162260293305195624270730660081949), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<float>(-0.5), static_cast<float>(-1e+05)), static_cast<float>(0.00496944596485066055800109163256108604615568144080386919012831), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<float>(-0.75), static_cast<float>(-1e+10)), static_cast<float>(0.0000157080225184890546939710019277357161497407143903832703317801), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<float>(-0.875), static_cast<float>(1) / 1024), static_cast<float>(2.18674503176462374414944618968850352696579451638002110619287), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<float>(-0.875), static_cast<float>(1023)/1024), static_cast<float>(101.045289804941384100960063898569538919135722087486350366997), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<float>(2.25), static_cast<float>(1)/(1024*1024)), static_cast<float>(2.34379212133481347189068464680335815256364262507955635911656e-15), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<float>(5.5), static_cast<float>(3.125)), static_cast<float>(0.0583514045989371500460946536220735787163510569634133670181210), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<float>(-5) + static_cast<float>(1)/1024, static_cast<float>(2.125)), static_cast<float>(0.0267920938009571023702933210070984416052633027166975342895062), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<float>(-5.5), static_cast<float>(10)), static_cast<float>(597.577606961369169607937419869926705730305175364662688426534), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<float>(-10486074)/(1024*1024), static_cast<float>(1)/1024), static_cast<float>(1.41474005665181350367684623930576333542989766867888186478185e35), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<float>(-10486074)/(1024*1024), static_cast<float>(50)), static_cast<float>(1.07153277202900671531087024688681954238311679648319534644743e20), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<float>(144794)/1024, static_cast<float>(100)), static_cast<float>(2066.27694757392660413922181531984160871678224178890247540320), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<float>(-144794)/1024, static_cast<float>(100)), static_cast<float>(2066.27694672763190927440969155740243346136463461655104698748), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(2457)/1024, static_cast<float>(1)/1024), static_cast<float>(3.80739920118603335646474073457326714709615200130620574875292e-9), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(5.5), static_cast<float>(3217)/1024), static_cast<float>(0.0281933076257506091621579544064767140470089107926550720453038), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(-5.5), static_cast<float>(3217)/1024), static_cast<float>(-2.55820064470647911823175836997490971806135336759164272675969), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(-5.5), static_cast<float>(1e+04)), static_cast<float>(2.449843111985605522111159013846599118397e-03), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(5.5), static_cast<float>(1e+04)), static_cast<float>(0.00759343502722670361395585198154817047185480147294665270646578), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(5.5), static_cast<float>(1e+06)), static_cast<float>(-0.000747424248595630177396350688505919533097973148718960064663632), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(5.125), static_cast<float>(1e+06)), static_cast<float>(-0.000776600124835704280633640911329691642748783663198207360238214), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(5.875), static_cast<float>(1e+06)), static_cast<float>(-0.000466322721115193071631008581529503095819705088484386434589780), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(0.5), static_cast<float>(101)), static_cast<float>(0.0358874487875643822020496677692429287863419555699447066226409), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(-5.5), static_cast<float>(1e+04)), static_cast<float>(0.00244984311198560552211115901384659911839737686676766460822577), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(-5.5), static_cast<float>(1e+06)), static_cast<float>(0.000279243200433579511095229508894156656558211060453622750659554), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(-0.5), static_cast<float>(101)), static_cast<float>(0.0708184798097594268482290389188138201440114881159344944791454), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(-10486074) / (1024*1024), static_cast<float>(1)/1024), static_cast<float>(1.41474013160494695750009004222225969090304185981836460288562e35), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(-10486074) / (1024*1024), static_cast<float>(15)), static_cast<float>(-0.0902239288885423309568944543848111461724911781719692852541489), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(10486074) / (1024*1024), static_cast<float>(1e+02)), static_cast<float>(-0.0547064914615137807616774867984047583596945624129838091326863), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(10486074) / (1024*1024), static_cast<float>(2e+04)), static_cast<float>(-0.00556783614400875611650958980796060611309029233226596737701688), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<float>(-10486074) / (1024*1024), static_cast<float>(1e+02)), static_cast<float>(-0.0547613660316806551338637153942604550779513947674222863858713), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<float>(0.5), static_cast<float>(0.875)), static_cast<float>(0.558532231646608646115729767013630967055657943463362504577189), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<float>(0.5), static_cast<float>(1.125)), static_cast<float>(0.383621010650189547146769320487006220295290256657827220786527), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<float>(2.25), static_cast<float>(std::ldexp(1.0, -30))), static_cast<float>(5.62397392719283271332307799146649700147907612095185712015604e20), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<float>(5.5), static_cast<float>(3217)/1024), static_cast<float>(1.30623288775012596319554857587765179889689223531159532808379), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<float>(-5.5), static_cast<float>(10)), static_cast<float>(0.0000733045300798502164644836879577484533096239574909573072142667), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<float>(-5.5), static_cast<float>(100)), static_cast<float>(5.41274555306792267322084448693957747924412508020839543293369e-45), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<float>(10240)/1024, static_cast<float>(1)/1024), static_cast<float>(2.35522579263922076203415803966825431039900000000993410734978e38), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<float>(10240)/1024, static_cast<float>(10)), static_cast<float>(0.00161425530039067002345725193091329085443750382929208307802221), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<float>(144793)/1024, static_cast<float>(100)), static_cast<float>(1.39565245860302528069481472855619216759142225046370312329416e-6), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<float>(144793)/1024, static_cast<float>(200)), static_cast<float>(9.11950412043225432171915100042647230802198254567007382956336e-68), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<float>(0.5), static_cast<float>(1) / (1024*1024)), static_cast<float>(-817.033790261762580469303126467917092806755460418223776544122), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<float>(5.5), static_cast<float>(3.125)), static_cast<float>(-2.61489440328417468776474188539366752698192046890955453259866), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<float>(-5.5), static_cast<float>(3.125)), static_cast<float>(-0.0274994493896489729948109971802244976377957234563871795364056), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<float>(-5.5), static_cast<float>(1e+04)), static_cast<float>(-0.00759343502722670361395585198154817047185480147294665270646578), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<float>(-10486074) / (1024*1024), static_cast<float>(1)/1024), static_cast<float>(-1.50382374389531766117868938966858995093408410498915220070230e38), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<float>(-10486074) / (1024*1024), static_cast<float>(1e+02)), static_cast<float>(0.0583041891319026009955779707640455341990844522293730214223545), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<float>(141.75), static_cast<float>(1e+02)), static_cast<float>(-5.38829231428696507293191118661269920130838607482708483122068e9), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<float>(141.75), static_cast<float>(2e+04)), static_cast<float>(-0.00376577888677186194728129112270988602876597726657372330194186), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<float>(-141.75), static_cast<float>(1e+02)), static_cast<float>(-3.81009803444766877495905954105669819951653361036342457919021e9), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(0), static_cast<float>(0)), static_cast<float>(0), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(0), static_cast<float>(-10)), static_cast<float>(-10), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(-1), static_cast<float>(-1)), static_cast<float>(-1.2261911708835170708130609674719067527242483502207), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(0.875), static_cast<float>(-4)), static_cast<float>(-5.3190556182262405182189463092940736859067548232647), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(-0.625), static_cast<float>(8)), static_cast<float>(9.0419973860310100524448893214394562615252527557062), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(0.875), static_cast<float>(1e-05)), static_cast<float>(0.000010000000000127604166668510945638036143355898993088), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(10)/1024, static_cast<float>(1e+05)), static_cast<float>(100002.38431454899771096037307519328741455615271038), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(1), static_cast<float>(1e-20)), static_cast<float>(1.0000000000000000000000000000000000000000166666667e-20), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(1e-20), static_cast<float>(1e-20)), static_cast<float>(1.000000000000000e-20), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(400)/1024, static_cast<float>(1e+20)), static_cast<float>(1.0418143796499216839719289963154558027005142709763e20), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(0.5), static_cast<float>(2)), static_cast<float>(2.1765877052210673672479877957388515321497888026770), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(0.5), static_cast<float>(4)), static_cast<float>(4.2543274975235836861894752787874633017836785640477), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(0.5), static_cast<float>(6)), static_cast<float>(6.4588766202317746302999080620490579800463614807916), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(0.5), static_cast<float>(10)), static_cast<float>(10.697409951222544858346795279378531495869386960090), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(0.5), static_cast<float>(-2)), static_cast<float>(-2.1765877052210673672479877957388515321497888026770), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(0.5), static_cast<float>(-4)), static_cast<float>(-4.2543274975235836861894752787874633017836785640477), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(0.5), static_cast<float>(-6)), static_cast<float>(-6.4588766202317746302999080620490579800463614807916), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<float>(0.5), static_cast<float>(-10)), static_cast<float>(-10.697409951222544858346795279378531495869386960090), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<float>(0), static_cast<float>(0)), static_cast<float>(0), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<float>(0), static_cast<float>(-10)), static_cast<float>(-10), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<float>(-1), static_cast<float>(-1)), static_cast<float>(-0.84147098480789650665250232163029899962256306079837), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<float>(900) / 1024, static_cast<float>(-4)), static_cast<float>(-3.1756145986492562317862928524528520686391383168377), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<float>(-600) / 1024, static_cast<float>(8)), static_cast<float>(7.2473147180505693037677015377802777959345489333465), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<float>(800) / 1024, static_cast<float>(1e-05)), static_cast<float>(9.999999999898274739584436515967055859383969942432E-6), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<float>(100) / 1024, static_cast<float>(1e+05)), static_cast<float>(99761.153306972066658135668386691227343323331995888), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<float>(-0.5), static_cast<float>(1e+10)), static_cast<float>(9.3421545766487137036576748555295222252286528414669e9), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<float>(400) / 1024, static_cast<float>(ldexp(static_cast<double>(1), 66))), static_cast<float>(7.0886102721911705466476846969992069994308167515242e19), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0), static_cast<float>(1), static_cast<float>(-1)), static_cast<float>(-1.557407724654902230506974807458360173087), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0.4), static_cast<float>(0), static_cast<float>(-4)), static_cast<float>(-4.153623371196831087495427530365430979011), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(-0.6), static_cast<float>(0), static_cast<float>(8)), static_cast<float>(8.935930619078575123490612395578518914416), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0.25), static_cast<float>(0), static_cast<float>(0.5)), static_cast<float>(0.501246705365439492445236118603525029757890291780157969500480), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0), static_cast<float>(0), static_cast<float>(0.5)), static_cast<float>(0.5), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0), static_cast<float>(-2), static_cast<float>(0.5)), static_cast<float>(0.437501067017546278595664813509803743009132067629603474488486), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0), static_cast<float>(0.25), static_cast<float>(0.5)), static_cast<float>(0.510269830229213412212501938035914557628394166585442994564135), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0), static_cast<float>(0.75), static_cast<float>(0.5)), static_cast<float>(0.533293253875952645421201146925578536430596894471541312806165), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0), static_cast<float>(0.75), static_cast<float>(0.75)), static_cast<float>(0.871827580412760575085768367421866079353646112288567703061975), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0), static_cast<float>(1), static_cast<float>(0.25)), static_cast<float>(0.255341921221036266504482236490473678204201638800822621740476), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0), static_cast<float>(2), static_cast<float>(0.25)), static_cast<float>(0.261119051639220165094943572468224137699644963125853641716219), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0), static_cast<float>(1023)/1024, static_cast<float>(1.5)), static_cast<float>(13.2821612239764190363647953338544569682942329604483733197131), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0.5), static_cast<float>(0.5), static_cast<float>(-1)), static_cast<float>(-1.228014414316220642611298946293865487807), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0.5), static_cast<float>(0.5), static_cast<float>(1e+10)), static_cast<float>(1.536591003599172091573590441336982730551e+10), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0.75), static_cast<float>(-1e+05), static_cast<float>(10)), static_cast<float>(0.0347926099493147087821620459290460547131012904008557007934290), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0.875), static_cast<float>(-1e+10), static_cast<float>(10)), static_cast<float>(0.000109956202759561502329123384755016959364346382187364656768212), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0.875), static_cast<float>(-1e+10), static_cast<float>(1e+20)), static_cast<float>(1.00000626665567332602765201107198822183913978895904937646809e15), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0.875), static_cast<float>(-1e+10), static_cast<float>(1608)/1024), static_cast<float>(0.0000157080616044072676127333183571107873332593142625043567690379), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0.875), 1-static_cast<float>(1) / 1024, static_cast<float>(1e+20)), static_cast<float>(6.43274293944380717581167058274600202023334985100499739678963e21), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0.25), static_cast<float>(50), static_cast<float>(0.1)), static_cast<float>(0.124573770342749525407523258569507331686458866564082916835900), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<float>(0.25), static_cast<float>(1.125), static_cast<float>(1)), static_cast<float>(1.77299767784815770192352979665283069318388205110727241629752), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::expint(static_cast<float>(1)/1024), static_cast<float>(-6.35327933972759151358547423727042905862963067106751711596065L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<float>(0.125)), static_cast<float>(-1.37320852494298333781545045921206470808223543321810480716122L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<float>(0.5)), static_cast<float>(0.454219904863173579920523812662802365281405554352642045162818L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<float>(1)), static_cast<float>(1.89511781635593675546652093433163426901706058173270759164623L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<float>(50.5)), static_cast<float>(1.72763195602911805201155668940185673806099654090456049881069e20L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<float>(-1)/1024), static_cast<float>(-6.35523246483107180261445551935803221293763008553775821607264L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<float>(-0.125)), static_cast<float>(-1.62342564058416879145630692462440887363310605737209536579267L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<float>(-0.5)), static_cast<float>(-0.559773594776160811746795939315085235226846890316353515248293L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<float>(-1)), static_cast<float>(-0.219383934395520273677163775460121649031047293406908207577979L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<float>(-50.5)), static_cast<float>(-2.27237132932219350440719707268817831250090574830769670186618e-24L), eps * 5000);

   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(0, static_cast<float>(1)), static_cast<float>(1.L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(1, static_cast<float>(1)), static_cast<float>(2.L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(1, static_cast<float>(2)), static_cast<float>(4.L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(1, static_cast<float>(10)), static_cast<float>(20), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(1, static_cast<float>(100)), static_cast<float>(200), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(1, static_cast<float>(1e6)), static_cast<float>(2e6), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(10, static_cast<float>(30)), static_cast<float>(5.896624628001300E+17L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(10, static_cast<float>(1000)), static_cast<float>(1.023976960161280E+33L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(10, static_cast<float>(10)), static_cast<float>(8.093278209760000E+12L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(10, static_cast<float>(-10)), static_cast<float>(8.093278209760000E+12L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(3, static_cast<float>(-10)), static_cast<float>(-7.880000000000000E+3L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(3, static_cast<float>(-1000)), static_cast<float>(-7.999988000000000E+9L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(3, static_cast<float>(-1000000)), static_cast<float>(-7.999999999988000E+18L), 100 * eps);

   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(0.125)), static_cast<float>(-0.63277562349869525529352526763564627152686379131122L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(1023) / static_cast<float>(1024)), static_cast<float>(-1023.4228554489429786541032870895167448906103303056L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(1025) / static_cast<float>(1024)), static_cast<float>(1024.5772867695045940578681624248887776501597556226L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(0.5)), static_cast<float>(-1.46035450880958681288949915251529801246722933101258149054289L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(1.125)), static_cast<float>(8.5862412945105752999607544082693023591996301183069L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(2)), static_cast<float>(1.6449340668482264364724151666460251892189499012068L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(3.5)), static_cast<float>(1.1267338673170566464278124918549842722219969574036L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(4)), static_cast<float>(1.08232323371113819151600369654116790277475095191872690768298L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(4 + static_cast<float>(1) / 1024), static_cast<float>(1.08225596856391369799036835439238249195298434901488518878804L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(4.5)), static_cast<float>(1.05470751076145426402296728896028011727249383295625173068468L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(6.5)), static_cast<float>(1.01200589988852479610078491680478352908773213619144808841031L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(7.5)), static_cast<float>(1.00582672753652280770224164440459408011782510096320822989663L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(8.125)), static_cast<float>(1.0037305205308161603183307711439385250181080293472L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(16.125)), static_cast<float>(1.0000140128224754088474783648500235958510030511915L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(0)), static_cast<float>(-0.5L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(-0.125)), static_cast<float>(-0.39906966894504503550986928301421235400280637468895L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(-1)), static_cast<float>(-0.083333333333333333333333333333333333333333333333333L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(-2)), static_cast<float>(0L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(-2.5)), static_cast<float>(0.0085169287778503305423585670283444869362759902200745L), eps * 5000 * 3);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(-3)), static_cast<float>(0.0083333333333333333333333333333333333333333333333333L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(-4)), static_cast<float>(0), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(-20)), static_cast<float>(0), eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(-21)), static_cast<float>(-281.46014492753623188405797101449275362318840579710L), eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<float>(-30.125)), static_cast<float>(2.2762941726834511267740045451463455513839970804578e7L), eps * 5000 * 100);

   BOOST_CHECK_CLOSE(tr1::sph_bessel(0, static_cast<float>(0.1433600485324859619140625e-1)), static_cast<float>(0.9999657468461303487880990241993035937654e0),  eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_bessel(0, static_cast<float>(0.1760916970670223236083984375e-1)), static_cast<float>(0.9999483203249623334100130061926184665364e0),  eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_bessel(2, static_cast<float>(0.1433600485324859619140625e-1)), static_cast<float>(0.1370120120703995134662099191103188366059e-4),  eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_bessel(2, static_cast<float>(0.1760916970670223236083984375e-1)), static_cast<float>(0.2067173265753174063228459655801741280461e-4),  eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_bessel(7, static_cast<float>(0.1252804412841796875e3)), static_cast<float>(0.7887555711993028736906736576314283291289e-2),  eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_bessel(7, static_cast<float>(0.25554705810546875e3)), static_cast<float>(-0.1463292767579579943284849187188066532514e-2),  eps * 5000 * 100);

   BOOST_CHECK_CLOSE(tr1::sph_neumann(0, static_cast<float>(0.408089816570281982421875e0)), static_cast<float>(-0.2249212131304610409189209411089291558038e1), eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_neumann(0, static_cast<float>(0.6540834903717041015625e0)), static_cast<float>(-0.1213309779166084571756446746977955970241e1),   eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_neumann(2, static_cast<float>(0.408089816570281982421875e0)), static_cast<float>(-0.4541702641837159203058389758895634766256e2),   eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_neumann(2, static_cast<float>(0.6540834903717041015625e0)), static_cast<float>(-0.1156112621471167110574129561700037138981e2),   eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_neumann(10, static_cast<float>(0.1097540378570556640625e1)), static_cast<float>(-0.2427889658115064857278886600528596240123e9),   eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_neumann(10, static_cast<float>(0.30944411754608154296875e1)), static_cast<float>(-0.3394649246350136450439882104151313759251e4),   eps * 5000 * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::sph_legendre(3, 2, static_cast<float>(0.5)), static_cast<float>(0.2061460599687871330692286791802688341213L), eps * 5000);
   BOOST_CHECK_CLOSE_FRACTION(tr1::sph_legendre(40, 15, static_cast<float>(0.75)), static_cast<float>(-0.406036847302819452666908966769096223205057182668333862900509L), eps * 5000);
#endif
}

void test_values(double, const char* name)
{
   std::cout << "Testing type " << name << std::endl;

#ifndef TEST_LD
   double eps = boost::math::tools::epsilon<double>();
   BOOST_CHECK_CLOSE(tr1::acosh(std::cosh(0.5)), 0.5, 500 * eps);
   BOOST_CHECK_CLOSE(tr1::asinh(std::sinh(0.5)), 0.5, 500 * eps);
   BOOST_CHECK_CLOSE(tr1::atanh(std::tanh(0.5)), 0.5, 500 * eps);

   BOOST_CHECK_CLOSE(tr1::cbrt(1.5 * 1.5 * 1.5), 1.5, 500 * eps);

   BOOST_CHECK(tr1::copysign(1.0, 1.0) == 1.0);
   BOOST_CHECK(tr1::copysign(1.0, -1.0) == -1.0);
   BOOST_CHECK(tr1::copysign(-1.0, 1.0) == 1.0);
   BOOST_CHECK(tr1::copysign(-1.0, -1.0) == -1.0);

   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<double>(0.125)), static_cast<double>(0.85968379519866618260697055347837660181302041685015L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<double>(0.5)), static_cast<double>(0.47950012218695346231725334610803547126354842424204L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<double>(1)), static_cast<double>(0.15729920705028513065877936491739074070393300203370L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<double>(5)), static_cast<double>(1.5374597944280348501883434853833788901180503147234e-12L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<double>(-0.125)), static_cast<double>(1.1403162048013338173930294465216233981869795831498L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<double>(-0.5)), static_cast<double>(1.5204998778130465376827466538919645287364515757580L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<double>(0)), static_cast<double>(1), eps * 1000);

   BOOST_CHECK_CLOSE(tr1::erf(static_cast<double>(0.125)), static_cast<double>(0.14031620480133381739302944652162339818697958314985L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<double>(0.5)), static_cast<double>(0.52049987781304653768274665389196452873645157575796L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<double>(1)), static_cast<double>(0.84270079294971486934122063508260925929606699796630L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<double>(5)), static_cast<double>(0.9999999999984625402055719651498116565146166211099L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<double>(-0.125)), static_cast<double>(-0.14031620480133381739302944652162339818697958314985L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<double>(-0.5)), static_cast<double>(-0.52049987781304653768274665389196452873645157575796L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<double>(0)), static_cast<double>(0), eps * 1000);

   BOOST_CHECK_CLOSE(tr1::log1p(static_cast<double>(0.582029759883880615234375e0)), static_cast<double>(0.4587086807259736626531803258754840111707e0L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::expm1(static_cast<double>(0.582029759883880615234375e0)), static_cast<double>(0.7896673415707786528734865994546559029663e0L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::log1p(static_cast<double>(-0.2047410048544406890869140625e-1)), static_cast<double>(-0.2068660038044094868521052319477265955827e-1L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::expm1(static_cast<double>(-0.2047410048544406890869140625e-1)), static_cast<double>(-0.2026592921724753704129022027337835687888e-1L), eps * 1000);

   BOOST_CHECK_EQUAL(tr1::fmax(0.1, -0.1), 0.1);
   BOOST_CHECK_EQUAL(tr1::fmin(0.1, -0.1), -0.1);

   BOOST_CHECK_CLOSE(tr1::hypot(1.0, 3.0), std::sqrt(10.0), eps * 500);

   BOOST_CHECK_CLOSE(tr1::lgamma(static_cast<double>(3.5)), static_cast<double>(1.2009736023470742248160218814507129957702389154682L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgamma(static_cast<double>(0.125)), static_cast<double>(2.0194183575537963453202905211670995899482809521344L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgamma(static_cast<double>(-0.125)), static_cast<double>(2.1653002489051702517540619481440174064962195287626L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgamma(static_cast<double>(-3.125)), static_cast<double>(0.1543111276840418242676072830970532952413339012367L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgamma(static_cast<double>(-53249.0/1024)), static_cast<double>(-149.43323093420259741100038126078721302600128285894L), 5000 * eps);

   BOOST_CHECK_CLOSE(tr1::tgamma(static_cast<double>(3.5)), static_cast<double>(3.3233509704478425511840640312646472177454052302295L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::tgamma(static_cast<double>(0.125)), static_cast<double>(7.5339415987976119046992298412151336246104195881491L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::tgamma(static_cast<double>(-0.125)), static_cast<double>(-8.7172188593831756100190140408231437691829605421405L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::tgamma(static_cast<double>(-3.125)), static_cast<double>(1.1668538708507675587790157356605097019141636072094L), 5000 * eps);

#ifdef BOOST_HAS_LONG_LONG
   BOOST_CHECK(tr1::llround(2.5) == 3LL);
   BOOST_CHECK(tr1::llround(2.25) == 2LL);
#endif
   BOOST_CHECK(tr1::lround(2.5) == 3L);
   BOOST_CHECK(tr1::lround(2.25) == 2L);
   BOOST_CHECK(tr1::round(2.5) == 3.0);
   BOOST_CHECK(tr1::round(2.25) == 2.0);

   BOOST_CHECK(tr1::nextafter(1.0, 2.0) > 1.0);
   BOOST_CHECK(tr1::nextafter(1.0, -2.0) < 1.0);
   BOOST_CHECK(tr1::nextafter(tr1::nextafter(1.0, 2.0), -2.0) == 1.0);
   BOOST_CHECK(tr1::nextafter(tr1::nextafter(1.0, -2.0), 2.0) == 1.0);
   BOOST_CHECK(tr1::nextafter(1.0, 2.0) > 1.0);
   BOOST_CHECK(tr1::nextafter(1.0, -2.0) < 1.0);
   BOOST_CHECK(tr1::nextafter(tr1::nextafter(1.0, 2.0), -2.0) == 1.0);
   BOOST_CHECK(tr1::nextafter(tr1::nextafter(1.0, -2.0), 2.0) == 1.0);

   BOOST_CHECK(tr1::trunc(2.5) == 2.0);
   BOOST_CHECK(tr1::trunc(2.25) == 2.0);

   //
   // Now the TR1 functions:
   //
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(4, 5, static_cast<double>(0.5L)), static_cast<double>(88.31510416666666666666666666666666666667L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(10, 0, static_cast<double>(2.5L)), static_cast<double>(-0.8802526766660982969576719576719576719577L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(10, 1, static_cast<double>(4.5L)), static_cast<double>(1.564311458042689732142857142857142857143L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(10, 6, static_cast<double>(8.5L)), static_cast<double>(20.51596541066649098875661375661375661376L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(10, 12, static_cast<double>(12.5L)), static_cast<double>(-199.5560968456234671241181657848324514991L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(50, 40, static_cast<double>(12.5L)), static_cast<double>(-4.996769495006119488583146995907246595400e16L), eps * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(1, static_cast<double>(0.5L)), static_cast<double>(0.5L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(4, static_cast<double>(0.5L)), static_cast<double>(-0.3307291666666666666666666666666666666667L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(7, static_cast<double>(0.5L)), static_cast<double>(-0.5183392237103174603174603174603174603175L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(20, static_cast<double>(0.5L)), static_cast<double>(0.3120174870800154148915399248893113634676L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(50, static_cast<double>(0.5L)), static_cast<double>(-0.3181388060269979064951118308575628226834L), eps * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(1, static_cast<double>(-0.5L)), static_cast<double>(1.5L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(4, static_cast<double>(-0.5L)), static_cast<double>(3.835937500000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(7, static_cast<double>(-0.5L)), static_cast<double>(7.950934709821428571428571428571428571429L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(20, static_cast<double>(-0.5L)), static_cast<double>(76.12915699869631476833699787070874048223L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(50, static_cast<double>(-0.5L)), static_cast<double>(2307.428631277506570629232863491518399720L), eps * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(1, static_cast<double>(4.5L)), static_cast<double>(-3.500000000000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(4, static_cast<double>(4.5L)), static_cast<double>(0.08593750000000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(7, static_cast<double>(4.5L)), static_cast<double>(-1.036928013392857142857142857142857142857L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(20, static_cast<double>(4.5L)), static_cast<double>(1.437239150257817378525582974722170737587L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(50, static_cast<double>(4.5L)), static_cast<double>(-0.7795068145562651416494321484050019245248L), eps * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendre(4, 2, static_cast<double>(0.5L)), static_cast<double>(4.218750000000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendre(7, 5, static_cast<double>(0.5L)), static_cast<double>(5696.789530152175143607977274672800795328L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendre(4, 2, static_cast<double>(-0.5L)), static_cast<double>(4.218750000000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendre(7, 5, static_cast<double>(-0.5L)), static_cast<double>(5696.789530152175143607977274672800795328L), eps * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::legendre(1, static_cast<double>(0.5L)), static_cast<double>(0.5L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::legendre(4, static_cast<double>(0.5L)), static_cast<double>(-0.2890625000000000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::legendre(7, static_cast<double>(0.5L)), static_cast<double>(0.2231445312500000000000000000000000000000L), eps * 100);
   BOOST_CHECK_CLOSE_FRACTION(tr1::legendre(40, static_cast<double>(0.5L)), static_cast<double>(-0.09542943523261546936538467572384923220258L), eps * 100);

   double sv = eps / 1024;

   BOOST_CHECK_CLOSE(tr1::beta(static_cast<double>(1), static_cast<double>(1)), static_cast<double>(1), eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::beta(static_cast<double>(1), static_cast<double>(4)), static_cast<double>(0.25), eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::beta(static_cast<double>(4), static_cast<double>(1)), static_cast<double>(0.25), eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::beta(sv, static_cast<double>(4)), 1/sv, eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::beta(static_cast<double>(4), sv), 1/sv, eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::beta(static_cast<double>(4), static_cast<double>(20)), static_cast<double>(0.00002823263692828910220214568040654997176736L), eps * 20 * 100);
   BOOST_CHECK_CLOSE(tr1::beta(static_cast<double>(0.0125L), static_cast<double>(0.000023L)), static_cast<double>(43558.24045647538375006349016083320744662L), eps * 20 * 100);

   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<double>(0)), static_cast<double>(1.5707963267948966192313216916397514420985846996876), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<double>(0.125)), static_cast<double>(1.5769867712158131421244030532288080803822271060839), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<double>(0.25)), static_cast<double>(1.5962422221317835101489690714979498795055744578951), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<double>(300)/1024), static_cast<double>(1.6062331054696636704261124078746600894998873503208), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<double>(400)/1024), static_cast<double>(1.6364782007562008756208066125715722889067992997614), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<double>(-0.5)), static_cast<double>(1.6857503548125960428712036577990769895008008941411), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<double>(-0.75)), static_cast<double>(1.9109897807518291965531482187613425592531451316788), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<double>(-1)), static_cast<double>(1), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<double>(0)), static_cast<double>(1.5707963267948966192313216916397514420985846996876), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<double>(100) / 1024), static_cast<double>(1.5670445330545086723323795143598956428788609133377), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<double>(200) / 1024), static_cast<double>(1.5557071588766556854463404816624361127847775545087), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<double>(300) / 1024), static_cast<double>(1.5365278991162754883035625322482669608948678755743), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<double>(400) / 1024), static_cast<double>(1.5090417763083482272165682786143770446401437564021), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<double>(-0.5)), static_cast<double>(1.4674622093394271554597952669909161360253617523272), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<double>(-600) / 1024), static_cast<double>(1.4257538571071297192428217218834579920545946473778), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<double>(-800) / 1024), static_cast<double>(1.2927868476159125056958680222998765985004489572909), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<double>(-900) / 1024), static_cast<double>(1.1966864890248739524112920627353824133420353430982), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<double>(0.2), static_cast<double>(0)), static_cast<double>(1.586867847454166237308008033828114192951), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<double>(0.4), static_cast<double>(0)), static_cast<double>(1.639999865864511206865258329748601457626), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<double>(0), static_cast<double>(0)), static_cast<double>(1.57079632679489661923132169163975144209858469968755291048747), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<double>(0), static_cast<double>(0.5)), static_cast<double>(2.221441469079183123507940495030346849307), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<double>(0.3), static_cast<double>(-4)), static_cast<double>(0.712708870925620061597924858162260293305195624270730660081949), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<double>(-0.5), static_cast<double>(-1e+05)), static_cast<double>(0.00496944596485066055800109163256108604615568144080386919012831), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<double>(-0.75), static_cast<double>(-1e+10)), static_cast<double>(0.0000157080225184890546939710019277357161497407143903832703317801), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<double>(-0.875), static_cast<double>(1) / 1024), static_cast<double>(2.18674503176462374414944618968850352696579451638002110619287), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<double>(-0.875), static_cast<double>(1023)/1024), static_cast<double>(101.045289804941384100960063898569538919135722087486350366997), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<double>(2.25), static_cast<double>(1)/(1024*1024)), static_cast<double>(2.34379212133481347189068464680335815256364262507955635911656e-15), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<double>(5.5), static_cast<double>(3.125)), static_cast<double>(0.0583514045989371500460946536220735787163510569634133670181210), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<double>(-5) + static_cast<double>(1)/1024, static_cast<double>(2.125)), static_cast<double>(0.0267920938009571023702933210070984416052633027166975342895062), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<double>(-5.5), static_cast<double>(10)), static_cast<double>(597.577606961369169607937419869926705730305175364662688426534), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<double>(-10486074)/(1024*1024), static_cast<double>(1)/1024), static_cast<double>(1.41474005665181350367684623930576333542989766867888186478185e35), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<double>(-10486074)/(1024*1024), static_cast<double>(50)), static_cast<double>(1.07153277202900671531087024688681954238311679648319534644743e20), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<double>(144794)/1024, static_cast<double>(100)), static_cast<double>(2066.27694757392660413922181531984160871678224178890247540320), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<double>(-144794)/1024, static_cast<double>(100)), static_cast<double>(2066.27694672763190927440969155740243346136463461655104698748), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(2457)/1024, static_cast<double>(1)/1024), static_cast<double>(3.80739920118603335646474073457326714709615200130620574875292e-9), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(5.5), static_cast<double>(3217)/1024), static_cast<double>(0.0281933076257506091621579544064767140470089107926550720453038), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(-5.5), static_cast<double>(3217)/1024), static_cast<double>(-2.55820064470647911823175836997490971806135336759164272675969), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(-5.5), static_cast<double>(1e+04)), static_cast<double>(2.449843111985605522111159013846599118397e-03), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(5.5), static_cast<double>(1e+04)), static_cast<double>(0.00759343502722670361395585198154817047185480147294665270646578), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(5.5), static_cast<double>(1e+06)), static_cast<double>(-0.000747424248595630177396350688505919533097973148718960064663632), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(5.125), static_cast<double>(1e+06)), static_cast<double>(-0.000776600124835704280633640911329691642748783663198207360238214), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(5.875), static_cast<double>(1e+06)), static_cast<double>(-0.000466322721115193071631008581529503095819705088484386434589780), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(0.5), static_cast<double>(101)), static_cast<double>(0.0358874487875643822020496677692429287863419555699447066226409), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(-5.5), static_cast<double>(1e+04)), static_cast<double>(0.00244984311198560552211115901384659911839737686676766460822577), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(-5.5), static_cast<double>(1e+06)), static_cast<double>(0.000279243200433579511095229508894156656558211060453622750659554), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(-0.5), static_cast<double>(101)), static_cast<double>(0.0708184798097594268482290389188138201440114881159344944791454), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(-10486074) / (1024*1024), static_cast<double>(1)/1024), static_cast<double>(1.41474013160494695750009004222225969090304185981836460288562e35), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(-10486074) / (1024*1024), static_cast<double>(15)), static_cast<double>(-0.0902239288885423309568944543848111461724911781719692852541489), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(10486074) / (1024*1024), static_cast<double>(1e+02)), static_cast<double>(-0.0547064914615137807616774867984047583596945624129838091326863), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(10486074) / (1024*1024), static_cast<double>(2e+04)), static_cast<double>(-0.00556783614400875611650958980796060611309029233226596737701688), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<double>(-10486074) / (1024*1024), static_cast<double>(1e+02)), static_cast<double>(-0.0547613660316806551338637153942604550779513947674222863858713), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<double>(0.5), static_cast<double>(0.875)), static_cast<double>(0.558532231646608646115729767013630967055657943463362504577189), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<double>(0.5), static_cast<double>(1.125)), static_cast<double>(0.383621010650189547146769320487006220295290256657827220786527), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<double>(2.25), static_cast<double>(std::ldexp(1.0, -30))), static_cast<double>(5.62397392719283271332307799146649700147907612095185712015604e20), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<double>(5.5), static_cast<double>(3217)/1024), static_cast<double>(1.30623288775012596319554857587765179889689223531159532808379), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<double>(-5.5), static_cast<double>(10)), static_cast<double>(0.0000733045300798502164644836879577484533096239574909573072142667), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<double>(-5.5), static_cast<double>(100)), static_cast<double>(5.41274555306792267322084448693957747924412508020839543293369e-45), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<double>(10240)/1024, static_cast<double>(1)/1024), static_cast<double>(2.35522579263922076203415803966825431039900000000993410734978e38), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<double>(10240)/1024, static_cast<double>(10)), static_cast<double>(0.00161425530039067002345725193091329085443750382929208307802221), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<double>(144793)/1024, static_cast<double>(100)), static_cast<double>(1.39565245860302528069481472855619216759142225046370312329416e-6), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<double>(144793)/1024, static_cast<double>(200)), static_cast<double>(9.11950412043225432171915100042647230802198254567007382956336e-68), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<double>(0.5), static_cast<double>(1) / (1024*1024)), static_cast<double>(-817.033790261762580469303126467917092806755460418223776544122), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<double>(5.5), static_cast<double>(3.125)), static_cast<double>(-2.61489440328417468776474188539366752698192046890955453259866), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<double>(-5.5), static_cast<double>(3.125)), static_cast<double>(-0.0274994493896489729948109971802244976377957234563871795364056), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<double>(-5.5), static_cast<double>(1e+04)), static_cast<double>(-0.00759343502722670361395585198154817047185480147294665270646578), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<double>(-10486074) / (1024*1024), static_cast<double>(1)/1024), static_cast<double>(-1.50382374389531766117868938966858995093408410498915220070230e38), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<double>(-10486074) / (1024*1024), static_cast<double>(1e+02)), static_cast<double>(0.0583041891319026009955779707640455341990844522293730214223545), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<double>(141.75), static_cast<double>(1e+02)), static_cast<double>(-5.38829231428696507293191118661269920130838607482708483122068e9), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<double>(141.75), static_cast<double>(2e+04)), static_cast<double>(-0.00376577888677186194728129112270988602876597726657372330194186), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<double>(-141.75), static_cast<double>(1e+02)), static_cast<double>(-3.81009803444766877495905954105669819951653361036342457919021e9), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(0), static_cast<double>(0)), static_cast<double>(0), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(0), static_cast<double>(-10)), static_cast<double>(-10), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(-1), static_cast<double>(-1)), static_cast<double>(-1.2261911708835170708130609674719067527242483502207), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(0.875), static_cast<double>(-4)), static_cast<double>(-5.3190556182262405182189463092940736859067548232647), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(-0.625), static_cast<double>(8)), static_cast<double>(9.0419973860310100524448893214394562615252527557062), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(0.875), static_cast<double>(1e-05)), static_cast<double>(0.000010000000000127604166668510945638036143355898993088), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(10)/1024, static_cast<double>(1e+05)), static_cast<double>(100002.38431454899771096037307519328741455615271038), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(1), static_cast<double>(1e-20)), static_cast<double>(1.0000000000000000000000000000000000000000166666667e-20), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(1e-20), static_cast<double>(1e-20)), static_cast<double>(1.000000000000000e-20), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(400)/1024, static_cast<double>(1e+20)), static_cast<double>(1.0418143796499216839719289963154558027005142709763e20), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(0.5), static_cast<double>(2)), static_cast<double>(2.1765877052210673672479877957388515321497888026770), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(0.5), static_cast<double>(4)), static_cast<double>(4.2543274975235836861894752787874633017836785640477), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(0.5), static_cast<double>(6)), static_cast<double>(6.4588766202317746302999080620490579800463614807916), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(0.5), static_cast<double>(10)), static_cast<double>(10.697409951222544858346795279378531495869386960090), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(0.5), static_cast<double>(-2)), static_cast<double>(-2.1765877052210673672479877957388515321497888026770), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(0.5), static_cast<double>(-4)), static_cast<double>(-4.2543274975235836861894752787874633017836785640477), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(0.5), static_cast<double>(-6)), static_cast<double>(-6.4588766202317746302999080620490579800463614807916), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<double>(0.5), static_cast<double>(-10)), static_cast<double>(-10.697409951222544858346795279378531495869386960090), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<double>(0), static_cast<double>(0)), static_cast<double>(0), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<double>(0), static_cast<double>(-10)), static_cast<double>(-10), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<double>(-1), static_cast<double>(-1)), static_cast<double>(-0.84147098480789650665250232163029899962256306079837), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<double>(900) / 1024, static_cast<double>(-4)), static_cast<double>(-3.1756145986492562317862928524528520686391383168377), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<double>(-600) / 1024, static_cast<double>(8)), static_cast<double>(7.2473147180505693037677015377802777959345489333465), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<double>(800) / 1024, static_cast<double>(1e-05)), static_cast<double>(9.999999999898274739584436515967055859383969942432E-6), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<double>(100) / 1024, static_cast<double>(1e+05)), static_cast<double>(99761.153306972066658135668386691227343323331995888), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<double>(-0.5), static_cast<double>(1e+10)), static_cast<double>(9.3421545766487137036576748555295222252286528414669e9), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<double>(400) / 1024, ldexp(static_cast<double>(1), 66)), static_cast<double>(7.0886102721911705466476846969992069994308167515242e19), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0), static_cast<double>(1), static_cast<double>(-1)), static_cast<double>(-1.557407724654902230506974807458360173087), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0.4), static_cast<double>(0), static_cast<double>(-4)), static_cast<double>(-4.153623371196831087495427530365430979011), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(-0.6), static_cast<double>(0), static_cast<double>(8)), static_cast<double>(8.935930619078575123490612395578518914416), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0.25), static_cast<double>(0), static_cast<double>(0.5)), static_cast<double>(0.501246705365439492445236118603525029757890291780157969500480), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0), static_cast<double>(0), static_cast<double>(0.5)), static_cast<double>(0.5), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0), static_cast<double>(-2), static_cast<double>(0.5)), static_cast<double>(0.437501067017546278595664813509803743009132067629603474488486), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0), static_cast<double>(0.25), static_cast<double>(0.5)), static_cast<double>(0.510269830229213412212501938035914557628394166585442994564135), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0), static_cast<double>(0.75), static_cast<double>(0.5)), static_cast<double>(0.533293253875952645421201146925578536430596894471541312806165), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0), static_cast<double>(0.75), static_cast<double>(0.75)), static_cast<double>(0.871827580412760575085768367421866079353646112288567703061975), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0), static_cast<double>(1), static_cast<double>(0.25)), static_cast<double>(0.255341921221036266504482236490473678204201638800822621740476), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0), static_cast<double>(2), static_cast<double>(0.25)), static_cast<double>(0.261119051639220165094943572468224137699644963125853641716219), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0), static_cast<double>(1023)/1024, static_cast<double>(1.5)), static_cast<double>(13.2821612239764190363647953338544569682942329604483733197131), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0.5), static_cast<double>(0.5), static_cast<double>(-1)), static_cast<double>(-1.228014414316220642611298946293865487807), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0.5), static_cast<double>(0.5), static_cast<double>(1e+10)), static_cast<double>(1.536591003599172091573590441336982730551e+10), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0.75), static_cast<double>(-1e+05), static_cast<double>(10)), static_cast<double>(0.0347926099493147087821620459290460547131012904008557007934290), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0.875), static_cast<double>(-1e+10), static_cast<double>(10)), static_cast<double>(0.000109956202759561502329123384755016959364346382187364656768212), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0.875), static_cast<double>(-1e+10), static_cast<double>(1e+20)), static_cast<double>(1.00000626665567332602765201107198822183913978895904937646809e15), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0.875), static_cast<double>(-1e+10), static_cast<double>(1608)/1024), static_cast<double>(0.0000157080616044072676127333183571107873332593142625043567690379), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0.875), 1-static_cast<double>(1) / 1024, static_cast<double>(1e+20)), static_cast<double>(6.43274293944380717581167058274600202023334985100499739678963e21), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0.25), static_cast<double>(50), static_cast<double>(0.1)), static_cast<double>(0.124573770342749525407523258569507331686458866564082916835900), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<double>(0.25), static_cast<double>(1.125), static_cast<double>(1)), static_cast<double>(1.77299767784815770192352979665283069318388205110727241629752), eps * 5000);

   BOOST_CHECK_CLOSE(tr1::expint(static_cast<double>(1)/1024), static_cast<double>(-6.35327933972759151358547423727042905862963067106751711596065L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<double>(0.125)), static_cast<double>(-1.37320852494298333781545045921206470808223543321810480716122L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<double>(0.5)), static_cast<double>(0.454219904863173579920523812662802365281405554352642045162818L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<double>(1)), static_cast<double>(1.89511781635593675546652093433163426901706058173270759164623L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<double>(50.5)), static_cast<double>(1.72763195602911805201155668940185673806099654090456049881069e20L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<double>(-1)/1024), static_cast<double>(-6.35523246483107180261445551935803221293763008553775821607264L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<double>(-0.125)), static_cast<double>(-1.62342564058416879145630692462440887363310605737209536579267L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<double>(-0.5)), static_cast<double>(-0.559773594776160811746795939315085235226846890316353515248293L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<double>(-1)), static_cast<double>(-0.219383934395520273677163775460121649031047293406908207577979L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<double>(-50.5)), static_cast<double>(-2.27237132932219350440719707268817831250090574830769670186618e-24L), eps * 5000);

   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(0, static_cast<double>(1)), static_cast<double>(1.L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(1, static_cast<double>(1)), static_cast<double>(2.L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(1, static_cast<double>(2)), static_cast<double>(4.L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(1, static_cast<double>(10)), static_cast<double>(20), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(1, static_cast<double>(100)), static_cast<double>(200), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(1, static_cast<double>(1e6)), static_cast<double>(2e6), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(10, static_cast<double>(30)), static_cast<double>(5.896624628001300E+17L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(10, static_cast<double>(1000)), static_cast<double>(1.023976960161280E+33L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(10, static_cast<double>(10)), static_cast<double>(8.093278209760000E+12L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(10, static_cast<double>(-10)), static_cast<double>(8.093278209760000E+12L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(3, static_cast<double>(-10)), static_cast<double>(-7.880000000000000E+3L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(3, static_cast<double>(-1000)), static_cast<double>(-7.999988000000000E+9L), 100 * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(3, static_cast<double>(-1000000)), static_cast<double>(-7.999999999988000E+18L), 100 * eps);

   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(0.125)), static_cast<double>(-0.63277562349869525529352526763564627152686379131122L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(1023) / static_cast<double>(1024)), static_cast<double>(-1023.4228554489429786541032870895167448906103303056L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(1025) / static_cast<double>(1024)), static_cast<double>(1024.5772867695045940578681624248887776501597556226L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(0.5)), static_cast<double>(-1.46035450880958681288949915251529801246722933101258149054289L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(1.125)), static_cast<double>(8.5862412945105752999607544082693023591996301183069L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(2)), static_cast<double>(1.6449340668482264364724151666460251892189499012068L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(3.5)), static_cast<double>(1.1267338673170566464278124918549842722219969574036L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(4)), static_cast<double>(1.08232323371113819151600369654116790277475095191872690768298L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(4 + static_cast<double>(1) / 1024), static_cast<double>(1.08225596856391369799036835439238249195298434901488518878804L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(4.5)), static_cast<double>(1.05470751076145426402296728896028011727249383295625173068468L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(6.5)), static_cast<double>(1.01200589988852479610078491680478352908773213619144808841031L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(7.5)), static_cast<double>(1.00582672753652280770224164440459408011782510096320822989663L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(8.125)), static_cast<double>(1.0037305205308161603183307711439385250181080293472L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(16.125)), static_cast<double>(1.0000140128224754088474783648500235958510030511915L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(0)), static_cast<double>(-0.5L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(-0.125)), static_cast<double>(-0.39906966894504503550986928301421235400280637468895L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(-1)), static_cast<double>(-0.083333333333333333333333333333333333333333333333333L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(-2)), static_cast<double>(0L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(-2.5)), static_cast<double>(0.0085169287778503305423585670283444869362759902200745L), eps * 5000 * 3);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(-3)), static_cast<double>(0.0083333333333333333333333333333333333333333333333333L), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(-4)), static_cast<double>(0), eps * 5000);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(-20)), static_cast<double>(0), eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(-21)), static_cast<double>(-281.46014492753623188405797101449275362318840579710L), eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<double>(-30.125)), static_cast<double>(2.2762941726834511267740045451463455513839970804578e7L), eps * 5000 * 100);

   BOOST_CHECK_CLOSE(tr1::sph_bessel(0, static_cast<double>(0.1433600485324859619140625e-1)), static_cast<double>(0.9999657468461303487880990241993035937654e0),  eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_bessel(0, static_cast<double>(0.1760916970670223236083984375e-1)), static_cast<double>(0.9999483203249623334100130061926184665364e0),  eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_bessel(2, static_cast<double>(0.1433600485324859619140625e-1)), static_cast<double>(0.1370120120703995134662099191103188366059e-4),  eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_bessel(2, static_cast<double>(0.1760916970670223236083984375e-1)), static_cast<double>(0.2067173265753174063228459655801741280461e-4),  eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_bessel(7, static_cast<double>(0.1252804412841796875e3)), static_cast<double>(0.7887555711993028736906736576314283291289e-2),  eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_bessel(7, static_cast<double>(0.25554705810546875e3)), static_cast<double>(-0.1463292767579579943284849187188066532514e-2),  eps * 5000 * 100);

   BOOST_CHECK_CLOSE(tr1::sph_neumann(0, static_cast<double>(0.408089816570281982421875e0)), static_cast<double>(-0.2249212131304610409189209411089291558038e1), eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_neumann(0, static_cast<double>(0.6540834903717041015625e0)), static_cast<double>(-0.1213309779166084571756446746977955970241e1),   eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_neumann(2, static_cast<double>(0.408089816570281982421875e0)), static_cast<double>(-0.4541702641837159203058389758895634766256e2),   eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_neumann(2, static_cast<double>(0.6540834903717041015625e0)), static_cast<double>(-0.1156112621471167110574129561700037138981e2),   eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_neumann(10, static_cast<double>(0.1097540378570556640625e1)), static_cast<double>(-0.2427889658115064857278886600528596240123e9),   eps * 5000 * 100);
   BOOST_CHECK_CLOSE(tr1::sph_neumann(10, static_cast<double>(0.30944411754608154296875e1)), static_cast<double>(-0.3394649246350136450439882104151313759251e4),   eps * 5000 * 100);

   BOOST_CHECK_CLOSE_FRACTION(tr1::sph_legendre(3, 2, static_cast<double>(0.5)), static_cast<double>(0.2061460599687871330692286791802688341213L), eps * 5000);
   BOOST_CHECK_CLOSE_FRACTION(tr1::sph_legendre(40, 15, static_cast<double>(0.75)), static_cast<double>(-0.406036847302819452666908966769096223205057182668333862900509L), eps * 5000);
#endif
}

void test_values(long double, const char* name)
{
   (void)name;
#ifdef TEST_LD
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   std::cout << "Testing type " << name << std::endl;

   long double eps = boost::math::tools::epsilon<long double>();
   BOOST_CHECK_CLOSE(tr1::acoshl(std::cosh(0.5L)), 0.5L, 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::asinhl(std::sinh(0.5L)), 0.5L, 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::atanhl(std::tanh(0.5L)), 0.5L, 5000 * eps);

   BOOST_CHECK_CLOSE(tr1::cbrtl(1.5L * 1.5L * 1.5L), 1.5L, 5000 * eps);

   BOOST_CHECK(tr1::copysignl(1.0L, 1.0L) == 1.0L);
   BOOST_CHECK(tr1::copysignl(1.0L, -1.0L) == -1.0L);
   BOOST_CHECK(tr1::copysignl(-1.0L, 1.0L) == 1.0L);
   BOOST_CHECK(tr1::copysignl(-1.0L, -1.0L) == -1.0L);

   BOOST_CHECK_CLOSE(tr1::erfcl(static_cast<long double>(0.125)), static_cast<long double>(0.85968379519866618260697055347837660181302041685015L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfcl(static_cast<long double>(0.5)), static_cast<long double>(0.47950012218695346231725334610803547126354842424204L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfcl(static_cast<long double>(1)), static_cast<long double>(0.15729920705028513065877936491739074070393300203370L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfcl(static_cast<long double>(5)), static_cast<long double>(1.5374597944280348501883434853833788901180503147234e-12L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfcl(static_cast<long double>(-0.125)), static_cast<long double>(1.1403162048013338173930294465216233981869795831498L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfcl(static_cast<long double>(-0.5)), static_cast<long double>(1.5204998778130465376827466538919645287364515757580L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfcl(static_cast<long double>(0)), static_cast<long double>(1), eps * 1000);

   BOOST_CHECK_CLOSE(tr1::erfl(static_cast<long double>(0.125)), static_cast<long double>(0.14031620480133381739302944652162339818697958314985L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfl(static_cast<long double>(0.5)), static_cast<long double>(0.52049987781304653768274665389196452873645157575796L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfl(static_cast<long double>(1)), static_cast<long double>(0.84270079294971486934122063508260925929606699796630L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfl(static_cast<long double>(5)), static_cast<long double>(0.9999999999984625402055719651498116565146166211099L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfl(static_cast<long double>(-0.125)), static_cast<long double>(-0.14031620480133381739302944652162339818697958314985L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfl(static_cast<long double>(-0.5)), static_cast<long double>(-0.52049987781304653768274665389196452873645157575796L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfl(static_cast<long double>(0)), static_cast<long double>(0), eps * 1000);

   BOOST_CHECK_CLOSE(tr1::log1pl(static_cast<long double>(0.582029759883880615234375e0)), static_cast<long double>(0.4587086807259736626531803258754840111707e0L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::expm1l(static_cast<long double>(0.582029759883880615234375e0)), static_cast<long double>(0.7896673415707786528734865994546559029663e0L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::log1pl(static_cast<long double>(-0.2047410048544406890869140625e-1)), static_cast<long double>(-0.2068660038044094868521052319477265955827e-1L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::expm1l(static_cast<long double>(-0.2047410048544406890869140625e-1)), static_cast<long double>(-0.2026592921724753704129022027337835687888e-1L), eps * 1000);

   BOOST_CHECK_EQUAL(tr1::fmaxl(0.1L, -0.1L), 0.1L);
   BOOST_CHECK_EQUAL(tr1::fminl(0.1L, -0.1L), -0.1L);

   BOOST_CHECK_CLOSE(tr1::hypotl(1.0L, 3.0L), std::sqrt(10.0L), eps * 500);

   BOOST_CHECK_CLOSE(tr1::lgammal(static_cast<long double>(3.5)), static_cast<long double>(1.2009736023470742248160218814507129957702389154682L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgammal(static_cast<long double>(0.125)), static_cast<long double>(2.0194183575537963453202905211670995899482809521344L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgammal(static_cast<long double>(-0.125)), static_cast<long double>(2.1653002489051702517540619481440174064962195287626L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgammal(static_cast<long double>(-3.125)), static_cast<long double>(0.1543111276840418242676072830970532952413339012367L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgammal(static_cast<long double>(-53249.0/1024)), static_cast<long double>(-149.43323093420259741100038126078721302600128285894L), 5000 * eps);

   BOOST_CHECK_CLOSE(tr1::tgammal(static_cast<long double>(3.5)), static_cast<long double>(3.3233509704478425511840640312646472177454052302295L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::tgammal(static_cast<long double>(0.125)), static_cast<long double>(7.5339415987976119046992298412151336246104195881491L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::tgammal(static_cast<long double>(-0.125)), static_cast<long double>(-8.7172188593831756100190140408231437691829605421405L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::tgammal(static_cast<long double>(-3.125)), static_cast<long double>(1.1668538708507675587790157356605097019141636072094L), 5000 * eps);

#ifdef BOOST_HAS_LONG_LONG
   BOOST_CHECK(tr1::llroundl(2.5L) == 3LL);
   BOOST_CHECK(tr1::llroundl(2.25L) == 2LL);
#endif
   BOOST_CHECK(tr1::lroundl(2.5L) == 3L);
   BOOST_CHECK(tr1::lroundl(2.25L) == 2L);
   BOOST_CHECK(tr1::roundl(2.5L) == 3.0L);
   BOOST_CHECK(tr1::roundl(2.25L) == 2.0L);

   BOOST_CHECK(tr1::nextafterl(1.0L, 2.0L) > 1.0L);
   BOOST_CHECK(tr1::nextafterl(1.0L, -2.0L) < 1.0L);
   BOOST_CHECK(tr1::nextafterl(tr1::nextafterl(1.0L, 2.0L), -2.0L) == 1.0L);
   BOOST_CHECK(tr1::nextafterl(tr1::nextafterl(1.0L, -2.0L), 2.0L) == 1.0L);
   BOOST_CHECK(tr1::nextafterl(1.0L, 2.0L) > 1.0L);
   BOOST_CHECK(tr1::nextafterl(1.0L, -2.0L) < 1.0L);
   BOOST_CHECK(tr1::nextafterl(tr1::nextafterl(1.0L, 2.0L), -2.0L) == 1.0L);
   BOOST_CHECK(tr1::nextafterl(tr1::nextafterl(1.0L, -2.0L), 2.0L) == 1.0L);

   BOOST_CHECK(tr1::truncl(2.5L) == 2.0L);
   BOOST_CHECK(tr1::truncl(2.25L) == 2.0L);

   //
   // And again but without the "l" suffix on the function names:
   //
   BOOST_CHECK_CLOSE(tr1::acosh(std::cosh(0.5L)), 0.5L, 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::asinh(std::sinh(0.5L)), 0.5L, 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::atanh(std::tanh(0.5L)), 0.5L, 5000 * eps);

   BOOST_CHECK_CLOSE(tr1::cbrt(1.5L * 1.5L * 1.5L), 1.5L, 5000 * eps);

   BOOST_CHECK(tr1::copysign(1.0L, 1.0L) == 1.0L);
   BOOST_CHECK(tr1::copysign(1.0L, -1.0L) == -1.0L);
   BOOST_CHECK(tr1::copysign(-1.0L, 1.0L) == 1.0L);
   BOOST_CHECK(tr1::copysign(-1.0L, -1.0L) == -1.0L);

   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<long double>(0.125)), static_cast<long double>(0.85968379519866618260697055347837660181302041685015L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<long double>(0.5)), static_cast<long double>(0.47950012218695346231725334610803547126354842424204L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<long double>(1)), static_cast<long double>(0.15729920705028513065877936491739074070393300203370L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<long double>(5)), static_cast<long double>(1.5374597944280348501883434853833788901180503147234e-12L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<long double>(-0.125)), static_cast<long double>(1.1403162048013338173930294465216233981869795831498L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<long double>(-0.5)), static_cast<long double>(1.5204998778130465376827466538919645287364515757580L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erfc(static_cast<long double>(0)), static_cast<long double>(1), eps * 1000);

   BOOST_CHECK_CLOSE(tr1::erf(static_cast<long double>(0.125)), static_cast<long double>(0.14031620480133381739302944652162339818697958314985L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<long double>(0.5)), static_cast<long double>(0.52049987781304653768274665389196452873645157575796L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<long double>(1)), static_cast<long double>(0.84270079294971486934122063508260925929606699796630L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<long double>(5)), static_cast<long double>(0.9999999999984625402055719651498116565146166211099L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<long double>(-0.125)), static_cast<long double>(-0.14031620480133381739302944652162339818697958314985L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<long double>(-0.5)), static_cast<long double>(-0.52049987781304653768274665389196452873645157575796L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::erf(static_cast<long double>(0)), static_cast<long double>(0), eps * 1000);

   BOOST_CHECK_CLOSE(tr1::log1p(static_cast<long double>(0.582029759883880615234375e0)), static_cast<long double>(0.4587086807259736626531803258754840111707e0L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::expm1(static_cast<long double>(0.582029759883880615234375e0)), static_cast<long double>(0.7896673415707786528734865994546559029663e0L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::log1p(static_cast<long double>(-0.2047410048544406890869140625e-1)), static_cast<long double>(-0.2068660038044094868521052319477265955827e-1L), eps * 1000);
   BOOST_CHECK_CLOSE(tr1::expm1(static_cast<long double>(-0.2047410048544406890869140625e-1)), static_cast<long double>(-0.2026592921724753704129022027337835687888e-1L), eps * 1000);

   BOOST_CHECK_EQUAL(tr1::fmax(0.1L, -0.1L), 0.1L);
   BOOST_CHECK_EQUAL(tr1::fmin(0.1L, -0.1L), -0.1L);

   BOOST_CHECK_CLOSE(tr1::hypot(1.0L, 3.0L), std::sqrt(10.0L), eps * 500);

   BOOST_CHECK_CLOSE(tr1::lgamma(static_cast<long double>(3.5)), static_cast<long double>(1.2009736023470742248160218814507129957702389154682L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgamma(static_cast<long double>(0.125)), static_cast<long double>(2.0194183575537963453202905211670995899482809521344L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgamma(static_cast<long double>(-0.125)), static_cast<long double>(2.1653002489051702517540619481440174064962195287626L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgamma(static_cast<long double>(-3.125)), static_cast<long double>(0.1543111276840418242676072830970532952413339012367L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::lgamma(static_cast<long double>(-53249.0/1024)), static_cast<long double>(-149.43323093420259741100038126078721302600128285894L), 5000 * eps);

   BOOST_CHECK_CLOSE(tr1::tgamma(static_cast<long double>(3.5)), static_cast<long double>(3.3233509704478425511840640312646472177454052302295L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::tgamma(static_cast<long double>(0.125)), static_cast<long double>(7.5339415987976119046992298412151336246104195881491L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::tgamma(static_cast<long double>(-0.125)), static_cast<long double>(-8.7172188593831756100190140408231437691829605421405L), 5000 * eps);
   BOOST_CHECK_CLOSE(tr1::tgamma(static_cast<long double>(-3.125)), static_cast<long double>(1.1668538708507675587790157356605097019141636072094L), 5000 * eps);

#ifdef BOOST_HAS_LONG_LONG
   BOOST_CHECK(tr1::llround(2.5L) == 3LL);
   BOOST_CHECK(tr1::llround(2.25L) == 2LL);
#endif
   BOOST_CHECK(tr1::lround(2.5L) == 3L);
   BOOST_CHECK(tr1::lround(2.25L) == 2L);
   BOOST_CHECK(tr1::round(2.5L) == 3.0L);
   BOOST_CHECK(tr1::round(2.25L) == 2.0L);

   BOOST_CHECK(tr1::nextafter(1.0L, 2.0L) > 1.0L);
   BOOST_CHECK(tr1::nextafter(1.0L, -2.0L) < 1.0L);
   BOOST_CHECK(tr1::nextafter(tr1::nextafter(1.0L, 2.0L), -2.0L) == 1.0L);
   BOOST_CHECK(tr1::nextafter(tr1::nextafter(1.0L, -2.0L), 2.0L) == 1.0L);
   BOOST_CHECK(tr1::nextafter(1.0L, 2.0L) > 1.0L);
   BOOST_CHECK(tr1::nextafter(1.0L, -2.0L) < 1.0L);
   BOOST_CHECK(tr1::nextafter(tr1::nextafter(1.0L, 2.0L), -2.0L) == 1.0L);
   BOOST_CHECK(tr1::nextafter(tr1::nextafter(1.0L, -2.0L), 2.0L) == 1.0L);

   BOOST_CHECK(tr1::trunc(2.5L) == 2.0L);
   BOOST_CHECK(tr1::trunc(2.25L) == 2.0L);

   //
   // Now for the TR1 math functions:
   //
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerrel(4L, 5L, static_cast<long double>(0.5L)), static_cast<long double>(88.31510416666666666666666666666666666667L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerrel(10L, 0L, static_cast<long double>(2.5L)), static_cast<long double>(-0.8802526766660982969576719576719576719577L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerrel(10L, 1L, static_cast<long double>(4.5L)), static_cast<long double>(1.564311458042689732142857142857142857143L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerrel(10L, 6L, static_cast<long double>(8.5L)), static_cast<long double>(20.51596541066649098875661375661375661376L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerrel(10L, 12L, static_cast<long double>(12.5L)), static_cast<long double>(-199.5560968456234671241181657848324514991L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerrel(50L, 40L, static_cast<long double>(12.5L)), static_cast<long double>(-4.996769495006119488583146995907246595400e16L), eps * 100L);

   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerrel(1L, static_cast<long double>(0.5L)), static_cast<long double>(0.5L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerrel(4L, static_cast<long double>(0.5L)), static_cast<long double>(-0.3307291666666666666666666666666666666667L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerrel(7L, static_cast<long double>(0.5L)), static_cast<long double>(-0.5183392237103174603174603174603174603175L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerrel(20L, static_cast<long double>(0.5L)), static_cast<long double>(0.3120174870800154148915399248893113634676L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerrel(50L, static_cast<long double>(0.5L)), static_cast<long double>(-0.3181388060269979064951118308575628226834L), eps * 100L);

   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerrel(1L, static_cast<long double>(-0.5L)), static_cast<long double>(1.5L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerrel(4L, static_cast<long double>(-0.5L)), static_cast<long double>(3.835937500000000000000000000000000000000L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerrel(7L, static_cast<long double>(-0.5L)), static_cast<long double>(7.950934709821428571428571428571428571429L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerrel(20L, static_cast<long double>(-0.5L)), static_cast<long double>(76.12915699869631476833699787070874048223L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerrel(50L, static_cast<long double>(-0.5L)), static_cast<long double>(2307.428631277506570629232863491518399720L), eps * 100L);

   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerrel(1L, static_cast<long double>(4.5L)), static_cast<long double>(-3.500000000000000000000000000000000000000L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerrel(4L, static_cast<long double>(4.5L)), static_cast<long double>(0.08593750000000000000000000000000000000000L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerrel(7L, static_cast<long double>(4.5L)), static_cast<long double>(-1.036928013392857142857142857142857142857L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerrel(20L, static_cast<long double>(4.5L)), static_cast<long double>(1.437239150257817378525582974722170737587L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerrel(50L, static_cast<long double>(4.5L)), static_cast<long double>(-0.7795068145562651416494321484050019245248L), eps * 100L);

   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendrel(4L, 2L, static_cast<long double>(0.5L)), static_cast<long double>(4.218750000000000000000000000000000000000L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendrel(7L, 5L, static_cast<long double>(0.5L)), static_cast<long double>(5696.789530152175143607977274672800795328L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendrel(4L, 2L, static_cast<long double>(-0.5L)), static_cast<long double>(4.218750000000000000000000000000000000000L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendrel(7L, 5L, static_cast<long double>(-0.5L)), static_cast<long double>(5696.789530152175143607977274672800795328L), eps * 100L);

   BOOST_CHECK_CLOSE_FRACTION(tr1::legendrel(1L, static_cast<long double>(0.5L)), static_cast<long double>(0.5L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::legendrel(4L, static_cast<long double>(0.5L)), static_cast<long double>(-0.2890625000000000000000000000000000000000L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::legendrel(7L, static_cast<long double>(0.5L)), static_cast<long double>(0.2231445312500000000000000000000000000000L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::legendrel(40L, static_cast<long double>(0.5L)), static_cast<long double>(-0.09542943523261546936538467572384923220258L), eps * 100L);

   long double sv = eps / 1024;
   BOOST_CHECK_CLOSE(tr1::betal(static_cast<long double>(1L), static_cast<long double>(1L)), static_cast<long double>(1L), eps * 20L * 100L);
   BOOST_CHECK_CLOSE(tr1::betal(static_cast<long double>(1L), static_cast<long double>(4L)), static_cast<long double>(0.25L), eps * 20L * 100L);
   BOOST_CHECK_CLOSE(tr1::betal(static_cast<long double>(4L), static_cast<long double>(1L)), static_cast<long double>(0.25L), eps * 20L * 100L);
   BOOST_CHECK_CLOSE(tr1::betal(sv, static_cast<long double>(4L)), 1/sv, eps * 20L * 100L);
   BOOST_CHECK_CLOSE(tr1::betal(static_cast<long double>(4L), sv), 1/sv, eps * 20L * 100L);
   BOOST_CHECK_CLOSE(tr1::betal(static_cast<long double>(4L), static_cast<long double>(20L)), static_cast<long double>(0.00002823263692828910220214568040654997176736L), eps * 20L * 100L);
   BOOST_CHECK_CLOSE(tr1::betal(static_cast<long double>(0.0125L), static_cast<long double>(0.000023L)), static_cast<long double>(43558.24045647538375006349016083320744662L), eps * 20L * 100L);

   BOOST_CHECK_CLOSE(tr1::comp_ellint_1l(static_cast<long double>(0L)), static_cast<long double>(1.5707963267948966192313216916397514420985846996876L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1l(static_cast<long double>(0.125L)), static_cast<long double>(1.5769867712158131421244030532288080803822271060839L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1l(static_cast<long double>(0.25L)), static_cast<long double>(1.5962422221317835101489690714979498795055744578951L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1l(static_cast<long double>(300L)/1024L), static_cast<long double>(1.6062331054696636704261124078746600894998873503208L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1l(static_cast<long double>(400L)/1024L), static_cast<long double>(1.6364782007562008756208066125715722889067992997614L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1l(static_cast<long double>(-0.5L)), static_cast<long double>(1.6857503548125960428712036577990769895008008941411L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1l(static_cast<long double>(-0.75L)), static_cast<long double>(1.9109897807518291965531482187613425592531451316788L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::comp_ellint_2l(static_cast<long double>(-1L)), static_cast<long double>(1L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2l(static_cast<long double>(0L)), static_cast<long double>(1.5707963267948966192313216916397514420985846996876L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2l(static_cast<long double>(100L) / 1024L), static_cast<long double>(1.5670445330545086723323795143598956428788609133377L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2l(static_cast<long double>(200L) / 1024L), static_cast<long double>(1.5557071588766556854463404816624361127847775545087L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2l(static_cast<long double>(300L) / 1024L), static_cast<long double>(1.5365278991162754883035625322482669608948678755743L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2l(static_cast<long double>(400L) / 1024L), static_cast<long double>(1.5090417763083482272165682786143770446401437564021L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2l(static_cast<long double>(-0.5L)), static_cast<long double>(1.4674622093394271554597952669909161360253617523272L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2l(static_cast<long double>(-600L) / 1024L), static_cast<long double>(1.4257538571071297192428217218834579920545946473778L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2l(static_cast<long double>(-800L) / 1024L), static_cast<long double>(1.2927868476159125056958680222998765985004489572909L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2l(static_cast<long double>(-900L) / 1024L), static_cast<long double>(1.1966864890248739524112920627353824133420353430982L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::comp_ellint_3l(static_cast<long double>(0.2L), static_cast<long double>(0L)), static_cast<long double>(1.586867847454166237308008033828114192951L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3l(static_cast<long double>(0.4L), static_cast<long double>(0L)), static_cast<long double>(1.639999865864511206865258329748601457626L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3l(static_cast<long double>(0L), static_cast<long double>(0L)), static_cast<long double>(1.57079632679489661923132169163975144209858469968755291048747L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3l(static_cast<long double>(0L), static_cast<long double>(0.5L)), static_cast<long double>(2.221441469079183123507940495030346849307L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3l(static_cast<long double>(0.3L), static_cast<long double>(-4L)), static_cast<long double>(0.712708870925620061597924858162260293305195624270730660081949L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3l(static_cast<long double>(-0.5L), static_cast<long double>(-1e+05L)), static_cast<long double>(0.00496944596485066055800109163256108604615568144080386919012831L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3l(static_cast<long double>(-0.75L), static_cast<long double>(-1e+10L)), static_cast<long double>(0.0000157080225184890546939710019277357161497407143903832703317801L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3l(static_cast<long double>(-0.875L), static_cast<long double>(1L) / 1024L), static_cast<long double>(2.18674503176462374414944618968850352696579451638002110619287L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3l(static_cast<long double>(-0.875L), static_cast<long double>(1023L)/1024L), static_cast<long double>(101.045289804941384100960063898569538919135722087486350366997L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::cyl_bessel_il(static_cast<long double>(2.25L), static_cast<long double>(1L)/(1024*1024L)), static_cast<long double>(2.34379212133481347189068464680335815256364262507955635911656e-15L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_il(static_cast<long double>(5.5L), static_cast<long double>(3.125L)), static_cast<long double>(0.0583514045989371500460946536220735787163510569634133670181210L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_il(static_cast<long double>(-5L) + static_cast<long double>(1L)/1024L, static_cast<long double>(2.125L)), static_cast<long double>(0.0267920938009571023702933210070984416052633027166975342895062L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_il(static_cast<long double>(-5.5L), static_cast<long double>(10L)), static_cast<long double>(597.577606961369169607937419869926705730305175364662688426534L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_il(static_cast<long double>(-10486074L)/(1024*1024L), static_cast<long double>(1L)/1024L), static_cast<long double>(1.41474005665181350367684623930576333542989766867888186478185e35L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_il(static_cast<long double>(-10486074L)/(1024*1024L), static_cast<long double>(50L)), static_cast<long double>(1.07153277202900671531087024688681954238311679648319534644743e20L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_il(static_cast<long double>(144794L)/1024L, static_cast<long double>(100L)), static_cast<long double>(2066.27694757392660413922181531984160871678224178890247540320L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_il(static_cast<long double>(-144794L)/1024L, static_cast<long double>(100L)), static_cast<long double>(2066.27694672763190927440969155740243346136463461655104698748L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(2457L)/1024L, static_cast<long double>(1L)/1024L), static_cast<long double>(3.80739920118603335646474073457326714709615200130620574875292e-9L), eps * 50000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(5.5L), static_cast<long double>(3217L)/1024L), static_cast<long double>(0.0281933076257506091621579544064767140470089107926550720453038L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(-5.5L), static_cast<long double>(3217L)/1024L), static_cast<long double>(-2.55820064470647911823175836997490971806135336759164272675969L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(-5.5L), static_cast<long double>(1e+04L)), static_cast<long double>(2.449843111985605522111159013846599118397e-03L), eps * 50000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(5.5L), static_cast<long double>(1e+04L)), static_cast<long double>(0.00759343502722670361395585198154817047185480147294665270646578L), eps * 5000L);
   //BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(5.5L), static_cast<long double>(1e+06L)), static_cast<long double>(-0.000747424248595630177396350688505919533097973148718960064663632L), eps * 50000L);
   //BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(5.125L), static_cast<long double>(1e+06L)), static_cast<long double>(-0.000776600124835704280633640911329691642748783663198207360238214L), eps * 50000L);
   //BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(5.875L), static_cast<long double>(1e+06L)), static_cast<long double>(-0.000466322721115193071631008581529503095819705088484386434589780L), eps * 50000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(0.5L), static_cast<long double>(101L)), static_cast<long double>(0.0358874487875643822020496677692429287863419555699447066226409L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(-5.5L), static_cast<long double>(1e+04L)), static_cast<long double>(0.00244984311198560552211115901384659911839737686676766460822577L), eps * 50000L);
   //BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(-5.5L), static_cast<long double>(1e+06L)), static_cast<long double>(0.000279243200433579511095229508894156656558211060453622750659554L), eps * 50000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(-0.5L), static_cast<long double>(101L)), static_cast<long double>(0.0708184798097594268482290389188138201440114881159344944791454L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(-10486074L) / (1024*1024L), static_cast<long double>(1L)/1024L), static_cast<long double>(1.41474013160494695750009004222225969090304185981836460288562e35L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(-10486074L) / (1024*1024L), static_cast<long double>(15L)), static_cast<long double>(-0.0902239288885423309568944543848111461724911781719692852541489L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(10486074L) / (1024*1024L), static_cast<long double>(1e+02L)), static_cast<long double>(-0.0547064914615137807616774867984047583596945624129838091326863L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(10486074L) / (1024*1024L), static_cast<long double>(2e+04L)), static_cast<long double>(-0.00556783614400875611650958980796060611309029233226596737701688L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_jl(static_cast<long double>(-10486074L) / (1024*1024L), static_cast<long double>(1e+02L)), static_cast<long double>(-0.0547613660316806551338637153942604550779513947674222863858713L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kl(static_cast<long double>(0.5L), static_cast<long double>(0.875L)), static_cast<long double>(0.558532231646608646115729767013630967055657943463362504577189L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kl(static_cast<long double>(0.5L), static_cast<long double>(1.125L)), static_cast<long double>(0.383621010650189547146769320487006220295290256657827220786527L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kl(static_cast<long double>(2.25L), static_cast<long double>(std::ldexp(1.0L, -30L))), static_cast<long double>(5.62397392719283271332307799146649700147907612095185712015604e20L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kl(static_cast<long double>(5.5L), static_cast<long double>(3217L)/1024L), static_cast<long double>(1.30623288775012596319554857587765179889689223531159532808379L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kl(static_cast<long double>(-5.5L), static_cast<long double>(10L)), static_cast<long double>(0.0000733045300798502164644836879577484533096239574909573072142667L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kl(static_cast<long double>(-5.5L), static_cast<long double>(100L)), static_cast<long double>(5.41274555306792267322084448693957747924412508020839543293369e-45L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kl(static_cast<long double>(10240L)/1024L, static_cast<long double>(1L)/1024L), static_cast<long double>(2.35522579263922076203415803966825431039900000000993410734978e38L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kl(static_cast<long double>(10240L)/1024L, static_cast<long double>(10L)), static_cast<long double>(0.00161425530039067002345725193091329085443750382929208307802221L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kl(static_cast<long double>(144793L)/1024L, static_cast<long double>(100L)), static_cast<long double>(1.39565245860302528069481472855619216759142225046370312329416e-6L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_kl(static_cast<long double>(144793L)/1024L, static_cast<long double>(200L)), static_cast<long double>(9.11950412043225432171915100042647230802198254567007382956336e-68L), eps * 7000L);

   BOOST_CHECK_CLOSE(tr1::cyl_neumannl(static_cast<long double>(0.5L), static_cast<long double>(1L) / (1024*1024L)), static_cast<long double>(-817.033790261762580469303126467917092806755460418223776544122L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannl(static_cast<long double>(5.5L), static_cast<long double>(3.125L)), static_cast<long double>(-2.61489440328417468776474188539366752698192046890955453259866L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannl(static_cast<long double>(-5.5L), static_cast<long double>(3.125L)), static_cast<long double>(-0.0274994493896489729948109971802244976377957234563871795364056L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannl(static_cast<long double>(-5.5L), static_cast<long double>(1e+04L)), static_cast<long double>(-0.00759343502722670361395585198154817047185480147294665270646578L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannl(static_cast<long double>(-10486074L) / (1024*1024L), static_cast<long double>(1L)/1024L), static_cast<long double>(-1.50382374389531766117868938966858995093408410498915220070230e38L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannl(static_cast<long double>(-10486074L) / (1024*1024L), static_cast<long double>(1e+02L)), static_cast<long double>(0.0583041891319026009955779707640455341990844522293730214223545L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannl(static_cast<long double>(141.75L), static_cast<long double>(1e+02L)), static_cast<long double>(-5.38829231428696507293191118661269920130838607482708483122068e9L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannl(static_cast<long double>(141.75L), static_cast<long double>(2e+04L)), static_cast<long double>(-0.00376577888677186194728129112270988602876597726657372330194186L), eps * 50000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumannl(static_cast<long double>(-141.75L), static_cast<long double>(1e+02L)), static_cast<long double>(-3.81009803444766877495905954105669819951653361036342457919021e9L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(0L), static_cast<long double>(0L)), static_cast<long double>(0L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(0L), static_cast<long double>(-10L)), static_cast<long double>(-10L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(-1L), static_cast<long double>(-1L)), static_cast<long double>(-1.2261911708835170708130609674719067527242483502207L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(0.875L), static_cast<long double>(-4L)), static_cast<long double>(-5.3190556182262405182189463092940736859067548232647L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(-0.625L), static_cast<long double>(8L)), static_cast<long double>(9.0419973860310100524448893214394562615252527557062L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(0.875L), static_cast<long double>(1e-05L)), static_cast<long double>(0.000010000000000127604166668510945638036143355898993088L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(10L)/1024L, static_cast<long double>(1e+05L)), static_cast<long double>(100002.38431454899771096037307519328741455615271038L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(1L), static_cast<long double>(1e-20L)), static_cast<long double>(1.0000000000000000000000000000000000000000166666667e-20L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(1e-20L), static_cast<long double>(1e-20L)), static_cast<long double>(1.000000000000000e-20L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(400L)/1024L, static_cast<long double>(1e+20L)), static_cast<long double>(1.0418143796499216839719289963154558027005142709763e20L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(0.5L), static_cast<long double>(2L)), static_cast<long double>(2.1765877052210673672479877957388515321497888026770L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(0.5L), static_cast<long double>(4L)), static_cast<long double>(4.2543274975235836861894752787874633017836785640477L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(0.5L), static_cast<long double>(6L)), static_cast<long double>(6.4588766202317746302999080620490579800463614807916L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(0.5L), static_cast<long double>(10L)), static_cast<long double>(10.697409951222544858346795279378531495869386960090L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(0.5L), static_cast<long double>(-2L)), static_cast<long double>(-2.1765877052210673672479877957388515321497888026770L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(0.5L), static_cast<long double>(-4L)), static_cast<long double>(-4.2543274975235836861894752787874633017836785640477L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(0.5L), static_cast<long double>(-6L)), static_cast<long double>(-6.4588766202317746302999080620490579800463614807916L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1l(static_cast<long double>(0.5L), static_cast<long double>(-10L)), static_cast<long double>(-10.697409951222544858346795279378531495869386960090L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::ellint_2l(static_cast<long double>(0L), static_cast<long double>(0L)), static_cast<long double>(0L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2l(static_cast<long double>(0L), static_cast<long double>(-10L)), static_cast<long double>(-10L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2l(static_cast<long double>(-1L), static_cast<long double>(-1L)), static_cast<long double>(-0.84147098480789650665250232163029899962256306079837L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2l(static_cast<long double>(900L) / 1024L, static_cast<long double>(-4L)), static_cast<long double>(-3.1756145986492562317862928524528520686391383168377L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2l(static_cast<long double>(-600L) / 1024L, static_cast<long double>(8L)), static_cast<long double>(7.2473147180505693037677015377802777959345489333465L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2l(static_cast<long double>(800L) / 1024L, static_cast<long double>(1e-05L)), static_cast<long double>(9.999999999898274739584436515967055859383969942432E-6L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2l(static_cast<long double>(100L) / 1024L, static_cast<long double>(1e+05L)), static_cast<long double>(99761.153306972066658135668386691227343323331995888L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2l(static_cast<long double>(-0.5L), static_cast<long double>(1e+10L)), static_cast<long double>(9.3421545766487137036576748555295222252286528414669e9L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2l(static_cast<long double>(400L) / 1024L, ldexp(static_cast<long double>(1L), 66L)), static_cast<long double>(7.0886102721911705466476846969992069994308167515242e19L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0L), static_cast<long double>(1L), static_cast<long double>(-1L)), static_cast<long double>(-1.557407724654902230506974807458360173087L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0.4L), static_cast<long double>(0L), static_cast<long double>(-4L)), static_cast<long double>(-4.153623371196831087495427530365430979011L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(-0.6L), static_cast<long double>(0L), static_cast<long double>(8L)), static_cast<long double>(8.935930619078575123490612395578518914416L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0.25L), static_cast<long double>(0L), static_cast<long double>(0.5L)), static_cast<long double>(0.501246705365439492445236118603525029757890291780157969500480L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0L), static_cast<long double>(0L), static_cast<long double>(0.5L)), static_cast<long double>(0.5L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0L), static_cast<long double>(-2L), static_cast<long double>(0.5L)), static_cast<long double>(0.437501067017546278595664813509803743009132067629603474488486L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0L), static_cast<long double>(0.25L), static_cast<long double>(0.5L)), static_cast<long double>(0.510269830229213412212501938035914557628394166585442994564135L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0L), static_cast<long double>(0.75L), static_cast<long double>(0.5L)), static_cast<long double>(0.533293253875952645421201146925578536430596894471541312806165L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0L), static_cast<long double>(0.75L), static_cast<long double>(0.75L)), static_cast<long double>(0.871827580412760575085768367421866079353646112288567703061975L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0L), static_cast<long double>(1L), static_cast<long double>(0.25L)), static_cast<long double>(0.255341921221036266504482236490473678204201638800822621740476L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0L), static_cast<long double>(2L), static_cast<long double>(0.25L)), static_cast<long double>(0.261119051639220165094943572468224137699644963125853641716219L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0L), static_cast<long double>(1023L)/1024L, static_cast<long double>(1.5L)), static_cast<long double>(13.2821612239764190363647953338544569682942329604483733197131L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0.5L), static_cast<long double>(0.5L), static_cast<long double>(-1L)), static_cast<long double>(-1.228014414316220642611298946293865487807L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0.5L), static_cast<long double>(0.5L), static_cast<long double>(1e+10L)), static_cast<long double>(1.536591003599172091573590441336982730551e+10L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0.75L), static_cast<long double>(-1e+05L), static_cast<long double>(10L)), static_cast<long double>(0.0347926099493147087821620459290460547131012904008557007934290L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0.875L), static_cast<long double>(-1e+10L), static_cast<long double>(10L)), static_cast<long double>(0.000109956202759561502329123384755016959364346382187364656768212L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0.875L), static_cast<long double>(-1e+10L), static_cast<long double>(1e+20L)), static_cast<long double>(1.00000626665567332602765201107198822183913978895904937646809e15L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0.875L), static_cast<long double>(-1e+10L), static_cast<long double>(1608L)/1024L), static_cast<long double>(0.0000157080616044072676127333183571107873332593142625043567690379L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0.875L), 1-static_cast<long double>(1L) / 1024L, static_cast<long double>(1e+20L)), static_cast<long double>(6.43274293944380717581167058274600202023334985100499739678963e21L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0.25L), static_cast<long double>(50L), static_cast<long double>(0.1L)), static_cast<long double>(0.124573770342749525407523258569507331686458866564082916835900L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3l(static_cast<long double>(0.25L), static_cast<long double>(1.125L), static_cast<long double>(1L)), static_cast<long double>(1.77299767784815770192352979665283069318388205110727241629752L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::expintl(static_cast<long double>(1L)/1024L), static_cast<long double>(-6.35327933972759151358547423727042905862963067106751711596065L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expintl(static_cast<long double>(0.125L)), static_cast<long double>(-1.37320852494298333781545045921206470808223543321810480716122L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expintl(static_cast<long double>(0.5L)), static_cast<long double>(0.454219904863173579920523812662802365281405554352642045162818L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expintl(static_cast<long double>(1L)), static_cast<long double>(1.89511781635593675546652093433163426901706058173270759164623L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expintl(static_cast<long double>(50.5L)), static_cast<long double>(1.72763195602911805201155668940185673806099654090456049881069e20L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expintl(static_cast<long double>(-1L)/1024L), static_cast<long double>(-6.35523246483107180261445551935803221293763008553775821607264L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expintl(static_cast<long double>(-0.125L)), static_cast<long double>(-1.62342564058416879145630692462440887363310605737209536579267L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expintl(static_cast<long double>(-0.5L)), static_cast<long double>(-0.559773594776160811746795939315085235226846890316353515248293L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expintl(static_cast<long double>(-1L)), static_cast<long double>(-0.219383934395520273677163775460121649031047293406908207577979L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expintl(static_cast<long double>(-50.5L)), static_cast<long double>(-2.27237132932219350440719707268817831250090574830769670186618e-24L), eps * 5000L);

   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitel(0L, static_cast<long double>(1L)), static_cast<long double>(1.L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitel(1L, static_cast<long double>(1L)), static_cast<long double>(2.L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitel(1L, static_cast<long double>(2L)), static_cast<long double>(4.L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitel(1L, static_cast<long double>(10L)), static_cast<long double>(20L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitel(1L, static_cast<long double>(100L)), static_cast<long double>(200L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitel(1L, static_cast<long double>(1e6L)), static_cast<long double>(2e6L), 100L * eps);
   //BOOST_CHECK_CLOSE_FRACTION(tr1::hermitel(10L, static_cast<long double>(30L)), static_cast<long double>(5.896624628001300E+17L), 1000L * eps);
   //BOOST_CHECK_CLOSE_FRACTION(tr1::hermitel(10L, static_cast<long double>(1000L)), static_cast<long double>(1.023976960161280E+33L), 1000L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitel(10L, static_cast<long double>(10L)), static_cast<long double>(8.093278209760000E+12L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitel(10L, static_cast<long double>(-10L)), static_cast<long double>(8.093278209760000E+12L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitel(3L, static_cast<long double>(-10L)), static_cast<long double>(-7.880000000000000E+3L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitel(3L, static_cast<long double>(-1000L)), static_cast<long double>(-7.999988000000000E+9L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermitel(3L, static_cast<long double>(-1000000L)), static_cast<long double>(-7.999999999988000E+18L), 100L * eps);

   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(0.125L)), static_cast<long double>(-0.63277562349869525529352526763564627152686379131122L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(1023L) / static_cast<long double>(1024L)), static_cast<long double>(-1023.4228554489429786541032870895167448906103303056L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(1025L) / static_cast<long double>(1024L)), static_cast<long double>(1024.5772867695045940578681624248887776501597556226L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(0.5L)), static_cast<long double>(-1.46035450880958681288949915251529801246722933101258149054289L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(1.125L)), static_cast<long double>(8.5862412945105752999607544082693023591996301183069L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(2L)), static_cast<long double>(1.6449340668482264364724151666460251892189499012068L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(3.5L)), static_cast<long double>(1.1267338673170566464278124918549842722219969574036L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(4L)), static_cast<long double>(1.08232323371113819151600369654116790277475095191872690768298L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(4L + static_cast<long double>(1L) / 1024L), static_cast<long double>(1.08225596856391369799036835439238249195298434901488518878804L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(4.5L)), static_cast<long double>(1.05470751076145426402296728896028011727249383295625173068468L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(6.5L)), static_cast<long double>(1.01200589988852479610078491680478352908773213619144808841031L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(7.5L)), static_cast<long double>(1.00582672753652280770224164440459408011782510096320822989663L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(8.125L)), static_cast<long double>(1.0037305205308161603183307711439385250181080293472L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(16.125L)), static_cast<long double>(1.0000140128224754088474783648500235958510030511915L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(0L)), static_cast<long double>(-0.5L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(-0.125L)), static_cast<long double>(-0.39906966894504503550986928301421235400280637468895L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(-1L)), static_cast<long double>(-0.083333333333333333333333333333333333333333333333333L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(-2L)), static_cast<long double>(0L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(-2.5L)), static_cast<long double>(0.0085169287778503305423585670283444869362759902200745L), eps * 5000L * 3L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(-3L)), static_cast<long double>(0.0083333333333333333333333333333333333333333333333333L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(-4L)), static_cast<long double>(0L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(-20L)), static_cast<long double>(0L), eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(-21L)), static_cast<long double>(-281.46014492753623188405797101449275362318840579710L), eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::riemann_zetal(static_cast<long double>(-30.125L)), static_cast<long double>(2.2762941726834511267740045451463455513839970804578e7L), eps * 5000L * 100L);

   BOOST_CHECK_CLOSE(tr1::sph_bessell(0L, static_cast<long double>(0.1433600485324859619140625e-1L)), static_cast<long double>(0.9999657468461303487880990241993035937654e0L),  eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_bessell(0L, static_cast<long double>(0.1760916970670223236083984375e-1L)), static_cast<long double>(0.9999483203249623334100130061926184665364e0L),  eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_bessell(2L, static_cast<long double>(0.1433600485324859619140625e-1L)), static_cast<long double>(0.1370120120703995134662099191103188366059e-4L),  eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_bessell(2L, static_cast<long double>(0.1760916970670223236083984375e-1L)), static_cast<long double>(0.2067173265753174063228459655801741280461e-4L),  eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_bessell(7L, static_cast<long double>(0.1252804412841796875e3L)), static_cast<long double>(0.7887555711993028736906736576314283291289e-2L),  eps * 50000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_bessell(7L, static_cast<long double>(0.25554705810546875e3L)), static_cast<long double>(-0.1463292767579579943284849187188066532514e-2L),  eps * 5000L * 100L);

   BOOST_CHECK_CLOSE(tr1::sph_neumannl(0L, static_cast<long double>(0.408089816570281982421875e0L)), static_cast<long double>(-0.2249212131304610409189209411089291558038e1L), eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_neumannl(0L, static_cast<long double>(0.6540834903717041015625e0L)), static_cast<long double>(-0.1213309779166084571756446746977955970241e1L),   eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_neumannl(2L, static_cast<long double>(0.408089816570281982421875e0L)), static_cast<long double>(-0.4541702641837159203058389758895634766256e2L),   eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_neumannl(2L, static_cast<long double>(0.6540834903717041015625e0L)), static_cast<long double>(-0.1156112621471167110574129561700037138981e2L),   eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_neumannl(10L, static_cast<long double>(0.1097540378570556640625e1L)), static_cast<long double>(-0.2427889658115064857278886600528596240123e9L),   eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_neumannl(10L, static_cast<long double>(0.30944411754608154296875e1L)), static_cast<long double>(-0.3394649246350136450439882104151313759251e4L),   eps * 5000L * 100L);

   BOOST_CHECK_CLOSE_FRACTION(tr1::sph_legendrel(3L, 2L, static_cast<long double>(0.5L)), static_cast<long double>(0.2061460599687871330692286791802688341213L), eps * 5000L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::sph_legendrel(40L, 15L, static_cast<long double>(0.75L)), static_cast<long double>(-0.406036847302819452666908966769096223205057182668333862900509L), eps * 5000L);

   //
   // Now all over again but without the "f" suffix on the function names this time:
   //
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(4L, 5L, static_cast<long double>(0.5L)), static_cast<long double>(88.31510416666666666666666666666666666667L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(10L, 0L, static_cast<long double>(2.5L)), static_cast<long double>(-0.8802526766660982969576719576719576719577L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(10L, 1L, static_cast<long double>(4.5L)), static_cast<long double>(1.564311458042689732142857142857142857143L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(10L, 6L, static_cast<long double>(8.5L)), static_cast<long double>(20.51596541066649098875661375661375661376L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(10L, 12L, static_cast<long double>(12.5L)), static_cast<long double>(-199.5560968456234671241181657848324514991L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_laguerre(50L, 40L, static_cast<long double>(12.5L)), static_cast<long double>(-4.996769495006119488583146995907246595400e16L), eps * 100L);

   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(1L, static_cast<long double>(0.5L)), static_cast<long double>(0.5L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(4L, static_cast<long double>(0.5L)), static_cast<long double>(-0.3307291666666666666666666666666666666667L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(7L, static_cast<long double>(0.5L)), static_cast<long double>(-0.5183392237103174603174603174603174603175L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(20L, static_cast<long double>(0.5L)), static_cast<long double>(0.3120174870800154148915399248893113634676L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(50L, static_cast<long double>(0.5L)), static_cast<long double>(-0.3181388060269979064951118308575628226834L), eps * 100L);

   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(1L, static_cast<long double>(-0.5L)), static_cast<long double>(1.5L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(4L, static_cast<long double>(-0.5L)), static_cast<long double>(3.835937500000000000000000000000000000000L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(7L, static_cast<long double>(-0.5L)), static_cast<long double>(7.950934709821428571428571428571428571429L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(20L, static_cast<long double>(-0.5L)), static_cast<long double>(76.12915699869631476833699787070874048223L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(50L, static_cast<long double>(-0.5L)), static_cast<long double>(2307.428631277506570629232863491518399720L), eps * 100L);

   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(1L, static_cast<long double>(4.5L)), static_cast<long double>(-3.500000000000000000000000000000000000000L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(4L, static_cast<long double>(4.5L)), static_cast<long double>(0.08593750000000000000000000000000000000000L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(7L, static_cast<long double>(4.5L)), static_cast<long double>(-1.036928013392857142857142857142857142857L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(20L, static_cast<long double>(4.5L)), static_cast<long double>(1.437239150257817378525582974722170737587L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::laguerre(50L, static_cast<long double>(4.5L)), static_cast<long double>(-0.7795068145562651416494321484050019245248L), eps * 100L);

   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendre(4L, 2L, static_cast<long double>(0.5L)), static_cast<long double>(4.218750000000000000000000000000000000000L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendre(7L, 5L, static_cast<long double>(0.5L)), static_cast<long double>(5696.789530152175143607977274672800795328L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendre(4L, 2L, static_cast<long double>(-0.5L)), static_cast<long double>(4.218750000000000000000000000000000000000L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::assoc_legendre(7L, 5L, static_cast<long double>(-0.5L)), static_cast<long double>(5696.789530152175143607977274672800795328L), eps * 100L);

   BOOST_CHECK_CLOSE_FRACTION(tr1::legendre(1L, static_cast<long double>(0.5L)), static_cast<long double>(0.5L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::legendre(4L, static_cast<long double>(0.5L)), static_cast<long double>(-0.2890625000000000000000000000000000000000L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::legendre(7L, static_cast<long double>(0.5L)), static_cast<long double>(0.2231445312500000000000000000000000000000L), eps * 100L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::legendre(40L, static_cast<long double>(0.5L)), static_cast<long double>(-0.09542943523261546936538467572384923220258L), eps * 100L);

   BOOST_CHECK_CLOSE(tr1::beta(static_cast<long double>(1L), static_cast<long double>(1L)), static_cast<long double>(1L), eps * 20L * 100L);
   BOOST_CHECK_CLOSE(tr1::beta(static_cast<long double>(1L), static_cast<long double>(4L)), static_cast<long double>(0.25L), eps * 20L * 100L);
   BOOST_CHECK_CLOSE(tr1::beta(static_cast<long double>(4L), static_cast<long double>(1L)), static_cast<long double>(0.25L), eps * 20L * 100L);
   BOOST_CHECK_CLOSE(tr1::beta(sv, static_cast<long double>(4L)), 1/sv, eps * 20L * 100L);
   BOOST_CHECK_CLOSE(tr1::beta(static_cast<long double>(4L), sv), 1/sv, eps * 20L * 100L);
   BOOST_CHECK_CLOSE(tr1::beta(static_cast<long double>(4L), static_cast<long double>(20L)), static_cast<long double>(0.00002823263692828910220214568040654997176736L), eps * 20L * 100L);
   BOOST_CHECK_CLOSE(tr1::beta(static_cast<long double>(0.0125L), static_cast<long double>(0.000023L)), static_cast<long double>(43558.24045647538375006349016083320744662L), eps * 20L * 100L);

   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<long double>(0L)), static_cast<long double>(1.5707963267948966192313216916397514420985846996876L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<long double>(0.125L)), static_cast<long double>(1.5769867712158131421244030532288080803822271060839L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<long double>(0.25L)), static_cast<long double>(1.5962422221317835101489690714979498795055744578951L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<long double>(300L)/1024L), static_cast<long double>(1.6062331054696636704261124078746600894998873503208L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<long double>(400L)/1024L), static_cast<long double>(1.6364782007562008756208066125715722889067992997614L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<long double>(-0.5L)), static_cast<long double>(1.6857503548125960428712036577990769895008008941411L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_1(static_cast<long double>(-0.75L)), static_cast<long double>(1.9109897807518291965531482187613425592531451316788L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<long double>(-1L)), static_cast<long double>(1L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<long double>(0L)), static_cast<long double>(1.5707963267948966192313216916397514420985846996876L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<long double>(100L) / 1024L), static_cast<long double>(1.5670445330545086723323795143598956428788609133377L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<long double>(200L) / 1024L), static_cast<long double>(1.5557071588766556854463404816624361127847775545087L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<long double>(300L) / 1024L), static_cast<long double>(1.5365278991162754883035625322482669608948678755743L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<long double>(400L) / 1024L), static_cast<long double>(1.5090417763083482272165682786143770446401437564021L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<long double>(-0.5L)), static_cast<long double>(1.4674622093394271554597952669909161360253617523272L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<long double>(-600L) / 1024L), static_cast<long double>(1.4257538571071297192428217218834579920545946473778L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<long double>(-800L) / 1024L), static_cast<long double>(1.2927868476159125056958680222998765985004489572909L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_2(static_cast<long double>(-900L) / 1024L), static_cast<long double>(1.1966864890248739524112920627353824133420353430982L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<long double>(0.2L), static_cast<long double>(0L)), static_cast<long double>(1.586867847454166237308008033828114192951L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<long double>(0.4L), static_cast<long double>(0L)), static_cast<long double>(1.639999865864511206865258329748601457626L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<long double>(0L), static_cast<long double>(0L)), static_cast<long double>(1.57079632679489661923132169163975144209858469968755291048747L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<long double>(0L), static_cast<long double>(0.5L)), static_cast<long double>(2.221441469079183123507940495030346849307L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<long double>(0.3L), static_cast<long double>(-4L)), static_cast<long double>(0.712708870925620061597924858162260293305195624270730660081949L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<long double>(-0.5L), static_cast<long double>(-1e+05L)), static_cast<long double>(0.00496944596485066055800109163256108604615568144080386919012831L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<long double>(-0.75L), static_cast<long double>(-1e+10L)), static_cast<long double>(0.0000157080225184890546939710019277357161497407143903832703317801L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<long double>(-0.875L), static_cast<long double>(1L) / 1024L), static_cast<long double>(2.18674503176462374414944618968850352696579451638002110619287L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::comp_ellint_3(static_cast<long double>(-0.875L), static_cast<long double>(1023L)/1024L), static_cast<long double>(101.045289804941384100960063898569538919135722087486350366997L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<long double>(2.25L), static_cast<long double>(1L)/(1024*1024L)), static_cast<long double>(2.34379212133481347189068464680335815256364262507955635911656e-15L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<long double>(5.5L), static_cast<long double>(3.125L)), static_cast<long double>(0.0583514045989371500460946536220735787163510569634133670181210L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<long double>(-5L) + static_cast<long double>(1L)/1024L, static_cast<long double>(2.125L)), static_cast<long double>(0.0267920938009571023702933210070984416052633027166975342895062L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<long double>(-5.5L), static_cast<long double>(10L)), static_cast<long double>(597.577606961369169607937419869926705730305175364662688426534L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<long double>(-10486074L)/(1024*1024L), static_cast<long double>(1L)/1024L), static_cast<long double>(1.41474005665181350367684623930576333542989766867888186478185e35L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<long double>(-10486074L)/(1024*1024L), static_cast<long double>(50L)), static_cast<long double>(1.07153277202900671531087024688681954238311679648319534644743e20L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<long double>(144794L)/1024L, static_cast<long double>(100L)), static_cast<long double>(2066.27694757392660413922181531984160871678224178890247540320L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_i(static_cast<long double>(-144794L)/1024L, static_cast<long double>(100L)), static_cast<long double>(2066.27694672763190927440969155740243346136463461655104698748L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(2457L)/1024L, static_cast<long double>(1L)/1024L), static_cast<long double>(3.80739920118603335646474073457326714709615200130620574875292e-9L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(5.5L), static_cast<long double>(3217L)/1024L), static_cast<long double>(0.0281933076257506091621579544064767140470089107926550720453038L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(-5.5L), static_cast<long double>(3217L)/1024L), static_cast<long double>(-2.55820064470647911823175836997490971806135336759164272675969L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(-5.5L), static_cast<long double>(1e+04L)), static_cast<long double>(2.449843111985605522111159013846599118397e-03L), eps * 50000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(5.5L), static_cast<long double>(1e+04L)), static_cast<long double>(0.00759343502722670361395585198154817047185480147294665270646578L), eps * 5000L);
   //BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(5.5L), static_cast<long double>(1e+06L)), static_cast<long double>(-0.000747424248595630177396350688505919533097973148718960064663632L), eps * 50000L);
   //BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(5.125L), static_cast<long double>(1e+06L)), static_cast<long double>(-0.000776600124835704280633640911329691642748783663198207360238214L), eps * 50000L);
   //BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(5.875L), static_cast<long double>(1e+06L)), static_cast<long double>(-0.000466322721115193071631008581529503095819705088484386434589780L), eps * 50000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(0.5L), static_cast<long double>(101L)), static_cast<long double>(0.0358874487875643822020496677692429287863419555699447066226409L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(-5.5L), static_cast<long double>(1e+04L)), static_cast<long double>(0.00244984311198560552211115901384659911839737686676766460822577L), eps * 50000L);
   //BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(-5.5L), static_cast<long double>(1e+06L)), static_cast<long double>(0.000279243200433579511095229508894156656558211060453622750659554L), eps * 50000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(-0.5L), static_cast<long double>(101L)), static_cast<long double>(0.0708184798097594268482290389188138201440114881159344944791454L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(-10486074L) / (1024*1024L), static_cast<long double>(1L)/1024L), static_cast<long double>(1.41474013160494695750009004222225969090304185981836460288562e35L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(-10486074L) / (1024*1024L), static_cast<long double>(15L)), static_cast<long double>(-0.0902239288885423309568944543848111461724911781719692852541489L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(10486074L) / (1024*1024L), static_cast<long double>(1e+02L)), static_cast<long double>(-0.0547064914615137807616774867984047583596945624129838091326863L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(10486074L) / (1024*1024L), static_cast<long double>(2e+04L)), static_cast<long double>(-0.00556783614400875611650958980796060611309029233226596737701688L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_j(static_cast<long double>(-10486074L) / (1024*1024L), static_cast<long double>(1e+02L)), static_cast<long double>(-0.0547613660316806551338637153942604550779513947674222863858713L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<long double>(0.5L), static_cast<long double>(0.875L)), static_cast<long double>(0.558532231646608646115729767013630967055657943463362504577189L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<long double>(0.5L), static_cast<long double>(1.125L)), static_cast<long double>(0.383621010650189547146769320487006220295290256657827220786527L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<long double>(2.25L), static_cast<long double>(std::ldexp(1.0L, -30L))), static_cast<long double>(5.62397392719283271332307799146649700147907612095185712015604e20L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<long double>(5.5L), static_cast<long double>(3217L)/1024L), static_cast<long double>(1.30623288775012596319554857587765179889689223531159532808379L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<long double>(-5.5L), static_cast<long double>(10L)), static_cast<long double>(0.0000733045300798502164644836879577484533096239574909573072142667L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<long double>(-5.5L), static_cast<long double>(100L)), static_cast<long double>(5.41274555306792267322084448693957747924412508020839543293369e-45L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<long double>(10240L)/1024L, static_cast<long double>(1L)/1024L), static_cast<long double>(2.35522579263922076203415803966825431039900000000993410734978e38L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<long double>(10240L)/1024L, static_cast<long double>(10L)), static_cast<long double>(0.00161425530039067002345725193091329085443750382929208307802221L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<long double>(144793L)/1024L, static_cast<long double>(100L)), static_cast<long double>(1.39565245860302528069481472855619216759142225046370312329416e-6L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_bessel_k(static_cast<long double>(144793L)/1024L, static_cast<long double>(200L)), static_cast<long double>(9.11950412043225432171915100042647230802198254567007382956336e-68L), eps * 7000L);

   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<long double>(0.5L), static_cast<long double>(1L) / (1024*1024L)), static_cast<long double>(-817.033790261762580469303126467917092806755460418223776544122L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<long double>(5.5L), static_cast<long double>(3.125L)), static_cast<long double>(-2.61489440328417468776474188539366752698192046890955453259866L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<long double>(-5.5L), static_cast<long double>(3.125L)), static_cast<long double>(-0.0274994493896489729948109971802244976377957234563871795364056L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<long double>(-5.5L), static_cast<long double>(1e+04L)), static_cast<long double>(-0.00759343502722670361395585198154817047185480147294665270646578L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<long double>(-10486074L) / (1024*1024L), static_cast<long double>(1L)/1024L), static_cast<long double>(-1.50382374389531766117868938966858995093408410498915220070230e38L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<long double>(-10486074L) / (1024*1024L), static_cast<long double>(1e+02L)), static_cast<long double>(0.0583041891319026009955779707640455341990844522293730214223545L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<long double>(141.75L), static_cast<long double>(1e+02L)), static_cast<long double>(-5.38829231428696507293191118661269920130838607482708483122068e9L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<long double>(141.75L), static_cast<long double>(2e+04L)), static_cast<long double>(-0.00376577888677186194728129112270988602876597726657372330194186L), eps * 50000L);
   BOOST_CHECK_CLOSE(tr1::cyl_neumann(static_cast<long double>(-141.75L), static_cast<long double>(1e+02L)), static_cast<long double>(-3.81009803444766877495905954105669819951653361036342457919021e9L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(0L), static_cast<long double>(0L)), static_cast<long double>(0L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(0L), static_cast<long double>(-10L)), static_cast<long double>(-10L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(-1L), static_cast<long double>(-1L)), static_cast<long double>(-1.2261911708835170708130609674719067527242483502207L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(0.875L), static_cast<long double>(-4L)), static_cast<long double>(-5.3190556182262405182189463092940736859067548232647L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(-0.625L), static_cast<long double>(8L)), static_cast<long double>(9.0419973860310100524448893214394562615252527557062L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(0.875L), static_cast<long double>(1e-05L)), static_cast<long double>(0.000010000000000127604166668510945638036143355898993088L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(10L)/1024L, static_cast<long double>(1e+05L)), static_cast<long double>(100002.38431454899771096037307519328741455615271038L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(1L), static_cast<long double>(1e-20L)), static_cast<long double>(1.0000000000000000000000000000000000000000166666667e-20L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(1e-20L), static_cast<long double>(1e-20L)), static_cast<long double>(1.000000000000000e-20L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(400L)/1024L, static_cast<long double>(1e+20L)), static_cast<long double>(1.0418143796499216839719289963154558027005142709763e20L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(0.5L), static_cast<long double>(2L)), static_cast<long double>(2.1765877052210673672479877957388515321497888026770L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(0.5L), static_cast<long double>(4L)), static_cast<long double>(4.2543274975235836861894752787874633017836785640477L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(0.5L), static_cast<long double>(6L)), static_cast<long double>(6.4588766202317746302999080620490579800463614807916L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(0.5L), static_cast<long double>(10L)), static_cast<long double>(10.697409951222544858346795279378531495869386960090L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(0.5L), static_cast<long double>(-2L)), static_cast<long double>(-2.1765877052210673672479877957388515321497888026770L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(0.5L), static_cast<long double>(-4L)), static_cast<long double>(-4.2543274975235836861894752787874633017836785640477L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(0.5L), static_cast<long double>(-6L)), static_cast<long double>(-6.4588766202317746302999080620490579800463614807916L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_1(static_cast<long double>(0.5L), static_cast<long double>(-10L)), static_cast<long double>(-10.697409951222544858346795279378531495869386960090L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<long double>(0L), static_cast<long double>(0L)), static_cast<long double>(0L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<long double>(0L), static_cast<long double>(-10L)), static_cast<long double>(-10L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<long double>(-1L), static_cast<long double>(-1L)), static_cast<long double>(-0.84147098480789650665250232163029899962256306079837L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<long double>(900L) / 1024L, static_cast<long double>(-4L)), static_cast<long double>(-3.1756145986492562317862928524528520686391383168377L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<long double>(-600L) / 1024L, static_cast<long double>(8L)), static_cast<long double>(7.2473147180505693037677015377802777959345489333465L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<long double>(800L) / 1024L, static_cast<long double>(1e-05L)), static_cast<long double>(9.999999999898274739584436515967055859383969942432E-6L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<long double>(100L) / 1024L, static_cast<long double>(1e+05L)), static_cast<long double>(99761.153306972066658135668386691227343323331995888L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<long double>(-0.5L), static_cast<long double>(1e+10L)), static_cast<long double>(9.3421545766487137036576748555295222252286528414669e9L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_2(static_cast<long double>(400L) / 1024L, ldexp(static_cast<long double>(1L), 66L)), static_cast<long double>(7.0886102721911705466476846969992069994308167515242e19L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0L), static_cast<long double>(1L), static_cast<long double>(-1L)), static_cast<long double>(-1.557407724654902230506974807458360173087L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0.4L), static_cast<long double>(0L), static_cast<long double>(-4L)), static_cast<long double>(-4.153623371196831087495427530365430979011L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(-0.6L), static_cast<long double>(0L), static_cast<long double>(8L)), static_cast<long double>(8.935930619078575123490612395578518914416L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0.25L), static_cast<long double>(0L), static_cast<long double>(0.5L)), static_cast<long double>(0.501246705365439492445236118603525029757890291780157969500480L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0L), static_cast<long double>(0L), static_cast<long double>(0.5L)), static_cast<long double>(0.5L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0L), static_cast<long double>(-2L), static_cast<long double>(0.5L)), static_cast<long double>(0.437501067017546278595664813509803743009132067629603474488486L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0L), static_cast<long double>(0.25L), static_cast<long double>(0.5L)), static_cast<long double>(0.510269830229213412212501938035914557628394166585442994564135L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0L), static_cast<long double>(0.75L), static_cast<long double>(0.5L)), static_cast<long double>(0.533293253875952645421201146925578536430596894471541312806165L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0L), static_cast<long double>(0.75L), static_cast<long double>(0.75L)), static_cast<long double>(0.871827580412760575085768367421866079353646112288567703061975L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0L), static_cast<long double>(1L), static_cast<long double>(0.25L)), static_cast<long double>(0.255341921221036266504482236490473678204201638800822621740476L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0L), static_cast<long double>(2L), static_cast<long double>(0.25L)), static_cast<long double>(0.261119051639220165094943572468224137699644963125853641716219L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0L), static_cast<long double>(1023L)/1024L, static_cast<long double>(1.5L)), static_cast<long double>(13.2821612239764190363647953338544569682942329604483733197131L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0.5L), static_cast<long double>(0.5L), static_cast<long double>(-1L)), static_cast<long double>(-1.228014414316220642611298946293865487807L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0.5L), static_cast<long double>(0.5L), static_cast<long double>(1e+10L)), static_cast<long double>(1.536591003599172091573590441336982730551e+10L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0.75L), static_cast<long double>(-1e+05L), static_cast<long double>(10L)), static_cast<long double>(0.0347926099493147087821620459290460547131012904008557007934290L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0.875L), static_cast<long double>(-1e+10L), static_cast<long double>(10L)), static_cast<long double>(0.000109956202759561502329123384755016959364346382187364656768212L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0.875L), static_cast<long double>(-1e+10L), static_cast<long double>(1e+20L)), static_cast<long double>(1.00000626665567332602765201107198822183913978895904937646809e15L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0.875L), static_cast<long double>(-1e+10L), static_cast<long double>(1608L)/1024L), static_cast<long double>(0.0000157080616044072676127333183571107873332593142625043567690379L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0.875L), 1-static_cast<long double>(1L) / 1024L, static_cast<long double>(1e+20L)), static_cast<long double>(6.43274293944380717581167058274600202023334985100499739678963e21L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0.25L), static_cast<long double>(50L), static_cast<long double>(0.1L)), static_cast<long double>(0.124573770342749525407523258569507331686458866564082916835900L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::ellint_3(static_cast<long double>(0.25L), static_cast<long double>(1.125L), static_cast<long double>(1L)), static_cast<long double>(1.77299767784815770192352979665283069318388205110727241629752L), eps * 5000L);

   BOOST_CHECK_CLOSE(tr1::expint(static_cast<long double>(1L)/1024L), static_cast<long double>(-6.35327933972759151358547423727042905862963067106751711596065L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<long double>(0.125L)), static_cast<long double>(-1.37320852494298333781545045921206470808223543321810480716122L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<long double>(0.5L)), static_cast<long double>(0.454219904863173579920523812662802365281405554352642045162818L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<long double>(1L)), static_cast<long double>(1.89511781635593675546652093433163426901706058173270759164623L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<long double>(50.5L)), static_cast<long double>(1.72763195602911805201155668940185673806099654090456049881069e20L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<long double>(-1L)/1024L), static_cast<long double>(-6.35523246483107180261445551935803221293763008553775821607264L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<long double>(-0.125L)), static_cast<long double>(-1.62342564058416879145630692462440887363310605737209536579267L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<long double>(-0.5L)), static_cast<long double>(-0.559773594776160811746795939315085235226846890316353515248293L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<long double>(-1L)), static_cast<long double>(-0.219383934395520273677163775460121649031047293406908207577979L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::expint(static_cast<long double>(-50.5L)), static_cast<long double>(-2.27237132932219350440719707268817831250090574830769670186618e-24L), eps * 5000L);

   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(0L, static_cast<long double>(1L)), static_cast<long double>(1.L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(1L, static_cast<long double>(1L)), static_cast<long double>(2.L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(1L, static_cast<long double>(2L)), static_cast<long double>(4.L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(1L, static_cast<long double>(10L)), static_cast<long double>(20L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(1L, static_cast<long double>(100L)), static_cast<long double>(200L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(1L, static_cast<long double>(1e6L)), static_cast<long double>(2e6L), 100L * eps);
   //BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(10L, static_cast<long double>(30L)), static_cast<long double>(5.896624628001300E+17L), 100L * eps);
   //BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(10L, static_cast<long double>(1000L)), static_cast<long double>(1.023976960161280E+33L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(10L, static_cast<long double>(10L)), static_cast<long double>(8.093278209760000E+12L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(10L, static_cast<long double>(-10L)), static_cast<long double>(8.093278209760000E+12L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(3L, static_cast<long double>(-10L)), static_cast<long double>(-7.880000000000000E+3L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(3L, static_cast<long double>(-1000L)), static_cast<long double>(-7.999988000000000E+9L), 100L * eps);
   BOOST_CHECK_CLOSE_FRACTION(tr1::hermite(3L, static_cast<long double>(-1000000L)), static_cast<long double>(-7.999999999988000E+18L), 100L * eps);

   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(0.125L)), static_cast<long double>(-0.63277562349869525529352526763564627152686379131122L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(1023L) / static_cast<long double>(1024L)), static_cast<long double>(-1023.4228554489429786541032870895167448906103303056L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(1025L) / static_cast<long double>(1024L)), static_cast<long double>(1024.5772867695045940578681624248887776501597556226L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(0.5L)), static_cast<long double>(-1.46035450880958681288949915251529801246722933101258149054289L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(1.125L)), static_cast<long double>(8.5862412945105752999607544082693023591996301183069L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(2L)), static_cast<long double>(1.6449340668482264364724151666460251892189499012068L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(3.5L)), static_cast<long double>(1.1267338673170566464278124918549842722219969574036L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(4L)), static_cast<long double>(1.08232323371113819151600369654116790277475095191872690768298L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(4L + static_cast<long double>(1L) / 1024L), static_cast<long double>(1.08225596856391369799036835439238249195298434901488518878804L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(4.5L)), static_cast<long double>(1.05470751076145426402296728896028011727249383295625173068468L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(6.5L)), static_cast<long double>(1.01200589988852479610078491680478352908773213619144808841031L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(7.5L)), static_cast<long double>(1.00582672753652280770224164440459408011782510096320822989663L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(8.125L)), static_cast<long double>(1.0037305205308161603183307711439385250181080293472L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(16.125L)), static_cast<long double>(1.0000140128224754088474783648500235958510030511915L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(0L)), static_cast<long double>(-0.5L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(-0.125L)), static_cast<long double>(-0.39906966894504503550986928301421235400280637468895L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(-1L)), static_cast<long double>(-0.083333333333333333333333333333333333333333333333333L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(-2L)), static_cast<long double>(0L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(-2.5L)), static_cast<long double>(0.0085169287778503305423585670283444869362759902200745L), eps * 5000L * 3L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(-3L)), static_cast<long double>(0.0083333333333333333333333333333333333333333333333333L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(-4L)), static_cast<long double>(0L), eps * 5000L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(-20L)), static_cast<long double>(0L), eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(-21L)), static_cast<long double>(-281.46014492753623188405797101449275362318840579710L), eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::riemann_zeta(static_cast<long double>(-30.125L)), static_cast<long double>(2.2762941726834511267740045451463455513839970804578e7L), eps * 5000L * 100L);

   BOOST_CHECK_CLOSE(tr1::sph_bessel(0L, static_cast<long double>(0.1433600485324859619140625e-1L)), static_cast<long double>(0.9999657468461303487880990241993035937654e0L),  eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_bessel(0L, static_cast<long double>(0.1760916970670223236083984375e-1L)), static_cast<long double>(0.9999483203249623334100130061926184665364e0L),  eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_bessel(2L, static_cast<long double>(0.1433600485324859619140625e-1L)), static_cast<long double>(0.1370120120703995134662099191103188366059e-4L),  eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_bessel(2L, static_cast<long double>(0.1760916970670223236083984375e-1L)), static_cast<long double>(0.2067173265753174063228459655801741280461e-4L),  eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_bessel(7L, static_cast<long double>(0.1252804412841796875e3L)), static_cast<long double>(0.7887555711993028736906736576314283291289e-2L),  eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_bessel(7L, static_cast<long double>(0.25554705810546875e3L)), static_cast<long double>(-0.1463292767579579943284849187188066532514e-2L),  eps * 5000L * 100L);

   BOOST_CHECK_CLOSE(tr1::sph_neumann(0L, static_cast<long double>(0.408089816570281982421875e0L)), static_cast<long double>(-0.2249212131304610409189209411089291558038e1L), eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_neumann(0L, static_cast<long double>(0.6540834903717041015625e0L)), static_cast<long double>(-0.1213309779166084571756446746977955970241e1L),   eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_neumann(2L, static_cast<long double>(0.408089816570281982421875e0L)), static_cast<long double>(-0.4541702641837159203058389758895634766256e2L),   eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_neumann(2L, static_cast<long double>(0.6540834903717041015625e0L)), static_cast<long double>(-0.1156112621471167110574129561700037138981e2L),   eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_neumann(10L, static_cast<long double>(0.1097540378570556640625e1L)), static_cast<long double>(-0.2427889658115064857278886600528596240123e9L),   eps * 5000L * 100L);
   BOOST_CHECK_CLOSE(tr1::sph_neumann(10L, static_cast<long double>(0.30944411754608154296875e1L)), static_cast<long double>(-0.3394649246350136450439882104151313759251e4L),   eps * 5000L * 100L);

   BOOST_CHECK_CLOSE_FRACTION(tr1::sph_legendre(3L, 2L, static_cast<long double>(0.5L)), static_cast<long double>(0.2061460599687871330692286791802688341213L), eps * 5000L);
   BOOST_CHECK_CLOSE_FRACTION(tr1::sph_legendre(40L, 15L, static_cast<long double>(0.75L)), static_cast<long double>(-0.406036847302819452666908966769096223205057182668333862900509L), eps * 5000L);
#endif
#endif
}

BOOST_AUTO_TEST_CASE( test_main )
{
#ifndef TEST_LD
   test_values(1.0f, "float");
   test_values(1.0, "double");
#else
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_values(1.0L, "long double");
#endif
#endif

}

