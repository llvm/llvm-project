// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/tools/config.hpp>
#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/next.hpp>  // for has_denorm_now
#include <boost/math/tools/stats.hpp>
#include "../include_private/boost/math/tools/test.hpp"
#include <boost/math/constants/constants.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class Real, class T>
void do_test_gamma(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && (!defined(TGAMMA_FUNCTION_TO_TEST) || !defined(LGAMMA_FUNCTION_TO_TEST)))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type);
#ifdef TGAMMA_FUNCTION_TO_TEST
   pg funcp = TGAMMA_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::tgamma<value_type>;
#else
   pg funcp = boost::math::tgamma;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test tgamma against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0),
      extract_result<Real>(1));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "tgamma", test_name);
   //
   // test lgamma against data:
   //
#ifdef LGAMMA_FUNCTION_TO_TEST
   funcp = LGAMMA_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   funcp = boost::math::lgamma<value_type>;
#else
   funcp = boost::math::lgamma;
#endif
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0),
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "lgamma", test_name);

   std::cout << std::endl;
#endif
}

template <class Real, class T>
void do_test_gammap1m1(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(TGAMMA1PM1_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type);
#ifdef TGAMMA1PM1_FUNCTION_TO_TEST
   pg funcp = TGAMMA1PM1_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::tgamma1pm1<value_type>;
#else
   pg funcp = boost::math::tgamma1pm1;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test tgamma1pm1 against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0),
      extract_result<Real>(1));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "tgamma1pm1", test_name);
   std::cout << std::endl;
#endif
}

template <class T>
void test_gamma(T, const char* name)
{
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // The contents are as follows, each row of data contains
   // three items, input value, gamma and lgamma:
   //
   // gamma and lgamma at integer and half integer values:
   // std::array<std::array<T, 3>, N> factorials;
   //
   // gamma and lgamma for z near 0:
   // std::array<std::array<T, 3>, N> near_0;
   //
   // gamma and lgamma for z near 1:
   // std::array<std::array<T, 3>, N> near_1;
   //
   // gamma and lgamma for z near 2:
   // std::array<std::array<T, 3>, N> near_2;
   //
   // gamma and lgamma for z near -10:
   // std::array<std::array<T, 3>, N> near_m10;
   //
   // gamma and lgamma for z near -55:
   // std::array<std::array<T, 3>, N> near_m55;
   //
   // The last two cases are chosen more or less at random,
   // except that one is even and the other odd, and both are
   // at negative poles.  The data near zero also tests near
   // a pole, the data near 1 and 2 are to probe lgamma as
   // the result -> 0.
   //
#  include "test_gamma_data.ipp"

   do_test_gamma<T>(factorials, name, "factorials");
   do_test_gamma<T>(near_0, name, "near 0");
   do_test_gamma<T>(near_1, name, "near 1");
   do_test_gamma<T>(near_2, name, "near 2");
   do_test_gamma<T>(near_m10, name, "near -10");
   do_test_gamma<T>(near_m55, name, "near -55");

   //
   // And now tgamma1pm1 which computes gamma(1+dz)-1:
   //
   do_test_gammap1m1<T>(gammap1m1_data, name, "tgamma1pm1(dz)");
}

template <class T>
void test_spots(T, const char* name)
{
   BOOST_MATH_STD_USING
   
   std::cout << "Testing type " << name << std::endl;
   //
   // basic sanity checks, tolerance is 50 epsilon expressed as a percentage:
   //
   T tolerance = boost::math::tools::epsilon<T>() * 5000;
   //
   // Extra tolerance for real_concept checks which use less accurate code:
   //
   T extra_tol = boost::is_floating_point<T>::value ? 1 : 20;

   BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(3.5)), static_cast<T>(3.3233509704478425511840640312646472177454052302295L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(0.125)), static_cast<T>(7.5339415987976119046992298412151336246104195881491L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(-0.125)), static_cast<T>(-8.7172188593831756100190140408231437691829605421405L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(-3.125)), static_cast<T>(1.1668538708507675587790157356605097019141636072094L), tolerance);
   // Lower tolerance on this one, is only really needed on Linux x86 systems, result is mostly down to std lib accuracy:
   BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(-53249.0 / 1024)), static_cast<T>(-1.2646559519067605488251406578743995122462767733517e-65L), tolerance * 3);

   // Very small values, from a bug report by Rocco Romeo:
   BOOST_CHECK_CLOSE(::boost::math::tgamma(ldexp(static_cast<T>(1), -12)), static_cast<T>(4095.42302574977164107280305038926932586783813167844235368772L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(ldexp(static_cast<T>(1), -14)), static_cast<T>(16383.4228446989052821887834066513143241996925504706815681204L), tolerance * 2);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(ldexp(static_cast<T>(1), -25)), static_cast<T>(3.35544314227843645746319656372890833248893111091576093784981e7L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(ldexp(static_cast<T>(1), -27)), static_cast<T>(1.34217727422784342467508497080056807355928046680073490038257e8L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(ldexp(static_cast<T>(1), -29)), static_cast<T>(5.36870911422784336940727488260481582524683632281496706906706e8L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(ldexp(static_cast<T>(1), -35)), static_cast<T>(3.43597383674227843351272524573929605605651956475300480712955e10L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(ldexp(static_cast<T>(1), -54)), static_cast<T>(1.80143985094819834227843350984671942971248427509141008005685e16L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(ldexp(static_cast<T>(1), -64)), static_cast<T>(1.84467440737095516154227843350984671394471047428598176073616e19L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(ldexp(static_cast<T>(1), -66)), static_cast<T>(7.37869762948382064634227843350984671394068921181531525785592922800e19L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(ldexp(static_cast<T>(1), -33)), static_cast<T>(8.58993459142278433521360841138215453639282914047157884932317481977e9L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(4 / boost::math::tools::max_value<T>()), boost::math::tools::max_value<T>() / 4, tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(-ldexp(static_cast<T>(1), -12)), static_cast<T>(-4096.57745718775464971331294488248972086965434176847741450728L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(-ldexp(static_cast<T>(1), -14)), static_cast<T>(-16384.5772760354695939336148831283410381037202353359487504624L), tolerance * 2);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(-ldexp(static_cast<T>(1), -25)), static_cast<T>(-3.35544325772156943776992988569766723938420508937071533029983e7L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(-ldexp(static_cast<T>(1), -27)), static_cast<T>(-1.34217728577215672270574319043497450577151370942651414968627e8L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(-ldexp(static_cast<T>(1), -29)), static_cast<T>(-5.36870912577215666743793215770406791630514293641886249382012e8L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(-ldexp(static_cast<T>(1), -34)), static_cast<T>(-1.71798691845772156649591034966100693794360502123447124928244e10L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(-ldexp(static_cast<T>(1), -54)), static_cast<T>(-1.80143985094819845772156649015329155101490229157245556564920e16L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(-ldexp(static_cast<T>(1), -64)), static_cast<T>(-1.84467440737095516165772156649015328606601289230246224694513e19L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(-ldexp(static_cast<T>(1), -66)), static_cast<T>(-7.37869762948382064645772156649015328606199162983179574406439e19L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(-ldexp(static_cast<T>(1), -33)), static_cast<T>(-8.58993459257721566501667413261977598620193488449233402857632e9L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(-4 / boost::math::tools::max_value<T>()), -boost::math::tools::max_value<T>() / 4, tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(-1 + ldexp(static_cast<T>(1), -22)), static_cast<T>(-4.19430442278467170746130758391572421252211886167956799318843e6L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(-1 - ldexp(static_cast<T>(1), -22)), static_cast<T>(4.19430357721600151046968956086404748206205391186399889108944e6L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(-4 + ldexp(static_cast<T>(1), -20)), static_cast<T>(43690.7294216755534842491085530510391932288379640970386378756L), tolerance * extra_tol);
   BOOST_CHECK_CLOSE(::boost::math::tgamma(-4 - ldexp(static_cast<T>(1), -20)), static_cast<T>(-43690.6039118698506165317137699180871126338425941292693705533L), tolerance * extra_tol);
   if(boost::math::tools::digits<T>() > 50)
   {
      BOOST_CHECK_CLOSE(::boost::math::tgamma(-1 + ldexp(static_cast<T>(1), -44)), static_cast<T>(-1.75921860444164227843350985473932247549232492467032584051825e13L), tolerance);
      BOOST_CHECK_CLOSE(::boost::math::tgamma(-1 - ldexp(static_cast<T>(1), -44)), static_cast<T>(1.75921860444155772156649016131144377791001546933519242218430e13L), tolerance);
      BOOST_CHECK_CLOSE(::boost::math::tgamma(-4 + ldexp(static_cast<T>(1), -44)), static_cast<T>(7.33007751850729421569517998006564998020333048893618664936994e11L), tolerance * extra_tol);
      BOOST_CHECK_CLOSE(::boost::math::tgamma(-4 - ldexp(static_cast<T>(1), -44)), static_cast<T>(-7.33007751850603911763815347967171096249288790373790093559568e11L), tolerance * extra_tol);
   }
   if(boost::math::tools::digits<T>() > 60)
   {
      BOOST_CHECK_CLOSE(::boost::math::tgamma(-1 + ldexp(static_cast<T>(1), -55)), static_cast<T>(-3.60287970189639684227843350984671785799289582631555600561524e16L), tolerance);
      BOOST_CHECK_CLOSE(::boost::math::tgamma(-1 - ldexp(static_cast<T>(1), -55)), static_cast<T>(3.60287970189639675772156649015328997929531384279596450489170e16L), tolerance * 3);
      BOOST_CHECK_CLOSE(::boost::math::tgamma(-4 + ldexp(static_cast<T>(1), -55)), static_cast<T>(1.50119987579016539608823618465835611632004877549994080474627e15L), tolerance * extra_tol);
      BOOST_CHECK_CLOSE(::boost::math::tgamma(-4 - ldexp(static_cast<T>(1), -55)), static_cast<T>(-1.50119987579016527057843048200831672241827850458884790004313e15L), tolerance * extra_tol);
   }

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4127)
#endif
   // Test bug fixes in tgamma:
   if(std::numeric_limits<T>::max_exponent10 > 244)
   {
      BOOST_CHECK_CLOSE(::boost::math::tgamma(static_cast<T>(142.75)), static_cast<T>(7.8029496083318133344429227511387928576820621466e244L), tolerance * 4);
   }
#ifdef _MSC_VER
#pragma warning(pop)
#endif
   // An extra fudge factor for real_concept which has a less accurate tgamma:
   T tolerance_tgamma_extra = std::numeric_limits<T>::is_specialized ? 1 : 10;

   int sign = 1;
   BOOST_CHECK_CLOSE(::boost::math::lgamma(static_cast<T>(3.5), &sign), static_cast<T>(1.2009736023470742248160218814507129957702389154682L), tolerance);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(static_cast<T>(0.125), &sign), static_cast<T>(2.0194183575537963453202905211670995899482809521344L), tolerance);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(static_cast<T>(-0.125), &sign), static_cast<T>(2.1653002489051702517540619481440174064962195287626L), tolerance);
   BOOST_CHECK(sign == -1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(static_cast<T>(-3.125), &sign), static_cast<T>(0.1543111276840418242676072830970532952413339012367L), tolerance * tolerance_tgamma_extra);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(static_cast<T>(-53249.0 / 1024), &sign), static_cast<T>(-149.43323093420259741100038126078721302600128285894L), tolerance);
   BOOST_CHECK(sign == -1);
   // Very small values, from a bug report by Rocco Romeo:
   BOOST_CHECK_CLOSE(::boost::math::lgamma(ldexp(static_cast<T>(1), -12), &sign), log(static_cast<T>(4095.42302574977164107280305038926932586783813167844235368772L)), tolerance);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(ldexp(static_cast<T>(1), -14), &sign), log(static_cast<T>(16383.4228446989052821887834066513143241996925504706815681204L)), tolerance);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(ldexp(static_cast<T>(1), -25), &sign), log(static_cast<T>(3.35544314227843645746319656372890833248893111091576093784981e7L)), tolerance);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(ldexp(static_cast<T>(1), -27), &sign), log(static_cast<T>(1.34217727422784342467508497080056807355928046680073490038257e8L)), tolerance);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(ldexp(static_cast<T>(1), -29), &sign), log(static_cast<T>(5.36870911422784336940727488260481582524683632281496706906706e8L)), tolerance);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(ldexp(static_cast<T>(1), -35), &sign), log(static_cast<T>(3.43597383674227843351272524573929605605651956475300480712955e10L)), tolerance);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(ldexp(static_cast<T>(1), -54), &sign), log(static_cast<T>(1.80143985094819834227843350984671942971248427509141008005685e16L)), tolerance);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(ldexp(static_cast<T>(1), -64), &sign), log(static_cast<T>(1.84467440737095516154227843350984671394471047428598176073616e19L)), tolerance);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(ldexp(static_cast<T>(1), -66), &sign), log(static_cast<T>(7.37869762948382064634227843350984671394068921181531525785592922800e19L)), tolerance);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(ldexp(static_cast<T>(1), -33), &sign), log(static_cast<T>(8.58993459142278433521360841138215453639282914047157884932317481977e9L)), tolerance);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(4 / boost::math::tools::max_value<T>(), &sign), log(boost::math::tools::max_value<T>() / 4), tolerance);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(-ldexp(static_cast<T>(1), -12), &sign), log(-static_cast<T>(-4096.57745718775464971331294488248972086965434176847741450728L)), tolerance);
   BOOST_CHECK(sign == -1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(-ldexp(static_cast<T>(1), -14), &sign), log(-static_cast<T>(-16384.5772760354695939336148831283410381037202353359487504624L)), tolerance);
   BOOST_CHECK(sign == -1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(-ldexp(static_cast<T>(1), -25), &sign), log(-static_cast<T>(-3.35544325772156943776992988569766723938420508937071533029983e7L)), tolerance);
   BOOST_CHECK(sign == -1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(-ldexp(static_cast<T>(1), -27), &sign), log(-static_cast<T>(-1.34217728577215672270574319043497450577151370942651414968627e8L)), tolerance);
   BOOST_CHECK(sign == -1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(-ldexp(static_cast<T>(1), -29), &sign), log(-static_cast<T>(-5.36870912577215666743793215770406791630514293641886249382012e8L)), tolerance);
   BOOST_CHECK(sign == -1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(-ldexp(static_cast<T>(1), -34), &sign), log(-static_cast<T>(-1.71798691845772156649591034966100693794360502123447124928244e10L)), tolerance);
   BOOST_CHECK(sign == -1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(-ldexp(static_cast<T>(1), -54), &sign), log(-static_cast<T>(-1.80143985094819845772156649015329155101490229157245556564920e16L)), tolerance);
   BOOST_CHECK(sign == -1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(-ldexp(static_cast<T>(1), -64), &sign), log(-static_cast<T>(-1.84467440737095516165772156649015328606601289230246224694513e19L)), tolerance);
   BOOST_CHECK(sign == -1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(-ldexp(static_cast<T>(1), -66), &sign), log(-static_cast<T>(-7.37869762948382064645772156649015328606199162983179574406439e19L)), tolerance);
   BOOST_CHECK(sign == -1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(-ldexp(static_cast<T>(1), -33), &sign), log(-static_cast<T>(-8.58993459257721566501667413261977598620193488449233402857632e9L)), tolerance);
   BOOST_CHECK(sign == -1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(-4 / boost::math::tools::max_value<T>(), &sign), log(boost::math::tools::max_value<T>() / 4), tolerance);
   BOOST_CHECK(sign == -1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(-1 + ldexp(static_cast<T>(1), -22), &sign), log(static_cast<T>(4.19430442278467170746130758391572421252211886167956799318843e6L)), tolerance);
   BOOST_CHECK(sign == -1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(-1 - ldexp(static_cast<T>(1), -22), &sign), log(static_cast<T>(4.19430357721600151046968956086404748206205391186399889108944e6L)), tolerance);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(-4 + ldexp(static_cast<T>(1), -20), &sign), log(static_cast<T>(43690.7294216755534842491085530510391932288379640970386378756L)), tolerance * extra_tol);
   BOOST_CHECK(sign == 1);
   BOOST_CHECK_CLOSE(::boost::math::lgamma(-4 - ldexp(static_cast<T>(1), -20), &sign), log(static_cast<T>(43690.6039118698506165317137699180871126338425941292693705533L)), tolerance * extra_tol);
   BOOST_CHECK(sign == -1);
   if(boost::math::tools::digits<T>() > 50)
   {
      BOOST_CHECK_CLOSE(::boost::math::lgamma(-1 + ldexp(static_cast<T>(1), -44), &sign), log(static_cast<T>(1.75921860444164227843350985473932247549232492467032584051825e13L)), tolerance);
      BOOST_CHECK(sign == -1);
      BOOST_CHECK_CLOSE(::boost::math::lgamma(-1 - ldexp(static_cast<T>(1), -44), &sign), log(static_cast<T>(1.75921860444155772156649016131144377791001546933519242218430e13L)), tolerance);
      BOOST_CHECK(sign == 1);
      BOOST_CHECK_CLOSE(::boost::math::lgamma(-4 + ldexp(static_cast<T>(1), -44), &sign), log(static_cast<T>(7.33007751850729421569517998006564998020333048893618664936994e11L)), tolerance * extra_tol);
      BOOST_CHECK(sign == 1);
      BOOST_CHECK_CLOSE(::boost::math::lgamma(-4 - ldexp(static_cast<T>(1), -44), &sign), log(static_cast<T>(7.33007751850603911763815347967171096249288790373790093559568e11L)), tolerance * extra_tol);
      BOOST_CHECK(sign == -1);
   }
   if(boost::math::tools::digits<T>() > 60)
   {
      BOOST_CHECK_CLOSE(::boost::math::lgamma(-1 + ldexp(static_cast<T>(1), -55), &sign), log(static_cast<T>(3.60287970189639684227843350984671785799289582631555600561524e16L)), tolerance);
      BOOST_CHECK(sign == -1);
      BOOST_CHECK_CLOSE(::boost::math::lgamma(-1 - ldexp(static_cast<T>(1), -55), &sign), log(static_cast<T>(3.60287970189639675772156649015328997929531384279596450489170e16L)), tolerance);
      BOOST_CHECK(sign == 1);
      BOOST_CHECK_CLOSE(::boost::math::lgamma(-4 + ldexp(static_cast<T>(1), -55), &sign), log(static_cast<T>(1.50119987579016539608823618465835611632004877549994080474627e15L)), tolerance * extra_tol);
      BOOST_CHECK(sign == 1);
      BOOST_CHECK_CLOSE(::boost::math::lgamma(-4 - ldexp(static_cast<T>(1), -55), &sign), log(static_cast<T>(1.50119987579016527057843048200831672241827850458884790004313e15L)), tolerance * extra_tol);
      BOOST_CHECK(sign == -1);
   }

   #ifndef BOOST_MATH_HAS_GPU_SUPPORT
   if(boost::math::detail::has_denorm_now<T>() && std::numeric_limits<T>::has_infinity && (boost::math::isinf)(1 / std::numeric_limits<T>::denorm_min()))
   {
      BOOST_CHECK_EQUAL(boost::math::tgamma(-std::numeric_limits<T>::denorm_min()), -std::numeric_limits<T>::infinity());
      BOOST_CHECK_EQUAL(boost::math::tgamma(std::numeric_limits<T>::denorm_min()), std::numeric_limits<T>::infinity());
   }
   #endif
   //
   // Extra large values for lgamma, see https://github.com/boostorg/math/issues/242
   //
   if (boost::math::tools::digits<T>() >= std::numeric_limits<double>::digits)
   {
      BOOST_CHECK_CLOSE(::boost::math::lgamma(ldexp(static_cast<T>(11103367432951928LL), 32)), static_cast<T>(2.7719825960021351251696385101478518546793793286704974382373670822285114741208958e27L), tolerance);
      BOOST_CHECK_CLOSE(::boost::math::lgamma(ldexp(static_cast<T>(11103367432951928LL), 62)), static_cast<T>(4.0411767712186990905102512019058204792570873633363159e36L), tolerance);
      BOOST_CHECK_CLOSE(::boost::math::lgamma(ldexp(static_cast<T>(11103367432951928LL), 326)), static_cast<T>(3.9754720509185529233002820161357111676582583112671658e116L), tolerance);
   }
   //
   // Super small values may cause spurious overflow:
   //
   if (std::numeric_limits<T>::is_specialized && boost::math::detail::has_denorm_now<T>())
   {
      T value = (std::numeric_limits<T>::min)();
      while (value != 0)
      {
         BOOST_CHECK((boost::math::isfinite)(boost::math::lgamma(value)));
         value /= 2;
      }
   }
}

