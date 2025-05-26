// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/tools/config.hpp>

#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#include <boost/math/concepts/real_concept.hpp>
#endif

#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/expint.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
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

template <class T>
T expint_wrapper(T n, T z)
{
#ifdef EN_FUNCTION_TO_TEST
   return EN_FUNCTION_TO_TEST(
      boost::math::itrunc(n), z);
#else
   return boost::math::expint(
      boost::math::itrunc(n), z);
#endif
}

template <class Real, class T>
void do_test_expint(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(EN_FUNCTION_TO_TEST))
   //
   // test En(T) against data:
   //
   using namespace std;
   typedef Real                   value_type;

   std::cout << test_name << " with type " << type_name << std::endl;

   typedef value_type (*pg)(value_type, value_type);
#if defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = expint_wrapper<value_type>;
#else
   pg funcp = expint_wrapper;
#endif

   boost::math::tools::test_result<value_type> result;
   //
   // test expint against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0, 1),
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "expint (En)", test_name);
   std::cout << std::endl;
#endif
}

template <class Real, class T>
void do_test_expint_Ei(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(EI_FUNCTION_TO_TEST))
   //
   // test Ei(T) against data:
   //
   using namespace std;
   typedef Real                   value_type;

   std::cout << test_name << " with type " << type_name << std::endl;

   typedef value_type (*pg)(value_type);
#ifdef EI_FUNCTION_TO_TEST
   pg funcp = EI_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::expint<value_type>;
#else
   pg funcp = boost::math::expint;
#endif

   boost::math::tools::test_result<value_type> result;
   //
   // test expint against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data,
      bind_func<Real>(funcp, 0),
      extract_result<Real>(1));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "expint (Ei)", test_name);
#endif
}

template <class T>
void test_expint(T, const char* name)
{
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
#include "expint_data.ipp"
   do_test_expint<T>(expint_data, name, "Exponential Integral En");
#include "expint_small_data.ipp"
   do_test_expint<T>(expint_small_data, name, "Exponential Integral En: small z values");
#include "expint_1_data.ipp"
   do_test_expint<T>(expint_1_data, name, "Exponential Integral E1");
#include "expinti_data.ipp"
   do_test_expint_Ei<T>(expinti_data, name, "Exponential Integral Ei");

   if(boost::math::tools::log_max_value<T>() > 100)
   {
#include "expinti_data_double.ipp"
      do_test_expint_Ei<T>(expinti_data_double, name, "Exponential Integral Ei: double exponent range");
   }
#if (defined(LDBL_MAX_10_EXP) && (LDBL_MAX_10_EXP > 2000)) || defined(TEST_UDT)
   if(boost::math::tools::log_max_value<T>() > 1000)
   {
#include "expinti_data_long.ipp"
      do_test_expint_Ei<T>(expinti_data_long, name, "Exponential Integral Ei: long exponent range");
   }
#endif
}

template <class T>
void test_spots(T, const char* t)
{
   std::cout << "Testing basic sanity checks for type " << t << std::endl;
   //
   // Basic sanity checks, tolerance is 100 epsilon 
   // expressed as a percentage:
   //
   T tolerance = boost::math::tools::epsilon<T>() * 100 *
      (boost::is_floating_point<T>::value ? 500 : 500);
   //
   // En:
   //
   BOOST_CHECK_CLOSE(::boost::math::expint(0, static_cast<T>(1)/1024), static_cast<T>(1023.0004881223430781283448725609773366468629307172L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(0, static_cast<T>(0.125F)), static_cast<T>(7.0599752206767632229191371458324058897760385999252L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(0, static_cast<T>(0.5F)), static_cast<T>(1.2130613194252668472075990699823609068838362709744L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(0, static_cast<T>(1.5F)), static_cast<T>(0.14875344009895321928885364717600834756144775290739L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(0, static_cast<T>(4.5F)), static_cast<T>(0.0024686658973871792213651409526512283936753927790603L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(0, static_cast<T>(50.0F)), static_cast<T>(3.8574996959278355660346856330540251495056653024605e-24L), tolerance);

   BOOST_CHECK_CLOSE(::boost::math::expint(1, static_cast<T>(1)/1024), static_cast<T>(6.3552324648310718026144555193580322129376300855378L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(1, static_cast<T>(0.125F)), static_cast<T>(1.6234256405841687914563069246244088736331060573721L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(1, static_cast<T>(0.5F)), static_cast<T>(0.55977359477616081174679593931508523522684689031635L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(1, static_cast<T>(1.5F)), static_cast<T>(0.10001958240663265190190933991166697826173000614035L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(1, static_cast<T>(4.5F)), static_cast<T>(0.0020734007547146144328855938695797884889319725701443L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(1, static_cast<T>(50.0F)), static_cast<T>(3.7832640295504590186989678540212857803028931862511e-24L), tolerance);

   BOOST_CHECK_CLOSE(::boost::math::expint(2, static_cast<T>(1)/1024), static_cast<T>(0.99281763247803906867747111039220635198625517639812L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(2, static_cast<T>(0.125F)), static_cast<T>(0.67956869751157430393285377765099962701786656781914L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(2, static_cast<T>(0.5F)), static_cast<T>(0.32664386232455301773040156533363783582849469032901L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(2, static_cast<T>(1.5F)), static_cast<T>(0.073100786538480851080416460896512053949576620150553L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(2, static_cast<T>(4.5F)), static_cast<T>(0.0017786931420265415481579618738214795713453909401220L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(2, static_cast<T>(50.0F)), static_cast<T>(3.7117833188688273667858889516369684601386058104706e-24L), tolerance);

   BOOST_CHECK_CLOSE(::boost::math::expint(5, static_cast<T>(1)/1024), static_cast<T>(0.24967471743034509414923673526350536071348482068601L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(5, static_cast<T>(0.125F)), static_cast<T>(0.21195078838966585733668853784460504343264488743527L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(5, static_cast<T>(0.5F)), static_cast<T>(0.13097731169586484777931864012654136046214618229236L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(5, static_cast<T>(1.5F)), static_cast<T>(0.038529924425495155395971538166055731456940831714065L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(5, static_cast<T>(4.5F)), static_cast<T>(0.0012311157382296328534406162790653865883418172939975L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(5, static_cast<T>(50.0F)), static_cast<T>(3.5124400631332394056901125740903320085753990490695e-24L), tolerance);

   BOOST_CHECK_CLOSE(::boost::math::expint(22, static_cast<T>(1)/1024), static_cast<T>(0.047570244582119027573000641518739337055625638422014L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(22, static_cast<T>(0.125F)), static_cast<T>(0.041762730174712898120606793100375713866392079558609L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(22, static_cast<T>(0.5F)), static_cast<T>(0.028178840877963713109230354192609362941651630844684L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(22, static_cast<T>(1.5F)), static_cast<T>(0.0098864453561701486668317763826728620676449196215016L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(22, static_cast<T>(4.5F)), static_cast<T>(0.00043257793497205419001613830279995985617548506505159L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(22, static_cast<T>(50.0F)), static_cast<T>(2.6900194251201629500599598206345018300567305625080e-24L), tolerance);
   //
   // Ei:
   //
   BOOST_CHECK_CLOSE(::boost::math::expint(static_cast<T>(1)/1024), static_cast<T>(-6.35327933972759151358547423727042905862963067106751711596065L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(static_cast<T>(0.125)), static_cast<T>(-1.37320852494298333781545045921206470808223543321810480716122L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(static_cast<T>(0.5)), static_cast<T>(0.454219904863173579920523812662802365281405554352642045162818L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(static_cast<T>(1)), static_cast<T>(1.89511781635593675546652093433163426901706058173270759164623L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(static_cast<T>(50.5)), static_cast<T>(1.72763195602911805201155668940185673806099654090456049881069e20L), tolerance);

   BOOST_CHECK_CLOSE(::boost::math::expint(static_cast<T>(-1)/1024), static_cast<T>(-6.35523246483107180261445551935803221293763008553775821607264L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(static_cast<T>(-0.125)), static_cast<T>(-1.62342564058416879145630692462440887363310605737209536579267L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(static_cast<T>(-0.5)), static_cast<T>(-0.559773594776160811746795939315085235226846890316353515248293L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(static_cast<T>(-1)), static_cast<T>(-0.219383934395520273677163775460121649031047293406908207577979L), tolerance);
   BOOST_CHECK_CLOSE(::boost::math::expint(static_cast<T>(-50.5)), static_cast<T>(-2.27237132932219350440719707268817831250090574830769670186618e-24L), tolerance);
}

