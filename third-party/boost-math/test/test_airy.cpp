//  Copyright John Maddock 2012
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef SYCL_LANGUAGE_VERSION
#include <pch_light.hpp>
#endif

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/airy.hpp>

#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
#include <boost/math/concepts/real_concept.hpp>
#endif

#include <boost/array.hpp>
#include <iostream>
#include <iomanip>

#ifdef _MSC_VER
#  pragma warning(disable : 4756 4127) // overflow in constant arithmetic
// Constants are too big for float case, but this doesn't matter for test.
#endif

//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the Airy functions.  
// These are basically just a few spot tests, since the underlying implementation
// is the same as for the Bessel functions.
//
template <class T>
void test_airy(T, const char* name)
{
   std::cout << "Testing type " << name << std::endl;

   static const std::array<std::array<T, 5>, 8> data = 
   {{
      // Values are x, Ai, Bi, Ai', Bi'.
      // Calculated from functions.wolfram.com.
      {{ 0, static_cast<T>(0.355028053887817239260063186004183176397979174199177240583327L), static_cast<T>(0.614926627446000735150922369093613553594728188648596505040879L), static_cast<T>(-0.258819403792806798405183560189203963479091138354934582210002L), static_cast<T>(0.448288357353826357914823710398828390866226799212262061082809L) }},
      {{ 2, static_cast<T>(0.0349241304232743791353220807918076097610602138975832071886699L), static_cast<T>(3.29809499997821471028060442522345242200397596340362078768292L), static_cast<T>(-0.0530903844336536317039991858787034912485609900458779926304030L), static_cast<T>(4.10068204993288988938203407917793529439024461377513711983771L) }},
      {{ 3.5, static_cast<T>(0.00258409878698963496327714478330027845019631096464789073425468L), static_cast<T>(33.0555067546114794142573129819217946861266278143724855010951L), static_cast<T>(-0.00500441396795258283203024967883836790716278377542974739809987L), static_cast<T>(59.1643195813609870345742121795224602529063343063320217229383L) }},
      {{ 30.5, static_cast<T>(2.04253461666926015963302258449952681287194929082634196439857e-50L), static_cast<T>(1.41092256201712698413071856671990267572381736486242769613348e48L), static_cast<T>(-1.12969466271996376602221721397236452019046628449363103927812e-49L), static_cast<T>(7.78046629440950280615411694840600772185863288897663300007879e48L) }},
      {{ -2, static_cast<T>(0.227407428201685575991924436037873799460772225417096716495790L), static_cast<T>(-0.412302587956398488083234054611461042034534834472404728823877L), static_cast<T>(0.618259020741691041406264291332475282915777945124146942159898L), static_cast<T>(0.278795166921169522685097569410983241403000593451631000239732L) }},
      {{ -3.5, static_cast<T>(-0.375533823140431911934396951580170239543268576378264063902563L), static_cast<T>(0.168939837481058611843442769540943269911562243926304070915824L), static_cast<T>(-0.343443433454048146287937374098698857094194220958713294264017L), static_cast<T>(-0.693116284907288801752443612670580462699702668014978495343697L) }},
      {{ -30.25, static_cast<T>(-0.236900428491559731119806902381433570300271552218956227857722L), static_cast<T>(0.0418614989839147441219283537268101850442118139150748075124111L), static_cast<T>(-0.232197332372689274917364138870840123428255785603863926579897L), static_cast<T>(-1.30261375952554697768880873095691151213006925411329957440342L) }},
      {{ -300.5, static_cast<T>(-0.117584304761702008955433457721670950077867326839023449546057L), static_cast<T>(0.0673518035440918806545676478730240310819758211028871823560384L), static_cast<T>(-1.16763702262473888724431076711846459336993902544926162874376L), static_cast<T>(-2.03826035550300900666977975504950803669545593208969273694133L) }},
   }};

   T tol = boost::math::tools::epsilon<T>() * 800;
   if (boost::math::tools::digits<T>() > 100)
      tol *= 2;

   #ifdef SYCL_LANGUAGE_VERSION
   tol *= 5;
   #endif

   for(unsigned i = 0; i < data.size(); ++i)
   {
      BOOST_CHECK_CLOSE_FRACTION(data[i][1], boost::math::airy_ai(data[i][0]), tol);
      if(boost::math::isfinite(data[i][2]))
         BOOST_CHECK_CLOSE_FRACTION(data[i][2], boost::math::airy_bi(data[i][0]), tol);
      BOOST_CHECK_CLOSE_FRACTION(data[i][3], boost::math::airy_ai_prime(data[i][0]), tol);
      if(boost::math::isfinite(data[i][4]))
         BOOST_CHECK_CLOSE_FRACTION(data[i][4], boost::math::airy_bi_prime(data[i][0]), tol);
   }
}


BOOST_AUTO_TEST_CASE( test_main )
{
#ifdef TEST_GSL
   gsl_set_error_handler_off();
#endif
   BOOST_MATH_CONTROL_FP;

#ifndef BOOST_MATH_BUGGY_LARGE_FLOAT_CONSTANTS
   test_airy(0.1F, "float");
#endif
   test_airy(0.1, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_airy(0.1L, "long double");
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
   test_airy(boost::math::concepts::real_concept(0), "real_concept");
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
}




