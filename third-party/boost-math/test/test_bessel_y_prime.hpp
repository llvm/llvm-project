//  Copyright (c) 2013 Anton Bikineev
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#include <boost/math/concepts/real_concept.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/array.hpp>
#include "functor.hpp"
#include <boost/math/concepts/real_concept.hpp>

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#  define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class Real, class T>
void do_test_cyl_neumann_y_prime(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(BESSEL_YP_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type);
#ifdef BESSEL_YP_FUNCTION_TO_TEST
   pg funcp = BESSEL_YP_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::cyl_neumann_prime<value_type, value_type>;
#else
   pg funcp = boost::math::cyl_neumann_prime;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

#include <boost/math/concepts/real_concept.hpp>
   //
   // test cyl_neumann against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func<Real>(funcp, 0, 1), 
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "cyl_neumann_prime", test_name);
   std::cout << std::endl;
#endif
}

template <class T>
T cyl_neumann_prime_int_wrapper(T v, T x)
{
#ifdef BESSEL_YNP_FUNCTION_TO_TEST
   return static_cast<T>(BESSEL_YNP_FUNCTION_TO_TEST(boost::math::itrunc(v), x));
#else
   return static_cast<T>(boost::math::cyl_neumann_prime(boost::math::itrunc(v), x));
#endif
}

template <class Real, class T>
void do_test_cyl_neumann_y_prime_int(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(BESSEL_YNP_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type);
#if defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = cyl_neumann_prime_int_wrapper<value_type>;
#else
   pg funcp = cyl_neumann_prime_int_wrapper;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test cyl_neumann derivative against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func<Real>(funcp, 0, 1), 
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "cyl_neumann_prime (integer orders)", test_name);
   std::cout << std::endl;
#endif
}

template <class Real, class T>
void do_test_sph_neumann_y_prime(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(BESSEL_YSP_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(unsigned, value_type);
#ifdef BESSEL_YPS_FUNCTION_TO_TEST
   pg funcp = BESSEL_YPS_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::sph_neumann_prime<value_type>;
#else
   pg funcp = boost::math::sph_neumann_prime;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test sph_neumann against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func_int1<Real>(funcp, 0, 1), 
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "sph_neumann_prime", test_name);
   std::cout << std::endl;
#endif
}

template <class T>
void test_bessel_prime(T, const char* name)
{
   using std::ldexp;
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // The contents are as follows, each row of data contains
   // three items, input value a, input value b and Y'(a, b):
   // 
    // function values calculated on wolframalpha.com
    static const std::array<std::array<typename table_type<T>::type, 3>, 9> y0_prime_data = {{
        {{ SC_(0.0), SC_(1.0), SC_(0.7812128213002887165471500000479648205499063907164) }},
        {{ SC_(0.0), SC_(2.0), SC_(0.1070324315409375468883707722774766366874808982351) }},
        {{ SC_(0.0), SC_(4.0), SC_(-0.397925710557100005253979972450791852271189181623) }},
        {{ SC_(0.0), SC_(8.0), SC_(0.15806046173124749425555526618748355032734404952671) }},
        {{ SC_(0.0), SC_(1e-05), SC_(63661.97727536548515747484843924772510915025447869) }},
        {{ SC_(0.0), SC_(1e-10), SC_(6.366197723675813431507891842844626117090808311905e9) }},
        {{ SC_(0.0), SC_(1e-20), SC_(6.366197723675813430755350534900574482790569176554e19) }},
        {{ SC_(0.0), SC_(1e+03), SC_(0.0247843312923517789148623560971412909386318548649) }},
        {{ SC_(0.0), SC_(1e+05), SC_(-0.00171921035008825630099494523539897102954509505) }}
    }};
    static const std::array<std::array<typename table_type<T>::type, 3>, 9> y1_prime_data = {{
        {{ SC_(1.0), SC_(1.0), SC_(0.8694697855159656745300767660714799833777239138071) }},
        {{ SC_(1.0), SC_(2.0), SC_(0.5638918884202138930407919788658961916118796762034) }},
        {{ SC_(1.0), SC_(4.0), SC_(-0.116422166964339993217130127559851181308289885304) }},
        {{ SC_(1.0), SC_(8.0), SC_(0.24327904710397215730926780877205580306573293697226) }},
        {{ SC_(1.0), SC_(1e-10), SC_(6.366197723675813430034640215574901912821641347643e19) }},
        {{ SC_(1.0), SC_(1e-20), SC_(6.366197723675813430755350534900574481363849370436e39) }},
        {{ SC_(1.0), SC_(1e+01), SC_(0.030769624862904003032131529943867767819086460209939) }},
        {{ SC_(1.0), SC_(1e+03), SC_(0.004740702308915165178688123821762396300797636752) }},
        {{ SC_(1.0), SC_(1e+05), SC_(0.00184674896676156322177773107486310726913857253) }}
    }};
    static const std::array<std::array<typename table_type<T>::type, 3>, 10> yn_prime_data = {{
        {{ SC_(2.0), SC_(1e-20), SC_(2.546479089470325372302140213960229792551354331847e60) }},
        {{ SC_(5.0), SC_(10.0), SC_(-0.21265103571277493482623417349611996600573875672875) }},
        {{ SC_(-5.0), SC_(1e+06), SC_(0.00072596421871030053058120610033601018452750251) }},
        {{ SC_(10.0), SC_(10.0), SC_(0.16051488637815838440809874678012991818716553338993) }},
        {{ SC_(10.0), SC_(1e-10), SC_(1.1828049049433493390095436658120487349235941485975e119) }},
        {{ SC_(-10.0), SC_(1e+06), SC_(-0.00033107967471992097725245404942310474516318425) }},
        {{ SC_(1e+02), SC_(5.0), SC_(1.0156878983956300357005118672219842696133568692723e117) }},
        {{ SC_(1e+03), SC_(1e+05), SC_(0.00128310308817651270517132752369325022363869159) }},
        {{ SC_(-1e+03), SC_(7e+02), SC_(1.9243675144213106227065036295645482241938721428442e77) }},
        {{ SC_(-25.0), SC_(8.0), SC_(-1.0191840913424144032043561764980932223038174827996e9) }}
    }};
    static const std::array<std::array<T, 3>, 11> yv_prime_data = {{
        {{ SC_(0.5), T(1) / (1024*1024), SC_(4.283610118295381639304989276580713877375759e8) }},
        {{ SC_(5.5), SC_(3.125), SC_(3.46903134947470280592767672475643312107258) }},
        {{ SC_(-5.5), SC_(3.125), SC_(-0.04142495199637659623440832639970224440469) }},
        {{ SC_(-5.5), SC_(1e+04), SC_(0.00245022241637437956702428797044365092097074) }},
        {{ T(-10486074) / (1024*1024), T(1)/1024, SC_(1.539961618935582531021699399508514975292038639e42) }},
        {{ T(-10486074) / (1024*1024), SC_(1e+02), SC_(-0.054782042073650048917092191171177791880141278121) }},
        {{ SC_(141.75), SC_(1e+02), SC_(5.3859930471571245788582581390871501852536045509e9) }},
        {{ SC_(141.75), SC_(2e+04), SC_(-0.0042010736481689878858599823347897260616269998902) }},
        {{ SC_(-141.75), SC_(1e+02), SC_(3.8084722070683992315593455637944657331085673830e9) }},
        {{ SC_(8.5), boost::math::constants::pi<T>() * 4, SC_(0.014516314554743677558496402742690038592727861) }},
        {{ SC_(-8.5), boost::math::constants::pi<T>() * 4, SC_(-0.194590144622675911618596506265006877277073804) }},
    }};
    static const std::array<std::array<T, 3>, 7> yv_prime_large_data = {{
#if LDBL_MAX_10_EXP > 326
        {{ SC_(0.5), static_cast<T>(std::ldexp(0.5, -683)), SC_(2.868703194735890254207338863894358862705699335892099e308) }},
#else
        {{ SC_(0.5), static_cast<T>(std::ldexp(0.5, -400)), SC_(4.6822269214637968690651040333526494618220547616350e180) }},
#endif
        {{ SC_(-0.5), static_cast<T>(std::ldexp(0.5, -683)), SC_(3.5741154998461284276309443770923823816821202344841e102) }},
        {{ SC_(0.0), static_cast<T>(std::ldexp(1.0, -53)), SC_(5.73416113922265864550047623401604244038331542638719289e15) }},
        {{ SC_(1.0), static_cast<T>(std::ldexp(1.0, -53)), SC_(5.164873193977108862252341626669725460073766e31) }},
        {{ SC_(2.0), static_cast<T>(std::ldexp(1.0, -53)), SC_(1.8608416793448936781963026443824482966468761e48) }},
        {{ SC_(3.0), static_cast<T>(std::ldexp(1.0, -53)), SC_(1.0056583072431781406772110820260315331263726e65) }},
        {{ SC_(10.0), static_cast<T>(std::ldexp(1.0, -53)), SC_(3.74455823365114672304576809031094538692683400e184) }},
    }};

    do_test_cyl_neumann_y_prime<T>(y0_prime_data, name, "Y'0: Mathworld Data");
    do_test_cyl_neumann_y_prime<T>(y1_prime_data, name, "Y'1: Mathworld Data");
    do_test_cyl_neumann_y_prime<T>(yn_prime_data, name, "Y'n: Mathworld Data");
    do_test_cyl_neumann_y_prime_int<T>(y0_prime_data, name, "Y'0: Mathworld Data (Integer Version)");
    do_test_cyl_neumann_y_prime_int<T>(y1_prime_data, name, "Y'1: Mathworld Data (Integer Version)");
    do_test_cyl_neumann_y_prime_int<T>(yn_prime_data, name, "Y'n: Mathworld Data (Integer Version)");
    do_test_cyl_neumann_y_prime<T>(yv_prime_data, name, "Y'v: Mathworld Data");
    if(yv_prime_large_data[0][1] != 0)
      do_test_cyl_neumann_y_prime<T>(yv_prime_large_data, name, "Y'v: Mathworld Data (large values)");

#include "bessel_y01_prime_data.ipp"
    do_test_cyl_neumann_y_prime<T>(bessel_y01_prime_data, name, "Y'0 and Y'1: Random Data");
#include "bessel_yn_prime_data.ipp"
    do_test_cyl_neumann_y_prime<T>(bessel_yn_prime_data, name, "Y'n: Random Data");
#include "bessel_yv_prime_data.ipp"
    do_test_cyl_neumann_y_prime<T>(bessel_yv_prime_data, name, "Y'v: Random Data");

#include "sph_neumann_prime_data.ipp"
    do_test_sph_neumann_y_prime<T>(sph_neumann_prime_data, name, "y': Random Data");

    //
    // More cases for full test coverage:
    //
    BOOST_CHECK_THROW(boost::math::cyl_neumann_prime(T(2.5), T(0)), std::domain_error);
    BOOST_CHECK_THROW(boost::math::cyl_neumann_prime(T(2.5), T(-1)), std::domain_error);
    BOOST_CHECK_THROW(boost::math::sph_neumann_prime(2, T(0)), std::domain_error);
    BOOST_CHECK_THROW(boost::math::sph_neumann_prime(2, T(-1)), std::domain_error);

    BOOST_IF_CONSTEXPR(std::numeric_limits<T>::has_infinity && (std::numeric_limits<T>::min_exponent < -1072))
    {
       static const std::array<std::array<T, 3>, 5> yv_prime_coverage_data = { {
          {{ SC_(170.25), SC_(2), SC_(4.1990285871978876642542582761856953686528755802132926772620e306) }},
    #if LDBL_MAX_10_EXP > 4936
          {{ SC_(14.25), ldexp(T(1), -1072), SC_(1.830622575805420777640505999291582751497710200210963689466e4936)}},
    #else
          {{ SC_(14.25), ldexp(T(1), -1072), std::numeric_limits<T>::infinity() }},
    #endif
          {{ SC_(14.25), ldexp(T(1), -1074), std::numeric_limits<T>::infinity() }},
          {{ SC_(15.25), ldexp(T(1), -1074), std::numeric_limits<T>::infinity() }},
          {{ SC_(30.25), ldexp(T(1), -1045), std::numeric_limits<T>::infinity() }},
       } };
       do_test_cyl_neumann_y_prime<T>(yv_prime_coverage_data, name, "y': Extra coverage data");
    }
   static const std::array<std::array<T, 3>, 1> sph_prime_coverage_data = { {
         // (SphericalBesselY[-1, 5/2] - (SphericalBesselY(0, 5/2)+5/2 * SphericalBesselY[1, 5/2])/(5/2))/2
      {{ SC_(0.0), SC_(2.5), SC_(0.1112058791540732032473814343996886423728680128280382077091151343) }},
   } };
   do_test_sph_neumann_y_prime<T>(sph_prime_coverage_data, name, "y': Extra coverage data");
}

