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

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#  define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class Real, class T>
void do_test_cyl_bessel_j_prime(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(BESSEL_JP_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type);
#ifdef BESSEL_JP_FUNCTION_TO_TEST
   pg funcp = BESSEL_JP_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::cyl_bessel_j_prime<value_type, value_type>;
#else
   pg funcp = boost::math::cyl_bessel_j_prime;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test cyl_bessel_j against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func<Real>(funcp, 0, 1), 
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "cyl_bessel_j_prime", test_name);
   std::cout << std::endl;
#endif
}

template <class T>
T cyl_bessel_j_prime_int_wrapper(T v, T x)
{
#ifdef BESSEL_JPN_FUNCTION_TO_TEST
   return static_cast<T>(BESSEL_JPN_FUNCTION_TO_TEST(boost::math::itrunc(v), x));
#else
   return static_cast<T>(boost::math::cyl_bessel_j_prime(boost::math::itrunc(v), x));
#endif
}


template <class Real, class T>
void do_test_cyl_bessel_j_prime_int(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(BESSEL_JPN_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type);
#if defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = cyl_bessel_j_prime_int_wrapper<value_type>;
#else
   pg funcp = cyl_bessel_j_prime_int_wrapper;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test cyl_bessel_j against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func<Real>(funcp, 0, 1), 
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "cyl_bessel_j_prime (integer orders)", test_name);
   std::cout << std::endl;
#endif
}

template <class Real, class T>
void do_test_sph_bessel_j_prime(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(BESSEL_JPS_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(unsigned, value_type);
#ifdef BESSEL_JPS_FUNCTION_TO_TEST
   pg funcp = BESSEL_JPS_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::sph_bessel_prime<value_type>;
#else
   pg funcp = boost::math::sph_bessel_prime;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test sph_bessel against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func_int1<Real>(funcp, 0, 1), 
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "sph_bessel_prime", test_name);
   std::cout << std::endl;
#endif
}

template <class T>
void test_bessel_prime(T, const char* name)
{
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // The contents are as follows, each row of data contains
   // three items, input value a, input value b and J'(a, b):
   // 
    // function values calculated on http://functions.wolfram.com/
    static const std::array<std::array<typename table_type<T>::type, 3>, 8> j0_data = {{
       {{ SC_(0.0), SC_(0.0), SC_(0.0) }},
        {{ SC_(0.0), SC_(1.0), SC_(-0.440050585744933515959682203718914913127) }},
        {{ SC_(0.0), SC_(-2.0), SC_(0.576724807756873387202448242269137086920) }},
        {{ SC_(0.0), SC_(4.0), SC_(0.06604332802354913614318542080327502873) }},
        {{ SC_(0.0), SC_(-8.0), SC_(0.2346363468539146243812766515904546115488) }},
        {{ SC_(0.0), SC_(1e-05), SC_(-0.499999999993750000000026041666666612413194e-5) }},
        {{ SC_(0.0), SC_(1e-10), SC_(-0.499999999999999999999375000000000000000000e-10) }},
        {{ SC_(0.0), SC_(-1e+01), SC_(0.0434727461688614366697487680258592883062724) }},
    }};
    static const std::array<std::array<T, 3>, 6> j0_tricky = {{
        // Big numbers make the accuracy of std::sin the limiting factor:
       {{ SC_(0.0), SC_(1e+03), SC_(-0.00472831190708952391757607190121691628542) }},
        {{ SC_(0.0), SC_(1e+05), SC_(-0.0018467575628825677163621239671142157437) }},
        // test at the regular Bessel roots:
        {{ SC_(0.0), T(2521642)/(1024 * 1024), SC_(-0.519147572225778564548541576612898453392794) }},
        {{ SC_(0.0), T(5788221)/(1024 * 1024), SC_(0.34026483151709114336072749629487476476084) }},
        {{ SC_(0.0), T(9074091)/(1024 * 1024), SC_(-0.271452311894657014854145327490965399410) }},
        {{ SC_(0.0), T(12364320)/(1024 * 1024), SC_(0.2324598316641066033541448467171088144257742) }}
    }};    

    static const std::array<std::array<typename table_type<T>::type, 3>, 8> j1_data = {{
        {{ SC_(1.0), SC_(0.0), SC_(0.5) }},
        {{ SC_(1.0), SC_(1.0), SC_(0.325147100813033035490035322383748307781902) }},
        {{ SC_(1.0), SC_(-2.0), SC_(-0.064471624737201025549396666484619917634997) }},
        {{ SC_(1.0), SC_(4.0), SC_(-0.38063897785796008825079441325087928479376) }},
        {{ SC_(1.0), SC_(-8.0), SC_(0.1423212637808145780432098264031651746248) }},
        {{ SC_(1.0), SC_(1e-05), SC_(0.499999999981250000000130208333332953559028) }},
        {{ SC_(1.0), SC_(1e-10), SC_(0.499999999999999999998125000000000000000001) }},
        {{ SC_(1.0), SC_(-1e+01), SC_(-0.250283039068234478864735739287914682660226) }},
    }};
    static const std::array<std::array<T, 3>, 5> j1_tricky = {{
        // Big numbers make the accuracy of std::sin the limiting factor:
        {{ SC_(1.0), SC_(1e+03), SC_(0.024781957840513085037413155043792491869881) }},
        {{ SC_(1.0), SC_(1e+05), SC_(-0.0017192195838116010182477650983128728897) }},
        // test at the regular Bessel roots:
        // calculated as (BesselJ[0, (4017834)/(1024*1024)] - BesselJ[2, (4017834)/(1024*1024)]) / 2 etc...
        {{ SC_(1.0), T(4017834)/(1024*1024), SC_(-0.402759487867380763351480272317936413309077788308634756237208644) }},
        {{ SC_(1.0), T(7356375)/(1024*1024), SC_(0.3001157854852255749972773530255671995928674390130509220099020091) }},
        {{ SC_(1.0), T(10667654)/(1024*1024), SC_(-0.249704889304504742750238623220064969087694831235737808091251358) }},
    }};

    static const std::array<std::array<typename table_type<T>::type, 3>, 17> jn_data = {{
        {{ SC_(-1.0), SC_(1.25), SC_(-0.237407477015380891294270728091739255408276776418956604373517363) }},
        {{ SC_(2.0), SC_(0.0), SC_(0.0) }},
        {{ SC_(-2.0), SC_(0.0), SC_(0.0) }},
        {{ SC_(2.0), SC_(1e-02), SC_(0.0024999583335286453993061206954067979960633680470591696456970138) }},
        {{ SC_(5.0), SC_(10.0), SC_(-0.102571922008611714904101858221407144485053455162064084296429679) }},
        {{ SC_(5.0), SC_(-10.0), SC_(-0.102571922008611714904101858221407144485053455162064084296429679) }},
        {{ SC_(-5.0), SC_(1e+06), SC_(-0.000331052451300704410558585953452327198846151723732312153110559) }},
        {{ SC_(5.0), SC_(1e+06), SC_(0.000331052451300704410558585953452327198846151723732312153110559) }},
        {{ SC_(-5.0), SC_(-1.0), SC_(-0.001227850313053782886909720690402218190791576453229129020988612) }},
        {{ SC_(10.0), SC_(10.0), SC_(0.0843695786317611882484905127333769830416507936391564732626938971) }},
        {{ SC_(10.0), SC_(-10.0), SC_(-0.0843695786317611882484905127333769830416507936391564732626938971) }},
        {{ SC_(10.0), SC_(-5.0), SC_(-0.002584677844854739252065488546761161575681058069442826487392210) }},
        {{ SC_(-10.0), SC_(1e+06), SC_(-0.000725951803719324335038787573389363596244265216142949209996916) }},
        {{ SC_(10.0), SC_(1e+06), SC_(-0.000725951803719324335038787573389363596244265216142949209996916) }},
        {{ SC_(100.0), SC_(80.0), SC_(3.5036060582489177538508950593467499997755458120649269214352e-6) }},
        {{ SC_(1000.0), SC_(100000.0), SC_(-0.002172446977760839340985075822746577648557209118811950682915414) }},
        {{ SC_(10.0), SC_(1e-100), SC_(2.691144455467372134038800705467372134038800705467372134038e-909) }},
    }};
    do_test_cyl_bessel_j_prime<T>(j0_data, name, "Bessel J0': Mathworld Data");
    do_test_cyl_bessel_j_prime<T>(j0_tricky, name, "Bessel J0': Mathworld Data (Tricky cases)");
    do_test_cyl_bessel_j_prime<T>(j1_data, name, "Bessel J1': Mathworld Data");
    do_test_cyl_bessel_j_prime<T>(j1_tricky, name, "Bessel J1': Mathworld Data (tricky cases)");
    do_test_cyl_bessel_j_prime<T>(jn_data, name, "Bessel JN': Mathworld Data");

    do_test_cyl_bessel_j_prime_int<T>(j0_data, name, "Bessel J0': Mathworld Data (Integer Version)");
    do_test_cyl_bessel_j_prime_int<T>(j0_tricky, name, "Bessel J0': Mathworld Data (Tricky cases) (Integer Version)");
    do_test_cyl_bessel_j_prime_int<T>(j1_data, name, "Bessel J1': Mathworld Data (Integer Version)");
    do_test_cyl_bessel_j_prime_int<T>(j1_tricky, name, "Bessel J1': Mathworld Data (tricky cases) (Integer Version)");
    do_test_cyl_bessel_j_prime_int<T>(jn_data, name, "Bessel JN': Mathworld Data (Integer Version)");

    static const std::array<std::array<T, 3>, 21> jv_data = {{
         {{ T(22.5), T(0), SC_(0.0) }},
         {{ T(2457)/1024, T(1)/1024, SC_(9.35477929043111040277363766198320562099360690e-6) }},
         {{ SC_(5.5), T(3217)/1024, SC_(0.042165579369684463582791278988393873) }},
         {{ SC_(-5.5), T(3217)/1024, SC_(3.361570113176257957139775812778503494) }},
         {{ SC_(-5.5), SC_(1e+04), SC_(0.007593311396019034252155600098309836289) }},
         {{ SC_(5.5), SC_(1e+04), SC_(-0.00245022241637437956702428797044365092) }},
         {{ SC_(5.5), SC_(1e+06), SC_(-0.000279242826717266554062248256927185394) }},
         {{ SC_(5.125), SC_(1e+06), SC_(0.0001830632695189459708211614700642271) }},
         {{ SC_(5.875), SC_(1e+06), SC_(-0.0006474276718101871487286860109203539) }},
         {{ SC_(0.5), SC_(101.0), SC_(0.070640819172197226936337703929857171981702865) }},
         {{ SC_(-5.5), SC_(1e+04), SC_(0.007593311396019034252155600098309836289) }},
         {{ SC_(-5.5), SC_(1e+06), SC_(-0.0007474243882060190346457525218941411076) }},
         {{ SC_(-0.5), SC_(101.0), SC_(-0.036238035321276062532981494694583591262302408) }},
         {{ T(-10486074) / (1024*1024), T(1)/512, SC_(-7.0724447469115535625316241941528969321944e35) }},
         {{ T(-10486074) / (1024*1024), SC_(15.0), SC_(-0.15994088796049823354364759206656917967697690) }},
         {{ T(10486074) / (1024*1024), SC_(1e+02), SC_(-0.05778764167290516644655950658602424434253) }},
         {{ T(10486074) / (1024*1024), SC_(2e+04), SC_(-0.00091101010794789360775314125410690740803) }},
         {{ T(-10486074) / (1024*1024), SC_(1e+02), SC_(-0.057736130385111563671838499496767877709471701) }},
         {{ SC_(1.5), T(8034)/1024, SC_(0.2783550354042687982259490073096357) }},
         {{ SC_(8.5), boost::math::constants::pi<T>() * 4, SC_(-0.194590144622675911618596506265006877277074) }},
         {{ SC_(-8.5), boost::math::constants::pi<T>() * 4, SC_(-0.014516314554743677558496402742690038592728) }},
    }};
    do_test_cyl_bessel_j_prime<T>(jv_data, name, "Bessel J': Mathworld Data");
    static const std::array<std::array<T, 3>, 4> jv_large_data = {{
#if LDBL_MAX_10_EXP > 308
      {{ SC_(-0.5), static_cast<T>(std::ldexp(0.5, -683)), SC_(-2.8687031947358902542073388638943588627056993e308) }},
#else
      {{ SC_(-0.5), static_cast<T>(std::ldexp(0.5, -450)), SC_(-1.7688953183288445554095310240218576026580197125814e203) }},
#endif
      {{ SC_(256.0), SC_(512.0), SC_(0.032286467266411904239327492993951594201583145) }},
      {{ SC_(-256.0), SC_(8.0), SC_(4.6974301387555891979202431551474684165419e-352) }},
      {{ SC_(-2.5), SC_(4.0), SC_(-0.3580070651681080294136741901878543615958139) }},
    }};
    if(jv_large_data[0][1] != 0)
      do_test_cyl_bessel_j_prime<T>(jv_large_data, name, "Bessel J': Mathworld Data (large values)");

#include "bessel_j_prime_int_data.ipp"
    do_test_cyl_bessel_j_prime<T>(bessel_j_prime_int_data, name, "Bessel JN': Random Data");

#include "bessel_j_prime_data.ipp"
    do_test_cyl_bessel_j_prime<T>(bessel_j_prime_data, name, "Bessel J': Random Data");

#include "bessel_j_prime_large_data.ipp"
    do_test_cyl_bessel_j_prime<T>(bessel_j_prime_large_data, name, "Bessel J': Random Data (Tricky large values)");

#include "sph_bessel_prime_data.ipp"
    do_test_sph_bessel_j_prime<T>(sph_bessel_prime_data, name, "Bessel j': Random Data");

    //
    // Some special cases:
    //
    BOOST_CHECK_EQUAL(boost::math::cyl_bessel_j_prime(T(1), T(0)), T(0.5));
    BOOST_CHECK_EQUAL(boost::math::cyl_bessel_j_prime(T(-1), T(0)), T(-0.5));
    BOOST_CHECK_EQUAL(boost::math::cyl_bessel_j_prime(T(2), T(0)), T(0));

    //
    // Special cases that are errors:
    //
    BOOST_MATH_CHECK_THROW(boost::math::sph_bessel_prime(1, T(0)), std::domain_error);
    BOOST_MATH_CHECK_THROW(boost::math::sph_bessel_prime(100000, T(0)), std::domain_error);
    BOOST_MATH_CHECK_THROW(boost::math::cyl_bessel_j_prime(T(-2.5), T(0)), std::domain_error);
    BOOST_MATH_CHECK_THROW(boost::math::cyl_bessel_j_prime(T(-2.5), T(-2)), std::domain_error);
    BOOST_MATH_CHECK_THROW(boost::math::cyl_bessel_j_prime(T(2.5), T(-2)), std::domain_error);
    BOOST_MATH_CHECK_THROW(boost::math::sph_bessel_prime(2, T(-2)), std::domain_error);
}

