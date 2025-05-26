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
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#  define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class T>
T cyl_bessel_i_prime_int_wrapper(T v, T x)
{
#ifdef BESSEL_IPN_FUNCTION_TO_TEST
   return static_cast<T>(
      BESSEL_IPN_FUNCTION_TO_TEST(
      boost::math::itrunc(v), x));
#else
   return static_cast<T>(
      boost::math::cyl_bessel_i_prime(
      boost::math::itrunc(v), x));
#endif
}

template <class Real, class T>
void do_test_cyl_bessel_i_prime(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(BESSEL_IP_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type);
#ifdef BESSEL_IP_FUNCTION_TO_TEST
   pg funcp = BESSEL_IP_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::cyl_bessel_i_prime<value_type, value_type>;
#else
   pg funcp = boost::math::cyl_bessel_i_prime;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test cyl_bessel_i_prime against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func<Real>(funcp, 0, 1), 
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "cyl_bessel_i_prime", test_name);
   std::cout << std::endl;
#endif
}

template <class Real, class T>
void do_test_cyl_bessel_i_prime_int(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(BESSEL_IPN_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type);
#if defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = cyl_bessel_i_prime_int_wrapper<value_type>;
#else
   pg funcp = cyl_bessel_i_prime_int_wrapper;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test cyl_bessel_i_prime against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func<Real>(funcp, 0, 1), 
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "cyl_bessel_i_prime (integer orders)", test_name);
   std::cout << std::endl;
#endif
}

template <class T>
void test_bessel(T, const char* name)
{
    BOOST_MATH_STD_USING
    // function values calculated on wolframalpha.com
    static const std::array<std::array<T, 3>, 10> i0_prime_data = {{
        {{ SC_(0.0), SC_(0.0), SC_(0.0) }},
        {{ SC_(0.0), SC_(1.0), SC_(0.565159103992485027207696027609863307328899621621) }},
        {{ SC_(0.0), SC_(-2.0), SC_(-1.590636854637329063382254424999666247954478159496) }},
        {{ SC_(0.0), SC_(4.0), SC_(9.75946515370444990947519256731268090005597033325) }},
        {{ SC_(0.0), SC_(-7.0), SC_(-156.039092869955453462390580660711155630031052042) }},
        {{ SC_(0.0), T(1) / 1024, SC_(0.000488281308207663226432087816784315537514225208473395) }},
        {{ SC_(0.0), T(SC_(1.0)) / (1024*1024), SC_(4.76837158203179210108624277276025646653133998635957e-7) }},
        {{ SC_(0.0), SC_(-1.0), SC_(-0.565159103992485027207696027609863307328899621621) }},
        {{ SC_(0.0), SC_(100.0), SC_(1.068369390338162481206145763224295265446122844056e42) }},
        {{ SC_(0.0), SC_(200.0), SC_(2.034581549332062703427427977139069503896611616811e85) }},
    }};
    static const std::array<std::array<T, 3>, 10> i1_prime_data = {{
        {{ SC_(1.0), SC_(0.0), SC_(0.5) }},
        {{ SC_(1.0), SC_(1.0), SC_(0.700906773759523308390548597604854230278770689734) }},
        {{ SC_(1.0), SC_(-2.0), SC_(1.484266875017402735746077228311700229308602023038) }},
        {{ SC_(1.0), SC_(4.0), SC_(8.86205566371021801898747204138893227239862401112) }},
        {{ SC_(1.0), SC_(-8.0), SC_(377.579973623984772900011405549855040764360549303) }},
        {{ SC_(1.0), T(SC_(1.0))/1024, SC_(0.500000178813946168551133736709856567600996119560422) }},
        {{ SC_(1.0), T(SC_(1.0))/(1024*1024), SC_(0.500000000000170530256582434815189962442052310320626) }},
        {{ SC_(1.0), SC_(-10.0), SC_(2548.6177980961290060357079567493748381638767230173) }},
        {{ SC_(1.0), SC_(100.0), SC_(1.063068013227692198707659399971251708633941964885e42) }},
        {{ SC_(1.0), SC_(200.0), SC_(2.029514265663064306024535986893764274813192678064e85) }},
    }};
    static const std::array<std::array<T, 3>, 11> in_prime_data = {{
        {{ SC_(-2.0), SC_(0.0), SC_(0.0) }},
        {{ SC_(2.0), T(SC_(1.0))/(1024*1024), SC_(2.38418579101598640072416185021877537269848820467704e-7) }},
        {{ SC_(5.0), SC_(10.0), SC_(837.8963945578616177877239800250165373153505679130) }},
        {{ SC_(-5.0), SC_(100.0), SC_(9.434574089052212641696648538570565739509851953166e41) }},
        {{ SC_(-5.0), SC_(-1.0), SC_(0.001379804441262006949232714689824343061461150959112) }},
        {{ SC_(10.0), SC_(20.0), SC_(3.887291476282816593964936516887942731146400030466e6) }},
        {{ SC_(10.0), SC_(-5.0), SC_(-0.0101556299784624552653083645193223022003797137770) }},
        {{ SC_(1e+02), SC_(9.0), SC_(3.0600487816519872979909028718737622834061708443e-92) }},
        {{ SC_(1e+02), SC_(80.0), SC_(7.43545006374466980237328068214707549314834217923e8) }},
        {{ SC_(-100.0), SC_(-200.0), SC_(-4.8578174816088978115191937982677942557681326349558e74) }},
        {{ SC_(10.0), SC_(1e-100), SC_(2.6911444554673721340388007054673721340e-909) }},
    }};
    static const std::array<std::array<T, 3>, 10> iv_prime_data = {{
        {{ SC_(2.25), T(1)/(1024*1024), SC_(5.5296993766970839641084373853875345330202883e-9) }},
        {{ SC_(5.5), SC_(3.125), SC_(0.11607917746126037030394881599790144677553715606) }},
        {{ T(-5) + T(1)/1024, SC_(2.125), SC_(0.0001131925559871199041270456478317398625046249903864372470342210384462922281) }},
        {{ SC_(-5.5), SC_(10.0), SC_(659.1902595786901927596924259811320437384361101) }},
        {{ SC_(-5.5), SC_(100.0), SC_(9.191476042191556775282339209385028823905941708e41) }},
        {{ T(-10486074)/(1024*1024), T(1)/1024, SC_(-1.44873720736417608945635957884937466861026978539e39) }},
        {{ T(-10486074)/(1024*1024), SC_(50.0), SC_(1.082410021443599516897183930739816215073642812109e20) }},
        {{ T(144794)/1024, SC_(100.0), SC_(3575.11008553328328036816705258135747714241715202) }},
        {{ T(144794)/1024, SC_(200.0), SC_(2.7358895637426974377937620224627094172800852276956e64) }},
        {{ T(-144794)/1024, SC_(100.0), SC_(3575.11008700037933897402396449269857968451879323) }},
    }};
    static const std::array<std::array<T, 3>, 5> iv_prime_large_data = {{
        {{ SC_(-1.0), static_cast<T>(ldexp(0.5, -512)), SC_(0.5) }},
        {{ SC_(1.0),  static_cast<T>(ldexp(0.5, -512)), SC_(0.5) }},
        {{ SC_(1.125),  static_cast<T>(ldexp(0.5, -512)), SC_(2.42025162605150606399395900489934587657244145536315936432966315563638e-20) }},
        {{ SC_(0.5), static_cast<T>(ldexp(0.5, -683)), SC_(3.5741154998461284276309443770923823816821202344841143399486401387635e102) }},
#if LDBL_MAX_10_EXP > 326
        {{ SC_(-1.125), static_cast<T>(ldexp(0.5, -512)), SC_(4.0715272050947359203430409041001937149343363573066460226173390878707e327) }},
#else
        { { SC_(-1.125), static_cast<T>(ldexp(0.5, -512)), std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : boost::math::tools::max_value<T>() } },
#endif
    }};

    do_test_cyl_bessel_i_prime<T>(i0_prime_data, name, "Bessel I'0: Mathworld Data");
    do_test_cyl_bessel_i_prime<T>(i1_prime_data, name, "Bessel I'1: Mathworld Data");
    do_test_cyl_bessel_i_prime<T>(in_prime_data, name, "Bessel I'n: Mathworld Data");

    do_test_cyl_bessel_i_prime_int<T>(i0_prime_data, name, "Bessel I'0: Mathworld Data (Integer Version)");
    do_test_cyl_bessel_i_prime_int<T>(i1_prime_data, name, "Bessel I'1: Mathworld Data (Integer Version)");
    do_test_cyl_bessel_i_prime_int<T>(in_prime_data, name, "Bessel I'n: Mathworld Data (Integer Version)");

    do_test_cyl_bessel_i_prime<T>(iv_prime_data, name, "Bessel I'v: Mathworld Data");

#include "bessel_i_prime_int_data.ipp"
    do_test_cyl_bessel_i_prime<T>(bessel_i_prime_int_data, name, "Bessel I'n: Random Data");
#include "bessel_i_prime_data.ipp"
    do_test_cyl_bessel_i_prime<T>(bessel_i_prime_data, name, "Bessel I'v: Random Data");

    if(0 != static_cast<T>(ldexp(static_cast<T>(0.5), -700)))
      do_test_cyl_bessel_i_prime<T>(iv_prime_large_data, name, "Bessel I'v: Mathworld Data (large values)");

    //
    // Special cases for extra coverage:
    //
    BOOST_CHECK_THROW(boost::math::cyl_bessel_i_prime(T(2.5), T(-1)), std::domain_error);
    BOOST_CHECK_THROW(boost::math::cyl_bessel_i_prime(T(0.25), T(0)), std::domain_error);
}

