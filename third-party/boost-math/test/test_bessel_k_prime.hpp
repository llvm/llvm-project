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
T cyl_bessel_k_prime_int_wrapper(T v, T x)
{
#ifdef BESSEL_KPN_FUNCTION_TO_TEST
   return static_cast<T>(
      BESSEL_KPN_FUNCTION_TO_TEST(
      boost::math::itrunc(v), x));
#else
   return static_cast<T>(
      boost::math::cyl_bessel_k_prime(
      boost::math::itrunc(v), x));
#endif
}

template <class Real, class T>
void do_test_cyl_bessel_k_prime(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(BESSEL_KP_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type);
#ifdef BESSEL_KP_FUNCTION_TO_TEST
   pg funcp = BESSEL_KP_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::cyl_bessel_k_prime<value_type, value_type>;
#else
   pg funcp = boost::math::cyl_bessel_k_prime;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test cyl_bessel_k_prime against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func<Real>(funcp, 0, 1), 
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "cyl_bessel_k_prime", test_name);
   std::cout << std::endl;
#endif
}

template <class Real, class T>
void do_test_cyl_bessel_k_prime_int(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(BESSEL_KPN_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type, value_type);
#if defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = cyl_bessel_k_prime_int_wrapper<value_type>;
#else
   pg funcp = cyl_bessel_k_prime_int_wrapper;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test cyl_bessel_k_prime against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func<Real>(funcp, 0, 1), 
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "cyl_bessel_k_prime (integer orders)", test_name);
   std::cout << std::endl;
#endif
}

template <class T>
void test_bessel(T, const char* name)
{
    // function values calculated on wolframalpha.com
    static const std::array<std::array<T, 3>, 9> k0_prime_data = {{
        {{ SC_(0.0), SC_(1.0), SC_(-0.60190723019723457473754000153561733926158688996810646) }},
        {{ SC_(0.0), SC_(2.0), SC_(-0.1398658818165224272845988070354110238872345848415155) }},
        {{ SC_(0.0), SC_(4.0), SC_(-0.012483498887268431470384179980806068483841584988625846) }},
        {{ SC_(0.0), SC_(8.0), SC_(-0.00015536921180500113391686245062247462111706512287261616) }},
        {{ SC_(0.0), T(std::ldexp(1.0, -15)), SC_(-32767.99983195283164326474413165391397251047283415776) }},
        {{ SC_(0.0), T(std::ldexp(1.0, -30)), SC_(-1.0737418239999999900300302857268733281035379954421507e9) }},
        {{ SC_(0.0), T(std::ldexp(1.0, -60)), SC_(-1.1529215046068469759999999999999999816966019886812660e18) }},
        {{ SC_(0.0), SC_(50.0), SC_(-3.44410222671755561259185303591267155099677251348256880e-23) }},
        {{ SC_(0.0), SC_(100.0), SC_(-4.6798537356369092865625442420243353079749435469433535e-45) }},
    }};
    static const std::array<std::array<T, 3>, 9> k1_prime_data = {{
        {{ SC_(1.0), SC_(1.0), SC_(-1.0229316684379429080731673807482263753978066381947669) }},
        {{ SC_(1.0), SC_(2.0), SC_(-0.1838268136577946492950189784501873449419439168095666) }},
        {{ SC_(1.0), SC_(4.0), SC_(-0.014280550807670132137341240975035006345969420135630802) }},
        {{ SC_(1.0), SC_(8.0), SC_(-0.00016589185669844052883619221502648724960693850919283604) }},
        {{ SC_(1.0), T(std::ldexp(1.0, -15)), SC_(-1.0737418290065696140247028419519880092107054138744140e9) }},
        {{ SC_(1.0), T(std::ldexp(1.0, -30)), SC_(-1.1529215046068469862051734662283858692135761720165778e18) }},
        {{ SC_(1.0), T(std::ldexp(1.0, -60)), SC_(-1.329227995784915872903807060280344596602381174627566e36) }},
        {{ SC_(1.0), SC_(50.0), SC_(-3.47904979432384662617251257307120566286496082789299947e-23) }},
        {{ SC_(1.0), SC_(100.0), SC_(-4.7034267665322711118046307319041297088872889209115474e-45) }},
    }};
    static const std::array<std::array<T, 3>, 9> kn_prime_data = {{
        {{ SC_(2.0), T(std::ldexp(1.0, -30)), SC_(-4.951760157141521099596496895999999995073222803776904e27) }},
        {{ SC_(5.0), SC_(10.0), SC_(-0.0000666323621535481236223011866087784024278980735437002384) }},
        {{ SC_(-5.0), SC_(100.0), SC_(-5.3060798744208349930861060378887340340201141387578377e-45) }},
        {{ SC_(10.0), SC_(10.0), SC_(-0.00232426413420145080508626300083871228780582972491498296) }},
        {{ SC_(10.0), T(std::ldexp(1.0, -30)), SC_(-4.0637928602074079595570948641288439020852370470244381e108) }},
        {{ SC_(-10.0), SC_(1.0), SC_(-1.8171379399979651461891429013401068319174853467388121e9) }},
        {{ SC_(100.0), SC_(5.0), SC_(-1.4097486373570936520327835736048715219413065916411893e117) }},
        {{ SC_(100.0), SC_(80.0), SC_(-1.34557011017664184003144916855685180771861680634827508e-11) }},
        {{ SC_(-129.0), SC_(200.0), SC_(-4.3110345255133348027545113739271337415489367194240537230455182e-71) }},
    }};
    static const std::array<std::array<T, 3>, 11> kv_prime_data = {{
        {{ SC_(0.5), SC_(0.875), SC_(-0.8776935068732421581818610624499915196588910540138553643355820) }},
        {{ SC_(0.5), SC_(1.125), SC_(-0.5541192376058293458786667962590089848709748151724170966916495) }},
        {{ SC_(2.25), T(std::ldexp(1.0, -30)), SC_(-1.358706605110306964608847299464328015299661532e30) }},
        {{ SC_(5.5), T(3217)/1024, SC_(-2.6903757178739422729800670428157504611799055394319992629519699) }},
        {{ SC_(-5.5), SC_(10.0), SC_(-0.000086479759593318257340087317655128751755482676477180134416784728) }},
        {{ SC_(-5.5), SC_(100.0), SC_(-5.4478425565190604625309457442097587701859746312164196732075323e-45) }},
        {{ T(10240)/1024, T(1)/1024, SC_(-2.411751224440479729811903506282248205762559999997965494837863222e42) }},
        {{ T(10240)/1024, SC_(10.0), SC_(-0.002324264134201450805086263000838712287805829724914982961118625775) }},
        {{ T(144793)/1024, SC_(100.0), SC_(-2.419425330672365273534646536102117722944744737761477017402710069e-6) }},
        {{ T(144793)/1024, SC_(200.0), SC_(-1.1183699286601178683373775100500418982738064865504029155187086e-67) }},
        {{ T(-144793)/1024, SC_(50.0), SC_(-3.906473504308773541933992099338237076647113693807893258840087e42) }},
    }};
    static const std::array<std::array<T, 3>, 5> kv_prime_large_data = {{
        {{ SC_(-0.5), static_cast<T>(ldexp(0.5, -512)), SC_(-2.75176667129887692508287667455879592490037256500173136025362e231) }},
        {{ SC_(0.5),  static_cast<T>(ldexp(0.5, -512)), SC_(-2.75176667129887692508287667455879592490037256500173136025362e231) }},
#if LDBL_MAX_10_EXP > 328
        {{ SC_(-1.125), static_cast<T>(ldexp(0.5, -512)), SC_(-1.67123513518264734700327664054002130440723e328) }},
        {{ SC_(1.125),  static_cast<T>(ldexp(0.5, -512)), SC_(-1.67123513518264734700327664054002130440723e328) }},
        {{ SC_(0.5),  static_cast<T>(ldexp(0.5, -683)), SC_(-4.5061484409559214227217449664854025793393e308) }},
#else
        { { SC_(-1.125), static_cast<T>(ldexp(0.5, -512)), std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : -boost::math::tools::max_value<T>() } },
        { { SC_(1.125), static_cast<T>(ldexp(0.5, -512)), std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : -boost::math::tools::max_value<T>() } },
        { { SC_(0.5), static_cast<T>(ldexp(0.5, -683)), std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : -boost::math::tools::max_value<T>() } },
#endif
    }};

    do_test_cyl_bessel_k_prime<T>(k0_prime_data, name, "Bessel K'0: Mathworld Data");
    do_test_cyl_bessel_k_prime<T>(k1_prime_data, name, "Bessel K'1: Mathworld Data");
    do_test_cyl_bessel_k_prime<T>(kn_prime_data, name, "Bessel K'n: Mathworld Data");

    do_test_cyl_bessel_k_prime_int<T>(k0_prime_data, name, "Bessel K'0: Mathworld Data (Integer Version)");
    do_test_cyl_bessel_k_prime_int<T>(k1_prime_data, name, "Bessel K'1: Mathworld Data (Integer Version)");
    do_test_cyl_bessel_k_prime_int<T>(kn_prime_data, name, "Bessel K'n: Mathworld Data (Integer Version)");

    do_test_cyl_bessel_k_prime<T>(kv_prime_data, name, "Bessel K'v: Mathworld Data");
    if(0 != static_cast<T>(ldexp(0.5, -512)))
      do_test_cyl_bessel_k_prime<T>(kv_prime_large_data, name, "Bessel K'v: Mathworld Data (large values)");
#include "bessel_k_prime_int_data.ipp"
    do_test_cyl_bessel_k_prime<T>(bessel_k_prime_int_data, name, "Bessel K'n: Random Data");
#include "bessel_k_prime_data.ipp"
    do_test_cyl_bessel_k_prime<T>(bessel_k_prime_data, name, "Bessel K'v: Random Data");
    //
    // Extra cases for full test coverage:
    //
    BOOST_CHECK_THROW(boost::math::cyl_bessel_k_prime(T(2.5), T(0)), std::domain_error);
    BOOST_CHECK_THROW(boost::math::cyl_bessel_k_prime(T(2), T(0)), std::domain_error);

    BOOST_IF_CONSTEXPR(std::numeric_limits<T>::has_infinity && std::numeric_limits<T>::min_exponent < -860)
    {
       static const std::array<std::array<T, 3>, 8> coverage_data = { {
#if LDBL_MAX_10_EXP > 310
           {{ SC_(20.0), static_cast<T>(ldexp(T(1), -45)), SC_(-3.793503044583520787322174911740831752794438746336004555076e308) }},
           {{ SC_(20.125), static_cast<T>(ldexp(T(1), -45)), SC_(-2.979220621533376610700938325572770408419207521624698386062e310) }},
#else
           {{ SC_(20.0), static_cast<T>(ldexp(T(1), -45)), -std::numeric_limits<T>::infinity() }},
           {{ SC_(20.125), static_cast<T>(ldexp(T(1), -45)), -std::numeric_limits<T>::infinity() }},
#endif
#if LDBL_MAX_10_EXP > 346
           {{ SC_(20.0), static_cast<T>(ldexp(T(1), -51)), SC_(-3.227155487331667007856383940813742118802894409545345203104e346) }},
           {{ SC_(20.125), static_cast<T>(ldexp(T(1), -51)), SC_(-4.262404050083097364466577035647085801041781477814968803189e348) }},
#else
           {{ SC_(20.0), static_cast<T>(ldexp(T(1), -51)), -std::numeric_limits<T>::infinity() }},
           {{ SC_(20.125), static_cast<T>(ldexp(T(1), -51)), -std::numeric_limits<T>::infinity() }},
#endif
#if LDBL_MAX_10_EXP > 4971
           {{ SC_(20.0), static_cast<T>(ldexp(T(1), -778)), SC_(-2.15657066125095338369788943003323297569772178814715617602e4942) }},
           {{ SC_(20.125), static_cast<T>(ldexp(T(1), -778)), SC_(-6.46694658438021098575183049117626387183087801364084017400e4971) }},
#else
           {{ SC_(20.0), static_cast<T>(ldexp(T(1), -778)), -std::numeric_limits<T>::infinity() }},
           {{ SC_(20.125), static_cast<T>(ldexp(T(1), -778)), -std::numeric_limits<T>::infinity() }},
#endif
#if LDBL_MAX_10_EXP > 5493
           {{ SC_(20.0), static_cast<T>(ldexp(T(1), -860)), SC_(-5.09819245599453059425108127687500966644642217657888061634e5460) }},
           {{ SC_(20.125), static_cast<T>(ldexp(T(1), -860)), SC_(-1.86169813082884487070647108868007382252541071831005047607e5493) }},
#else
           {{ SC_(20.0), static_cast<T>(ldexp(T(1), -860)), -std::numeric_limits<T>::infinity() }},
           {{ SC_(20.125), static_cast<T>(ldexp(T(1), -860)), -std::numeric_limits<T>::infinity() }},
#endif
       } };
       do_test_cyl_bessel_k_prime<T>(coverage_data, name, "Bessel K': Extra Coverage");
    }

}

