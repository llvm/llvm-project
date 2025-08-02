/*
 * Copyright Evan Miller, 2020
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#define BOOST_TEST_MAIN
#define NOMINMAX
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/ellint_rf.hpp>
#include <boost/math/special_functions/jacobi_elliptic.hpp>
#include <boost/math/special_functions/jacobi_theta.hpp>
#include <boost/math/special_functions/zeta.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class Real, class T>
void do_test_jacobi_theta1(const T& data, const char* type_name, const char* test_name) {
   typedef Real                   value_type;
   typedef value_type (*pg)(value_type, value_type);
   std::cout << "Testing: " << test_name << std::endl;
#ifdef JACOBI_THETA1_FUNCTION_TO_TEST
   pg fp2 = JACOBI_THETA1_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg fp2 = boost::math::jacobi_theta1<value_type, value_type>;
#else
   pg fp2 = boost::math::jacobi_theta1;
#endif
   boost::math::tools::test_result<value_type> result;
   result = boost::math::tools::test_hetero<Real>(
           data,
           bind_func<Real>(fp2, 0, 1),
           extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(),
           type_name, "jacobi_theta1", test_name);

   std::cout << std::endl;
}

template <class Real, class T>
void do_test_jacobi_theta2(const T& data, const char* type_name, const char* test_name) {
   typedef Real                   value_type;
   typedef value_type (*pg)(value_type, value_type);
   std::cout << "Testing: " << test_name << std::endl;
#ifdef JACOBI_THETA2_FUNCTION_TO_TEST
   pg fp2 = JACOBI_THETA2_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg fp2 = boost::math::jacobi_theta2<value_type, value_type>;
#else
   pg fp2 = boost::math::jacobi_theta2;
#endif
   boost::math::tools::test_result<value_type> result;
   result = boost::math::tools::test_hetero<Real>(
           data,
           bind_func<Real>(fp2, 0, 1),
           extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(),
           type_name, "jacobi_theta2", test_name);

   std::cout << std::endl;
}

template <class Real, class T>
void do_test_jacobi_theta3(const T& data, const char* type_name, const char* test_name) {
   typedef Real                   value_type;
   typedef value_type (*pg)(value_type, value_type);
   std::cout << "Testing: " << test_name << std::endl;
#ifdef JACOBI_THETA3_FUNCTION_TO_TEST
   pg fp2 = JACOBI_THETA3_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg fp2 = boost::math::jacobi_theta3<value_type, value_type>;
#else
   pg fp2 = boost::math::jacobi_theta3;
#endif
   boost::math::tools::test_result<value_type> result;
   result = boost::math::tools::test_hetero<Real>(
           data,
           bind_func<Real>(fp2, 0, 1),
           extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(),
           type_name, "jacobi_theta3", test_name);

   std::cout << std::endl;
}

template <class Real, class T>
void do_test_jacobi_theta4(const T& data, const char* type_name, const char* test_name) {
   typedef Real                   value_type;
   typedef value_type (*pg)(value_type, value_type);
   std::cout << "Testing: " << test_name << std::endl;
#ifdef JACOBI_THETA4_FUNCTION_TO_TEST
   pg fp2 = JACOBI_THETA4_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg fp2 = boost::math::jacobi_theta4<value_type, value_type>;
#else
   pg fp2 = boost::math::jacobi_theta4;
#endif
   boost::math::tools::test_result<value_type> result;
   result = boost::math::tools::test_hetero<Real>(
           data,
           bind_func<Real>(fp2, 0, 1),
           extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(),
           type_name, "jacobi_theta4", test_name);

   std::cout << std::endl;
}

template <class Real, class T>
void do_test_jacobi_theta_tau(const T& data, const char* type_name, const char* test_name) {
   typedef Real                   value_type;
   typedef value_type (*pg)(value_type, value_type);
   std::cout << "Testing: " << test_name << std::endl;
#if defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg fp1 = boost::math::jacobi_theta1tau<value_type, value_type>;
   pg fp2 = boost::math::jacobi_theta2tau<value_type, value_type>;
   pg fp3 = boost::math::jacobi_theta3tau<value_type, value_type>;
   pg fp4 = boost::math::jacobi_theta4tau<value_type, value_type>;
#else
   pg fp1 = boost::math::jacobi_theta1tau;
   pg fp2 = boost::math::jacobi_theta2tau;
   pg fp3 = boost::math::jacobi_theta3tau;
   pg fp4 = boost::math::jacobi_theta4tau;
#endif
   boost::math::tools::test_result<value_type> result;

   result = boost::math::tools::test_hetero<Real>(
           data,
           bind_func<Real>(fp1, 0, 1),
           extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(),
           type_name, "jacobi_theta1tau", test_name);

   result = boost::math::tools::test_hetero<Real>(
           data,
           bind_func<Real>(fp2, 0, 1),
           extract_result<Real>(3));
   handle_test_result(result, data[result.worst()], result.worst(),
           type_name, "jacobi_theta2tau", test_name);

   result = boost::math::tools::test_hetero<Real>(
           data,
           bind_func<Real>(fp3, 0, 1),
           extract_result<Real>(4));
   handle_test_result(result, data[result.worst()], result.worst(),
           type_name, "jacobi_theta3tau", test_name);

   result = boost::math::tools::test_hetero<Real>(
           data,
           bind_func<Real>(fp4, 0, 1),
           extract_result<Real>(5));
   handle_test_result(result, data[result.worst()], result.worst(),
           type_name, "jacobi_theta4tau", test_name);

   std::cout << std::endl;
}

template <typename T>
void test_spots(T, const char* type_name)
{
    // Function values calculated on https://wolframalpha.com/
    // EllipticTheta[1, z, q]
    static const std::array<std::array<typename table_type<T>::type, 3>, 22> data1 = {{
        {{ SC_(0.25), SC_(0.5), SC_(0.1540230688155610552510349122197994458164480291364308) }},
        {{ SC_(0.5), SC_(0.5), SC_(0.402768575854814314826394321410682828786027207014725) }},
        {{ SC_(1.0), SC_(0.5), SC_(1.330378498179274650272750199052730280058943456725878763411) }},
        {{ SC_(2.0), SC_(0.5), SC_(1.632025902952598833772353216268208997235004608799433589257) }},
        {{ SC_(4.0), SC_(0.5), SC_(-1.02330632494166272025903454492708687979388431668700575889) }},
        {{ SC_(10.0), SC_(0.5), SC_(-0.506725689219604598643739369857898454980617737596340) }},

        {{ SC_(0.25), SC_(0.0078125), SC_(0.147082536469061213804178649159394420990352754783257117514) }},
        {{ SC_(0.5), SC_(0.0078125), SC_(0.285031930001354337576834900191853014429316815453397057542) }},
        {{ SC_(1.0), SC_(0.0078125), SC_(0.500336519612853406200885943502694674511080381314798446615) }},
        {{ SC_(2.0), SC_(0.0078125), SC_(0.540681625286624428872671041984657483294251553554422250931) }},
        {{ SC_(4.0), SC_(0.0078125), SC_(-0.44997798288122032252205899314355127447793127666845934004) }},
        {{ SC_(10.0), SC_(0.0078125), SC_(-0.32344103052261772036606444750254248788211904932304982822) }},

        {{ SC_(1.5), SC_(0.9375), SC_(6.455616598043074010374387709748431776849714249419980311428) }},
        {{ SC_(1.5), SC_(0.96875), SC_(8.494748959742732967152146642136570398472098411842769875811) }},
        {{ SC_(1.5), SC_(0.984375), SC_(10.27394960641515008799474668007745853274050682503839614462) }},
        {{ SC_(1.5), SC_(0.9921875), SC_(10.56322418602486655878408856907764433121023675569880599668) }},

        {{ SC_(0.0), SC_(0.0078125), SC_(0.0) }},
        {{ SC_(0.0), SC_(0.5),      SC_(0.0) }},
        {{ SC_(0.0), SC_(0.9375),   SC_(0.0) }},
        {{ SC_(0.0), SC_(0.96875),  SC_(0.0) }},
        {{ SC_(0.0), SC_(0.984375), SC_(0.0) }},
        {{ SC_(0.0), SC_(0.9921875), SC_(0.0) }},
    }};

    // EllipticTheta[2, z, q]
    static const std::array<std::array<typename table_type<T>::type, 3>, 22> data2 = {{
        {{ SC_(0.25), SC_(0.5), SC_(1.945359087094512287818938605108992884433591043123906291186) }},
        {{ SC_(0.5), SC_(0.5), SC_(1.484216087659583107499509464625356597654932790316228596683) }},
        {{ SC_(1.0), SC_(0.5), SC_(0.500198138514456200618643558666164246520575297293771869190) }},
        {{ SC_(2.0), SC_(0.5), SC_(-0.31816282165462356641101267196568721591143313305914629995) }},
        {{ SC_(4.0), SC_(0.5), SC_(-0.73416812893190753892245332974270105112878210782122749389) }},
        {{ SC_(10.0), SC_(0.5), SC_(-1.32067302962326803616213092008760707610192812421263609239) }},

        {{ SC_(0.25), SC_(0.0078125), SC_(0.576145327104766930654951565363812166642791552142749996541) }},
        {{ SC_(0.5), SC_(0.0078125), SC_(0.521816280475855206768414007009226207248962173223429288460) }},
        {{ SC_(1.0), SC_(0.0078125), SC_(0.321229744663905222607893889592775291250157432381753805004) }},
        {{ SC_(2.0), SC_(0.0078125), SC_(-0.24740754322178854446297728714315695332313383557595139996) }},
        {{ SC_(4.0), SC_(0.0078125), SC_(-0.38862819739105110692686845441417046539892089911988041243) }},
        {{ SC_(10.0), SC_(0.0078125), SC_(-0.49890931813624527541778224812869556711737103111567319268) }},

        {{ SC_(3.0), SC_(0.9375), SC_(-5.11392816786538016035153925334241975210394670067130953352) }},
        {{ SC_(3.0), SC_(0.96875), SC_(-5.29012912680048642016398403857878652305705819627897707515) }},
        {{ SC_(3.0), SC_(0.984375), SC_(-3.95437444890235969862463149250591376362876865922981295660) }},
        {{ SC_(3.0), SC_(0.9921875), SC_(-1.55309936234390798246955842243578030972976727248025068776) }},

        {{ SC_(0.0), SC_(0.0078125), SC_(0.594639849222534631954791856184512118943710280851563329851) }},
        {{ SC_(0.0), SC_(0.5), SC_(2.128931250513027558591613402575350180853805396958448940968) }},
        {{ SC_(0.0), SC_(0.9375), SC_(6.976947123071698246084428957201676843908839940030780606850) }},
        {{ SC_(0.0), SC_(0.96875), SC_(9.947454796978382607130245293535173220560623896730670936855) }},
        {{ SC_(0.0), SC_(0.984375), SC_(14.12398706491126638681068088410435889335374416476169364474) }},
        {{ SC_(0.0), SC_(0.9921875), SC_(20.01377050922054864039693105446212500742825758308336203414) }},
    }};

    // EllipticTheta[3, z, q]
    static const std::array<std::array<typename table_type<T>::type, 3>, 22> data3 = {{
        {{ SC_(0.25), SC_(0.5), SC_(1.945383919743612326705943032930976804537995814958244156964) }},
        {{ SC_(0.5), SC_(0.5), SC_(1.484396862425166928164115914328477415075581759665236164625) }},
        {{ SC_(1.0), SC_(0.5), SC_(0.505893885730484607919474452677852065978820023168006719298) }},
        {{ SC_(2.0), SC_(0.5), SC_(0.331435978324530423856445870208355989399154547338364678855) }},
        {{ SC_(4.0), SC_(0.5), SC_(0.736474899103717622792604836948158645655031914452730542597) }},
        {{ SC_(10.0), SC_(0.5), SC_(1.320991123572679837556511698539830878932277973655257733968) }},

        {{ SC_(0.25), SC_(0.0078125), SC_(1.013712231555102950279020764520600278509143561359073286704) }},
        {{ SC_(0.5), SC_(0.0078125), SC_(1.008442220428654137020348371834353416731512133091516925412) }},
        {{ SC_(1.0), SC_(0.0078125), SC_(0.993497700808926421501904684196885527471205379480110676883) }},
        {{ SC_(2.0), SC_(0.0078125), SC_(0.989786817339946335270527719280367827578464526769825119042) }},
        {{ SC_(4.0), SC_(0.0078125), SC_(0.997726554836621271192712333586918701234666824689471509074) }},
        {{ SC_(10.0), SC_(0.0078125), SC_(1.006376277246758468079371354566456368238591100751823541324) }},

        {{ SC_(3.0), SC_(0.9375), SC_(5.113928167865380160351539253342419752103946700671309533529) }},
        {{ SC_(3.0), SC_(0.96875), SC_(5.290129126800486420163984038578786523057058196278977075158) }},
        {{ SC_(3.0), SC_(0.984375), SC_(3.954374448902359698624631492505913763628768659229812956604) }},
        {{ SC_(3.0), SC_(0.9921875), SC_(1.553099362343907982469558422435780309729767272480250687761) }},

        {{ SC_(0.0), SC_(0.0078125), SC_(1.015625007450580597140668559497101271987479437621158995242) }},
        {{ SC_(0.0), SC_(0.5), SC_(2.128936827211877158669458548544951324612516539940878092889) }},
        {{ SC_(0.0), SC_(0.9375), SC_(6.976947123071698246084428957201676843908839940030780606850) }},
        {{ SC_(0.0), SC_(0.96875), SC_(9.947454796978382607130245293535173220560623896730670936855) }},
        {{ SC_(0.0), SC_(0.984375), SC_(14.12398706491126638681068088410435889335374416476169364474) }},
        {{ SC_(0.0), SC_(0.9921875), SC_(20.01377050922054864039693105446212500742825758308336203414) }},
    }};

    // EllipticTheta[4, z, q]
    static const std::array<std::array<typename table_type<T>::type, 3>, 20> data4 = {{
        {{ SC_(0.25), SC_(0.5), SC_(0.189666257078605856907477593562312286776776156459895303534) }},
        {{ SC_(0.5), SC_(0.5), SC_(0.411526533253405515206323680892825857445581901774756902114) }},
        {{ SC_(1.0), SC_(0.5), SC_(1.330686328485433289294314954726283002076056588770122570003) }},
        {{ SC_(2.0), SC_(0.5), SC_(1.632130562351990831100773069064726698266264072056233877584) }},
        {{ SC_(4.0), SC_(0.5), SC_(1.024161147731330827103564503229671566751499815308607281771) }},
        {{ SC_(10.0), SC_(0.5), SC_(0.512267623558970547872956225451763774220817155272291299008) }},

        {{ SC_(0.25), SC_(0.0078125), SC_(0.986287776496028802869709593974763104913490530471778951141) }},
        {{ SC_(0.5), SC_(0.0078125), SC_(0.991557773370274771280909909075180928934876164736866576523) }},
        {{ SC_(1.0), SC_(0.0078125), SC_(1.006502289451024620679171207071841795644771897492095505061) }},
        {{ SC_(2.0), SC_(0.0078125), SC_(1.010213180491934207237038406874868068814754041886231359135) }},
        {{ SC_(4.0), SC_(0.0078125), SC_(1.002273430893140443692155306258136796608858362734966941965) }},
        {{ SC_(10.0), SC_(0.0078125), SC_(0.993623712815089968927968772900270119031604138870447126142) }},

        {{ SC_(1.5), SC_(0.9375), SC_(6.455616598043074010374387709748431776849714249419980311428) }},
        {{ SC_(1.5), SC_(0.96875), SC_(8.494748959742732967152146642136570398472098411842769875811) }},
        {{ SC_(1.5), SC_(0.984375), SC_(10.27394960641515008799474668007745853274050682503839614462) }},
        {{ SC_(1.5), SC_(0.9921875), SC_(10.56322418602486655878408856907764433121023675569880599668) }},

        {{ SC_(0.0), SC_(0.0078125), SC_(0.984375007450580596706987690502899498384498317273182227148) }},
        {{ SC_(0.0), SC_(0.5), SC_(0.121124208002580502460849293181867505809858246820960597233) }},
        {{ SC_(0.0), SC_(0.9375), SC_(3.4752705802238602772154431927173524635861732774491547e-16) }},
        {{ SC_(0.0), SC_(0.96875), SC_(3.5224799214778391114074447929704379790977217720041291e-33) }}
        // {{ SC_(0.0), SC_(0.984375), SC_(0.0) }},
        // {{ SC_(0.0), SC_(0.9921875), SC_(0.0) }},
    }};

    do_test_jacobi_theta1<T>(data1, type_name, "Jacobi Theta 1: WolframAlpha Data");
    do_test_jacobi_theta2<T>(data2, type_name, "Jacobi Theta 2: WolframAlpha Data");
    do_test_jacobi_theta3<T>(data3, type_name, "Jacobi Theta 3: WolframAlpha Data");
    do_test_jacobi_theta4<T>(data4, type_name, "Jacobi Theta 4: WolframAlpha Data");

#include "jacobi_theta_data.ipp"

    do_test_jacobi_theta_tau<T>(jacobi_theta_data, type_name, "Jacobi Theta: Random Data");

#include "jacobi_theta_small_tau.ipp"

    do_test_jacobi_theta_tau<T>(jacobi_theta_small_tau_data, type_name, "Jacobi Theta: Random Data (Small Tau)");
}

#define _check_close(a, b, eps) \
    if (abs(a) < 2 * eps * eps || abs(b) < 2 * eps * eps) { \
        BOOST_CHECK_SMALL((a) - (b), eps); \
    } else { \
        BOOST_CHECK_CLOSE_FRACTION(a, b, eps); \
    }

template <typename RealType>
inline void test_periodicity(RealType z, RealType q, RealType eps) {
    using namespace boost::math;

    _check_close(
            jacobi_theta1(z, q),
            jacobi_theta1(z + constants::two_pi<RealType>(), q),
            eps);

    _check_close(
            jacobi_theta2(z, q),
            jacobi_theta2(z + constants::two_pi<RealType>(), q), 
            eps);

    _check_close(
            jacobi_theta3(z, q),
            jacobi_theta3(z + constants::pi<RealType>(), q), 
            eps);

    _check_close(
            jacobi_theta4(z, q),
            jacobi_theta4(z + constants::pi<RealType>(), q), 
            eps);
}

// DLMF 20.2(iii) Translation of the Argument by Half-Periods
template <typename RealType>
inline void test_argument_translation(RealType z, RealType q, RealType eps) {
    using namespace boost::math;

    _check_close( // DLMF 20.2.11
            jacobi_theta1(z, q),
            -jacobi_theta2(z + constants::half_pi<RealType>(), q), 
            eps);

    _check_close( // DLMF 20.2.12
            jacobi_theta2(z, q),
            jacobi_theta1(z + constants::half_pi<RealType>(), q), 
            eps);

    _check_close( // DLMF 20.2.13
            jacobi_theta3(z, q),
            jacobi_theta4(z + constants::half_pi<RealType>(), q), 
            eps);

    _check_close( // DLMF 20.2.14
            jacobi_theta4(z, q),
            jacobi_theta3(z + constants::half_pi<RealType>(), q), 
            eps);
}

// DLMF 20.7(i) Sums of Squares
template <typename RealType>
inline void test_sums_of_squares(RealType z, RealType q, RealType eps) {
    using namespace boost::math;

    _check_close( // DLMF 20.7.1
            jacobi_theta3(RealType(0), q) * jacobi_theta3(z, q),
            hypot(
                jacobi_theta4(RealType(0), q) * jacobi_theta4(z, q),
                jacobi_theta2(RealType(0), q) * jacobi_theta2(z, q)),
            eps);

    _check_close( // DLMF 20.7.2
            jacobi_theta3(RealType(0), q) * jacobi_theta4(z, q),
            hypot(
                jacobi_theta2(RealType(0), q) * jacobi_theta1(z, q),
                jacobi_theta4(RealType(0), q) * jacobi_theta3(z, q)),
            eps);

    _check_close( // DLMF 20.7.3
            jacobi_theta2(RealType(0), q) * jacobi_theta4(z, q),
            hypot(
                jacobi_theta3(RealType(0), q) * jacobi_theta1(z, q),
                jacobi_theta4(RealType(0), q) * jacobi_theta2(z, q)),
            eps);

    _check_close( // DLMF 20.7.4
            jacobi_theta2(RealType(0), q) * jacobi_theta3(z, q),
            hypot(
                jacobi_theta4(RealType(0), q) * jacobi_theta1(z, q),
                jacobi_theta3(RealType(0), q) * jacobi_theta2(z, q)),
            eps);

    _check_close( // DLMF 20.7.5 (no z)
            jacobi_theta3(RealType(0), q) * jacobi_theta3(RealType(0), q),
            hypot(
                jacobi_theta2(RealType(0), q) * jacobi_theta2(RealType(0), q),
                jacobi_theta4(RealType(0), q) * jacobi_theta4(RealType(0), q)),
            eps);
}

// DLMF 20.7(ii) Addition Formulas
template <typename RealType>
inline void test_addition_formulas(RealType z, RealType w, RealType q, RealType eps) {
    using namespace boost::math;

    _check_close( // DLMF 20.7.6
            jacobi_theta4(RealType(0), q) * jacobi_theta4(RealType(0), q) *
            jacobi_theta1(w + z, q) * jacobi_theta1(w - z, q),
            jacobi_theta3(w, q) * jacobi_theta3(w, q) * jacobi_theta2(z, q) * jacobi_theta2(z, q) -
            jacobi_theta2(w, q) * jacobi_theta2(w, q) * jacobi_theta3(z, q) * jacobi_theta3(z, q),
            eps);

    _check_close( // DLMF 20.7.7
            jacobi_theta4(RealType(0), q) * jacobi_theta4(RealType(0), q) *
            jacobi_theta2(w + z, q) * jacobi_theta2(w - z, q),
            jacobi_theta4(w, q) * jacobi_theta4(w, q) * jacobi_theta2(z, q) * jacobi_theta2(z, q) -
            jacobi_theta1(w, q) * jacobi_theta1(w, q) * jacobi_theta3(z, q) * jacobi_theta3(z, q),
            eps);

    _check_close( // DLMF 20.7.8
            jacobi_theta4(RealType(0), q) * jacobi_theta4(RealType(0), q) *
            jacobi_theta3(w + z, q) * jacobi_theta3(w - z, q),
            jacobi_theta4(w, q) * jacobi_theta4(w, q) * jacobi_theta3(z, q) * jacobi_theta3(z, q) -
            jacobi_theta1(w, q) * jacobi_theta1(w, q) * jacobi_theta2(z, q) * jacobi_theta2(z, q),
            eps);

    _check_close( // DLMF 20.7.9
            jacobi_theta4(RealType(0), q) * jacobi_theta4(RealType(0), q) *
            jacobi_theta4(w + z, q) * jacobi_theta4(w - z, q),
            jacobi_theta3(w, q) * jacobi_theta3(w, q) * jacobi_theta3(z, q) * jacobi_theta3(z, q) -
            jacobi_theta2(w, q) * jacobi_theta2(w, q) * jacobi_theta2(z, q) * jacobi_theta2(z, q),
            eps);
}

// DLMF 20.7(iii) Duplication Formula
template <typename RealType>
inline void test_duplication_formula(RealType z, RealType q, RealType eps) {
    using namespace boost::math;

    _check_close( // DLMF 20.7.10
            jacobi_theta1(z + z, q) * jacobi_theta2(RealType(0), q) * jacobi_theta3(RealType(0), q) * jacobi_theta4(RealType(0), q),
            RealType(2) * jacobi_theta1(z, q) * jacobi_theta2(z, q) * jacobi_theta3(z, q) * jacobi_theta4(z, q),
            eps);
}

// DLMF 20.7(iv) Transformations of Nome
template <typename RealType>
inline void test_transformations_of_nome(RealType z, RealType q, RealType eps) {
    using namespace boost::math;

    _check_close( // DLMF 20.7.11
            jacobi_theta1(z, q) * jacobi_theta2(z, q) * jacobi_theta4(z + z, q * q),
            jacobi_theta3(z, q) * jacobi_theta4(z, q) * jacobi_theta1(z + z, q * q),
            eps);

    _check_close( // DLMF 20.7.12
            jacobi_theta1(z, q * q) * jacobi_theta4(z, q * q) * jacobi_theta2(z, q),
            jacobi_theta2(z, q * q) * jacobi_theta3(z, q * q) * jacobi_theta1(z, q),
            eps);
}

// DLMF 20.7(v) Watson's Identities
template <typename RealType>
inline void test_watsons_identities(RealType z, RealType w, RealType q, RealType eps) {
    using namespace boost::math;

    // Rearrange DLMF equations to get q*q on each side of the equality

    _check_close( // DLMF 20.7.13
            jacobi_theta1(z, q) * jacobi_theta1(w, q)
            + jacobi_theta2(z + w, q * q) * jacobi_theta3(z - w, q * q),
            jacobi_theta3(z + w, q * q) * jacobi_theta2(z - w, q * q),
            eps);

    _check_close( // DLMF 20.7.14
            jacobi_theta3(z, q) * jacobi_theta3(w, q)
            - jacobi_theta2(z + w, q * q) * jacobi_theta2(z - w, q * q),
            jacobi_theta3(z + w, q * q) * jacobi_theta3(z - w, q * q),
            eps);

    _check_close( // MathWorld Eqn. 48
            jacobi_theta3(z, q) - jacobi_theta2(z + z, q*q*q*q),
            jacobi_theta3(z + z, q*q*q*q),
            eps);

    _check_close( // MathWorld Eqn. 49
            jacobi_theta4(z, q) + jacobi_theta2(z + z, q*q*q*q),
            jacobi_theta3(z + z, q*q*q*q),
            eps);
}

template <typename RealType>
inline void test_landen_transformations(RealType z, RealType tau, RealType eps) {
    // A and B below are the reciprocals of their DLMF definitions
    using namespace boost::math;

    // DLMF 20.7.15 (reciprocal)
    RealType A = jacobi_theta4tau(RealType(0), tau + tau);

    _check_close( // DLMF 20.7.16
            jacobi_theta1tau(z + z, tau + tau) * A,
            jacobi_theta1tau(z, tau) * jacobi_theta2tau(z, tau),
            eps);

    _check_close( // DLMF 20.7.17
            jacobi_theta2tau(z + z, tau + tau) * A,
            jacobi_theta1tau(constants::quarter_pi<RealType>() - z, tau) * jacobi_theta1tau(constants::quarter_pi<RealType>() + z, tau),
            eps);

    _check_close( // DLMF 20.7.18
            jacobi_theta3tau(z + z, tau + tau) * A,
            jacobi_theta3tau(constants::quarter_pi<RealType>() - z, tau) * jacobi_theta3tau(constants::quarter_pi<RealType>() + z, tau),
            eps);

    _check_close( // DLMF 20.7.19
            jacobi_theta4tau(z + z, tau + tau) * A,
            jacobi_theta3tau(z, tau) * jacobi_theta4tau(z, tau),
            eps);

    // DLMF 20.7.20 (reciprocal)
    RealType B = jacobi_theta3tau(RealType(0), tau) * jacobi_theta4tau(RealType(0), tau) * jacobi_theta3tau(constants::quarter_pi<RealType>(), tau);

    _check_close( // DLMF 20.7.21
            jacobi_theta1tau(4*z, 4*tau) * B,
            jacobi_theta1tau(z, tau) 
            * jacobi_theta1tau(constants::quarter_pi<RealType>() - z, tau) 
            * jacobi_theta1tau(constants::quarter_pi<RealType>() + z, tau)
            * jacobi_theta2tau(z, tau),
            eps);

    _check_close( // DLMF 20.7.22
            jacobi_theta2tau(4*z, 4*tau) * B,
            jacobi_theta2tau(constants::quarter_pi<RealType>()/2 - z, tau) 
            * jacobi_theta2tau(constants::quarter_pi<RealType>()/2 + z, tau) 
            * jacobi_theta2tau(constants::three_quarters_pi<RealType>()/2 - z, tau)
            * jacobi_theta2tau(constants::three_quarters_pi<RealType>()/2 + z, tau),
            eps);

    _check_close( // DLMF 20.7.23
            jacobi_theta3tau(4*z, 4*tau) * B,
            jacobi_theta3tau(constants::quarter_pi<RealType>()/2 - z, tau) 
            * jacobi_theta3tau(constants::quarter_pi<RealType>()/2 + z, tau) 
            * jacobi_theta3tau(constants::three_quarters_pi<RealType>()/2 - z, tau)
            * jacobi_theta3tau(constants::three_quarters_pi<RealType>()/2 + z, tau),
            eps);

    _check_close( // DLMF 20.7.24
            jacobi_theta4tau(4*z, 4*tau) * B,
            jacobi_theta4tau(z, tau)
            * jacobi_theta4tau(constants::quarter_pi<RealType>() - z, tau)
            * jacobi_theta4tau(constants::quarter_pi<RealType>() + z, tau)
            * jacobi_theta3tau(z, tau),
            eps);
}

template <typename RealType>
inline void test_special_values(RealType eps) {
    // https://mathworld.wolfram.com/JacobiThetaFunctions.html (Eq. 45)
    using namespace boost::math;
    BOOST_MATH_STD_USING

    _check_close(
            tgamma(RealType(0.75)) * jacobi_theta3tau(RealType(0), RealType(1)),
            sqrt(constants::root_pi<RealType>()),
            eps);

    _check_close(
            tgamma(RealType(1.25))
            * constants::root_pi<RealType>()
            * sqrt(sqrt(constants::root_two<RealType>()))
            * jacobi_theta3tau(RealType(0), constants::root_two<RealType>()),
            tgamma(RealType(1.125))
            * sqrt(tgamma(RealType(0.25))),
            eps);

    _check_close(
            tgamma(RealType(0.75))
            * sqrt(constants::root_two<RealType>())
            * jacobi_theta4tau(RealType(0), RealType(1)),
            sqrt(constants::root_pi<RealType>()),
            eps);
}

template <typename RealType>
inline void test_mellin_transforms(RealType s, RealType integration_eps, RealType result_eps) {
    using namespace boost::math;
    BOOST_MATH_STD_USING

    boost::math::quadrature::exp_sinh<RealType> integrator;

    auto f2 = [&, s](RealType t)
    {
        if (t*t == 0.f)
            return RealType(0);
        if (t > sqrt(sqrt((std::numeric_limits<RealType>::max)())))
            return RealType(0);

        return pow(t, s-1) * jacobi_theta2tau(RealType(0), t*t);
    };

    auto f3 = [&, s](RealType t)
    {
        if (t*t == 0.f)
            return RealType(0);
        if (t > sqrt(sqrt((std::numeric_limits<RealType>::max)())))
            return RealType(0);

        return pow(t, s-1) * jacobi_theta3m1tau(RealType(0), t*t);
    };

    auto f4 = [&, s](RealType t)
    {
        if (t*t == 0.f)
            return RealType(0);
        if (t > sqrt(sqrt((std::numeric_limits<RealType>::max)())))
            return RealType(0);

        return -pow(t, s-1) * jacobi_theta4m1tau(RealType(0), t*t);
    };

    _check_close( // DLMF 20.10.1
            integrator.integrate(f2, integration_eps),
            (pow(RealType(2), s) - 1) * pow(constants::pi<RealType>(), -0.5*s) * tgamma(0.5*s) * zeta(s),
            result_eps);

    _check_close( // DLMF 20.10.2
            integrator.integrate(f3, integration_eps),
            pow(constants::pi<RealType>(), -0.5*s) * tgamma(0.5*s) * zeta(s),
            result_eps);

    _check_close( // DLMF 20.10.3
            integrator.integrate(f4, integration_eps),
            (1 - pow(RealType(2), 1 - s)) * pow(constants::pi<RealType>(), -0.5*s) * tgamma(0.5*s) * zeta(s),
            result_eps);
}

template <typename RealType>
inline void test_laplace_transforms(RealType s, RealType integration_eps, RealType result_eps) {
    using namespace boost::math;
    BOOST_MATH_STD_USING

    RealType beta = -0.5;
    RealType l = sinh(abs(beta)) + 1.0;

    boost::math::quadrature::exp_sinh<RealType> integrator;

    auto f1 = [&, s, l, beta](RealType t)
    {
        return exp(-s*t) * jacobi_theta1tau(0.5 * beta * constants::pi<RealType>() / l,
                constants::pi<RealType>() * t / l / l);
    };

    auto f4 = [&, s, l, beta](RealType t)
    {
        return exp(-s*t) * jacobi_theta4tau(0.5 * beta * constants::pi<RealType>() / l,
                constants::pi<RealType>() * t / l / l);
    };

    _check_close( // DLMF 20.10.4 says the RHS should be negative?
            integrator.integrate(f1, integration_eps),
            l/sqrt(s)*sinh(beta*sqrt(s))/cosh(l*sqrt(s)),
            result_eps);

    _check_close( // DLMF 20.10.5
            integrator.integrate(f4, integration_eps),
            l/sqrt(s)*cosh(beta*sqrt(s))/sinh(l*sqrt(s)),
            result_eps);

    // TODO - DLMF defines two additional relations for theta2 and theta3, but
    // these do not match the computed values at all.
}

template <typename RealType>
inline void test_elliptic_functions(RealType z, RealType q, RealType eps) {
    using namespace boost::math;
    BOOST_MATH_STD_USING

    RealType t2 = jacobi_theta2(RealType(0), q);
    RealType t3 = jacobi_theta3(RealType(0), q);
    RealType t4 = jacobi_theta4(RealType(0), q);
    RealType k = t2 * t2 / (t3 * t3);
    RealType xi = z / (t3 * t3);

    _check_close( // DLMF 22.2.4
            jacobi_sn(k, z) *
            t2 * jacobi_theta4(xi, q),
            t3 * jacobi_theta1(xi, q),
            eps);

    _check_close( // DLMF 22.2.5
            jacobi_cn(k, z) *
            t2 * jacobi_theta4(xi, q),
            t4 * jacobi_theta2(xi, q),
            eps);

    _check_close( // DLMF 22.2.6
            jacobi_dn(k, z) *
            t3 * jacobi_theta4(xi, q),
            t4 * jacobi_theta3(xi, q),
            eps);

    _check_close( // DLMF 22.2.7
            jacobi_sd(k, z) *
            t2 * t4 * jacobi_theta3(xi, q),
            t3 * t3 * jacobi_theta1(xi, q),
            eps);

    _check_close( // DLMF 22.2.8
            jacobi_cd(k, z) *
            t2 * jacobi_theta3(xi, q),
            t3 * jacobi_theta2(xi, q),
            eps);

    _check_close( // DLMF 22.2.9
            jacobi_cd(k, z) *
            t2 * jacobi_theta3(xi, q),
            t3 * jacobi_theta2(xi, q),
            eps);

    _check_close( // DLMF 22.2.9
            jacobi_sc(k, z) *
            t4 * jacobi_theta2(xi, q),
            t3 * jacobi_theta1(xi, q),
            eps);
}

template <typename RealType>
inline void test_elliptic_integrals(RealType q, RealType eps) {
    using namespace boost::math;
    BOOST_MATH_STD_USING

    RealType t2 = jacobi_theta2(RealType(0), q);
    RealType t3 = jacobi_theta3(RealType(0), q);
    RealType t4 = jacobi_theta4(RealType(0), q);
    RealType k = t2*t2 / (t3*t3);
    RealType k1= t4*t4 / (t3*t3);
    
    if (t3*t3*t3*t3 != 0.f && t4*t4*t4*t4 != 0.f) {
        _check_close( // DLMF 20.9.4
                ellint_rf(RealType(0), t3*t3*t3*t3, t4*t4*t4*t4),
                constants::half_pi<RealType>(),
                eps);
    }

    if (k*k != 0.f && k1*k1 != 0.f) {
        _check_close( // DLMF 20.9.5
                ellint_rf(RealType(0), k1*k1, RealType(1))
                * log(q) / constants::pi<RealType>(),
                -ellint_rf(RealType(0), k*k, RealType(1)),
                eps);
    }
}
