// Copyright John Maddock 2006.
// Copyright Paul A. Bristow 2007, 2009
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/concepts/real_concept.hpp>
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"

#ifndef SC_
#define SC_(x) static_cast<typename table_type<T>::type>(BOOST_JOIN(x, L))
#endif

template <class Real, class T>
void do_test_digamma(const T& data, const char* type_name, const char* test_name)
{
#if !(defined(ERROR_REPORTING_MODE) && !defined(DIGAMMA_FUNCTION_TO_TEST))
   typedef Real                   value_type;

   typedef value_type (*pg)(value_type);
#ifdef DIGAMMA_FUNCTION_TO_TEST
   pg funcp = DIGAMMA_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = boost::math::digamma<value_type>;
#else
   pg funcp = boost::math::digamma;
#endif

   boost::math::tools::test_result<value_type> result;

   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

   //
   // test digamma against data:
   //
   result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func<Real>(funcp, 0), 
      extract_result<Real>(1));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "digamma", test_name);
   std::cout << std::endl;
#endif
}

template <class T>
void test_digamma(T, const char* name)
{
   //
   // The actual test data is rather verbose, so it's in a separate file
   //
   // The contents are as follows, each row of data contains
   // three items, input value a, input value b and erf(a, b):
   // 
#  include "digamma_data.ipp"

   do_test_digamma<T>(digamma_data, name, "Digamma Function: Large Values");

#  include "digamma_root_data.ipp"

   do_test_digamma<T>(digamma_root_data, name, "Digamma Function: Near the Positive Root");

#  include "digamma_small_data.ipp"

   do_test_digamma<T>(digamma_small_data, name, "Digamma Function: Near Zero");

#  include "digamma_neg_data.ipp"

   do_test_digamma<T>(digamma_neg_data, name, "Digamma Function: Negative Values");

    static const std::array<std::array<typename table_type<T>::type, 2>, 5> digamma_bugs = {{
       // Test cases from Rocco Romeo:
        {{ SC_(7.888609052210118054117285652827862296732064351090230047702789306640625e-31) /*std::ldexp(1.0, -100)*/, SC_(-1.26765060022822940149670320537657721566490153286060651209008e30) }},
        {{ SC_(-7.888609052210118054117285652827862296732064351090230047702789306640625e-31) /*-std::ldexp(1.0, -100)*/, SC_(1.26765060022822940149670320537542278433509846713939348790992e30) }},
        {{ SC_(1.0), SC_(-0.577215664901532860606512090082402431042159335939923598805767) }},
        {{ SC_(-0.99999904632568359375) /*static_cast<T>(-1) + static_cast<T>(std::ldexp(1.0, -20))*/, SC_(-1.04857557721314249602848739817764518743062133735858753112190e6) }},
        {{ SC_(-1.00000095367431640625) /*static_cast<T>(-1) - static_cast<T>(std::ldexp(1.0, -20))*/, SC_(1.04857642278181269259522681939281063878220298942888100442172e6) }},
    }};
   do_test_digamma<T>(digamma_bugs, name, "Digamma Function: Values near 0");

   static const std::array<std::array<typename table_type<T>::type, 2>, 40> digamma_integers = { {
      {{ SC_(1.0), SC_(-0.57721566490153286060651209008240243) }}, {{ SC_(2.0), SC_(0.42278433509846713939348790991759757) }}, {{ SC_(3.0), SC_(0.92278433509846713939348790991759757) }}, {{ SC_(4.0), SC_(1.2561176684318004727268212432509309) }}, {{ SC_(5.0), SC_(1.5061176684318004727268212432509309) }}, {{ SC_(6.0), SC_(1.7061176684318004727268212432509309) }}, {{ SC_(7.0), SC_(1.8727843350984671393934879099175976) }}, {{ SC_(8.0), SC_(2.0156414779556099965363450527747404) }}, {{ SC_(9.0), SC_(2.1406414779556099965363450527747404) }}, {{ SC_(10.0), SC_(2.2517525890667211076474561638858515) }}, {{ SC_(11.0), SC_(2.3517525890667211076474561638858515) }}, {{ SC_(12.0), SC_(2.4426616799758120167383652547949424) }}, {{ SC_(13.0), SC_(2.5259950133091453500716985881282758) }}, {{ SC_(14.0), SC_(2.6029180902322222731486216650513527) }}, {{ SC_(15.0), SC_(2.6743466616607937017200502364799241) }}, {{ SC_(16.0), SC_(2.7410133283274603683867169031465908) }}, {{ SC_(17.0), SC_(2.8035133283274603683867169031465908) }}, {{ SC_(18.0), SC_(2.8623368577392250742690698443230614) }}, {{ SC_(19.0), SC_(2.9178924132947806298246253998786169) }}, {{ SC_(20.0), SC_(2.9705239922421490508772569788259854) }}, {{ SC_(21.0), SC_(3.0205239922421490508772569788259854) }}, {{ SC_(22.0), SC_(3.0681430398611966699248760264450330) }}, {{ SC_(23.0), SC_(3.1135975853157421244703305718995784) }}, {{ SC_(24.0), SC_(3.1570758461853073418616349197256654) }}, {{ SC_(25.0), SC_(3.1987425128519740085283015863923321) }}, {{ SC_(26.0), SC_(3.2387425128519740085283015863923321) }}, {{ SC_(27.0), SC_(3.2772040513135124700667631248538705) }}, {{ SC_(28.0), SC_(3.3142410883505495071038001618909076) }}, {{ SC_(29.0), SC_(3.3499553740648352213895144476051933) }}, {{ SC_(30.0), SC_(3.3844381326855248765619282407086415) }}, {{ SC_(31.0), SC_(3.4177714660188582098952615740419749) }}, {{ SC_(32.0), SC_(3.4500295305349872421533260901710071) }}, {{ SC_(33.0), SC_(3.4812795305349872421533260901710071) }}, {{ SC_(34.0), SC_(3.5115825608380175451836291204740374) }}, {{ SC_(35.0), SC_(3.5409943255438998981248055910622727) }}, {{ SC_(36.0), SC_(3.5695657541153284695533770196337013) }}, {{ SC_(37.0), SC_(3.5973435318931062473311547974114791) }}, {{ SC_(38.0), SC_(3.6243705589201332743581818244385061) }}, {{ SC_(39.0), SC_(3.6506863483938174848844976139121903) }}, {{ SC_(40.0), SC_(3.6763273740348431259101386395532160) }}
   } };
   do_test_digamma<T>(digamma_integers, name, "Digamma Function: Integer arguments");

   static const std::array<std::array<typename table_type<T>::type, 2>, 41> digamma_half_integers = { {
      {{ SC_(0.5), SC_(-1.9635100260214234794409763329987556) }}, {{ SC_(1.5), SC_(0.036489973978576520559023667001244433) }}, {{ SC_(2.5), SC_(0.70315664064524318722569033366791110) }}, {{ SC_(3.5), SC_(1.1031566406452431872256903336679111) }}, {{ SC_(4.5), SC_(1.3888709263595289015114046193821968) }}, {{ SC_(5.5), SC_(1.6110931485817511237336268416044190) }}, {{ SC_(6.5), SC_(1.7929113303999329419154450234226009) }}, {{ SC_(7.5), SC_(1.9467574842460867880692911772687547) }}, {{ SC_(8.5), SC_(2.0800908175794201214026245106020880) }}, {{ SC_(9.5), SC_(2.1977378764029495331673303929550292) }}, {{ SC_(10.5), SC_(2.3030010342976863752725935508497661) }}, {{ SC_(11.5), SC_(2.3982391295357816133678316460878613) }}, {{ SC_(12.5), SC_(2.4851956512749120481504403417400352) }}, {{ SC_(13.5), SC_(2.5651956512749120481504403417400352) }}, {{ SC_(14.5), SC_(2.6392697253489861222245144158141093) }}, {{ SC_(15.5), SC_(2.7082352425903654325693420020210058) }}, {{ SC_(16.5), SC_(2.7727513716226234970854710342790703) }}, {{ SC_(17.5), SC_(2.8333574322286841031460770948851310) }}, {{ SC_(18.5), SC_(2.8905002893715412460032199520279881) }}, {{ SC_(19.5), SC_(2.9445543434255953000572740060820421) }}, {{ SC_(20.5), SC_(2.9958363947076465821085560573640934) }}, {{ SC_(21.5), SC_(3.0446168825125246308890438622421422) }}, {{ SC_(22.5), SC_(3.0911285104195013750750903738700492) }}, {{ SC_(23.5), SC_(3.1355729548639458195195348183144936) }}, {{ SC_(24.5), SC_(3.1781261463533075216471943927825787) }}, {{ SC_(25.5), SC_(3.2189424728839197665451535764560481) }}, {{ SC_(26.5), SC_(3.2581581591584295704667222039070285) }}, {{ SC_(27.5), SC_(3.2958940082150333440516278642843870) }}, {{ SC_(28.5), SC_(3.3322576445786697076879915006480234) }}, {{ SC_(29.5), SC_(3.3673453638769153217230792199462690) }}, {{ SC_(30.5), SC_(3.4012436689616610844349436267259300) }}, {{ SC_(31.5), SC_(3.4340305542075627237792059218078972) }}, {{ SC_(32.5), SC_(3.4657765859535944698109519535539290) }}, {{ SC_(33.5), SC_(3.4965458167228252390417211843231597) }}, {{ SC_(34.5), SC_(3.5263965629914819554596316320843538) }}, {{ SC_(35.5), SC_(3.5553820702378587670538345306350784) }}, {{ SC_(36.5), SC_(3.5835510843223658093073556573956418) }}, {{ SC_(37.5), SC_(3.6109483445963384120470816847929021) }}, {{ SC_(38.5), SC_(3.6376150112630050787137483514595687) }}, {{ SC_(39.5), SC_(3.6635890372370310527397223774335947) }}, {{ SC_(40.5), SC_(3.6889054929332335843852919976867593) }}
   } };
   do_test_digamma<T>(digamma_half_integers, name, "Digamma Function: Half integer arguments");

   BOOST_MATH_CHECK_THROW(boost::math::digamma(T(0)), std::domain_error);
   BOOST_MATH_CHECK_THROW(boost::math::digamma(T(-1)), std::domain_error);
   BOOST_MATH_CHECK_THROW(boost::math::digamma(T(-2)), std::domain_error);
}

