//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4224)
#endif

#include <boost/math/special_functions/digamma.hpp>
#include <boost/array.hpp>
#include <boost/lexical_cast.hpp>
#include "../../test/table_type.hpp"
#include "table_helper.hpp"
#include "performance.hpp"
#include <iostream>

int main()
{
   typedef double T;
#define SC_(x) static_cast<double>(x)
#  include "digamma_data.ipp"
#  include "digamma_root_data.ipp"
#  include "digamma_small_data.ipp"
#  include "digamma_neg_data.ipp"
    static const std::array<std::array<T, 2>, 5> digamma_bugs = {{
       // Test cases from Rocco Romeo:
        {{ static_cast<T>(std::ldexp(1.0, -100)), SC_(-1.26765060022822940149670320537657721566490153286060651209008e30) }},
        {{ static_cast<T>(-std::ldexp(1.0, -100)), SC_(1.26765060022822940149670320537542278433509846713939348790992e30) }},
        {{ static_cast<T>(1), SC_(-0.577215664901532860606512090082402431042159335939923598805767) }},
        {{ static_cast<T>(-1) + static_cast<T>(std::ldexp(1.0, -20)), SC_(-1.04857557721314249602848739817764518743062133735858753112190e6) }},
        {{ static_cast<T>(-1) - static_cast<T>(std::ldexp(1.0, -20)), SC_(1.04857642278181269259522681939281063878220298942888100442172e6) }},
    }};
   static const std::array<std::array<T, 2>, 40> digamma_integers = { {
      { 1, SC_(-0.57721566490153286060651209008240243) }, { 2, SC_(0.42278433509846713939348790991759757) }, { 3, SC_(0.92278433509846713939348790991759757) }, { 4, SC_(1.2561176684318004727268212432509309) }, { 5, SC_(1.5061176684318004727268212432509309) }, { 6, SC_(1.7061176684318004727268212432509309) }, { 7, SC_(1.8727843350984671393934879099175976) }, { 8, SC_(2.0156414779556099965363450527747404) }, { 9, SC_(2.1406414779556099965363450527747404) }, { SC_(10.0), SC_(2.2517525890667211076474561638858515) }, { SC_(11.0), SC_(2.3517525890667211076474561638858515) }, { SC_(12.0), SC_(2.4426616799758120167383652547949424) }, { SC_(13.0), SC_(2.5259950133091453500716985881282758) }, { SC_(14.0), SC_(2.6029180902322222731486216650513527) }, { SC_(15.0), SC_(2.6743466616607937017200502364799241) }, { SC_(16.0), SC_(2.7410133283274603683867169031465908) }, { SC_(17.0), SC_(2.8035133283274603683867169031465908) }, { SC_(18.0), SC_(2.8623368577392250742690698443230614) }, { SC_(19.0), SC_(2.9178924132947806298246253998786169) }, { SC_(20.0), SC_(2.9705239922421490508772569788259854) }, { SC_(21.0), SC_(3.0205239922421490508772569788259854) }, { SC_(22.0), SC_(3.0681430398611966699248760264450330) }, { SC_(23.0), SC_(3.1135975853157421244703305718995784) }, { SC_(24.0), SC_(3.1570758461853073418616349197256654) }, { SC_(25.0), SC_(3.1987425128519740085283015863923321) }, { SC_(26.0), SC_(3.2387425128519740085283015863923321) }, { SC_(27.0), SC_(3.2772040513135124700667631248538705) }, { SC_(28.0), SC_(3.3142410883505495071038001618909076) }, { SC_(29.0), SC_(3.3499553740648352213895144476051933) }, { SC_(30.0), SC_(3.3844381326855248765619282407086415) }, { SC_(31.0), SC_(3.4177714660188582098952615740419749) }, { SC_(32.0), SC_(3.4500295305349872421533260901710071) }, { SC_(33.0), SC_(3.4812795305349872421533260901710071) }, { SC_(34.0), SC_(3.5115825608380175451836291204740374) }, { SC_(35.0), SC_(3.5409943255438998981248055910622727) }, { SC_(36.0), SC_(3.5695657541153284695533770196337013) }, { SC_(37.0), SC_(3.5973435318931062473311547974114791) }, { SC_(38.0), SC_(3.6243705589201332743581818244385061) }, { SC_(39.0), SC_(3.6506863483938174848844976139121903) }, { SC_(40.0), SC_(3.6763273740348431259101386395532160) }
   } };
   static const std::array<std::array<T, 2>, 41> digamma_half_integers = { {
      { SC_(0.5), SC_(-1.9635100260214234794409763329987556) }, { SC_(1.5), SC_(0.036489973978576520559023667001244433) }, { SC_(2.5), SC_(0.70315664064524318722569033366791110) }, { SC_(3.5), SC_(1.1031566406452431872256903336679111) }, { SC_(4.5), SC_(1.3888709263595289015114046193821968) }, { SC_(5.5), SC_(1.6110931485817511237336268416044190) }, { SC_(6.5), SC_(1.7929113303999329419154450234226009) }, { SC_(7.5), SC_(1.9467574842460867880692911772687547) }, { SC_(8.5), SC_(2.0800908175794201214026245106020880) }, { SC_(9.5), SC_(2.1977378764029495331673303929550292) }, { SC_(10.5), SC_(2.3030010342976863752725935508497661) }, { SC_(11.5), SC_(2.3982391295357816133678316460878613) }, { SC_(12.5), SC_(2.4851956512749120481504403417400352) }, { SC_(13.5), SC_(2.5651956512749120481504403417400352) }, { SC_(14.5), SC_(2.6392697253489861222245144158141093) }, { SC_(15.5), SC_(2.7082352425903654325693420020210058) }, { SC_(16.5), SC_(2.7727513716226234970854710342790703) }, { SC_(17.5), SC_(2.8333574322286841031460770948851310) }, { SC_(18.5), SC_(2.8905002893715412460032199520279881) }, { SC_(19.5), SC_(2.9445543434255953000572740060820421) }, { SC_(20.5), SC_(2.9958363947076465821085560573640934) }, { SC_(21.5), SC_(3.0446168825125246308890438622421422) }, { SC_(22.5), SC_(3.0911285104195013750750903738700492) }, { SC_(23.5), SC_(3.1355729548639458195195348183144936) }, { SC_(24.5), SC_(3.1781261463533075216471943927825787) }, { SC_(25.5), SC_(3.2189424728839197665451535764560481) }, { SC_(26.5), SC_(3.2581581591584295704667222039070285) }, { SC_(27.5), SC_(3.2958940082150333440516278642843870) }, { SC_(28.5), SC_(3.3322576445786697076879915006480234) }, { SC_(29.5), SC_(3.3673453638769153217230792199462690) }, { SC_(30.5), SC_(3.4012436689616610844349436267259300) }, { SC_(31.5), SC_(3.4340305542075627237792059218078972) }, { SC_(32.5), SC_(3.4657765859535944698109519535539290) }, { SC_(33.5), SC_(3.4965458167228252390417211843231597) }, { SC_(34.5), SC_(3.5263965629914819554596316320843538) }, { SC_(35.5), SC_(3.5553820702378587670538345306350784) }, { SC_(36.5), SC_(3.5835510843223658093073556573956418) }, { SC_(37.5), SC_(3.6109483445963384120470816847929021) }, { SC_(38.5), SC_(3.6376150112630050787137483514595687) }, { SC_(39.5), SC_(3.6635890372370310527397223774335947) }, { SC_(40.5), SC_(3.6889054929332335843852919976867593) }
   } };

   add_data(digamma_data);
   add_data(digamma_root_data);
   add_data(digamma_small_data);
   add_data(digamma_neg_data);
   add_data(digamma_bugs);
   add_data(digamma_integers);
   add_data(digamma_half_integers);

   unsigned data_total = data.size();

   screen_data([](const std::vector<double>& v){  return boost::math::digamma(v[0]);  }, [](const std::vector<double>& v){ return v[1];  });


#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return gsl_sf_psi(v[0]);  }, [](const std::vector<double>& v){ return v[1];  });
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   screen_data([](const std::vector<double>& v){  return ::digamma(v[0]);  }, [](const std::vector<double>& v){ return v[1];  });
#endif

   unsigned data_used = data.size();
   std::string function = "digamma[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "digamma";

   double time = exec_timed_test([](const std::vector<double>& v){  return boost::math::digamma(v[0]);  });
   std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH))
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name());
#endif
   report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name());
   //
   // Boost again, but with promotion to long double turned off:
   //
#if !defined(COMPILER_COMPARISON_TABLES)
   if(sizeof(long double) != sizeof(double))
   {
      double time = exec_timed_test([](const std::vector<double>& v){  return boost::math::digamma(v[0], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_psi(v[0]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif
#if defined(TEST_RMATH) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return ::digamma(v[0]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "Rmath "  R_VERSION_STRING);
#endif

   return 0;
}

