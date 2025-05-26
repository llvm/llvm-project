//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning (disable : 4224)
#endif

#include <boost/math/special_functions/ellint_3.hpp>
#include <boost/array.hpp>
#include <boost/lexical_cast.hpp>
#include "../../test/table_type.hpp"
#include "table_helper.hpp"
#include "performance.hpp"
#include <iostream>

typedef double T;
#define SC_(x) static_cast<double>(x)
static const std::array<std::array<T, 4>, 65> data1 = { {
   { { SC_(1.0), SC_(-1.0), SC_(0.0), SC_(-1.557407724654902230506974807458360173087) } },
   { { SC_(0.0), SC_(-4.0), SC_(0.4), SC_(-4.153623371196831087495427530365430979011) } },
   { { SC_(0.0), SC_(8.0), SC_(-0.6), SC_(8.935930619078575123490612395578518914416) } },
   { { SC_(0.0), SC_(0.5), SC_(0.25), SC_(0.501246705365439492445236118603525029757890291780157969500480) } },
   { { SC_(0.0), SC_(0.5), SC_(0.0), SC_(0.5) } },
   { { SC_(-2.0), SC_(0.5), SC_(0.0), SC_(0.437501067017546278595664813509803743009132067629603474488486) } },
   { { SC_(0.25), SC_(0.5), SC_(0.0), SC_(0.510269830229213412212501938035914557628394166585442994564135) } },
   { { SC_(0.75), SC_(0.5), SC_(0.0), SC_(0.533293253875952645421201146925578536430596894471541312806165) } },
   { { SC_(0.75), SC_(0.75), SC_(0.0), SC_(0.871827580412760575085768367421866079353646112288567703061975) } },
   { { SC_(1.0), SC_(0.25), SC_(0.0), SC_(0.255341921221036266504482236490473678204201638800822621740476) } },
   { { SC_(2.0), SC_(0.25), SC_(0.0), SC_(0.261119051639220165094943572468224137699644963125853641716219) } },
   { { T(1023) / 1024, SC_(1.5), SC_(0.0), SC_(13.2821612239764190363647953338544569682942329604483733197131) } },
   { { SC_(0.5), SC_(-1.0), SC_(0.5), SC_(-1.228014414316220642611298946293865487807) } },
   { { SC_(0.5), SC_(1e+10), SC_(0.5), SC_(1.536591003599172091573590441336982730551e+10) } },
   { { SC_(-1e+05), SC_(10.0), SC_(0.75), SC_(0.0347926099493147087821620459290460547131012904008557007934290) } },
   { { SC_(-1e+10), SC_(10.0), SC_(0.875), SC_(0.000109956202759561502329123384755016959364346382187364656768212) } },
   { { SC_(-1e+10), SC_(1e+20), SC_(0.875), SC_(1.00000626665567332602765201107198822183913978895904937646809e15) } },
   { { SC_(-1e+10), T(1608) / 1024, SC_(0.875), SC_(0.0000157080616044072676127333183571107873332593142625043567690379) } },
   { { 1 - T(1) / 1024, SC_(1e+20), SC_(0.875), SC_(6.43274293944380717581167058274600202023334985100499739678963e21) } },
   { { SC_(50.0), SC_(0.125), SC_(0.25), SC_(0.196321043776719739372196642514913879497104766409504746018939) } },
   { { SC_(1.125), SC_(1.0), SC_(0.25), SC_(1.77299767784815770192352979665283069318388205110727241629752) } },
   { { SC_(1.125), SC_(10.0), SC_(0.25), SC_(0.662467818678976949597336360256848770217429434745967677192487) } },
   { { SC_(1.125), SC_(3.0), SC_(0.25), SC_(-0.142697285116693775525461312178015106079842313950476205580178) } },
   { { T(257) / 256, SC_(1.5), SC_(0.125), SC_(22.2699300473528164111357290313578126108398839810535700884237) } },
   { { T(257) / 256, SC_(21.5), SC_(0.125), SC_(-0.535406081652313940727588125663856894154526187713506526799429) } },
   // Bug cases from Rocco Romeo:
   { { SC_(-1E-170), boost::math::constants::pi<T>() / 4, SC_(1E-164), SC_(0.785398163397448309615660845819875721049292349843776455243736) } },
   { { SC_(-1E-170), boost::math::constants::pi<T>() / 4, SC_(-1E-164), SC_(0.785398163397448309615660845819875721049292349843776455243736) } },
   { { -ldexp(T(1.0), -52), boost::math::constants::pi<T>() / 4, SC_(0.9375), SC_(0.866032844934895872810905364370384153285798081574191920571016) } },
   { { -ldexp(T(1.0), -52), boost::math::constants::pi<T>() / 4, SC_(-0.9375), SC_(0.866032844934895872810905364370384153285798081574191920571016) } },
   { { std::numeric_limits<T>::max_exponent > 600 ? -ldexp(T(1), 604) : T(0), -ldexp(T(1), -816), ldexp(T(1), -510), std::numeric_limits<T>::max_exponent > 600 ? SC_(-2.28835573409367516299079046268930870596307631872422530813192e-246) : SC_(-2.28835573409367516299079046268930870596307631872422530813192e-246) } },
   { { std::numeric_limits<T>::max_exponent > 600 ? -ldexp(T(1), 604) : T(0), -ldexp(T(1), -816), -ldexp(T(1), -510), std::numeric_limits<T>::max_exponent > 600 ? SC_(-2.28835573409367516299079046268930870596307631872422530813192e-246) : SC_(-2.28835573409367516299079046268930870596307631872422530813192e-246) } },
   { { -ldexp(T(1), -622), -ldexp(T(1), -800), ldexp(T(1), -132), SC_(-1.49969681389563095481764443762806535353996169623910829793733e-241) } },
   { { -ldexp(T(1), -622), -ldexp(T(1), -800), -ldexp(T(1), -132), SC_(-1.49969681389563095481764443762806535353996169623910829793733e-241) } },
   { { -ldexp(T(1), -562), ldexp(T(1), -140), ldexp(T(1), -256), SC_(7.174648137343063403129495466444370592154941142407760751e-43) } },
   { { -ldexp(T(1), -562), -ldexp(T(1), -140), ldexp(T(1), -256), SC_(-7.17464813734306340312949546644437059215494114240776075e-43) } },
   { { ldexp(T(1), -688), -ldexp(T(1), -243), ldexp(T(1), -193), SC_(-7.07474928033336903711649944600608732865822749854620171e-74) } },
   { { -ldexp(T(1), -688), -ldexp(T(1), -243), ldexp(T(1), -193), SC_(-7.07474928033336903711649944600608732865822749854620171e-74) } },
   // Special cases where k = 0:
   { { SC_(0.5), SC_(1.0), SC_(0.0), SC_(1.17881507892743738986863357869566288974084658835353613038547) } },
   { { SC_(-0.5), SC_(1.0), SC_(0.0), SC_(0.888286691263535380266337576823783210424994266596287990733270) } },
   { { SC_(0.5), SC_(-1.0), SC_(0.0), SC_(-1.17881507892743738986863357869566288974084658835353613038547) } },
   { { SC_(-0.5), SC_(-1.0), SC_(0.0), SC_(-0.888286691263535380266337576823783210424994266596287990733270) } },
   // k == 0 and phi > pi/2:
   { { SC_(0.5), SC_(2.0), SC_(0.0), SC_(3.03379730757207227653600089552126882582809860566558143254794) } },
   { { SC_(-0.5), SC_(2.0), SC_(0.0), SC_(1.57453655812023739911111328195028658229986230310938753315640) } },
   { { SC_(0.5), SC_(-2.0), SC_(0.0), SC_(-3.03379730757207227653600089552126882582809860566558143254794) } },
   { { SC_(-0.5), SC_(-2.0), SC_(0.0), SC_(-1.57453655812023739911111328195028658229986230310938753315640) } },
   // Special cases where k = 1:
   { { SC_(0.5), SC_(1.0), SC_(1.0), SC_(1.4830998734200773326887632776553375078936815318419194718912351) } },
   { { SC_(-0.5), SC_(1.0), SC_(1.0), SC_(1.07048347329000030842347009377117215811122412769516781788253) } },
   { { SC_(0.5), SC_(-1.0), SC_(1.0), SC_(-1.4830998734200773326887632776553375078936815318419194718912) } },
   { { SC_(-0.5), SC_(-1.0), SC_(1.0), SC_(-1.07048347329000030842347009377117215811122412769516781788253) } },
   // special cases where v = 1:
   { { SC_(1.0), SC_(0.5), SC_(0.5), SC_(0.55225234291197632914658859230278152249148960801635386133501) } },
   { { SC_(1.0), SC_(-0.5), SC_(0.5), SC_(-0.55225234291197632914658859230278152249148960801635386133501) } },
   { { SC_(1.0), SC_(2.0), SC_(0.5), SC_(-2.87534521505997989921579168327307068134740792740155171368532) } },
   { { SC_(1.0), SC_(-2.0), SC_(0.5), SC_(2.87534521505997989921579168327307068134740792740155171368532) } },
   { { SC_(1.0), SC_(2.0), ldexp(T(1), -200), SC_(-2.18503986326151899164330610231368254343201774622766316456295) } },
   { { SC_(1.0), SC_(-2.0), ldexp(T(1), -200), SC_(2.18503986326151899164330610231368254343201774622766316456295) } },
   { { SC_(1.0), ldexp(T(1.0), -150), ldexp(T(1), -200), SC_(7.006492321624085354618647916449580656401309709382578858e-46) } },
   { { SC_(1.0), -ldexp(T(1.0), -150), ldexp(T(1), -200), SC_(-7.006492321624085354618647916449580656401309709382578858e-46) } },
   // Previously unsupported region with v > 1 and |phi| > PI/2, this is the only region
   // with high-ish error rates caused by argument reduction by Pi:
   { { SC_(20.0), ldexp(T(1647611), -19), SC_(0.5), SC_(0.000975940902692994840122139131147517258405256880370413541280) } },
   { { SC_(20.0), -ldexp(T(1647611), -19), SC_(0.5), SC_(-0.000975940902692994840122139131147517258405256880370413541280) } },
   { { T(1.0) + ldexp(T(1), -6), ldexp(T(889085), -19), SC_(0.5), SC_(-27.1647225624906589308619292363045712770651414487085887109197) } },
   { { T(1.0) + ldexp(T(1), -6), -ldexp(T(889085), -19), SC_(0.5), SC_(27.1647225624906589308619292363045712770651414487085887109197) } },
   // Phi = 0:
   { { SC_(1.0), SC_(0.0), SC_(0.5), SC_(0.0) } },
   { { SC_(-1.0), SC_(0.0), SC_(0.5), SC_(0.0) } },
   { { SC_(100.0), SC_(0.0), SC_(0.5), SC_(0.0) } },
   { { SC_(-100.0), SC_(0.0), SC_(0.5), SC_(0.0) } },
   } };


int main()
{
#include "ellint_pi3_data.ipp"
#include "ellint_pi3_large_data.ipp"

   add_data(data1);
   add_data(ellint_pi3_data);
   add_data(ellint_pi3_large_data);

   unsigned data_total = data.size();

   std::cout << "Screening boost data:\n";
   screen_data([](const std::vector<double>& v){  return boost::math::ellint_3(v[2], v[0], v[1]);  }, [](const std::vector<double>& v){ return v[3];  });


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   std::cout << "Screening libstdc++ data:\n";
      screen_data([](const std::vector<double>& v){  return std::tr1 ::ellint_3(v[2], -v[0], v[1]);  }, [](const std::vector<double>& v){ return v[3];  });
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   std::cout << "Screening libstdc++ data:\n";
      screen_data([](const std::vector<double>& v){  return gsl_sf_ellint_P(v[1], v[2], -v[0], GSL_PREC_DOUBLE);  }, [](const std::vector<double>& v){ return v[3];  });
#endif

   unsigned data_used = data.size();
   std::string function = "ellint_3[br](" + boost::lexical_cast<std::string>(data_used) + "/" + boost::lexical_cast<std::string>(data_total) + " tests selected)";
   std::string function_short = "ellint_3";

   double time;

   time = exec_timed_test([](const std::vector<double>& v){  return boost::math::ellint_3(v[2], v[0], v[1]);  });
   std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_LIBSTDCXX))
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name());
#endif
   report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name());
   //
   // Boost again, but with promotion to long double turned off:
   //
#if !defined(COMPILER_COMPARISON_TABLES)
   if(sizeof(long double) != sizeof(double))
   {
      time = exec_timed_test([](const std::vector<double>& v){  return boost::math::ellint_3(v[2], v[0], v[1], boost::math::policies::make_policy(boost::math::policies::promote_double<false>()));  });
      std::cout << time << std::endl;
#if !defined(COMPILER_COMPARISON_TABLES) && (defined(TEST_GSL) || defined(TEST_RMATH) || defined(TEST_LIBSTDCXX))
      report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, boost_name() + "[br]promote_double<false>");
#endif
      report_execution_time(time, std::string("Compiler Comparison on ") + std::string(platform_name()), function_short, compiler_name() + std::string("[br]") + boost_name() + "[br]promote_double<false>");
   }
#endif


#if defined(TEST_LIBSTDCXX) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return std::tr1::ellint_3(v[2], -v[0], v[1]);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "tr1/cmath");
#endif
#if defined(TEST_GSL) && !defined(COMPILER_COMPARISON_TABLES)
   time = exec_timed_test([](const std::vector<double>& v){  return gsl_sf_ellint_P(v[1], v[2], -v[0], GSL_PREC_DOUBLE);  });
   std::cout << time << std::endl;
   report_execution_time(time, std::string("Library Comparison with ") + std::string(compiler_name()) + std::string(" on ") + platform_name(), function, "GSL " GSL_VERSION);
#endif

   return 0;
}

