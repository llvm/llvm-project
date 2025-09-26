//  Copyright John Maddock 2013
//  Copyright Christopher Kormanyos 2013.
//  Copyright Paul A. Bristow 2013.

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#  pragma warning(disable : 4127) // conditional expression is constant.
#  pragma warning(disable : 4512) // assignment operator could not be generated.
#  pragma warning(disable : 4996) // use -D_SCL_SECURE_NO_WARNINGS.
#endif

//#include <pch_light.hpp> // commented out during testing.

//#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/cstdint.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/airy.hpp>
#include <boost/math/tools/test.hpp>

#include <boost/math/concepts/real_concept.hpp> // for real_concept

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/tools/floating_point_comparison.hpp>

#include <typeinfo>
#include <iostream>
#include <iomanip>

// #include <boost/math/tools/
//
// DESCRIPTION:
// ~~~~~~~~~~~~
//
// This file tests the functions that evaluate zeros (or roots) of Bessel, Neumann and Airy functions.

// Spot tests which compare our results with selected values computed
// using the online special function calculator at functions.wolfram.com,
// and values generated with Boost.Multiprecision at about 1000-bit or 100 decimal digits precision.

// We are most grateful for the invaluable
// Weisstein, Eric W. "Bessel Function Zeros." From MathWorld--A Wolfram Web Resource.
// http://mathworld.wolfram.com/BesselFunctionZeros.html
// and the newer http://www.wolframalpha.com/

// See also NIST Handbook of Mathematical Function http://dlmf.nist.gov/10.21
/*
Tests of cyl Bessel and cyl Neumann zeros.
==========================================

The algorithms for estimating the roots of both cyl. Bessel
as well as cyl. Neumann have the same cross-over points,
and also use expansions that have the same order of approximation.

Therefore, tests will be equally effective for both functions in the regions of order.

I have recently changed a critical cross-over in the algorithms
from a value of order of 1.2 to a value of order of 2.2.
In addition, there is a critical cross-over in the rank of the
zero from rank 1 to rank 2 and above. The first zero is
treated differently than the remaining ones.

The test cover various regions of order,
each one tested with several zeros:
  * Order 219/100: This checks a region just below a critical cutoff.
  * Order 221/100: This checks a region just above a critical cutoff.
  * Order 0: Something always tends to go wrong at zero.
  * Order 1/1000: A small order.
  * Order 71/19: Merely an intermediate order.
  * Order 7001/19: A medium-large order, small enough to retain moderate efficiency of calculation.

There are also a few selected high zeros
such as the 1000th zero for a few modest orders such as 71/19, etc.

Tests of Airy zeros.
====================

The Airy zeros algorithms use tabulated values for the first 10 zeros,
whereby algorithms are used for rank 11 and higher.
So testing the zeros of Ai and Bi from 1 through 20 handles
this cross-over nicely.

In addition, the algorithms for the estimates of the zeros
become increasingly accurate for larger, negative argument.

On the other hand, the zeros become increasingly close
for large, negative argument. So another nice test
involves testing pairs of zeros for different orders of
magnitude of the zeros, to insure that the program
properly resolves very closely spaced zeros.
*/


template <class RealType>
void test_bessel_zeros(RealType)
{
  // Basic sanity checks for finding zeros of Bessel and Airy function.
  // where template parameter RealType can be float, double, long double,
  // or real_concept, a prototype for user-defined floating-point types.

  // Parameter RealType is only used to communicate the RealType, float, double...
  // and is an arbitrary zero for all tests.
   RealType tolerance = 5 * (std::max)(
     static_cast<RealType>(boost::math::tools::epsilon<long double>()),
     boost::math::tools::epsilon<RealType>());
   std::cout << "Tolerance for type " << typeid(RealType).name()  << " is " << tolerance << "." << std::endl;
   //
   // An extra fudge factor for real_concept which has a less accurate tgamma:
   RealType tolerance_tgamma_extra = std::numeric_limits<RealType>::is_specialized ? 1 : 15;

   // http://www.wolframalpha.com/
   using boost::math::cyl_bessel_j_zero; // (nu, j)
   using boost::math::isnan;

  BOOST_MATH_CHECK_THROW(cyl_bessel_j_zero(static_cast<RealType>(0), 0), std::domain_error);
  BOOST_MATH_CHECK_THROW(cyl_bessel_j_zero(static_cast<RealType>(-1.5), 0), std::domain_error);

  // Abuse with infinity and max.
  if (std::numeric_limits<RealType>::has_infinity)
  {
    //BOOST_CHECK_EQUAL(cyl_bessel_j_zero(static_cast<RealType>(std::numeric_limits<RealType>::infinity()), 1),
    //  static_cast<RealType>(std::numeric_limits<RealType>::infinity()) );
    // unknown location(0): fatal error in "test_main":
    // class boost::exception_detail::clone_impl<struct boost::exception_detail::error_info_injector<class std::domain_error> >:
    // Error in function boost::math::cyl_bessel_j_zero<double>(double, int): Order argument is 1.#INF, but must be finite >= 0 !
    // Note that the reported type long double is not the type of the original call RealType,
    // but the promoted value, here long double, if applicable.
    BOOST_MATH_CHECK_THROW(cyl_bessel_j_zero(static_cast<RealType>(std::numeric_limits<RealType>::infinity()), 1),
     std::domain_error);
    BOOST_MATH_CHECK_THROW(cyl_bessel_j_zero(static_cast<RealType>(-std::numeric_limits<RealType>::infinity()), 1),
     std::domain_error);

  }
  // Test with maximum value of v that will cause evaluation error
  //BOOST_MATH_CHECK_THROW(cyl_bessel_j_zero(boost::math::tools::max_value<RealType>(), 1), std::domain_error);
  // unknown location(0): fatal error in "test_main":
  // class boost::exception_detail::clone_impl<struct boost::exception_detail::error_info_injector<class boost::math::evaluation_error> >:
  // Error in function boost::math::bessel_jy<double>(double,double): Order of Bessel function is too large to evaluate: got 3.4028234663852886e+038

  BOOST_MATH_CHECK_THROW(cyl_bessel_j_zero(boost::math::tools::max_value<RealType>(), 1), boost::math::evaluation_error);

  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(boost::math::tools::min_value<RealType>(), 1),
    static_cast<RealType>(2.4048255576957727686216318793264546431242449091460L), tolerance);

  BOOST_CHECK_CLOSE_FRACTION(-cyl_bessel_j_zero(boost::math::tools::min_value<RealType>(), 1),
    static_cast<RealType>(-2.4048255576957727686216318793264546431242449091460L), tolerance);

  // Checks on some spot values.

  // http://mathworld.wolfram.com/BesselFunctionZeros.html provides some spot values,
  // evaluation at 50 decimal digits using WoldramAlpha.

  /* Table[N[BesselJZero[0, n], 50], {n, 1, 5, 1}]
  n |
  1 | 2.4048255576957727686216318793264546431242449091460
  2 | 5.5200781102863106495966041128130274252218654787829
  3 | 8.6537279129110122169541987126609466855657952312754
  4 | 11.791534439014281613743044911925458922022924699695
  5 | 14.930917708487785947762593997388682207915850115633
  */
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(0), 1), static_cast<RealType>(2.4048255576957727686216318793264546431242449091460L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(0), 2), static_cast<RealType>(5.5200781102863106495966041128130274252218654787829L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(0), 3), static_cast<RealType>(8.6537279129110122169541987126609466855657952312754L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(0), 4), static_cast<RealType>(11.791534439014281613743044911925458922022924699695L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(0), 5), static_cast<RealType>(14.930917708487785947762593997388682207915850115633L), tolerance);

  { // Same test using the multiple zeros version.
    std::vector<RealType> zeros;
    cyl_bessel_j_zero(static_cast<RealType>(0.0), 1, 3, std::back_inserter(zeros) );
    BOOST_CHECK_CLOSE_FRACTION(zeros[0], static_cast<RealType>(2.4048255576957727686216318793264546431242449091460L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(zeros[1], static_cast<RealType>(5.5200781102863106495966041128130274252218654787829L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(zeros[2], static_cast<RealType>(8.6537279129110122169541987126609466855657952312754L), tolerance);
  }
  // 1/1000 a small order.
  /* Table[N[BesselJZero[1/1000, n], 50], {n, 1, 4, 1}]
  n |
1 | 2.4063682720422009275161970278295108254321633626292
2 | 5.5216426858401848664019464270992222126391378706092
3 | 8.6552960859298799453893840513333150237193779482071
4 | 11.793103797689738596231262077785930962647860975357

Table[N[BesselJZero[1/1000, n], 50], {n, 10, 20, 1}]
n |
10 | 30.636177039613574749066837922778438992469950755736
11 | 33.777390823252864715296422192027816488172667994611
12 | 36.918668992567585467000743488690258054442556198147
13 | 40.059996426251227493370316149043896483196561190610
14 | 43.201362392820317233698309483240359167380135262681
15 | 46.342759065846108737848449985452774243376260538634
16 | 49.484180603489984324820981438067325210499739716337
17 | 52.625622557085775090390071484188995092211215108718
18 | 55.767081479279692992978326069855684800673801918763
19 | 58.908554657366270044071505013449016741804538135905
20 | 62.050039927521244984641179233170843941940575857282

*/

  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(1)/1000, 1), static_cast<RealType>(2.4063682720422009275161970278295108254321633626292L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(1)/1000, 4), static_cast<RealType>(11.793103797689738596231262077785930962647860975357L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(1)/1000, 10), static_cast<RealType>(30.636177039613574749066837922778438992469950755736L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(1)/1000, 20), static_cast<RealType>(62.050039927521244984641179233170843941940575857282L), tolerance);

    /*
  Table[N[BesselJZero[1, n], 50], {n, 1, 4, 1}]
  n |
  1 | 3.8317059702075123156144358863081607665645452742878
  2 | 7.0155866698156187535370499814765247432763115029142
  3 | 10.173468135062722077185711776775844069819512500192
  4 | 13.323691936314223032393684126947876751216644731358
  */

  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(1), 1), static_cast<RealType>(3.8317059702075123156144358863081607665645452742878L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(1), 2), static_cast<RealType>(7.0155866698156187535370499814765247432763115029142L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(1), 3), static_cast<RealType>(10.173468135062722077185711776775844069819512500192L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(1), 4), static_cast<RealType>(13.323691936314223032393684126947876751216644731358L), tolerance);

  /*
  Table[N[BesselJZero[5, n], 50], {n, 1, 5, 1}]
  n |
  1 | 8.7714838159599540191228671334095605629810770148974
  2 | 12.338604197466943986082097644459004412683491122239
  3 | 15.700174079711671037587715595026422501346662246893
  4 | 18.980133875179921120770736748466932306588828411497
  5 | 22.217799896561267868824764947529187163096116704354
*/

  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(5), 1), static_cast<RealType>(8.7714838159599540191228671334095605629810770148974L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(5), 2), static_cast<RealType>(12.338604197466943986082097644459004412683491122239L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(5), 3), static_cast<RealType>(15.700174079711671037587715595026422501346662246893L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(5), 4), static_cast<RealType>(18.980133875179921120770736748466932306588828411497L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(5), 5), static_cast<RealType>(22.217799896561267868824764947529187163096116704354L), tolerance);

  // An intermediate order
  /*
  Table[N[BesselJZero[71/19, n], 50], {n, 1, 20, 1}]

  7.27317519383164895031856942622907655889631967016227,
  10.7248583088831417325361727458514166471107495990849,
  14.0185045994523881061204595580426602824274719315813,
  17.2524984591704171821624871665497773491959038386104,
  20.4566788740445175951802340838942858854605020778141,
  23.6436308971423452249455142271473195998540517250404,
  26.8196711402550877454213114709650192615223905192969,
  29.9883431174236747426791417966614320438788681941419,
  33.1517968976905208712508624699734452654447919661140,
  36.3114160002162074157243540350393860813165201842005,
  39.4681324675052365879451978080833378877659670320292,
  42.6225978013912364748550348312979540188444334802274,
  45.7752814645368477533902062078067265814959500124386,
  48.9265304891735661983677668174785539924717398947994,
  52.0766070453430027942797460418789248768734780634716,
  55.2257129449125713935942243278172656890590028901917,
  58.3740061015388864367751881504390252017351514189321,
  61.5216118730009652737267426593531362663909441035715,
  64.6686310537909303683464822148736607945659662871596,
  67.8151456196962909255567913755559511651114605854579
  */
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(71)/19, 1), static_cast<RealType>(7.27317519383164895031856942622907655889631967016227L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(71)/19, 4), static_cast<RealType>(17.2524984591704171821624871665497773491959038386104L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(71)/19, 10), static_cast<RealType>(36.3114160002162074157243540350393860813165201842005L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(71)/19, 20), static_cast<RealType>(67.8151456196962909255567913755559511651114605854579L), tolerance);
  /*

  Table[N[BesselJZero[7001/19, n], 50], {n, 1, 2, 1}]

1 | 381.92201523024489386917204470434842699154031135348
2 | 392.17508657648737502651299853099852567001239217724

Table[N[BesselJZero[7001/19, n], 50], {n, 19, 20, 1}]

19 | 491.67809669154347398205298745712766193052308172472
20 | 496.39435037938252557535375498577989720272298310802

  */
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(7001)/19, 1), static_cast<RealType>(381.92201523024489386917204470434842699154031135348L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(7001)/19, 2), static_cast<RealType>(392.17508657648737502651299853099852567001239217724L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(7001)/19, 20), static_cast<RealType>(496.39435037938252557535375498577989720272298310802L), tolerance);

  // Some non-integral tests.
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(3.73684210526315789473684210526315789473684210526315789L), 1), static_cast<RealType>(7.273175193831648950318569426229076558896319670162279791988152000556091140599946365217211157877052381L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(3.73684210526315789473684210526315789473684210526315789L), 20), static_cast<RealType>(67.81514561969629092555679137555595116511146058545787883557679231060644931096494584364894743334132014L), tolerance);

  // Some non-integral tests in 'tough' regions.
  // Order 219/100: This checks a region just below a critical cutoff.
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(219)/100, 1), static_cast<RealType>(5.37568854370623186731066365697341253761466705063679L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(219)/100, 2), static_cast<RealType>(8.67632060963888122764226633146460596009874991130394L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(219)/100, 20), static_cast<RealType>(65.4517712237598926858973399895944886397152223643028L), tolerance);
  // Order 221/100: This checks a region just above a critical cutoff.
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(221)/100, 1), static_cast<RealType>(5.40084731984998184087380740054933778965260387203942L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(221)/100, 2), static_cast<RealType>(8.70347906513509618445695740167369153761310106851599L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(221)/100, 20), static_cast<RealType>(65.4825314862621271716158606625527548818843845600782L), tolerance);

  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(7001)/19, 1), static_cast<RealType>(381.922015230244893869172044704348426991540311353476L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(7001)/19, 2), static_cast<RealType>(392.175086576487375026512998530998525670012392177242L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(7001)/19, 20), static_cast<RealType>(496.394350379382525575353754985779897202722983108025L), tolerance);

  // Zero'th cases.
  BOOST_MATH_CHECK_THROW(boost::math::cyl_bessel_j_zero(static_cast<RealType>(0), 0), std::domain_error); // Zero'th zero of J0(x).
  BOOST_CHECK(boost::math::cyl_bessel_j_zero(static_cast<RealType>(1), 0) == 0); // Zero'th zero of J1(x).
  BOOST_CHECK(boost::math::cyl_bessel_j_zero(static_cast<RealType>(2), 0) == 0); // Zero'th zero of J2(x).


  // Negative order cases.
  // Table[N[BesselJZero[-39, n], 51], {n, 1, 20, 1}]

  //  45.597624026432090522996531982029164361723758769649

    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(-39), 1), static_cast<RealType>(45.597624026432090522996531982029164361723758769649L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(-39), 2), static_cast<RealType>(50.930599960211455519691708196247756810739999585797L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(-39), 4), static_cast<RealType>(59.810708207036942166964205243063534405954475825070L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(-39), 10), static_cast<RealType>(82.490310026657839398140015188318580114553721419436L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(-39), 15), static_cast<RealType>(99.886172950858129702511715161572827825877395517083L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(-39), 20), static_cast<RealType>(116.73117751356457774415638043701531989536641098359L), tolerance);

   // Table[N[BesselJZero[-39 - (1/3), n], 51], {n, 1, 20, 1}]

   // 43.803165820025277290601047312311146608776920513241
   // 49.624678304306778749502719837270544976331123155017

    RealType v = static_cast<RealType>(-39);

    v -= boost::math::constants::third<RealType>();

   // BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(v, 1), static_cast<RealType>(43.803165820025277290601047312311146608776920513241L), tolerance);
  //  BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(-39) - static_cast<RealType>(1)/3, 1), static_cast<RealType>(43.803165820025277290601047312311146608776920513241L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(v, 2), static_cast<RealType>(49.624678304306778749502719837270544976331123155017L), tolerance * 4);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(-39) - static_cast<RealType>(0.333333333333333333333333333333333333333333333L), 5), static_cast<RealType>(62.911281619408963609400485687996804820400102193455L), tolerance * 4);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(-39) - static_cast<RealType>(0.333333333333333333333333333333333333333333333L), 10), static_cast<RealType>(81.705998611506506523381866527389118594062841737382L), tolerance * 4);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(static_cast<RealType>(-39) - static_cast<RealType>(0.333333333333333333333333333333333333333333333L), 20), static_cast<RealType>(116.05368337161392034833932554892349580959931408963L), tolerance * 4);


  // Table[N[BesselJZero[-1/3, n], 51], {n, 1, 20, 1}]
    // 1.86635085887389517154698498025466055044627209492336
    // 4.98785323143515872689263163814239463653891121063534
    v = - boost::math::constants::third<RealType>();
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(v, 1), static_cast<RealType>(1.86635085887389517154698498025466055044627209492336L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(v, 2), static_cast<RealType>(4.98785323143515872689263163814239463653891121063534L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(v, 5), static_cast<RealType>(14.4037758801360172217813556328092353168458341692115L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(v, 20), static_cast<RealType>(61.5239847181314647255554392599009248210564008120358L), tolerance);

  // Table[N[BesselJZero[-3 - (999999/1000000), n], 51], {n, 1, 20, 1}]
    // 0.666908567552422764702292353801313970109968787260547
    //7.58834489983121936102504707121493271448122800440112

    std::cout.precision(2 + std::numeric_limits<RealType>::digits * 3010/10000);
    v = -static_cast<RealType>(3);
    //std::cout << "v = " << v << std::endl;
    RealType d = static_cast<RealType>(999999)/1000000; // Value very near to unity.
    //std::cout << "d = " << d << std::endl;
    v -= d;
   // std::cout << "v = " << v << std::endl; //   v = -3.9999989999999999

    // 1st is much less accurate.
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(v, 1), static_cast<RealType>(0.666908567552422764702292353801313970109968787260547L), tolerance * 500000);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(v, 2), static_cast<RealType>(7.58834489983121936102504707121493271448122800440112L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(v, 5), static_cast<RealType>(17.6159678964372778134202353240221384945968807948928L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(v, 20), static_cast<RealType>(65.0669968910414433560468307554730940098734494938136L), tolerance);


    v = -static_cast<RealType>(1)/81799; // Largish prime, so small value.
    // std::cout << "v = " << v << std::endl; // v = -1.22251e-005

   // Table[N[BesselJZero[-1/81799, n], 51], {n, 1, 20, 1}]

    // 2.40480669570616362235270726259606288441474232101937
    //5.52005898213436490056801834487410496538653938730884
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(v, 1), static_cast<RealType>(2.40480669570616362235270726259606288441474232101937L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(v, 2), static_cast<RealType>(5.52005898213436490056801834487410496538653938730884L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(v, 5), static_cast<RealType>(14.9308985160466385806685583210609848822943295303368L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_bessel_j_zero(v, 20), static_cast<RealType>(62.0484499877253314338528593349200129641402661038743L), tolerance);

  // Confirm that negative m throws domain_error.
  BOOST_MATH_CHECK_THROW(boost::math::cyl_bessel_j_zero(static_cast<RealType>(0), -1), std::domain_error);
  // unknown location(0): fatal error in "test_main":
  // class boost::exception_detail::clone_impl<struct boost::exception_detail::error_info_injector<class std::domain_error> >:
  // Error in function boost::math::cyl_bessel_j_zero<double>(double, int): Requested the -1'th zero, but must be > 0 !

  // Confirm that a C-style ignore_all policy returns NaN for bad input.
  typedef boost::math::policies::policy<
    boost::math::policies::domain_error<boost::math::policies::ignore_error>,
    boost::math::policies::overflow_error<boost::math::policies::ignore_error>,
    boost::math::policies::underflow_error<boost::math::policies::ignore_error>,
    boost::math::policies::denorm_error<boost::math::policies::ignore_error>,
    boost::math::policies::pole_error<boost::math::policies::ignore_error>,
    boost::math::policies::evaluation_error<boost::math::policies::ignore_error>
              > ignore_all_policy;

  if (std::numeric_limits<RealType>::has_quiet_NaN)
  {
    BOOST_MATH_CHECK_THROW(cyl_bessel_j_zero(static_cast<RealType>(std::numeric_limits<RealType>::quiet_NaN()), 1), std::domain_error);
    // Check that bad m returns NaN if policy is no throws.
    BOOST_CHECK((boost::math::isnan<RealType>)(cyl_bessel_j_zero(std::numeric_limits<RealType>::quiet_NaN(), 1, ignore_all_policy())) );
    BOOST_MATH_CHECK_THROW(boost::math::cyl_bessel_j_zero(static_cast<RealType>(std::numeric_limits<RealType>::quiet_NaN()), -1), std::domain_error);
  }
  else
  { // real_concept bad m returns zero.
    //std::cout << boost::math::cyl_bessel_j_zero(static_cast<RealType>(0), -1, ignore_all_policy()) << std::endl; // 0 for real_concept.
      BOOST_CHECK_EQUAL(boost::math::cyl_bessel_j_zero(static_cast<RealType>(0), -1, ignore_all_policy() ), 0);
  }

  if (std::numeric_limits<RealType>::has_infinity)
  {
    BOOST_MATH_CHECK_THROW(cyl_bessel_j_zero(std::numeric_limits<RealType>::infinity(), 0), std::domain_error);
    BOOST_MATH_CHECK_THROW(cyl_bessel_j_zero(std::numeric_limits<RealType>::infinity(), 1), std::domain_error);
    // Check that NaN is returned if error ignored.
    BOOST_CHECK((boost::math::isnan<RealType>)(cyl_bessel_j_zero(std::numeric_limits<RealType>::infinity(), 1, ignore_all_policy())) );
  }

  // Tests of cyc_neumann zero function (BesselYZero in Wolfram) for spot values.
  /*
  Table[N[BesselYZero[0, n], 50], {n, 1, 5, 1}]
n |
1 | 0.89357696627916752158488710205833824122514686193001
2 | 3.9576784193148578683756771869174012814186037655636
3 | 7.0860510603017726976236245968203524689715103811778
4 | 10.222345043496417018992042276342187125994059613181
5 | 13.361097473872763478267694585713786426579135174880

Table[N[BesselYZero[0, n], 50], {n, 1, 5, 1}]

n | 
1 | 0.89357696627916752158488710205833824122514686193001
2 | 3.9576784193148578683756771869174012814186037655636
3 | 7.0860510603017726976236245968203524689715103811778
4 | 10.222345043496417018992042276342187125994059613181
5 | 13.361097473872763478267694585713786426579135174880

So K == Y

Table[N[BesselYZero[1, n], 50], {n, 1, 5, 1}]
n |
1 | 2.1971413260310170351490335626989662730530183315003
2 | 5.4296810407941351327720051908525841965837574760291
3 | 8.5960058683311689264296061801639678511029215669749
4 | 11.749154830839881243399421939922350714301165983279
5 | 14.897442128336725378844819156429870879807150630875

Table[N[BesselYZero[2, n], 50], {n, 1, 5, 1}]
n |
1 | 3.3842417671495934727014260185379031127323883259329
2 | 6.7938075132682675382911671098369487124493222183854
3 | 10.023477979360037978505391792081418280789658279097
4 | 13.209986710206416382780863125329852185107588501072
5 | 16.378966558947456561726714466123708444627678549687

*/
  // Some simple integer values.

  using boost::math::cyl_neumann_zero;
  // Bad rank m.
  BOOST_MATH_CHECK_THROW(cyl_neumann_zero(static_cast<RealType>(0), 0), std::domain_error); // 
  BOOST_MATH_CHECK_THROW(cyl_neumann_zero(static_cast<RealType>(0), -1), std::domain_error);

  if (std::numeric_limits<RealType>::has_quiet_NaN)
  {
    BOOST_MATH_CHECK_THROW(cyl_neumann_zero(std::numeric_limits<RealType>::quiet_NaN(), 1), std::domain_error);
    BOOST_MATH_CHECK_THROW(cyl_neumann_zero(static_cast<RealType>(0), -1), std::domain_error);
  }
  if (std::numeric_limits<RealType>::has_infinity)
  {
    BOOST_MATH_CHECK_THROW(cyl_neumann_zero(std::numeric_limits<RealType>::infinity(), 2), std::domain_error);
    BOOST_MATH_CHECK_THROW(cyl_neumann_zero(static_cast<RealType>(0), -1), std::domain_error);
  }
  // else no infinity tests.

  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(0), 1), static_cast<RealType>(0.89357696627916752158488710205833824122514686193001L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(1), 2), static_cast<RealType>(5.4296810407941351327720051908525841965837574760291L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(2), 3), static_cast<RealType>(10.023477979360037978505391792081418280789658279097L), tolerance);

  /*
  Table[N[BesselYZero[3, n], 50], {n, 1, 5, 1}]
  1 | 4.5270246611496438503700268671036276386651555486109
  2 | 8.0975537628604907044022139901128042290432231369075
  3 | 11.396466739595866739252048190629504945984969192535
  4 | 14.623077742393873174076722507725200649352970569915
  5 | 17.818455232945520262553239064736739443380352162752
  */

    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(3), 1), static_cast<RealType>(4.5270246611496438503700268671036276386651555486109L), tolerance * 2);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(3), 2), static_cast<RealType>(8.0975537628604907044022139901128042290432231369075L), tolerance * 2);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(3), 3), static_cast<RealType>(11.396466739595866739252048190629504945984969192535L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(3), 4), static_cast<RealType>(14.623077742393873174076722507725200649352970569915L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(3), 5), static_cast<RealType>(17.818455232945520262553239064736739443380352162752L), tolerance);


    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3), 1), static_cast<RealType>(4.5270246611496438503700268671036276386651555486109L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3), 2), static_cast<RealType>(8.0975537628604907044022139901128042290432231369075L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3), 3), static_cast<RealType>(11.396466739595866739252048190629504945984969192535L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3), 4), static_cast<RealType>(14.623077742393873174076722507725200649352970569915L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3), 5), static_cast<RealType>(17.818455232945520262553239064736739443380352162752L), tolerance);

    /*
      Table[N[BesselYZero[-5/2, n], 50], {n, 1, 5, 1}]
      n | y_(-2.5000000000000000000000000000000000000000000000000, n)
      1 | 5.7634591968945497914064666539527350764090876841674
      2 | 9.0950113304763551563376983279896952524009293663831
      3 | 12.322940970566582051969567925329726061189423834915
      4 | 5.7634591968945497914064666539527350764090876841674
      5 | 9.0950113304763551563376983279896952524009293663831
    */

    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-2.5), 1), static_cast<RealType>(5.7634591968945497914064666539527350764090876841674L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-2.5), 2), static_cast<RealType>(9.0950113304763551563376983279896952524009293663831L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-2.5), 3), static_cast<RealType>(12.322940970566582051969567925329726061189423834915L), tolerance);
    //BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-2.5), 4), static_cast<RealType>(5.7634591968945497914064666539527350764090876841674L), tolerance);
    //BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-2.5), 5), static_cast<RealType>(9.0950113304763551563376983279896952524009293663831L), tolerance);
    
    { // Repeat rest using multiple zeros version.
    std::vector<RealType> zeros;
    cyl_neumann_zero(static_cast<RealType>(0.0), 1, 3, std::back_inserter(zeros) );
    BOOST_CHECK_CLOSE_FRACTION(zeros[0], static_cast<RealType>(0.89357696627916752158488710205833824122514686193001L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(zeros[1], static_cast<RealType>(3.9576784193148578683756771869174012814186037655636L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(zeros[2], static_cast<RealType>(7.0860510603017726976236245968203524689715103811778L), tolerance);
  }
  // Order 0: Something always tends to go wrong at zero.

  /* Order 219/100: This checks accuracy in a region just below a critical cutoff.

  Table[N[BesselKZero[219/100, n], 50], {n, 1, 20, 4}]
1 | 3.6039149425338727979151181355741147312162055042157
5 | 16.655399111666833825247894251535326778980614938275
9 | 29.280564448169163756478439692311605757712873534942
13 | 41.870269811145814760551599481942750124112093564643
17 | 54.449180021209532654553613813754733514317929678038
  */

  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(219)/100, 1), static_cast<RealType>(3.6039149425338727979151181355741147312162055042157L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(219)/100, 5), static_cast<RealType>(16.655399111666833825247894251535326778980614938275L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(219)/100, 17), static_cast<RealType>(54.449180021209532654553613813754733514317929678038L), tolerance);

  /* Order 221/100: This checks a region just above a critical cutoff.
  Table[N[BesselYZero[220/100, n], 50], {n, 1, 20, 5}]
  1 | 3.6154383428745996706772556069431792744372398748425
  6 | 19.833435100254138641131431268153987585842088078470
  11 | 35.592602956438811360473753622212346081080817891225
  16 | 51.320322762482062633162699745957897178885350674038
  */

  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(220)/100, 1), static_cast<RealType>(3.6154383428745996706772556069431792744372398748425L), 2 * tolerance);
  // Note * 2 tolerance needed - using cpp_dec_float_50 it computes exactly, probably because of extra guard digits in multiprecision decimal version.
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(220)/100, 6), static_cast<RealType>(19.833435100254138641131431268153987585842088078470L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(220)/100, 11), static_cast<RealType>(35.592602956438811360473753622212346081080817891225L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(220)/100, 16), static_cast<RealType>(51.320322762482062633162699745957897178885350674038L), tolerance);

  /*  Order 1/1000: A small order.
  Table[N[BesselYZero[1/1000, n], 50], {n, 1, 20, 5}]
  1 | 0.89502371604431360670577815537297733265776195646969
  6 | 16.502492490954716850993456703662137628148182892787
  11 | 32.206774708309182755790609144739319753463907110990
  16 | 47.913467031941494147962476920863688176374357572509
  */

  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(1)/1000, 1), static_cast<RealType>(0.89502371604431360670577815537297733265776195646969L), 2 * tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(1)/1000, 6), static_cast<RealType>(16.5024924909547168509934567036621376281481828927870L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(1)/1000, 11), static_cast<RealType>(32.206774708309182755790609144739319753463907110990L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(1)/1000, 16), static_cast<RealType>(47.913467031941494147962476920863688176374357572509L), tolerance);

  /* Order 71/19: Merely an intermediate order.
  Table[N[BesselYZero[71/19, n], 50], {n, 1, 20, 5}]
  1 | 5.3527167881149432911848659069476821793319749146616
  6 | 22.051823727778538215953091664153117627848857279151
  11 | 37.890091170552491176745048499809370107665221628364
  16 | 53.651270581421816017744203789836444968181687858095
  */
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(71)/19, 1), static_cast<RealType>(5.3527167881149432911848659069476821793319749146616L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(71)/19, 6), static_cast<RealType>(22.051823727778538215953091664153117627848857279151L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(71)/19, 11), static_cast<RealType>(37.890091170552491176745048499809370107665221628364L), tolerance);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(71)/19, 16), static_cast<RealType>(53.651270581421816017744203789836444968181687858095L), tolerance);

  /* Order 7001/19: A medium-large order, small enough to retain moderate efficiency of calculation.

  Table[N[BesselYZero[7001/19, n], 50], {n, 1}]
   1 | 375.18866334770357669101711932706658671250621098115

  Table[N[BesselYZero[7001/19, n], 50], {n, 2}]
  Standard computation time exceeded :-(
  */
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(7001)/19, 1), static_cast<RealType>(375.18866334770357669101711932706658671250621098115L), tolerance);

 /* A high zero such as the 1000th zero for a modest order such as 71/19.
   Table[N[BesselYZero[71/19, n], 50], {n, 1000}]
   Standard computation time exceeded :-(
 */

  /*
  Test Negative orders cyl_neumann.

  Table[N[BesselYZero[-1, n], 50], {n, 1, 10, 1}]
  1 | 2.1971413260310170351490335626989662730530183315003
  2 | 5.4296810407941351327720051908525841965837574760291
  3 | 8.5960058683311689264296061801639678511029215669749
  4 | 11.749154830839881243399421939922350714301165983279
  5 | 14.897442128336725378844819156429870879807150630875
  6 | 18.043402276727855564304555507889508902163088324834
  7 | 21.188068934142213016142481528685423196935024604904
  8 | 24.331942571356912035992944051850129651414333340303
  9 | 27.475294980449223512212285525410668235700897307021
  10 | 30.618286491641114715761625696447448310277939570868
  11 | 33.761017796109325692471759911249650993879821495802
  16 | 49.472505679924095824128003887609267273294894411716
*/

  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-1), 1), static_cast<RealType>(2.1971413260310170351490335626989662730530183315003L), tolerance * 3);
  // Note this test passes at tolerance for float, double and long double, but fails for real_concept if tolerance <= 2.
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-1), 6), static_cast<RealType>(18.043402276727855564304555507889508902163088324834L), tolerance * 3);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-1), 11), static_cast<RealType>(33.761017796109325692471759911249650993879821495802L), tolerance * 3);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-1), 16), static_cast<RealType>(49.472505679924095824128003887609267273294894411716L), tolerance * 3);

 /*
  Table[N[BesselYZero[-2, n], 50], {n, 1, 20, 5}]
  1 | 3.3842417671495934727014260185379031127323883259329
  6 | 19.539039990286384411511740683423888947393156497603
  11 | 35.289793869635804143323234828826075805683602368473
  16 | 51.014128749483902310217774804582826908060740157564
  */

  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-2), 1), static_cast<RealType>(3.3842417671495934727014260185379031127323883259329L), tolerance * 3);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-2), 6), static_cast<RealType>(19.539039990286384411511740683423888947393156497603L), tolerance * 3);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-2), 11), static_cast<RealType>(35.289793869635804143323234828826075805683602368473L), tolerance * 3);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-2), 16), static_cast<RealType>(51.014128749483902310217774804582826908060740157564L), tolerance * 3);

/*
  Table[N[BesselYZero[-3, n], 51], {n, 1, 7, 1}]
  1 | 4.52702466114964385037002686710362763866515554861094
  2 | 8.09755376286049070440221399011280422904322313690750
  3 | 11.3964667395958667392520481906295049459849691925349
  4 | 14.6230777423938731740767225077252006493529705699150
  5 | 17.8184552329455202625532390647367394433803521627517
  6 | 20.9972847541877606834525058939528641630713169437070
  7 | 24.1662357585818282287385597668220226288453739040042
*/

  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3), 1), static_cast<RealType>(4.52702466114964385037002686710362763866515554861094L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3), 2), static_cast<RealType>(8.09755376286049070440221399011280422904322313690750L), tolerance);

  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3), 4), static_cast<RealType>(14.6230777423938731740767225077252006493529705699150L), tolerance);

/* Table[N[BesselKZero[-39, n], 51], {n, 1, 20, 5}]
  1 | 42.2362394762664681287397356668342141701037684436723
  6 | 65.8250353430045981408288669790173009159561533403819
  11 | 84.2674082411341814641248554679382420802125973458922
  16 | 101.589776978258493441843447810649346266014624868410
*/
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-39), 1), static_cast<RealType>(42.2362394762664681287397356668342141701037684436723L), tolerance );
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-39), 6), static_cast<RealType>(65.8250353430045981408288669790173009159561533403819L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-39), 11), static_cast<RealType>(84.2674082411341814641248554679382420802125973458922L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-39), 16), static_cast<RealType>(101.589776978258493441843447810649346266014624868410L), tolerance);

/* Table[N[BesselKZero[-39 -(1/3), n], 51], {n, 1, 20, 5}]
  1 | 39.3336965099558453809241429692683050137281997313679
  6 | 64.9038181444904768984884565999608291433823953030822
  11 | 83.4922341795560713832607574604255239776551554961143
  16 | 100.878386349724826125265571457142254077564666532665
*/

  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-39) - static_cast<RealType>(1)/3, 1), static_cast<RealType>(39.3336965099558453809241429692683050137281997313679L), tolerance * 4);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-39) - static_cast<RealType>(1)/3, 6), static_cast<RealType>(64.9038181444904768984884565999608291433823953030822L), tolerance * 4);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-39) - static_cast<RealType>(1)/3, 11), static_cast<RealType>(83.4922341795560713832607574604255239776551554961143L), tolerance * 4);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-39) - static_cast<RealType>(1)/3, 16), static_cast<RealType>(100.878386349724826125265571457142254077564666532665L), tolerance * 4);
/* Table[N[BesselKZero[-(1/3), n], 51], {n, 1, 20, 5}]
n | 
1 | 0.364442931311036254896373762996743259918847602789703
6 | 15.9741013584105984633772025789145590038676373673203
11 | 31.6799168750213003020847708007848147516190373648194
16 | 47.3871543280673235432396563497681616285970326011211
*/

    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(-static_cast<RealType>(1)/3, 1), static_cast<RealType>(0.364442931311036254896373762996743259918847602789703L), tolerance * 10);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(-static_cast<RealType>(1)/3, 6), static_cast<RealType>(15.9741013584105984633772025789145590038676373673203L), tolerance * 10);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(-static_cast<RealType>(1)/3, 11), static_cast<RealType>(31.6799168750213003020847708007848147516190373648194L), tolerance * 4);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(-static_cast<RealType>(1)/3, 16), static_cast<RealType>(47.3871543280673235432396563497681616285970326011211L), tolerance * 4);

/* Table[N[BesselKZero[-3 -(9999/10000), n], 51], {n, 1, 20, 5}]
  1 | 5.64546089250283694562642537496601708928630550185069
  2 | 9.36184180108088288881787970896747209376324330610979
  3 | 12.7303431758275183078115963473808796340618061355885
  4 | 15.9998152121877557837972245675029531998475502716021
  6 | 9.36184180108088288881787970896747209376324330610979
  11 | 25.6104419106589739931633042959774157385787405502820
  16 | 41.4361281441868132581487460354904567452973524446193
*/

    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3) -static_cast<RealType>(9999)/10000, 1), static_cast<RealType>(5.64546089250283694562642537496601708928630550185069L), tolerance * 4);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3) -static_cast<RealType>(9999)/10000, 2), static_cast<RealType>(9.36184180108088288881787970896747209376324330610979L), tolerance * 4);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3) -static_cast<RealType>(9999)/10000, 3), static_cast<RealType>(12.7303431758275183078115963473808796340618061355885L), tolerance * 4);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3) -static_cast<RealType>(9999)/10000, 4), static_cast<RealType>(15.9998152121877557837972245675029531998475502716021L), tolerance * 4);
 
/*  Table[N[BesselYZero[-3 -(9999/10000), n], 51], {n, 1, 7, 1}]
1 | 5.64546089250283694562642537496601708928630550185069
2 | 9.36184180108088288881787970896747209376324330610979
3 | 12.7303431758275183078115963473808796340618061355885
4 | 15.9998152121877557837972245675029531998475502716021

// but 5 is same as 1!!  Acknowledged as fault Wolfram [TS 6475] 26 Feb 13.

5 | 5.64546089250283694562642537496601708928630550184982
6 | 9.36184180108088288881787970896747209376324330610979
7 | 12.7303431758275183078115963473808796340618061355885

In[26]:= FindRoot[BesselY[-3 -9999/10000, r] == 0, {r, 3}]  for r = 2,3, 4, 5 = {r->5.64546}

In[26]:= FindRoot[BesselY[-3 -9999/10000, r] == 0, {r, 19}] = 19.2246

So no very accurate reference value for these.

Calculated using cpp_dec_float_50

  5.6454608925028369456264253749660170892863055018498
  9.3618418010808828888178797089674720937632433061099
  12.730343175827518307811596347380879634061806135589
  15.999815212187755783797224567502953199847550271602

  19.224610865671563344572152795434688888375602299773
  22.424988389021059116212186912990863561607855849204
  25.610441910658973993163304295977415738578740550282
  28.786066313968546073981640755202085944374967166411
  31.954857624676521867923579695253822854717613513587
    */
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3) -static_cast<RealType>(9999)/10000, 1), static_cast<RealType>(5.64546089250283694562642537496601708928630550185069L), tolerance * 4);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3) -static_cast<RealType>(9999)/10000, 2), static_cast<RealType>(9.36184180108088288881787970896747209376324330610979L), tolerance * 4);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3) -static_cast<RealType>(9999)/10000, 3), static_cast<RealType>(12.7303431758275183078115963473808796340618061355885L), tolerance * 4);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3) -static_cast<RealType>(9999)/10000, 4), static_cast<RealType>(15.9998152121877557837972245675029531998475502716021L), tolerance * 4);
//  
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3) -static_cast<RealType>(9999)/10000, 5), static_cast<RealType>(19.224610865671563344572152795434688888375602299773L), tolerance * 4);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3) -static_cast<RealType>(9999)/10000, 6), static_cast<RealType>(22.424988389021059116212186912990863561607855849204L), tolerance * 4);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3) -static_cast<RealType>(9999)/10000, 7), static_cast<RealType>(25.610441910658973993163304295977415738578740550282L), tolerance * 4);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3) -static_cast<RealType>(9999)/10000, 8), static_cast<RealType>(28.786066313968546073981640755202085944374967166411L), tolerance * 4);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-3) -static_cast<RealType>(9999)/10000, 9), static_cast<RealType>(31.954857624676521867923579695253822854717613513587L), tolerance * 4);


// Plot[BesselYZero[-7 - v, 1], {v, 0, 1}]  shows discontinuity at the mid-point between integers.

/* Table[N[BesselYZero[-7 - (4999/10000), n], 51], {n, 1, 4, 1}]
  1 | 3.59209698655443348407622952525352410710983745802573
  2 | 11.6573245781899449398248761667833391837824916603434
  3 | 15.4315262542144355217979771618575628291362029097236
  4 | 18.9232143766706670333395285892576635207736306576135
*/

/* Table[N[BesselYZero[-7 - (5001/10000), n], 51], {n, 1, 4, 1}]
  1 | 11.6567397956147934678808863468662427054245897492445
  2 | 15.4310521624769624067699131497395566368341140531722
  3 | 18.9227840182910629037411848072684247564491740961847
  4 | 22.2951449444372591060253508661432751300205474374696
*/

    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-7) -static_cast<RealType>(4999)/10000, 1), static_cast<RealType>(3.59209698655443348407622952525352410710983745802573L), tolerance * 2000);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-7) -static_cast<RealType>(4999)/10000, 2), static_cast<RealType>(11.6573245781899449398248761667833391837824916603434L), tolerance * 100);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-7) -static_cast<RealType>(4999)/10000, 3), static_cast<RealType>(15.4315262542144355217979771618575628291362029097236L), tolerance * 100);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-7) -static_cast<RealType>(4999)/10000, 4), static_cast<RealType>(18.9232143766706670333395285892576635207736306576135L), tolerance * 100);

    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-7) -static_cast<RealType>(5001)/10000, 1), static_cast<RealType>(11.6567397956147934678808863468662427054245897492445L), tolerance * 100);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-7) -static_cast<RealType>(5001)/10000, 2), static_cast<RealType>(15.4310521624769624067699131497395566368341140531722L), tolerance * 100);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-7) -static_cast<RealType>(5001)/10000, 3), static_cast<RealType>(18.9227840182910629037411848072684247564491740961847L), tolerance * 100);
    BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-7) -static_cast<RealType>(5001)/10000, 4), static_cast<RealType>(22.2951449444372591060253508661432751300205474374696L), tolerance * 100);

  //BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(-(static_cast<RealType>(-3)-static_cast<RealType>(99)/100), 5), 
  //  cyl_neumann_zero(+(static_cast<RealType>(-3)-static_cast<RealType>(99)/100), 5), tolerance * 100);
  {
    long double x = 1.L;
    BOOST_CHECK_CLOSE_FRACTION(
      cyl_neumann_zero(-(static_cast<RealType>(x)), 5), 
      cyl_neumann_zero(+(static_cast<RealType>(x)), 5), tolerance * 100);
  }
  {
    long double x = 2.L;
    BOOST_CHECK_CLOSE_FRACTION(
      cyl_neumann_zero(-(static_cast<RealType>(x)), 5), 
      cyl_neumann_zero(+(static_cast<RealType>(x)), 5), tolerance * 100);
  }
  {
    long double x = 3.L;
    BOOST_CHECK_CLOSE_FRACTION(
      cyl_neumann_zero(-(static_cast<RealType>(x)), 5), 
      cyl_neumann_zero(+(static_cast<RealType>(x)), 5), tolerance * 100);
  }
  // These are very close but not exactly same.
  //{
  //  RealType x = static_cast<RealType>(-3) -static_cast<RealType>(9999)/10000;
  //  BOOST_CHECK_CLOSE_FRACTION(
  //    cyl_neumann_zero(-(static_cast<RealType>(x)), 5), 
  //    cyl_neumann_zero(+(static_cast<RealType>(x)), 5), tolerance * 100);
  //  // 19.2242889 and 19.2246113
  //}
  //{

  //  RealType x = static_cast<RealType>(-3) -static_cast<RealType>(9999)/10000;
  //  BOOST_CHECK_CLOSE_FRACTION(
  //    cyl_neumann_zero(-(static_cast<RealType>(x)), 6), 
  //    cyl_neumann_zero(+(static_cast<RealType>(x)), 6), tolerance * 100);
  //  // 22.4246693 and 22.4249878
  //}



  //  2.5  18.6890354  17.1033592


/*Table[N[BesselYZero[-1/81799, n], 51], {n, 1, 10, 5}]

1 | 0.893559276290122922836047849416713592133322804889757
2 | 3.95765935645507004204986415533750122885237402118726
3 | 7.08603190350579828577279552434514387474680226004173
4 | 10.2223258629823064789904339889550588869985272176335
5 | 13.3610782840659145864973521693322670264135672594988
3 | 7.08603190350579828577279552434514387474680226004173
5 | 13.3610782840659145864973521693322670264135672594988
6 | 16.5009032471619898684110089652474861084220781491575
7 | 19.6412905039556082160052482410981245043314155416354
9 | 25.9229384536173175152381652048590136247796591153244
*/
   BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(-static_cast<RealType>(1)/81799, 1), static_cast<RealType>(0.893559276290122922836047849416713592133322804889757L), tolerance * 4);
  // Doesn't converge!
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(-static_cast<RealType>(1)/81799, 2), static_cast<RealType>(3.95765935645507004204986415533750122885237402118726L), tolerance * 4);
  BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(-static_cast<RealType>(1)/81799, 3), static_cast<RealType>(7.08603190350579828577279552434514387474680226004173L), tolerance * 4);
  /* try positive x
  Table[N[BesselYZero[1/81799, n], 51], {n, 1, 5, 1}]

  1 | 0.893594656187326273432267210617481926490785928764963
  2 | 3.95769748213950546166537901626409026826595687994956
  3 | 7.08607021707716361104064671367526817399129653285580
  4 | 10.2223642239960815612515914411615233651316361060338
  5 | 13.3611166636685056799674772287389749065996094266976
*/
   BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(1)/81799, 2), static_cast<RealType>(3.95769748213950546166537901626409026826595687994956L), tolerance * 4);
   BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(1)/81799, 3), static_cast<RealType>(7.08607021707716361104064671367526817399129653285580L), tolerance * 4);
   BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(-static_cast<RealType>(1)/81799, 4), static_cast<RealType>(10.2223258629823064789904339889550588869985272176335L), tolerance * 4);
   BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(-static_cast<RealType>(1)/81799, 5), static_cast<RealType>(13.3610782840659145864973521693322670264135672594988L), tolerance * 4);
   BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(-static_cast<RealType>(1)/81799, 6), static_cast<RealType>(16.5009032471619898684110089652474861084220781491575L), tolerance * 4);
   BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(-static_cast<RealType>(1)/81799, 9), static_cast<RealType>(25.9229384536173175152381652048590136247796591153244L), tolerance * 4);
   BOOST_CHECK_CLOSE_FRACTION(cyl_neumann_zero(static_cast<RealType>(-7) - static_cast<RealType>(1)/3, 1), static_cast<RealType>(7.3352783956690540155848592759652828459644819344081L), tolerance * 1000);

  // Test Data for airy_ai_zero and airy_bi_zero functions.

   using boost::math::airy_ai_zero; //

   using boost::math::isnan;

   BOOST_MATH_CHECK_THROW(airy_ai_zero<RealType>(0), std::domain_error);

  if (std::numeric_limits<RealType>::has_quiet_NaN)
  { // If ignore errors, return NaN.
    BOOST_CHECK((boost::math::isnan)(airy_ai_zero<RealType>(0, ignore_all_policy())));
    BOOST_CHECK((boost::math::isnan)(airy_ai_zero<RealType>((std::numeric_limits<unsigned>::min)() , ignore_all_policy())));
    // Can't abuse with NaN as won't compile.
    //BOOST_MATH_CHECK_THROW(airy_ai_zero<RealType>(std::numeric_limits<RealType>::quiet_NaN()), std::domain_error);
  }
  else
  { // real_concept NaN not available, so return zero.
    BOOST_CHECK_EQUAL(airy_ai_zero<RealType>(0, ignore_all_policy()), 0);
    // BOOST_CHECK_EQUAL(airy_ai_zero<RealType>(-1), 0); //  warning C4245: 'argument' : conversion from 'int' to 'unsigned int', signed/unsigned mismatch
  }

  BOOST_MATH_CHECK_THROW(airy_ai_zero<RealType>(-1), std::domain_error);
  if (std::numeric_limits<RealType>::digits && (std::numeric_limits<RealType>::digits < 100))
  {
     // Limited precision test value:
     BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>((std::numeric_limits<std::int32_t>::max)()), -static_cast<RealType>(4678579.33301973093739L), tolerance);
  }

  // Can't abuse with infinity because won't compile - no conversion.
  //if (std::numeric_limits<RealType>::has_infinity)
  //{
  //  BOOST_CHECK(isnan(airy_bi_zero<RealType>(-1)) );
  //}

  // WolframAlpha  Table[N[AiryAiZero[n], 51], {n, 1, 20, 1}]

  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(1), static_cast<RealType>(-2.33810741045976703848919725244673544063854014567239L), tolerance * 2 * tolerance_tgamma_extra);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(2), static_cast<RealType>(-4.08794944413097061663698870145739106022476469910853L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(3), static_cast<RealType>(-5.52055982809555105912985551293129357379721428061753L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(4), static_cast<RealType>(-6.78670809007175899878024638449617696605388247739349L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(5), static_cast<RealType>(-7.94413358712085312313828055579826853214067439697221L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(6), static_cast<RealType>(-9.02265085334098038015819083988008925652467753515608L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(7), static_cast<RealType>(-10.0401743415580859305945567373625180940429025691058L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(8), static_cast<RealType>(-11.0085243037332628932354396495901510167308253815040L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(9), static_cast<RealType>(-11.9360155632362625170063649029305843155778862321198L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(10), static_cast<RealType>(-12.8287767528657572004067294072418244773864155995734L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(11), static_cast<RealType>(-13.6914890352107179282956967794669205416653698092008L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(12), static_cast<RealType>(-14.5278299517753349820739814429958933787141648698348L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(13), static_cast<RealType>(-15.3407551359779968571462085134814867051175833202480L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(14), static_cast<RealType>(-16.1326851569457714393459804472025217905182723970763L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(15), static_cast<RealType>(-16.9056339974299426270352387706114765990900510950317L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(16), static_cast<RealType>(-17.6613001056970575092536503040180559521532186681200L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(17), static_cast<RealType>(-18.4011325992071154158613979295043367545938146060201L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(18), static_cast<RealType>(-19.1263804742469521441241486897324946890754583847531L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(19), static_cast<RealType>(-19.8381298917214997009475636160114041983356824945389L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(20), static_cast<RealType>(-20.5373329076775663599826814113081017453042180147375L), tolerance);

  // Table[N[AiryAiZero[n], 51], {n, 1000, 1001, 1}]

  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(1000), static_cast<RealType>(-281.031519612521552835336363963709689055717463965420L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(1001), static_cast<RealType>(-281.218889579130068414512015874511112547569713693446L), tolerance);

  // Table[N[AiryAiZero[n], 51], {n, 1000000, 1000001, 1}]
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(1000000), static_cast<RealType>(-28107.8319793795834876064419863203282898723750036048L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(1000001), static_cast<RealType>(-28107.8507179357979542838020057465277368471496446555L), tolerance);


  // Table[N[AiryAiZero[n], 51], {n, 1000000000, 1000000001, 1}]
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(1000000000), static_cast<RealType>(-2.81078366593344513918947921096193426320298300481145E+6L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_ai_zero<RealType>(1000000001), static_cast<RealType>(-2.81078366780730091663459728526906320267920607427246E+6L), tolerance);

  // Test Data for airy_bi
  using boost::math::airy_bi_zero;

  BOOST_MATH_CHECK_THROW(airy_bi_zero<RealType>(0), std::domain_error);

  if (std::numeric_limits<RealType>::has_quiet_NaN)
  { // return NaN.
    BOOST_CHECK((boost::math::isnan)(airy_bi_zero<RealType>(0, ignore_all_policy())));
    BOOST_CHECK((boost::math::isnan)(airy_bi_zero<RealType>((std::numeric_limits<unsigned>::min)() , ignore_all_policy())));
    // Can't abuse with NaN as won't compile.
    // BOOST_MATH_CHECK_THROW(airy_bi_zero<RealType>(std::numeric_limits<RealType>::quiet_NaN()), std::domain_error);
    // cannot convert parameter 1 from 'boost::math::concepts::real_concept' to 'unsigned int'.
  }
  else
  { // real_concept NaN not available, so return zero.
    BOOST_CHECK_EQUAL(airy_bi_zero<RealType>(0, ignore_all_policy()), 0);
    // BOOST_CHECK_EQUAL(airy_bi_zero<RealType>(-1), 0);
    //  warning C4245: 'argument' : conversion from 'int' to 'unsigned int', signed/unsigned mismatch.
    // If ignore the warning, interpreted as max unsigned:
    // check airy_bi_zero<RealType>(-1) == 0 has failed [-7.42678e+006 != 0]
  }

  BOOST_MATH_CHECK_THROW(airy_bi_zero<RealType>(-1), std::domain_error);
  if (std::numeric_limits<RealType>::digits && (std::numeric_limits<RealType>::digits < 100))
  {
     // Limited precision test value:
     BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>((std::numeric_limits<std::int32_t>::max)()), -static_cast<RealType>(4678579.33229351984573L), tolerance * 300);
  }

  // Can't abuse with infinity because won't compile - no conversion.
  //if (std::numeric_limits<RealType>::has_infinity)
  //{
  //  BOOST_CHECK(isnan(airy_bi_zero<RealType>(std::numeric_limits<RealType>::infinity)) );
  //}

  // Table[N[AiryBiZero[n], 51], {n, 1, 20, 1}]
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(1), static_cast<RealType>(-1.17371322270912792491997996247390210454364638917570L), tolerance * 4 * tolerance_tgamma_extra);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(2), static_cast<RealType>(-3.27109330283635271568022824016641380630093596910028L), tolerance * tolerance_tgamma_extra);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(3), static_cast<RealType>(-4.83073784166201593266770933990517817696614261732301L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(4), static_cast<RealType>(-6.16985212831025125983336452055593667996554943427563L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(5), static_cast<RealType>(-7.37676207936776371359995933044254122209152229939710L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(6), static_cast<RealType>(-8.49194884650938801344803949280977672860508755505546L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(7), static_cast<RealType>(-9.53819437934623888663298854515601962083907207638247L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(8), static_cast<RealType>(-10.5299135067053579244005555984531479995295775946214L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(9), static_cast<RealType>(-11.4769535512787794379234649247328196719482538148877L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(10), static_cast<RealType>(-12.3864171385827387455619015028632809482597983846856L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(11), static_cast<RealType>(-13.2636395229418055541107433243954907752411519609813L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(12), static_cast<RealType>(-14.1127568090686577915873097822240184716840428285509L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(13),  static_cast<RealType>(-14.9370574121541640402032143104909046396121763517782L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(14), static_cast<RealType>(-15.7392103511904827708949784797481833807180162767841L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(15), static_cast<RealType>(-16.5214195506343790539179499652105457167110310370581L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(16), static_cast<RealType>(-17.2855316245812425329342366922535392425279753602710L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(17), static_cast<RealType>(-18.0331132872250015721711125433391920008087291416406L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(18), static_cast<RealType>(-18.7655082844800810413429789236105128440267189551421L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(19), static_cast<RealType>(-19.4838801329892340136659986592413575122062977793610L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(20), static_cast<RealType>(-20.1892447853962024202253232258275360764649783583934L), tolerance);

 // Table[N[AiryBiZero[n], 51], {n, 1000, 1001, 1}]
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(1000), static_cast<RealType>(-280.937811203415240157883427412260300146245056425646L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(1001), static_cast<RealType>(-281.125212400956392021977771104562061554648675044114L), tolerance);

  // Table[N[AiryBiZero[n], 51], {n, 1000000, 1000001, 1}]
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(1000000), static_cast<RealType>(-28107.8226100991339342855024130953986989636667226163L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(1000001), static_cast<RealType>(-28107.8413486584714939255315213519230566014624895515L), tolerance);

  //Table[N[AiryBiZero[n], 51], {n, 1000000000, 1000000001, 1}]
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(1000000000), static_cast<RealType>(-2.81078366499651725023268820158218492845371527054171E+6L), tolerance);
  BOOST_CHECK_CLOSE_FRACTION(airy_bi_zero<RealType>(1000000001), static_cast<RealType>(-2.81078366687037302799011557215619265502627118526716E+6L), tolerance);

  // Check the multi-root versions.
  {
    unsigned int n_roots = 1U;
    std::vector<RealType> roots;
    boost::math::airy_ai_zero<RealType>(2U, n_roots, std::back_inserter(roots));
    BOOST_CHECK_CLOSE_FRACTION(roots[0], static_cast<RealType>(-4.08794944413097061663698870145739106022476469910853L), tolerance);
  }
  {
    unsigned int n_roots = 1U;
    std::vector<RealType> roots;
    boost::math::airy_bi_zero<RealType>(2U, n_roots, std::back_inserter(roots));
    BOOST_CHECK_CLOSE_FRACTION(roots[0], static_cast<RealType>(-3.27109330283635271568022824016641380630093596910028L), tolerance * tolerance_tgamma_extra);
  }
} // template <class RealType> void test_spots(RealType)

  #include <boost/multiprecision/cpp_dec_float.hpp>

BOOST_AUTO_TEST_CASE(test_main)
{
   test_bessel_zeros(0.1F);
   test_bessel_zeros(0.1);
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
   test_bessel_zeros(0.1L);
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
   test_bessel_zeros(boost::math::concepts::real_concept(0.1));
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
}
