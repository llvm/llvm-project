// Lambert W lookup table generator lambert_w_lookup_table_generator.cpp

//! \file
//! Output a table of precomputed array values for Lambert W0 and W-1,
//! and square roots and halves, and powers of e,
// as a .ipp file for use by lambert_w.hpp.

//! \details Output as long double precision (suffix L) using Boost.Multiprecision
//! to 34 decimal digits precision to cater for platforms that have 128-bit long double.
//! The function bisection can then use any built-in floating-point type,
//! which may have different precision and speed on different platforms.
//! The actual builtin floating-point type of the arrays is chosen by a
//! typedef in \modular-boost\libs\math\include\boost\math\special_functions\lambert_w.hpp
//! by default, for example: typedef double lookup_t;


// This includes lookup tables for both branches W0 and W-1.
// Only W-1 is needed by current code that uses JM rational Polynomials,
// but W0 is kept (for now) to allow comparison with the previous FKDVPB version
// that uses lookup for W0 branch as well as W-1.

#include <boost/config.hpp>
#include <boost/math/constants/constants.hpp> // For exp_minus_one == 3.67879441171442321595523770161460867e-01.
using boost::math::constants::exp_minus_one; // 0.36787944
using boost::math::constants::root_e; // 1.64872
#include <boost/multiprecision/cpp_bin_float.hpp>
using boost::multiprecision::cpp_bin_float_quad;
using boost::multiprecision::cpp_bin_float_50;

#include <iostream>
#include <fstream>
#include <typeinfo>

/*
typedef double lookup_t; // Type for lookup table (double or float?)

static constexpr std::size_t noof_sqrts = 12;
static constexpr lookup_t a[noof_sqrts] = // 0.6065 0.7788, 0.8824 ... 0.9997, sqrt of previous elements.
{
0.60653065971263342, 0.77880078307140487, 0.8824969025845954, 0.93941306281347579, 0.96923323447634408, 0.98449643700540841,
0.99221793826024351, 0.99610136947011749, 0.99804878110747547, 0.99902391418197566, 0.99951183793988937, 0.99975588917489722
};
static constexpr  lookup_t b[noof_sqrts] = // 0.5 0.25 0.125, 0.0625 ...  0.0002441, halves of previous elements.
{ 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625, 0.00048828125, 0.000244140625 };

static constexpr size_t noof_w0zs = 65;
static constexpr lookup_t g[noof_w0zs] = // lambert_w[k] for W0 branch. 0 2.7182 14.77 60.2566 ... 1.445e+29 3.990e+29.
{ 0., 2.7182818284590452, 14.7781121978613, 60.256610769563003, 218.39260013257696, 742.06579551288302, 2420.5727609564107, 7676.4321089992102,
23847.663896333826, 72927.755348178456, 220264.65794806717, 658615.558867176, 1953057.4970280471, 5751374.0961159665, 16836459.978306875, 49035260.58708166,
142177768.32812596, 410634196.81078007, 1181879444.4719492, 3391163718.300558, 9703303908.1958056, 27695130424.147509, 78868082614.895014, 224130479263.72476,
635738931116.24334, 1800122483434.6468, 5088969845149.8079, 14365302496248.563, 40495197800161.305, 114008694617177.22, 320594237445733.86,
900514339622670.18, 2526814725845782.2, 7083238132935230.1, 19837699245933466., 55510470830970076., 1.5520433569614703e+17, 4.3360826779369662e+17,
1.2105254067703227e+18, 3.3771426165357561e+18, 9.4154106734807994e+18, 2.6233583234732253e+19, 7.3049547543861044e+19, 2.032970971338619e+20,
5.6547040503180956e+20, 1.5720421975868293e+21, 4.3682149334771265e+21, 1.2132170565093317e+22, 3.3680332378068632e+22, 9.3459982052259885e+22,
2.5923527642935362e+23, 7.1876803203773879e+23, 1.99212416037262e+24, 5.5192924995054165e+24, 1.5286067837683347e+25, 4.2321318958281094e+25,
1.1713293177672778e+26, 3.2408603996214814e+26, 8.9641258264226028e+26, 2.4787141382364034e+27, 6.8520443388941057e+27, 1.8936217407781711e+28,
5.2317811346197018e+28, 1.4450833904658542e+29, 3.9904954117194348e+29
};

static constexpr std::size_t noof_wm1zs = 66;
static constexpr lookup_t e[noof_wm1zs] = // lambert_w[k] for W-1 branch. 2.7182 1. 0.3678 0.1353 0.04978 ... 4.359e-28 1.603e-28
{
2.7182818284590452, 1., 0.36787944117144232, 0.13533528323661269, 0.049787068367863943, 0.01831563888873418, 0.0067379469990854671,
0.0024787521766663584, 0.00091188196555451621, 0.00033546262790251184, 0.00012340980408667955, 4.5399929762484852e-05, 1.6701700790245659e-05,
6.1442123533282098e-06, 2.2603294069810543e-06, 8.3152871910356788e-07, 3.0590232050182579e-07, 1.1253517471925911e-07, 4.1399377187851667e-08,
1.5229979744712628e-08, 5.6027964375372675e-09, 2.0611536224385578e-09, 7.5825604279119067e-10, 2.7894680928689248e-10, 1.026187963170189e-10,
3.7751345442790977e-11, 1.3887943864964021e-11, 5.1090890280633247e-12, 1.8795288165390833e-12, 6.914400106940203e-13, 2.5436656473769229e-13,
9.3576229688401746e-14, 3.4424771084699765e-14, 1.2664165549094176e-14, 4.6588861451033974e-15, 1.713908431542013e-15, 6.3051167601469894e-16,
2.3195228302435694e-16, 8.5330476257440658e-17, 3.1391327920480296e-17, 1.1548224173015786e-17, 4.248354255291589e-18, 1.5628821893349888e-18,
5.7495222642935598e-19, 2.1151310375910805e-19, 7.7811322411337965e-20, 2.8625185805493936e-20, 1.0530617357553812e-20, 3.8739976286871871e-21,
1.4251640827409351e-21, 5.2428856633634639e-22, 1.9287498479639178e-22, 7.0954741622847041e-23, 2.6102790696677048e-23, 9.602680054508676e-24,
3.532628572200807e-24, 1.2995814250075031e-24, 4.7808928838854691e-25, 1.7587922024243116e-25, 6.4702349256454603e-26, 2.3802664086944006e-26,
8.7565107626965203e-27, 3.2213402859925161e-27, 1.185064864233981e-27, 4.359610000063081e-28, 1.6038108905486378e-28
};

lambert_w0 version of array from Fukushima

// lambert_w[k] for W-1 branch. 2.7182 1. 0.3678 0.1353 0.04978 ... 4.359e-28 1.603e-28
e: 2.7182818284590452, 1., 0.36787944117144232, 0.13533528323661269, 0.049787068367863943, 0.01831563888873418, 0.0067379469990854671,
0.0024787521766663584, 0.00091188196555451621, 0.00033546262790251184, 0.00012340980408667955, 4.5399929762484852e-05, 1.6701700790245659e-05,
6.1442123533282098e-06, 2.2603294069810543e-06, 8.3152871910356788e-07, 3.0590232050182579e-07, 1.1253517471925911e-07, 4.1399377187851667e-08,
1.5229979744712628e-08, 5.6027964375372675e-09, 2.0611536224385578e-09, 7.5825604279119067e-10, 2.7894680928689248e-10, 1.026187963170189e-10,
3.7751345442790977e-11, 1.3887943864964021e-11, 5.1090890280633247e-12, 1.8795288165390833e-12, 6.914400106940203e-13, 2.5436656473769229e-13,
9.3576229688401746e-14, 3.4424771084699765e-14, 1.2664165549094176e-14, 4.6588861451033974e-15, 1.713908431542013e-15, 6.3051167601469894e-16,
2.3195228302435694e-16, 8.5330476257440658e-17, 3.1391327920480296e-17, 1.1548224173015786e-17, 4.248354255291589e-18, 1.5628821893349888e-18,
5.7495222642935598e-19, 2.1151310375910805e-19, 7.7811322411337965e-20, 2.8625185805493936e-20, 1.0530617357553812e-20, 3.8739976286871871e-21,
1.4251640827409351e-21, 5.2428856633634639e-22, 1.9287498479639178e-22, 7.0954741622847041e-23, 2.6102790696677048e-23, 9.602680054508676e-24,
3.532628572200807e-24, 1.2995814250075031e-24, 4.7808928838854691e-25, 1.7587922024243116e-25, 6.4702349256454603e-26, 2.3802664086944006e-26,
8.7565107626965203e-27, 3.2213402859925161e-27, 1.185064864233981e-27, 4.359610000063081e-28, 1.6038108905486378e-28

// lambert_w[k] for W0 branch. 0 2.7182 14.77 60.2566 ... 1.445e+29 3.990e+29.

g: 0, 2.7182818284590452, 14.7781121978613, 60.256610769563003, 218.39260013257696, 742.06579551288302, 2420.5727609564107, 7676.4321089992102,
23847.663896333826, 72927.755348178456, 220264.65794806717, 658615.558867176, 1953057.4970280471, 5751374.0961159665, 16836459.978306875, 49035260.58708166,
142177768.32812596, 410634196.81078007, 1181879444.4719492, 3391163718.300558, 9703303908.1958056, 27695130424.147509, 78868082614.895014, 224130479263.72476,
635738931116.24334, 1800122483434.6468, 5088969845149.8079, 14365302496248.563, 40495197800161.305, 114008694617177.22, 320594237445733.86,
900514339622670.18, 2526814725845782.2, 7083238132935230.1, 19837699245933466, 55510470830970076, 1.5520433569614703e+17, 4.3360826779369662e+17,
1.2105254067703227e+18, 3.3771426165357561e+18, 9.4154106734807994e+18, 2.6233583234732253e+19, 7.3049547543861044e+19, 2.032970971338619e+20,
5.6547040503180956e+20, 1.5720421975868293e+21, 4.3682149334771265e+21, 1.2132170565093317e+22, 3.3680332378068632e+22, 9.3459982052259885e+22,
2.5923527642935362e+23, 7.1876803203773879e+23, 1.99212416037262e+24, 5.5192924995054165e+24, 1.5286067837683347e+25, 4.2321318958281094e+25,
1.1713293177672778e+26, 3.2408603996214814e+26, 8.9641258264226028e+26, 2.4787141382364034e+27, 6.8520443388941057e+27, 1.8936217407781711e+28,
5.2317811346197018e+28, 1.4450833904658542e+29, 3.9904954117194348e+29


lambert_wm1 version of arrays from Fukushima
e: 2.7182817459106445 7.3890557289123535 20.085535049438477 54.59814453125 148.41314697265625 403.42874145507813 1096.6329345703125 2980.957275390625 8103.08154296875 22026.458984375 59874.12109375 162754.734375 442413.21875 1202603.75 3269015.75 8886106 24154940 65659932 178482192 485164896 1318814848 3584910336 9744796672 26489102336 72004845568 195729457152 532047822848 1446255919104 3931331100672 10686465835008 29048824659968 78962889850880 214643389759488 583461240832000 1586012102852608 4311227773747200 11719131799748608 31855901283450880 86593318145753088 2.3538502982225101e+17 6.398428560008151e+17 1.7392731886358364e+18 4.7278345784949473e+18 1.2851586685678387e+19 3.493423319351296e+19 9.4961089747571704e+19 2.581309902546461e+20 7.0167278463083348e+20 1.9073443887231177e+21 5.1846992652160593e+21 1.4093473476000776e+22 3.831003235981371e+22 1.0413746376682761e+23 2.8307496154307266e+23 7.6947746628514896e+23 2.0916565667371597e+24 5.6857119515524837e+24 1.5455367020327599e+25 4.2012039964445827e+25 1.1420056438012293e+26 3.1042929865047826e+26 8.4383428037470738e+26 2.2937792813113457e+27 6.2351382164292627e+27

g: -0.36787945032119751 -0.27067059278488159 -0.14936122298240662 -0.073262564837932587 -0.033689741045236588 -0.014872515574097633 -0.0063831745646893978 -0.0026837014593183994 -0.0011106884339824319 -0.00045399941154755652 -0.00018371877376921475 -7.3730567237362266e-05 -2.9384291337919421e-05 -1.1641405762929935e-05 -4.5885362851549871e-06 -1.8005634956352878e-06 -7.0378973759943619e-07 -2.7413975089984888e-07 -1.0645318582191976e-07 -4.122309249510181e-08 -1.5923385277005764e-08 -6.1368328196920174e-09 -2.3602335641470518e-09 -9.0603280433754207e-10 -3.471987974901225e-10 -1.3283640853956058e-10 -5.0747316071575455e-11 -1.9360334516105304e-11 -7.3766357605586919e-12 -2.8072891233854591e-12 -1.0671687058344537e-12 -4.0525363013271809e-13 -1.5374336461045079e-13 -5.8272932648966574e-14 -2.206792725173521e-14 -8.3502896573240185e-15 -3.1572303958374423e-15 -1.192871523299666e-15 -4.5038112940094517e-16 -1.699343306816689e-16 -6.4078234365689933e-17 -2.4148019279880996e-17 -9.095073346605316e-18 -3.4237017961279004e-18 -1.2881348671140216e-18 -4.8440896082993503e-19 -1.8207810463107454e-19 -6.8407959442757565e-20 -2.569017156788846e-20 -9.6437611040447661e-21 -3.6186962678628536e-21 -1.357346940624028e-21 -5.0894276378983633e-22 -1.9076220526102576e-22 -7.1477077345829229e-23 -2.6773039821769189e-23 -1.0025130740057213e-23 -3.7527418826161672e-24 -1.4043593713279384e-24 -5.2539147015754201e-25 -1.9650207139502987e-25 -7.3474141096711539e-26 -2.7465588329293218e-26 -1.0264406957471058e-26


a: 1.6487212181091309 1.2840254306793213 1.1331484317779541 1.0644944906234741 1.0317434072494507 1.0157476663589478 1.007843017578125 1.0039138793945313 1.0019550323486328 1.0009770393371582 1.0004884004592896 1.000244140625

// These are common to both W0 and W-1
b: 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125 0.000244140625

*/

// Creates if no file exists, & uses default overwrite/ ios::replace.
//const char filename[] = // "lambert_w_lookup_table.ipp";  // Write to same folder as generator:
//"I:/modular-boost/libs/math/include/boost/math/special_functions/lambert_w_lookup_table.ipp";
const char filename[] = "lambert_w_lookup_table.ipp";

std::ofstream fout(filename, std::ios::out); // File output stream.

// 128-bit precision type (so that full precision if long double type uses 128-bit).
// typedef cpp_bin_float_quad table_lookup_t; // Output using max_digits10 for 37 decimal digit precision.
// (This is the precision for the tables output as a C++ program,
// not the precision used by the lambert_w.hpp, that defines another typedef lookup_t, default double.

typedef cpp_bin_float_50 table_lookup_t; // Compute tables to 50 decimal digit precision to avoid slight inaccuracy from repeated multiply.

// But Output using max_digits10 for 37 decimal digit precision.

int main()
{
  std::cout << "Lambert W table lookup values." << std::endl;
  if (!fout.is_open())
  {  // File failed to open OK.
    std::cerr << "Open file " << filename << " failed!" << std::endl;
    std::cerr << "errno " << errno << std::endl;
    return -1;
  }
  try
  {
    std::cout << "Lambert W test values writing to file " << filename << std::endl;
    int output_precision = std::numeric_limits<cpp_bin_float_quad>::max_digits10; // 37 decimal digits.
    fout.precision(output_precision);
    fout <<
      "// Copyright Paul A. Bristow 2017."   "\n"
      "// Distributed under the Boost Software License, Version 1.0." "\n"
      "// (See accompanying file LICENSE_1_0.txt" "\n"
      "// or copy at http://www.boost.org/LICENSE_1_0.txt)" "\n"
      "\n"
      "// " << filename << "\n\n"
      "// A collection of 128-bit precision integral z argument Lambert W values computed using "
      << output_precision << " decimal digits precision.\n"
      "// C++ floating-point precision is 128-bit long double.\n"
      "// Output as "
      << std::numeric_limits<table_lookup_t>::max_digits10
      << " decimal digits, suffixed L.\n"
      "\n"
      "// C++ floating-point type is provided by lambert_w.hpp typedef."  "\n"
      "// For example: typedef lookup_t double; (or float or long double)"  "\n"

      "\n"
      "// Written by " << __FILE__ << " " << __TIMESTAMP__ << "\n"
      << std::endl;

    fout << "// Sizes of arrays of z values for Lambert W[0], W[1] ... W[64]"
      "\"n""and W[-1], W[-2] ... W[-64]." << std::endl;

    fout << "\nnamespace boost {\nnamespace math {\nnamespace lambert_w_detail {\nnamespace lambert_w_lookup\n{ \n";

    static constexpr std::size_t noof_sqrts = 12;
    static constexpr std::size_t noof_halves = 12;
    fout << "static constexpr std::size_t  noof_sqrts = " << noof_sqrts << ";" << std::endl;
    fout << "static constexpr std::size_t  noof_halves = " << noof_halves << ";" << std::endl; // Common to both branches.

    static constexpr std::size_t noof_w0zs = 65; // F[k] 0 <= k <= 64. f[0] = F[0], f[64] = F[64]
    static constexpr std::size_t noof_w0es = 66; //  noof_w0zs +1 for gratuitous extra power.
    static constexpr std::size_t noof_wm1zs = 64; // G[k] 1 <= k <= 64. (W-1 = 0 would be z = -infinity, so not stored in array) g[0] == G[1], g[63] = G[64]
    static constexpr std::size_t noof_wm1es = 64; //

    fout << "static constexpr std::size_t  noof_w0es = " << noof_w0zs << ";" << std::endl;
    fout << "static constexpr std::size_t  noof_w0zs = " << noof_w0zs << ";" << std::endl;
    fout << "static constexpr std::size_t  noof_wm1es = " << noof_wm1zs << ";" << std::endl;
    fout << "static constexpr std::size_t  noof_wm1zs = " << noof_wm1zs << ";" << std::endl;

    // Defining actual lookup table sqrts of e^k, e^-k = 1/e, etc.
    table_lookup_t halves[noof_halves]; // 0.5 0.25 0.125, 0.0625 ...  0.0002441, halves of previous elements.
    table_lookup_t sqrtw0s[noof_sqrts]; // 0.6065 0.7788, 0.8824 ... 0.9997, sqrt of previous elements.
    table_lookup_t sqrtwm1s[noof_sqrts]; // 1.6487, 1.2840 1.1331  ... 1.00024 , sqrt of previous elements.
    table_lookup_t w0es[noof_w0es]; // lambert_w[k] for W0 branch. 2.7182, 1, 0.3678, 0.1353, ... 1.6038e-28
    table_lookup_t w0zs[noof_w0zs]; // lambert_w[k] for W0 branch. 0. , 2.7182, 14.77, 60.2566 ... 1.445e+29, 3.990e+29.
    table_lookup_t wm1es[noof_wm1es];  // lambert_w[k] for W-1 branch. 2.7182 7.38905 20.085 ... 6.235e+27
    table_lookup_t wm1zs[noof_wm1zs];  // lambert_w[k] for W-1 branch. -0.3678 ... -1.0264e-26

    // e values lambert_w[k] for W-1 branch. 2.7182 1. 0.3678 0.1353 0.04978 ... 4.359e-28 1.603e-28

    using boost::math::constants::e;
    using boost::math::constants::exp_minus_one;

    {  // z values for integral W F[k] and powers for W0 branch.
      table_lookup_t ej = 1; //
      w0es[0] = e<table_lookup_t>(); // e = 2.7182 exp(-1) - 1/e exp_minus_one = 0.36787944.
      w0es[1] = 1; // e^0
      w0zs[0] = 0; // F[0] = 0 or W0 branch.
      for (int j = 1, jj = 2; jj != noof_w0es; ++jj)
      {
        w0es[jj] = w0es[j] * exp_minus_one<table_lookup_t>(); // previous * 0.36787944.
        ej *= e<table_lookup_t>(); // * 2.7182
        w0zs[j] = j * ej; // For W0 branch.
        j = jj; // Previous.
      } // for
    }
    // Checks on accuracy of W0 exponents.

    // Checks on e power w0es

    // w0es[64] =       4.3596100000630809736231248158884615452e-28
    // N[e ^ -63, 37] = 4.359610000063080973623124815888459643*10^-28
    // So slight loss at last decimal place.

    // Checks on accuracy of z for integral W0 w0zs
    // w0zs[0] = 0,  z = -infinity expected? but = zero
    // w0zs[1] = 2.7182818284590452353602874713526623144
    // w0[2] z = 14.778112197861300454460854921150012956
    // w0zs[64] = 3.9904954117194348050619127737142022705e+29
    // N[productlog(0, 3.9904954117194348050619127737142022705 10^+29), 37]
    //   =  63.99999999999999999999999999999999547
    //   = 64.0 to 34 decimal digits, so exact. :-)

    { // z values for integral powers G[k] and e^-k for W-1 branch.
      // Fukushima indexing of G (k-1) differs by 1 from(k).
      // G[0] = -infinity, so his first item in array g[0] is -0.3678 which is G[1]
      // and last is g[63] = G[64] = 1.026e-26
      table_lookup_t e1 = 1. / e<table_lookup_t>(); // 1/e = 0.36787944117144233
      table_lookup_t ej = e1;
      wm1es[0] = e<table_lookup_t>(); // e = 2.7182
      wm1zs[0] = -e1; // -1/e = - 0.3678
      for (int j = 0, jj = 1; jj != noof_wm1zs; ++jj)
      {
        ej *= e1; // * 0.3678..
        wm1es[jj] = wm1es[j] * e<table_lookup_t>();
        wm1zs[jj] = -(jj + 1) * ej;
        j = jj; // Previous.
      } // for
    }

    // Checks on W-1 branch accuracy wm1es by comparing with Wolfram.
    // exp powers:
    // N[e ^ 1, 37]  2.718281828459045235360287471352662498
    // wm1es[0] =   2.7182818284590452353602874713526623144  - close enough.
    // N[e ^ 3, 37]         20.08553692318766774092852965458171790
    // computed wm1es[2]    2.0085536923187667740928529654581712847e+01L  OK
    // e ^ 66 =        4.6071866343312915426773184428060086893349003037096040 * 10^28
    // N[e ^ 66, 34] = 4.607186634331291542677318442806009 10^28
    // computed        4.6071866343312915426773184428059867859e+28L
    // N[e ^ 66, 37] = 4.607186634331291542677318442806008689*10^28
    // so suffering some loss of precision by repeated multiplication computation.
    // :-(

    // Repeat with cpp_bin_float_50 and correct to 37th decimal digit.
    //                 4.60718663433129154267731844280600868933490030370929982
    // output std::cout.precision(std::numeric_limits<cpp_bin_float_quad>::max_digits10) as 37 decimal digits.
    //                 4.6071866343312915426773184428060086893e+28L
    // N[e ^ 66, 37] = 4.607186634331291542677318442806008689*10^28
    // Agrees exactly for 37th place, so should be read in to nearest representable value.


    // Checks W-1 branch z values wm1zs
    // W-1[0] = -2.7067056647322538378799898994496883858e-01
    // w-1[1] = -1.4936120510359182893802724695018536337e-01

    // wm1zs[65] -1.4325445274604020119111357113179868158e-27

    // N[productlog(-1, -1.4325445274604020119111357113179868158* 10^-27), 37]
    // = -65.99999999999999999999999999999999955
    // = -66 accurately, so this is OK.
    // z = 66 * e^66 =
    // =N[-66*e ^ -66, 37]
    //           -1.432544527460402011911135711317986177*10^-27
    // wm1zs[65] -1.4325445274604020119111357113179868158e-27
    // which agrees well enough to 34 decimal digits.
    // last wm1zs[65] = 0 is unused.

    // Halves, common to both W0 and W-1.
    halves[0] = static_cast<table_lookup_t>(0.5); // Exactly representable.
    for (int j = 0; j != noof_sqrts -1; ++j)
    {
      halves[j+1] = halves[j] / 2;  // Half previous element (/2 will be optimised better?).
    } // for j

    // W0 sqrts
    sqrtw0s[0] = static_cast<table_lookup_t>(0.606530659712633423603799534991180453441918135487186955682L);
    for (int j = 0; j != noof_sqrts -1; ++j)
    {
      sqrtw0s[j+1] = sqrt(sqrtw0s[j]); //  Sqrt of previous element. sqrt(1/e), sqrt(sqrt(1/e)) ...
    } // for j

    // W-1 sqrts
    sqrtwm1s[0] = root_e<table_lookup_t>();
    for (int j = 0; j != noof_sqrts -1; ++j)
    {
      sqrtwm1s[j+1] = sqrt(sqrtwm1s[j]); //  Sqrt of previous element. sqrt(1/e), sqrt(sqrt(1/e)) ...
    } // for j

    // Output values as C++ arrays,
    // using static constexpr as static and constexpr as possible for platform.
    fout << std::noshowpoint; // Do show NOT trailing zeros for halves and sqrts values.

    fout <<
      "\n" "static constexpr lookup_t halves[noof_halves] = " //
      "\n" "{ // Common to Lambert W0 and W-1 (and exactly representable)." << "\n  ";
      for (int i = 0; i != noof_halves; i++)
      {
        fout << halves[i] << 'L';
        if (i != noof_halves - 1)
        { // Omit trailing comma on last element.
          fout << ", ";
        }
        else
        {
          fout << std::endl;
        }
      }
      fout << "}; // halves, 0.5, 0.25, ... 0.000244140625, common to W0 and W-1." << std::endl;

      fout <<
        "\n" "static constexpr lookup_t sqrtw0s[noof_sqrts] = " //
        "\n" "{  // For Lambert W0 only." << "\n  ";
      for (int i = 0; i != noof_sqrts; i++)
      {
        fout << sqrtw0s[i] << 'L';
        if (i != noof_sqrts - 1)
        { // Omit trailing comma on last element.
          fout << ", ";
        }
        else
        {
          fout << std::endl;
        }
      }
      fout << "}; // sqrtw0s" << std::endl;

      fout <<
        "\n" "static constexpr lookup_t sqrtwm1s[noof_sqrts] = " //
        "\n" "{ // For Lambert W-1 only." << "\n  ";
      for (int i = 0; i != noof_sqrts; i++)
      {
        fout << sqrtwm1s[i] << 'L';
        if (i != noof_sqrts - 1)
        { // Omit trailing comma on last element.
          fout << ", ";
        }
        else
        {
          fout << std::endl;
        }
      }
      fout << "}; // sqrtwm1s" << std::endl;

      fout << std::scientific // May be needed to avoid very large dddddddddddddddd.ddddddddddddddd output?
        << std::showpoint; // Do show trailing zeros for sqrts and halves.

      // Two W0 arrays

      fout << // W0 e values.
        // Fukushima code generates an extra unused power, but it is not output by using noof_w0zs instead of noof_w0es.
        "\n" "static constexpr lookup_t w0es[noof_w0zs] = " //
        "\n" "{ // Fukushima e powers array e[0] = 2.718, 1., e[2] = e^-1 = 0.135, e[3] = e^-2 = 0.133 ... e[64] = 4.3596100000630809736231248158884615452e-28." << "\n  ";
      for (int i = 0; i != noof_w0zs; i++)
      {
        fout << w0es[i] << 'L';
        if (i != noof_w0es - 1)
        { // Omit trailing comma on last element.
          fout << ", ";
        }
        if (i % 4 == 0)
        {
          fout << "\n  ";
        }
      }
      fout << "\n}; // w0es" << std::endl;

      fout << // W0 z values for W[1], .
        "\n" "static constexpr lookup_t w0zs[noof_w0zs] = " //
        "\n" "{ // z values for W[0], W[1], W[2] ... W[64] (Fukushima array Fk). " << "\n  ";
      for (int i = 0; i != noof_w0zs; i++)
      {
        fout << w0zs[i] << 'L';
        if (i != noof_w0zs - 1)
        { // Omit trailing comma on last element.
          fout << ", ";
        }
        if (i % 4 == 0)
        {
          fout << "\n  ";
        }
      }
      fout << "\n}; // w0zs" << std::endl;

      // Two arrays for w-1

      fout << // W-1 e values.
        "\n" "static constexpr lookup_t wm1es[noof_wm1es] = " //
        "\n" "{ // Fukushima e array e[0] = e^1 = 2.718, e[1] = e^2 = 7.39 ... e[64] = 4.60718e+28." << "\n  ";
      for (int i = 0; i != noof_wm1es; i++)
      {
        fout << wm1es[i] << 'L';
        if (i != noof_wm1es - 1)
        { // Omit trailing comma on last element.
          fout << ", ";
        }
        if (i % 4 == 0)
        {
          fout << "\n  ";
        }
      }
      fout << "\n}; // wm1es" << std::endl;

      fout << // Wm1 z values for integral K.
        "\n" "static constexpr lookup_t wm1zs[noof_wm1zs] = " //
        "\n" "{ // Fukushima G array of z values for integral K, (Fukushima Gk) g[0] (k = -1) = 1 ... g[64] = -1.0264389699511303e-26." << "\n  ";
      for (int i = 0; i != noof_wm1zs; i++)
      {
        fout << wm1zs[i] << 'L';
        if (i != noof_wm1zs - 1)
        { // Omit trailing comma on last element.
          fout << ", ";
        }
        if (i % 4 == 0)
        { // 4 values per line.
          fout << "\n  ";
        }
      }
      fout << "\n}; // wm1zs" << std::endl;

      fout << "} // namespace lambert_w_lookup\n} // namespace lambert_w_detail\n} // namespace math\n} // namespace boost" << std::endl;
    }
    catch (std::exception& ex)
    {
      std::cout << "Exception " << ex.what() << std::endl;
    }
    fout.close();
    return 0;

} // int main()

/*

Original arrays as output by Veberic/Fukushima code:

w0 branch

W-1 branch

e: 2.7182818284590451 7.3890560989306495 20.085536923187664 54.598150033144229 148.41315910257657 403.42879349273500 1096.6331584284583 2980.9579870417274 8103.0839275753815 22026.465794806707 59874.141715197788 162754.79141900383 442413.39200892020 1202604.2841647759 3269017.3724721079 8886110.5205078647 24154952.753575277 65659969.137330450 178482300.96318710 485165195.40978980 1318815734.4832134 3584912846.1315880 9744803446.2488918 26489122129.843441 72004899337.385788 195729609428.83853 532048240601.79797 1446257064291.4734 3931334297144.0371 10686474581524.447 29048849665247.383 78962960182680.578 214643579785915.75 583461742527454.00 1586013452313428.3 4311231547115188.5 11719142372802592. 31855931757113704. 86593400423993600. 2.3538526683701958e+17 6.3984349353005389e+17 1.7392749415204982e+18 4.7278394682293381e+18 1.2851600114359284e+19 3.4934271057485025e+19 9.4961194206024286e+19 2.5813128861900616e+20 7.0167359120976157e+20 1.9073465724950953e+21 5.1847055285870605e+21 1.4093490824269355e+22 3.8310080007165677e+22 1.0413759433029062e+23 2.8307533032746866e+23 7.6947852651419974e+23 2.0916594960129907e+24 5.6857199993359170e+24 1.5455389355900996e+25 4.2012104037905024e+25 1.1420073898156810e+26 3.1042979357019109e+26 8.4383566687414291e+26 2.2937831594696028e+27 6.2351490808115970e+27

g: -0.36787944117144233 -0.27067056647322540 -0.14936120510359185 -0.073262555554936742 -0.033689734995427351 -0.014872513059998156 -0.0063831737588816162 -0.0026837010232200957 -0.0011106882367801162 -0.00045399929762484866 -0.00018371870869270232 -7.3730548239938541e-05 -2.9384282290753722e-05 -1.1641402067449956e-05 -4.5885348075273889e-06 -1.8005627955081467e-06 -7.0378941219347870e-07 -2.7413963540482742e-07 -1.0645313231320814e-07 -4.1223072448771179e-08 -1.5923376898615014e-08 -6.1368298043116385e-09 -2.3602323152914367e-09 -9.0603229062698418e-10 -3.4719859662410078e-10 -1.3283631472964657e-10 -5.0747278046555293e-11 -1.9360320299432585e-11 -7.3766303773930841e-12 -2.8072868906520550e-12 -1.0671679036256938e-12 -4.0525329757101402e-13 -1.5374324278841227e-13 -5.8272886672428505e-14 -2.2067908660514491e-14 -8.3502821888768594e-15 -3.1572276215253082e-15 -1.1928704609782527e-15 -4.5038074274761624e-16 -1.6993417021166378e-16 -6.4078169762734621e-17 -2.4147993510032983e-17 -9.0950634616416589e-18 -3.4236981860988753e-18 -1.2881333612472291e-18 -4.8440839844747606e-19 -1.8207788854829806e-19 -6.8407875971564987e-20 -2.5690139750481013e-20 -9.6437492398196038e-21 -3.6186918227652047e-21 -1.3573451162272088e-21 -5.0894204288896066e-22 -1.9076194289884390e-22 -7.1476978375412793e-23 -2.6773000149758669e-23 -1.0025115553818592e-23 -3.7527362568743735e-24 -1.4043571811296988e-24 -5.2539064576179218e-25 -1.9650175744554385e-25 -7.3474021582506962e-26 -2.7465543000397468e-26 -1.0264389699511303e-26

a: 1.6487212707001282 1.2840254166877414 1.1331484530668263 1.0644944589178595 1.0317434074991028 1.0157477085866857 1.0078430972064480 1.0039138893383477 1.0019550335910028 1.0009770394924165 1.0004884004786945 1.0002441704297478

b: 0.50000000000000000 0.25000000000000000 0.12500000000000000 0.062500000000000000 0.031250000000000000 0.015625000000000000 0.0078125000000000000 0.0039062500000000000 0.0019531250000000000 0.00097656250000000000 0.00048828125000000000 0.00024414062500000000


*/


