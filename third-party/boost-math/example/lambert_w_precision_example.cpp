// Copyright Paul A. Bristow 2016, 2018.

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or
//  copy at http ://www.boost.org/LICENSE_1_0.txt).

//! Lambert W examples of controlling precision

// #define BOOST_MATH_INSTRUMENT_LAMBERT_W  // #define only for (much) diagnostic output.

#include <boost/config.hpp> // for BOOST_PLATFORM, BOOST_COMPILER,  BOOST_STDLIB ...
#include <boost/version.hpp>   // for BOOST_MSVC versions.
#include <boost/math/constants/constants.hpp> // For exp_minus_one == 3.67879441171442321595523770161460867e-01.
#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/next.hpp>  // for float_distance.
#include <boost/math/special_functions/relative_difference.hpp> // for relative and epsilon difference.

// Built-in/fundamental GCC float128 or Intel Quad 128-bit type, if available.
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp> // Not available for MSVC.
// sets BOOST_MP_USE_FLOAT128 for GCC
using boost::multiprecision::float128;
#endif //# NOT _MSC_VER

#include <boost/multiprecision/cpp_dec_float.hpp> // boost::multiprecision::cpp_dec_float_50
using boost::multiprecision::cpp_dec_float_50; // 50 decimal digits type.
using boost::multiprecision::cpp_dec_float_100; // 100 decimal digits type.

#include <boost/multiprecision/cpp_bin_float.hpp>
using boost::multiprecision::cpp_bin_float_double_extended;
using boost::multiprecision::cpp_bin_float_double;
using boost::multiprecision::cpp_bin_float_quad;
// For lambert_w function.
#include <boost/math/special_functions/lambert_w.hpp>
// using boost::math::lambert_w0;
// using boost::math::lambert_wm1;

#include <iostream>
#include <exception>
#include <stdexcept>
#include <string>
#include <limits>  // For std::numeric_limits.

int main()
{
  try
  {
    std::cout << "Lambert W examples of precision control." << std::endl;
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::cout << std::showpoint << std::endl; // Show any trailing zeros.

    using boost::math::constants::exp_minus_one;

    using boost::math::lambert_w0;
    using boost::math::lambert_wm1;

    // Error handling policy examples.
    using namespace boost::math::policies;
    using boost::math::policies::make_policy;
    using boost::math::policies::policy;
    using boost::math::policies::evaluation_error;
    using boost::math::policies::domain_error;
    using boost::math::policies::overflow_error;
    using boost::math::policies::throw_on_error;

//[lambert_w_precision_reference_w

    using boost::multiprecision::cpp_bin_float_50;
    using boost::math::float_distance;

    cpp_bin_float_50 z("10."); // Note use a decimal digit string, not a double 10.
    cpp_bin_float_50 r;
    std::cout.precision(std::numeric_limits<cpp_bin_float_50>::digits10);

    r = lambert_w0(z); // Default policy.
    std::cout << "lambert_w0(z) cpp_bin_float_50  = " << r << std::endl;
    //lambert_w0(z) cpp_bin_float_50  = 1.7455280027406993830743012648753899115352881290809
    //       [N[productlog[10], 50]] == 1.7455280027406993830743012648753899115352881290809
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    std::cout << "lambert_w0(z) static_cast from cpp_bin_float_50  = "
      << static_cast<double>(r) << std::endl;
    // double lambert_w0(z) static_cast from cpp_bin_float_50  = 1.7455280027406994
    // [N[productlog[10], 17]]                                == 1.7455280027406994
   std::cout << "bits different from Wolfram = "
     << static_cast<int>(float_distance(static_cast<double>(r), 1.7455280027406994))
     << std::endl; // 0


//] [/lambert_w_precision_reference_w]

//[lambert_w_precision_0
    std::cout.precision(std::numeric_limits<float>::max_digits10); // Show all potentially significant decimal digits,
    std::cout << std::showpoint << std::endl; // and show any significant trailing zeros too.

    float x = 10.;
    std::cout << "Lambert W (" << x << ") = " << lambert_w0(x) << std::endl;
//] [/lambert_w_precision_0]

/*
//[lambert_w_precision_output_0
Lambert W (10.0000000) = 1.74552800
//] [/lambert_w_precision_output_0]
*/
  { // Lambert W0 Halley step example
//[lambert_w_precision_1
    using boost::math::lambert_w_detail::lambert_w_halley_step;
    using boost::math::epsilon_difference;
    using boost::math::relative_difference;

    std::cout << std::showpoint << std::endl; // and show any significant trailing zeros too.
    std::cout.precision(std::numeric_limits<double>::max_digits10); // 17 decimal digits for double.

    cpp_bin_float_50 z50("1.23"); // Note: use a decimal digit string, not a double 1.23!
    double z = static_cast<double>(z50);
    cpp_bin_float_50 w50;
    w50 = lambert_w0(z50);
    std::cout.precision(std::numeric_limits<cpp_bin_float_50>::max_digits10); // 50 decimal digits.
    std::cout << "Reference Lambert W (" << z << ") =\n                                              "
      << w50 << std::endl;
    std::cout.precision(std::numeric_limits<double>::max_digits10); // 17 decimal digits for double.
    double wr = static_cast<double>(w50);
    std::cout << "Reference Lambert W (" << z << ") =    " << wr << std::endl;

    double w = lambert_w0(z);
    std::cout << "Rat/poly Lambert W  (" << z << ")  =   " << lambert_w0(z) << std::endl;
    // Add a Halley step to the value obtained from rational polynomial approximation.
    double ww = lambert_w_halley_step(lambert_w0(z), z);
    std::cout << "Halley Step Lambert W (" << z << ") =  " << lambert_w_halley_step(lambert_w0(z), z) << std::endl;

    std::cout << "absolute difference from Halley step = " << w - ww << std::endl;
    std::cout << "relative difference from Halley step = " << relative_difference(w, ww) << std::endl;
    std::cout << "epsilon difference from Halley step  = " << epsilon_difference(w, ww) << std::endl;
    std::cout << "epsilon for float =                    " << std::numeric_limits<double>::epsilon() << std::endl;
    std::cout << "bits different from Halley step  =     " << static_cast<int>(float_distance(w, ww)) << std::endl;
//] [/lambert_w_precision_1]


/*
//[lambert_w_precision_output_1
  Reference Lambert W (1.2299999999999999822364316059974953532218933105468750) =
  0.64520356959320237759035605255334853830173300262666480
  Reference Lambert W (1.2300000000000000) =    0.64520356959320235
  Rat/poly Lambert W  (1.2300000000000000)  =   0.64520356959320224
  Halley Step Lambert W (1.2300000000000000) =  0.64520356959320235
  absolute difference from Halley step = -1.1102230246251565e-16
  relative difference from Halley step = 1.7207329236029286e-16
  epsilon difference from Halley step  = 0.77494921535422934
  epsilon for float =                    2.2204460492503131e-16
  bits different from Halley step  =     1
//] [/lambert_w_precision_output_1]
*/

  } // Lambert W0 Halley step example

  { // Lambert W-1 Halley step example
    //[lambert_w_precision_2
    using boost::math::lambert_w_detail::lambert_w_halley_step;
    using boost::math::epsilon_difference;
    using boost::math::relative_difference;

    std::cout << std::showpoint << std::endl; // and show any significant trailing zeros too.
    std::cout.precision(std::numeric_limits<double>::max_digits10); // 17 decimal digits for double.

    cpp_bin_float_50 z50("-0.123"); // Note: use a decimal digit string, not a double -1.234!
    double z = static_cast<double>(z50);
    cpp_bin_float_50 wm1_50;
    wm1_50 = lambert_wm1(z50);
    std::cout.precision(std::numeric_limits<cpp_bin_float_50>::max_digits10); // 50 decimal digits.
    std::cout << "Reference Lambert W-1 (" << z << ") =\n                                                  "
      << wm1_50 << std::endl;
    std::cout.precision(std::numeric_limits<double>::max_digits10); // 17 decimal digits for double.
    double wr = static_cast<double>(wm1_50);
    std::cout << "Reference Lambert W-1 (" << z << ") =    " << wr << std::endl;

    double w = lambert_wm1(z);
    std::cout << "Rat/poly Lambert W-1 (" << z << ")  =    " << lambert_wm1(z) << std::endl;
    // Add a Halley step to the value obtained from rational polynomial approximation.
    double ww = lambert_w_halley_step(lambert_wm1(z), z);
    std::cout << "Halley Step Lambert W (" << z << ") =    " << lambert_w_halley_step(lambert_wm1(z), z) << std::endl;

    std::cout << "absolute difference from Halley step = " << w - ww << std::endl;
    std::cout << "relative difference from Halley step = " << relative_difference(w, ww) << std::endl;
    std::cout << "epsilon difference from Halley step  = " << epsilon_difference(w, ww) << std::endl;
    std::cout << "epsilon for float =                    " << std::numeric_limits<double>::epsilon() << std::endl;
    std::cout << "bits different from Halley step  =     " << static_cast<int>(float_distance(w, ww)) << std::endl;
    //] [/lambert_w_precision_2]
  }
  /*
  //[lambert_w_precision_output_2
    Reference Lambert W-1 (-0.12299999999999999822364316059974953532218933105468750) =
    -3.2849102557740360179084675531714935199110302996513384
    Reference Lambert W-1 (-0.12300000000000000) =    -3.2849102557740362
    Rat/poly Lambert W-1 (-0.12300000000000000)  =    -3.2849102557740357
    Halley Step Lambert W (-0.12300000000000000) =    -3.2849102557740362
    absolute difference from Halley step = 4.4408920985006262e-16
    relative difference from Halley step = 1.3519066740696092e-16
    epsilon difference from Halley step  = 0.60884463935795785
    epsilon for float =                    2.2204460492503131e-16
    bits different from Halley step  =     -1
  //] [/lambert_w_precision_output_2]
  */



  // Similar example using cpp_bin_float_quad (128-bit floating-point types).

  cpp_bin_float_quad zq = 10.;
  std::cout << "\nTest evaluation of cpp_bin_float_quad Lambert W(" << zq << ")"
    << std::endl;
  std::cout << std::setprecision(3) << "std::numeric_limits<cpp_bin_float_quad>::digits = " << std::numeric_limits<cpp_bin_float_quad>::digits << std::endl;
  std::cout << std::setprecision(3) << "std::numeric_limits<cpp_bin_float_quad>::epsilon() = " << std::numeric_limits<cpp_bin_float_quad>::epsilon() << std::endl;
  std::cout << std::setprecision(3) << "std::numeric_limits<cpp_bin_float_quad>::max_digits10 = " << std::numeric_limits<cpp_bin_float_quad>::max_digits10 << std::endl;
  std::cout << std::setprecision(3) << "std::numeric_limits<cpp_bin_float_quad>::digits10 = " << std::numeric_limits<cpp_bin_float_quad>::digits10 << std::endl;
  std::cout.precision(std::numeric_limits<cpp_bin_float_quad>::max_digits10);
  // All are same precision because double precision first approximation used before Halley.

  /*

      */

  { // Reference value for lambert_w0(10)
    cpp_dec_float_50 z("10");
    cpp_dec_float_50 r;
    std::cout.precision(std::numeric_limits<cpp_dec_float_50>::digits10);

    r = lambert_w0(z); // Default policy.
    std::cout << "lambert_w0(z) cpp_dec_float_50                                   = " << r << std::endl; //  0.56714329040978387299996866221035554975381578718651
    std::cout.precision(std::numeric_limits<cpp_bin_float_quad>::max_digits10);

    std::cout << "lambert_w0(z) cpp_dec_float_50 cast to quad (max_digits10(" << std::numeric_limits<cpp_bin_float_quad>::max_digits10 <<
      " )   = " << static_cast<cpp_bin_float_quad>(r) << std::endl;         // 1.7455280027406993830743012648753899115352881290809
    std::cout.precision(std::numeric_limits<cpp_bin_float_quad>::digits10); // 1.745528002740699383074301264875389837
    std::cout << "lambert_w0(z) cpp_dec_float_50 cast to quad (digits10(" << std::numeric_limits<cpp_bin_float_quad>::digits10 <<
      " )       = " << static_cast<cpp_bin_float_quad>(r) << std::endl;    // 1.74552800274069938307430126487539
    std::cout.precision(std::numeric_limits<cpp_bin_float_quad>::digits10 + 1); //

    std::cout << "lambert_w0(z) cpp_dec_float_50 cast to quad (digits10(" << std::numeric_limits<cpp_bin_float_quad>::digits10 <<
      " )       = " << static_cast<cpp_bin_float_quad>(r) << std::endl;    // 1.74552800274069938307430126487539

    // [N[productlog[10], 50]] == 1.7455280027406993830743012648753899115352881290809

    // [N[productlog[10], 37]] == 1.745528002740699383074301264875389912
    // [N[productlog[10], 34]] == 1.745528002740699383074301264875390
    // [N[productlog[10], 33]] == 1.74552800274069938307430126487539

    // lambert_w0(z) cpp_dec_float_50 cast to quad = 1.745528002740699383074301264875389837

    // lambert_w0(z) cpp_dec_float_50 = 1.7455280027406993830743012648753899115352881290809
    // lambert_w0(z) cpp_dec_float_50 cast to quad = 1.745528002740699383074301264875389837
    // lambert_w0(z) cpp_dec_float_50 cast to quad = 1.74552800274069938307430126487539
    }
  }
  catch (std::exception& ex)
  {
    std::cout << ex.what() << std::endl;
  }
}  // int main()




