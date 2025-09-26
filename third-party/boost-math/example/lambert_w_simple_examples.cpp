// Copyright Paul A. Bristow 2016, 2017.

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or
//  copy at http ://www.boost.org/LICENSE_1_0.txt).

// Build and run a simple examples of Lambert W function.

// Some macros that will show some(or much) diagnostic values if #defined.
//#define-able macros
//#define BOOST_MATH_INSTRUMENT_LAMBERT_W0 // W0 branch diagnostics.
//#define BOOST_MATH_INSTRUMENT_LAMBERT_Wm1 // W1 branch diagnostics.
//#define BOOST_MATH_INSTRUMENT_LAMBERT_W_HALLEY // Halley refinement diagnostics.
//#define BOOST_MATH_INSTRUMENT_LAMBERT_W_SCHROEDER // Schroeder refinement diagnostics.
//#define BOOST_MATH_INSTRUMENT_LAMBERT_W_TERMS // Number of terms used for near-singularity series.
//#define BOOST_MATH_INSTRUMENT_LAMBERT_W0_NOT_BUILTIN // higher than built-in precision types approximation and refinement.
//#define BOOST_MATH_INSTRUMENT_LAMBERT_W_SINGULARITY_SERIES // Show evaluation of series near branch singularity.
//#define BOOST_MATH_INSTRUMENT_LAMBERT_W_SMALL_Z_SERIES_ITERATIONS  // Show evaluation of series for small z.
//#define BOOST_MATH_INSTRUMENT_LAMBERT_W0_LOOKUP // Show results from lookup table.

#include <boost/config.hpp> // for BOOST_PLATFORM, BOOST_COMPILER,  BOOST_STDLIB ...
#include <boost/version.hpp>   // for BOOST_MSVC versions.
#include <boost/math/constants/constants.hpp> // For exp_minus_one == 3.67879441171442321595523770161460867e-01.
#include <boost/math/policies/policy.hpp>

#include <boost/multiprecision/cpp_dec_float.hpp> // boost::multiprecision::cpp_dec_float_50
using boost::multiprecision::cpp_dec_float_50; // 50 decimal digits type.
using boost::multiprecision::cpp_dec_float_100; // 100 decimal digits type.
using boost::multiprecision::backends::cpp_dec_float;
using boost::multiprecision::number;
typedef number<cpp_dec_float<1000> > cpp_dec_float_1000; // 1000 decimal digit types

#include <boost/multiprecision/cpp_bin_float.hpp>
using boost::multiprecision::cpp_bin_float_double; // == double
using boost::multiprecision::cpp_bin_float_double_extended; // 80-bit long double emulation.
using boost::multiprecision::cpp_bin_float_quad; // 128-bit quad precision.

//[lambert_w_simple_examples_includes
#include <boost/math/special_functions/lambert_w.hpp> // For lambert_w function.

using boost::math::lambert_w0;
using boost::math::lambert_wm1;
//] //[/lambert_w_simple_examples_includes]

#include <iostream>
// using std::cout;
// using std::endl;
#include <exception>
#include <stdexcept>
#include <string>
#include <limits>  // For std::numeric_limits.

//! Show value of z to the full possibly-significant max_digits10 precision of type T.
template<typename T>
void show_value(T z)
{
  std::streamsize precision = std::cout.precision(std::numeric_limits<T>::max_digits10);  // Save.
  std::cout.precision(std::numeric_limits<T>::max_digits10); // Show all possibly significant digits.
  std::ios::fmtflags flags(std::cout.flags());
  std::cout.setf(std::ios_base::showpoint); // Include any trailing zeros.
  std::cout << z;
  // Restore:
  std::cout.precision(precision);
  std::cout.flags(flags);
} // template<typename T> void show_value(T z)

int main()
{
  try
  {
    std::cout << "Lambert W simple examples." << std::endl;

    using boost::math::constants::exp_minus_one; //-1/e, the branch point, a singularity ~= -0.367879.

    // using statements needed for changing error handling policy.
    using boost::math::policies::policy;
    using boost::math::policies::make_policy;
    using boost::math::policies::evaluation_error;
    using boost::math::policies::domain_error;
    using boost::math::policies::overflow_error;
    using boost::math::policies::ignore_error;
    using boost::math::policies::throw_on_error;

  {
//[lambert_w_simple_examples_0
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    // Show all potentially significant decimal digits,
    std::cout << std::showpoint << std::endl;
    // and show significant trailing zeros too.

    double z = 10.;
    double r = lambert_w0(z); // Default policy for double.
    std::cout << "lambert_w0(z) = " << r << std::endl;
    // lambert_w0(z) = 1.7455280027406994
//] [/lambert_w_simple_examples_0]
  }
  {
    // Other floating-point types can be used too, here float.
    // It is convenient to use a function like `show_value`
    // to display all potentially significant decimal digits
    // for the type, including any significant trailing zeros.
    //[lambert_w_simple_examples_1
    float z = 10.F;
    float r;
    r = lambert_w0(z);        // Default policy digits10 = 7, digits2 = 24
    std::cout << "lambert_w0(";
    show_value(z);
    std::cout << ") = ";
    show_value(r);
    std::cout << std::endl;   // lambert_w0(10.0000000) = 1.74552798
   //] //[/lambert_w_simple_examples_1]
  }
   {
     // Example of an integer argument to lambert_w,
     // showing that an integer is correctly promoted to a double.
//[lambert_w_simple_examples_2
     std::cout.precision(std::numeric_limits<double>::max_digits10);
     double r = lambert_w0(10);                           // Pass an int argument "10" that should be promoted to double argument.
     std::cout << "lambert_w0(10) = " << r << std::endl;  // lambert_w0(10) = 1.7455280027406994
     double rp = lambert_w0(10);
     std::cout << "lambert_w0(10) = " << rp << std::endl;
     // lambert_w0(10) = 1.7455280027406994
     auto rr = lambert_w0(10);                            // C++11 needed.
     std::cout << "lambert_w0(10) = " << rr << std::endl;
     // lambert_w0(10) = 1.7455280027406994 too, showing that rr has been promoted to double.
//] //[/lambert_w_simple_examples_2]
   }
   {
     // Using multiprecision types to get much higher precision is painless.
     //[lambert_w_simple_examples_3
     cpp_dec_float_50 z("10");
     // Note construction using a decimal digit string "10",
     // NOT a floating-point double literal 10.
     cpp_dec_float_50 r;
     r = lambert_w0(z);
     std::cout << "lambert_w0("; show_value(z); std::cout << ") = ";
     show_value(r);
     std::cout << std::endl;
     // lambert_w0(10.000000000000000000000000000000000000000000000000000000000000000000000000000000) =
     //   1.7455280027406993830743012648753899115352881290809413313533156980404446940000000
     //] //[/lambert_w_simple_examples_3]
   }
   // Using multiprecision types to get multiprecision precision wrong!
   {
     //[lambert_w_simple_examples_4
     cpp_dec_float_50 z(0.7777777777777777777777777777777777777777777777777777777777777777777777777);
     // Compiler evaluates the nearest double-precision binary representation,
     // from the max_digits10 of the floating_point literal double 0.7777777777777777777777777777...,
     // so any extra digits in the multiprecision type
     // beyond max_digits10 (usually 17) are random and meaningless.
     cpp_dec_float_50 r;
     r = lambert_w0(z);
     std::cout << "lambert_w0(";
     show_value(z);
     std::cout << ") = "; show_value(r);
     std::cout << std::endl;
     // lambert_w0(0.77777777777777779011358916250173933804035186767578125000000000000000000000000000)
     //   = 0.48086152073210493501934682309060873341910109230469724725005039758139532631901386
     //] //[/lambert_w_simple_examples_4]
   }
   {
     //[lambert_w_simple_examples_4a
     cpp_dec_float_50 z(0.9); // Construct from floating_point literal double 0.9.
     cpp_dec_float_50 r;
     r = lambert_w0(0.9);
     std::cout << "lambert_w0(";
     show_value(z);
     std::cout << ") = "; show_value(r);
     std::cout << std::endl;
     // lambert_w0(0.90000000000000002220446049250313080847263336181640625000000000000000000000000000)
     //   = 0.52983296563343440510607251781038939952850341796875000000000000000000000000000000
     std::cout << "lambert_w0(0.9) = " << lambert_w0(static_cast<double>(0.9))
     // lambert_w0(0.9)
     //   = 0.52983296563343441
       << std::endl;
     //] //[/lambert_w_simple_examples_4a]
   }
   {
     // Using multiprecision types to get multiprecision precision right!
     //[lambert_w_simple_examples_4b
     cpp_dec_float_50 z("0.9");     // Construct from decimal digit string.
     cpp_dec_float_50 r;
     r = lambert_w0(z);
     std::cout << "lambert_w0(";
     show_value(z);
     std::cout << ") = "; show_value(r);
     std::cout << std::endl;
     // 0.90000000000000000000000000000000000000000000000000000000000000000000000000000000)
     // = 0.52983296563343441213336643954546304857788132269804249284012528304239956413801252
     //] //[/lambert_w_simple_examples_4b]
   }
   // Getting extreme precision (1000 decimal digits) Lambert W values.
   {
     std::cout.precision(std::numeric_limits<cpp_dec_float_1000>::digits10);
     cpp_dec_float_1000 z("2.0");
     cpp_dec_float_1000 r;
     r = lambert_w0(z);
     std::cout << "lambert_w0(z) = " << r << std::endl;
     // 0.8526055020137254913464724146953174668984533001514035087721073946525150656742630448965773783502494847334503972691804119834761668851953598826198984364998343940330324849743119327028383008883133161249045727544669202220292076639777316648311871183719040610274221013237163543451621208284315007250267190731048119566857455987975973474411544571619699938899354169616378479326962044241495398851839432070255805880208619490399218130868317114428351234208216131218024303904457925834743326836272959669122797896855064630871955955318383064292191644322931561534814178034773896739684452724587331245831001449498844495771266728242975586931792421997636537572767708722190588748148949667744956650966402600446780664924889043543203483210769017254907808218556111831854276511280553252641907484685164978750601216344998778097446525021666473925144772131644151718261199915247932015387685261438125313159125475113124470774926288823525823567568542843625471594347837868505309329628014463491611881381186810879712667681285740515197493390563
     // Wolfram alpha command N[productlog[0, 2.0],1000] gives the identical result:
     // 0.8526055020137254913464724146953174668984533001514035087721073946525150656742630448965773783502494847334503972691804119834761668851953598826198984364998343940330324849743119327028383008883133161249045727544669202220292076639777316648311871183719040610274221013237163543451621208284315007250267190731048119566857455987975973474411544571619699938899354169616378479326962044241495398851839432070255805880208619490399218130868317114428351234208216131218024303904457925834743326836272959669122797896855064630871955955318383064292191644322931561534814178034773896739684452724587331245831001449498844495771266728242975586931792421997636537572767708722190588748148949667744956650966402600446780664924889043543203483210769017254907808218556111831854276511280553252641907484685164978750601216344998778097446525021666473925144772131644151718261199915247932015387685261438125313159125475113124470774926288823525823567568542843625471594347837868505309329628014463491611881381186810879712667681285740515197493390563
   }
   {
//[lambert_w_simple_examples_error_policies
      // Define an error handling policy:
      typedef policy<
        domain_error<throw_on_error>,
        overflow_error<ignore_error> // possibly unwise?
      > my_throw_policy;

      std::cout.precision(std::numeric_limits<double>::max_digits10);
      // Show all potentially significant decimal digits,
      std::cout << std::showpoint << std::endl;
      // and show significant trailing zeros too.
      double z = +1;
      std::cout << "Lambert W (" << z << ") = " << lambert_w0(z) << std::endl;
      // Lambert W (1.0000000000000000) = 0.56714329040978384
      std::cout << "\nLambert W (" << z << ", my_throw_policy()) = "
        << lambert_w0(z, my_throw_policy()) << std::endl;
      // Lambert W (1.0000000000000000, my_throw_policy()) = 0.56714329040978384
    //] //[/lambert_w_simple_example_error_policies]
    }
    {
      // Show error reporting from passing a value to lambert_wm1 that is out of range,
      // (and probably was meant to be passed to lambert_0 instead).
//[lambert_w_simple_examples_out_of_range
      double z = +1.;
      double r = lambert_wm1(z);
      std::cout << "lambert_wm1(+1.) = " << r << std::endl;
 //] [/lambert_w_simple_examples_out_of_range]
     // Error in function boost::math::lambert_wm1<RealType>(<RealType>):
      // Argument z = 1 is out of range (z <= 0) for Lambert W-1 branch! (Try Lambert W0 branch?)
    }
  }
  catch (std::exception& ex)
  {
    std::cout << ex.what() << std::endl;
  }
}  // int main()

   /*

   Output:
//[lambert_w_simple_examples_error_message_1
Error in function boost::math::lambert_wm1<RealType>(<RealType>):
Argument z = 1 is out of range (z <= 0) for Lambert W-1 branch! (Try Lambert W0 branch?)
//] [/lambert_w_simple_examples_error_message_1]

   */
