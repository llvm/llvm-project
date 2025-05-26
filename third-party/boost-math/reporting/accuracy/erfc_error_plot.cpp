
//  (C) Copyright Nick Thompson 2020.
//  (C) Copyright John Maddock 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// We deliberately include some Unicode characters:
// 
// boost-no-inspect
//
#include <iostream>
#include <boost/math/tools/ulps_plot.hpp>
#include <boost/core/demangle.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
#endif

using namespace boost::multiprecision;

#ifndef TEST_TYPE
#define TEST_TYPE cpp_bin_float_50
#endif

std::string test_type_name(BOOST_STRINGIZE(TEST_TYPE));
std::string test_type_filename(BOOST_STRINGIZE(TEST_TYPE));

using boost::math::tools::ulps_plot;

int main() 
{
   std::string::size_type n;
   while ((n = test_type_filename.find_first_not_of("_qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890")) != std::string::npos)
   {
      test_type_filename[n] = '_';
   }

   using PreciseReal = boost::multiprecision::mpfr_float_100;
   using CoarseReal = TEST_TYPE;

   typedef boost::math::policies::policy<
      boost::math::policies::promote_float<false>,
      boost::math::policies::promote_double<false> >
      no_promote_policy;

   auto ai_coarse = [](CoarseReal const& x)->CoarseReal {
      return erfc(x);
   };
   auto ai_precise = [](PreciseReal const& x)->PreciseReal {
      return erfc(x);
   };

   std::string filename = "erfc_errors_";
   filename += test_type_filename;
   filename += ".svg";
   int samples = 100000;
   // How many pixels wide do you want your .svg?
   int width = 700;
   // Near a root, we have unbounded relative error. So for functions with roots, we define an ULP clip:
   PreciseReal clip = 40;
   // Should we perturb the abscissas? i.e., should we compute the high precision function f at x,
   // and the low precision function at the nearest representable x̂ to x?
   // Or should we compute both the high precision and low precision function at a low precision representable x̂?
   bool perturb_abscissas = false;
   auto plot = ulps_plot<decltype(ai_precise), PreciseReal, CoarseReal>(ai_precise, CoarseReal(-10), CoarseReal(30), samples, perturb_abscissas);
   // Note the argument chaining:
   plot.clip(clip).width(width);
   plot.background_color("white").font_color("black");
   // Sometimes it's useful to set a title, but in many cases it's more useful to just use a caption.
   std::string title = "Erfc ULP plot at " + test_type_name + " precision";
   plot.title(title);
   plot.vertical_lines(6);
   plot.add_fn(ai_coarse);
   // You can write the plot to a stream:
   //std::cout << plot;
   // Or to a file:
   plot.write(filename);
}
