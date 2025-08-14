//  (C) Copyright Nick Thompson 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Deliberately contains some unicode characters:
// 
// boost-no-inspect

#include <iostream>
#include <boost/math/tools/ulps_plot.hpp>
#include <boost/core/demangle.hpp>
#include <boost/math/special_functions/airy.hpp>

using boost::math::tools::ulps_plot;

int main() {
    using PreciseReal = long double;
    using CoarseReal = float;

    typedef boost::math::policies::policy<
      boost::math::policies::promote_float<false>,
      boost::math::policies::promote_double<false> >
      no_promote_policy;

    auto ai_coarse = [](CoarseReal x) {
        return boost::math::airy_ai<CoarseReal>(x, no_promote_policy());
    };
    auto ai_precise = [](PreciseReal x) {
        return boost::math::airy_ai<PreciseReal>(x, no_promote_policy());
    };

    std::string filename = "airy_ai_" + boost::core::demangle(typeid(CoarseReal).name()) + ".svg";
    int samples = 10000;
    // How many pixels wide do you want your .svg?
    int width = 700;
    // Near a root, we have unbounded relative error. So for functions with roots, we define an ULP clip:
    PreciseReal clip = 2.5;
    // Should we perturb the abscissas? i.e., should we compute the high precision function f at x,
    // and the low precision function at the nearest representable x̂ to x?
    // Or should we compute both the high precision and low precision function at a low precision representable x̂?
    bool perturb_abscissas = false;
    auto plot = ulps_plot<decltype(ai_precise), PreciseReal, CoarseReal>(ai_precise, CoarseReal(-3), CoarseReal(3), samples, perturb_abscissas);
    // Note the argument chaining:
    plot.clip(clip).width(width);
    // Sometimes it's useful to set a title, but in many cases it's more useful to just use a caption.
    //std::string title = "Airy Ai ULP plot at " + boost::core::demangle(typeid(CoarseReal).name()) + " precision";
    //plot.title(title);
    plot.vertical_lines(6);
    plot.add_fn(ai_coarse);
    // You can write the plot to a stream:
    //std::cout << plot;
    // Or to a file:
    plot.write(filename);

    // Don't like the default dark theme?
    plot.background_color("white").font_color("black");
    filename =  "airy_ai_" + boost::core::demangle(typeid(CoarseReal).name()) + "_white.svg";
    plot.write(filename);

    // Don't like the envelope?
    plot.ulp_envelope(false);
    filename =  "airy_ai_" + boost::core::demangle(typeid(CoarseReal).name()) + "_white_no_envelope.svg";
    plot.write(filename);
}
