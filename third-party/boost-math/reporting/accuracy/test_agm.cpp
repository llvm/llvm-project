//  (C) Copyright Nick Thompson 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <iostream>
#include <boost/math/tools/ulps_plot.hpp>
#include <boost/core/demangle.hpp>
#include <boost/math/tools/agm.hpp>
#include <boost/multiprecision/float128.hpp>

using boost::math::tools::ulps_plot;
using boost::math::tools::agm;

int main() {
    using PreciseReal = boost::multiprecision::float128;
    using CoarseReal = float;

    auto agm_coarse = [](CoarseReal x) {
        return agm<CoarseReal>(x, CoarseReal(1));
    };
    auto agm_precise = [](PreciseReal x) {
        return agm<PreciseReal>(x, PreciseReal(1));
    };

    std::string filename = "agm_" + boost::core::demangle(typeid(CoarseReal).name()) + ".svg";
    int samples = 2500;
    int width = 1100;
    PreciseReal clip = 100;
    auto plot = ulps_plot<decltype(agm_precise), PreciseReal, CoarseReal>(agm_precise, CoarseReal(0), CoarseReal(10000), samples);
    plot.clip(clip).width(width);
    std::string title = "AGM ULP plot at " + boost::core::demangle(typeid(CoarseReal).name()) + " precision";
    //plot.title(title);
    plot.vertical_lines(10);
    plot.add_fn(agm_coarse);
    plot.write(filename);
}
