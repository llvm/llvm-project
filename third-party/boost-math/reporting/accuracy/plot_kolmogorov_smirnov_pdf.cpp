//  Copyright Evan Miller 2020
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <boost/math/tools/ulps_plot.hpp>
#include <boost/core/demangle.hpp>
#include <boost/math/distributions/kolmogorov_smirnov.hpp>

using boost::math::tools::ulps_plot;

int main() {
    using PreciseReal = long double;
    using CoarseReal = float;

    boost::math::kolmogorov_smirnov_distribution<CoarseReal> dist_coarse(10);
    auto pdf_coarse = [&, dist_coarse](CoarseReal x) {
        return boost::math::pdf(dist_coarse, x);
    };
    boost::math::kolmogorov_smirnov_distribution<PreciseReal> dist_precise(10);
    auto pdf_precise = [&, dist_precise](PreciseReal x) {
        return boost::math::pdf(dist_precise, x);
    };

    int samples = 2500;
    int width = 800;
    PreciseReal clip = 100;

    std::string filename1 = "kolmogorov_smirnov_pdf_" + boost::core::demangle(typeid(CoarseReal).name()) + ".svg";
    auto plot1 = ulps_plot<decltype(pdf_precise), PreciseReal, CoarseReal>(pdf_precise, 0.0, 1.0, samples);
    plot1.clip(clip).width(width);
    std::string title1 = "Kolmogorov-Smirnov PDF (N=10) ULP plot at " + boost::core::demangle(typeid(CoarseReal).name()) + " precision";
    plot1.title(title1);
    plot1.vertical_lines(10);
    plot1.add_fn(pdf_coarse);
    plot1.write(filename1);
}
