//  (C) Copyright Evan Miller 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <boost/math/tools/ulps_plot.hpp>
#include <boost/core/demangle.hpp>
#include <boost/math/special_functions/jacobi_theta.hpp>

using boost::math::tools::ulps_plot;

int main() {
    using PreciseReal = long double;
    using CoarseReal = float;

    auto jacobi_theta1_coarse = [](CoarseReal q) {
        return boost::math::jacobi_theta1<CoarseReal>(5.0, q);
    };
    auto jacobi_theta1_precise = [](PreciseReal q) {
        return boost::math::jacobi_theta1<PreciseReal>(5.0, q);
    };

    auto jacobi_theta2_coarse = [](CoarseReal q) {
        return boost::math::jacobi_theta2<CoarseReal>(0.4, q);
    };
    auto jacobi_theta2_precise = [](PreciseReal q) {
        return boost::math::jacobi_theta2<PreciseReal>(0.4, q);
    };

    auto jacobi_theta3_coarse = [](CoarseReal q) {
        return boost::math::jacobi_theta3<CoarseReal>(0.4, q);
    };
    auto jacobi_theta3_precise = [](PreciseReal q) {
        return boost::math::jacobi_theta3<PreciseReal>(0.4, q);
    };

    auto jacobi_theta4_coarse = [](CoarseReal q) {
        return boost::math::jacobi_theta4<CoarseReal>(5.0, q);
    };
    auto jacobi_theta4_precise = [](PreciseReal q) {
        return boost::math::jacobi_theta4<PreciseReal>(5.0, q);
    };

    int samples = 2500;
    int width = 800;
    PreciseReal clip = 100;

    std::string filename1 = "jacobi_theta1q_" + boost::core::demangle(typeid(CoarseReal).name()) + ".svg";
    auto plot1 = ulps_plot<decltype(jacobi_theta1_precise), PreciseReal, CoarseReal>(jacobi_theta1_precise, CoarseReal(0), CoarseReal(0.999999), samples);
    plot1.clip(clip).width(width);
    std::string title1 = "jacobi_theta1(5.0, q) ULP plot at " + boost::core::demangle(typeid(CoarseReal).name()) + " precision";
    plot1.title(title1);
    plot1.vertical_lines(10);
    plot1.add_fn(jacobi_theta1_coarse);
    plot1.write(filename1);

    std::string filename2 = "jacobi_theta2q_" + boost::core::demangle(typeid(CoarseReal).name()) + ".svg";
    auto plot2 = ulps_plot<decltype(jacobi_theta2_precise), PreciseReal, CoarseReal>(jacobi_theta2_precise, CoarseReal(0), CoarseReal(0.999999), samples);
    plot2.clip(clip).width(width);
    std::string title2 = "jacobi_theta2(0.4, q) ULP plot at " + boost::core::demangle(typeid(CoarseReal).name()) + " precision";
    plot2.title(title2);
    plot2.vertical_lines(10);
    plot2.add_fn(jacobi_theta2_coarse);
    plot2.write(filename2);

    std::string filename3 = "jacobi_theta3q_" + boost::core::demangle(typeid(CoarseReal).name()) + ".svg";
    auto plot3 = ulps_plot<decltype(jacobi_theta3_precise), PreciseReal, CoarseReal>(jacobi_theta3_precise, CoarseReal(0), CoarseReal(0.999999), samples);
    plot3.clip(clip).width(width);
    std::string title3 = "jacobi_theta3(0.4, q) ULP plot at " + boost::core::demangle(typeid(CoarseReal).name()) + " precision";
    plot3.title(title3);
    plot3.vertical_lines(10);
    plot3.add_fn(jacobi_theta3_coarse);
    plot3.write(filename3);

    std::string filename4 = "jacobi_theta4q_" + boost::core::demangle(typeid(CoarseReal).name()) + ".svg";
    auto plot4 = ulps_plot<decltype(jacobi_theta4_precise), PreciseReal, CoarseReal>(jacobi_theta4_precise, CoarseReal(0), CoarseReal(0.999999), samples);
    plot4.clip(clip).width(width);
    std::string title4 = "jacobi_theta4(5.0, q) ULP plot at " + boost::core::demangle(typeid(CoarseReal).name()) + " precision";
    plot4.title(title4);
    plot4.vertical_lines(10);
    plot4.add_fn(jacobi_theta4_coarse);
    plot4.write(filename4);
}
