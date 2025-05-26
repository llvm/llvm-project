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

    CoarseReal q = 0.5;

    auto jacobi_theta1_coarse = [=](CoarseReal z) {
        return boost::math::jacobi_theta1<CoarseReal>(z, q);
    };
    auto jacobi_theta1_precise = [=](PreciseReal z) {
        return boost::math::jacobi_theta1<PreciseReal>(z, q);
    };
    auto jacobi_theta2_coarse = [=](CoarseReal z) {
        return boost::math::jacobi_theta2<CoarseReal>(z, q);
    };
    auto jacobi_theta2_precise = [=](PreciseReal z) {
        return boost::math::jacobi_theta2<PreciseReal>(z, q);
    };
    auto jacobi_theta3_coarse = [=](CoarseReal z) {
        return boost::math::jacobi_theta3m1<CoarseReal>(z, q);
    };
    auto jacobi_theta3_precise = [=](PreciseReal z) {
        return boost::math::jacobi_theta3m1<PreciseReal>(z, q);
    };
    auto jacobi_theta4_coarse = [=](CoarseReal z) {
        return boost::math::jacobi_theta4m1<CoarseReal>(z, q);
    };
    auto jacobi_theta4_precise = [=](PreciseReal z) {
        return boost::math::jacobi_theta4m1<PreciseReal>(z, q);
    };

    int samples = 2500;
    int width = 800;
    PreciseReal clip = 100;

    std::string filename1 = "jacobi_theta1_" + boost::core::demangle(typeid(CoarseReal).name()) + ".svg";
    auto plot1 = ulps_plot<decltype(jacobi_theta1_precise), PreciseReal, CoarseReal>(jacobi_theta1_precise, 0.0, boost::math::constants::two_pi<CoarseReal>(), samples);
    plot1.clip(clip).width(width);
    std::string title1 = "jacobi_theta1(x, 0.5) ULP plot at " + boost::core::demangle(typeid(CoarseReal).name()) + " precision";
    plot1.title(title1);
    plot1.vertical_lines(10);
    plot1.add_fn(jacobi_theta1_coarse);
    plot1.write(filename1);

    std::string filename2 = "jacobi_theta2_" + boost::core::demangle(typeid(CoarseReal).name()) + ".svg";
    auto plot2 = ulps_plot<decltype(jacobi_theta2_precise), PreciseReal, CoarseReal>(jacobi_theta2_precise, 0.0, boost::math::constants::two_pi<CoarseReal>(), samples);
    plot2.clip(clip).width(width);
    std::string title2 = "jacobi_theta2(x, 0.5) ULP plot at " + boost::core::demangle(typeid(CoarseReal).name()) + " precision";
    plot2.title(title2);
    plot2.vertical_lines(10);
    plot2.add_fn(jacobi_theta2_coarse);
    plot2.write(filename2);

    std::string filename3 = "jacobi_theta3_" + boost::core::demangle(typeid(CoarseReal).name()) + ".svg";
    auto plot3 = ulps_plot<decltype(jacobi_theta3_precise), PreciseReal, CoarseReal>(jacobi_theta3_precise, 0.0, boost::math::constants::two_pi<CoarseReal>(), samples);
    plot3.clip(clip).width(width);
    std::string title3 = "jacobi_theta3m1(x, 0.5) ULP plot at " + boost::core::demangle(typeid(CoarseReal).name()) + " precision";
    plot3.title(title3);
    plot3.vertical_lines(10);
    plot3.add_fn(jacobi_theta3_coarse);
    plot3.write(filename3);

    std::string filename4 = "jacobi_theta4_" + boost::core::demangle(typeid(CoarseReal).name()) + ".svg";
    auto plot4 = ulps_plot<decltype(jacobi_theta4_precise), PreciseReal, CoarseReal>(jacobi_theta4_precise, 0.0, boost::math::constants::two_pi<CoarseReal>(), samples);
    plot4.clip(clip).width(width);
    std::string title4 = "jacobi_theta4m1(x, 0.5) ULP plot at " + boost::core::demangle(typeid(CoarseReal).name()) + " precision";
    plot4.title(title4);
    plot4.vertical_lines(10);
    plot4.add_fn(jacobi_theta4_coarse);
    plot4.write(filename4);
}
