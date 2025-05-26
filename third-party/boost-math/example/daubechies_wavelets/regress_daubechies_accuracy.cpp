//  (C) Copyright Nick Thompson 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//


#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <cmath>
#include <vector>
#include <iomanip>
#include <boost/algorithm/string.hpp>
#include <boost/math/statistics/linear_regression.hpp>
#include <boost/assert.hpp>


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: ./regress_accuracy.x foo.csv\n";
        return 1;
    }
    std::string filename = std::string(argv[1]);
    std::ifstream ifs(filename.c_str());
    if (!ifs.good())
    {
        std::cerr << "Couldn't find file " << filename << "\n";
        return 1;
    }
    std::map<std::string, std::vector<double>> m;

    std::string header_line;
    std::getline(ifs, header_line);
    std::cout << "Header line = " << header_line << "\n";
    std::vector<std::string> header_strs;
    boost::split(header_strs, header_line, boost::is_any_of(","));
    for (auto & s : header_strs) {
        boost::algorithm::trim(s);
    }

    std::string line;
    std::vector<double> r;
    std::vector<double> matched_holder;
    std::vector<double> linear;
    std::vector<double> quadratic_b_spline;
    std::vector<double> cubic_b_spline;
    std::vector<double> quintic_b_spline;
    std::vector<double> cubic_hermite;
    std::vector<double> pchip;
    std::vector<double> makima;
    std::vector<double> fotaylor;
    std::vector<double> quintic_hermite;
    std::vector<double> sotaylor;
    std::vector<double> totaylor;
    std::vector<double> septic_hermite;
    while(std::getline(ifs, line))
    {
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(","));
        for (auto & s : strs)
        {
            boost::algorithm::trim(s);
        }
        std::vector<double> v(strs.size(), std::numeric_limits<double>::quiet_NaN());
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = std::stod(strs[i]);
        }
        r.push_back(v[0]);
        matched_holder.push_back(std::log2(v[1]));
        linear.push_back(std::log2(v[2]));
        quadratic_b_spline.push_back(std::log2(v[3]));
        cubic_b_spline.push_back(std::log2(v[4]));
        quintic_b_spline.push_back(std::log2(v[5]));
        cubic_hermite.push_back(std::log2(v[6]));
        pchip.push_back(std::log2(v[7]));
        makima.push_back(std::log2(v[8]));
        fotaylor.push_back(std::log2(v[9]));
        if (v.size() > 10) {
            quintic_hermite.push_back(std::log2(v[10]));
            sotaylor.push_back(std::log2(v[11]));
        }
        if (v.size() > 12) {
            totaylor.push_back(std::log2(v[12]));
            septic_hermite.push_back(std::log2(v[13]));
        }
    }

    std::cout << std::fixed << std::setprecision(16);
    auto q  = boost::math::statistics::simple_ordinary_least_squares_with_R_squared(r, matched_holder);
    BOOST_ASSERT(std::get<1>(q) < 0);
    std::cout << "Matched Holder    : " << std::get<0>(q) << " - " << std::abs(std::get<1>(q)) << "r, R^2 = " << std::get<2>(q) << "\n";

    q  = boost::math::statistics::simple_ordinary_least_squares_with_R_squared(r, linear);
    BOOST_ASSERT(std::get<1>(q) < 0);
    std::cout << "Linear            : " << std::get<0>(q) << " - " << std::abs(std::get<1>(q)) << "r, R^2 = " << std::get<2>(q) << "\n";

    q  = boost::math::statistics::simple_ordinary_least_squares_with_R_squared(r, quadratic_b_spline);
    BOOST_ASSERT(std::get<1>(q) < 0);
    std::cout << "Quadratic B-spline: " << std::get<0>(q) << " - " << std::abs(std::get<1>(q)) << "r, R^2 = " << std::get<2>(q) << "\n";

    q  = boost::math::statistics::simple_ordinary_least_squares_with_R_squared(r, cubic_b_spline);
    BOOST_ASSERT(std::get<1>(q) < 0);
    std::cout << "Cubic B-spline    : " << std::get<0>(q) << " - " << std::abs(std::get<1>(q)) << "r, R^2 = " << std::get<2>(q) << "\n";

    q  = boost::math::statistics::simple_ordinary_least_squares_with_R_squared(r, quintic_b_spline);
    BOOST_ASSERT(std::get<1>(q) < 0);
    std::cout << "Quintic B-spline  : " << std::get<0>(q) << " - " << std::abs(std::get<1>(q)) << "r, R^2 = " << std::get<2>(q) << "\n";

    q  = boost::math::statistics::simple_ordinary_least_squares_with_R_squared(r, cubic_hermite);
    BOOST_ASSERT(std::get<1>(q) < 0);
    std::cout << "Cubic Hermite     : " << std::get<0>(q) << " - " << std::abs(std::get<1>(q)) << "r, R^2 = " << std::get<2>(q) << "\n";

    q  = boost::math::statistics::simple_ordinary_least_squares_with_R_squared(r, pchip);
    BOOST_ASSERT(std::get<1>(q) < 0);
    std::cout << "PCHIP             : " << std::get<0>(q) << " - " << std::abs(std::get<1>(q)) << "r, R^2 = " << std::get<2>(q) << "\n";

    q  = boost::math::statistics::simple_ordinary_least_squares_with_R_squared(r, makima);
    BOOST_ASSERT(std::get<1>(q) < 0);
    std::cout << "Makima            : " << std::get<0>(q) << " - " << std::abs(std::get<1>(q)) << "r, R^2 = " << std::get<2>(q) << "\n";

    q  = boost::math::statistics::simple_ordinary_least_squares_with_R_squared(r, fotaylor);
    BOOST_ASSERT(std::get<1>(q) < 0);
    std::cout << "First-order Taylor: " << std::get<0>(q) << " - " << std::abs(std::get<1>(q)) << "r, R^2 = " << std::get<2>(q) << "\n";

    if (sotaylor.size() > 0)
    {
    q  = boost::math::statistics::simple_ordinary_least_squares_with_R_squared(r, quintic_hermite);
    BOOST_ASSERT(std::get<1>(q) < 0);
    std::cout << "Quintic Hermite   : " << std::get<0>(q) << " - " << std::abs(std::get<1>(q)) << "r, R^2 = " << std::get<2>(q) << "\n";

    q  = boost::math::statistics::simple_ordinary_least_squares_with_R_squared(r, sotaylor);
    BOOST_ASSERT(std::get<1>(q) < 0);
    std::cout << "2nd order Taylor  : " << std::get<0>(q) << " - " << std::abs(std::get<1>(q)) << "r, R^2 = " << std::get<2>(q) << "\n";

    }

    if (totaylor.size() > 0)
    {
    q  = boost::math::statistics::simple_ordinary_least_squares_with_R_squared(r, totaylor);
    BOOST_ASSERT(std::get<1>(q) < 0);
    std::cout << "3rd order Taylor  : " << std::get<0>(q) << " - " << std::abs(std::get<1>(q)) << "r, R^2 = " << std::get<2>(q) << "\n";

    q  = boost::math::statistics::simple_ordinary_least_squares_with_R_squared(r, septic_hermite);
    BOOST_ASSERT(std::get<1>(q) < 0);
    std::cout << "Septic Hermite    : " << std::get<0>(q) << " - " << std::abs(std::get<1>(q)) << "r, R^2 = " << std::get<2>(q) << "\n";

    }

}
