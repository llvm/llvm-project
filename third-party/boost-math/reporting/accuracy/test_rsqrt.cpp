//  (C) Copyright Nick Thompson 2020.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/float128.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/tools/ulps_plot.hpp>
#include <boost/math/special_functions/rsqrt.hpp>

int main()
{
    using boost::multiprecision::number;
    using PreciseReal = number<boost::multiprecision::mpfr_float_backend<1000>>;
    using CoarseReal = boost::multiprecision::float128;
    using boost::math::tools::ulps_plot;
    std::string filename = "rsqrt_quad_0_100.svg";
    int samples = 2500;
    int width = 1100;
    auto f = [](PreciseReal x) {
        using boost::math::rsqrt;
        return rsqrt(x);
    };
    auto plot03 = ulps_plot<decltype(f), PreciseReal, CoarseReal>(f, (std::numeric_limits<CoarseReal>::min)(), CoarseReal(100), samples);
    plot03.width(width);
    std::string title = "rsqrt ULPs plot at quad precision";
    plot03.title(title);
    plot03.vertical_lines(6);
    auto g = [](CoarseReal x) {
        return boost::math::rsqrt(x);
    };
    plot03.add_fn(g);
    plot03.write(filename);
}
