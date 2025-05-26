/*
 * Copyright Nick Thompson, 2020
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#include <cstdint>
#include <cmath>
#include <boost/math/quadrature/wavelet_transforms.hpp>


int main()
{
    using boost::math::quadrature::daubechies_wavelet_transform;
    double a = 1.3;
    auto f = [&a](double t) {
        if(t==0) {
            return double(0);
        }
        return std::sin(a/t);
    };

    auto Wf = daubechies_wavelet_transform<decltype(f), double, 8>(f);

    double s = 7;
    double t = 0;

    auto g = [&a](double t)->std::complex<double> {
        if (t==0) {
            return {0.0, 0.0};
        }
        return std::exp(std::complex<double>(0.0, a/t));
    };

    auto Wg = daubechies_wavelet_transform<decltype(g), double, 8>(g);
    std::cout << "W[f](s,t) = " << Wf(s,t) << "\n";
    std::cout << "W[g](s,t) = " << Wg(s, t) << "\n";
    std::cout << Wg(0.0, 3.5) << "\n";
    std::cout << Wf(0.0, 4.8) << "\n";
    std::cout << "W[f](-s,t) = " << Wf(-s, t) << "\n";
    std::cout << "W[g](-s,t) = " << Wg(-s, t) << "\n";

}
