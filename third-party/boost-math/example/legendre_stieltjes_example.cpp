// Copyright Nick Thompson 2017.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <string>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/legendre_stieltjes.hpp>

using boost::math::legendre_p;
using boost::math::legendre_p_zeros;
using boost::math::legendre_p_prime;
using boost::math::legendre_stieltjes;
using boost::multiprecision::cpp_bin_float_quad;
using boost::multiprecision::cpp_bin_float_100;
using boost::multiprecision::cpp_dec_float_100;

template<class Real>
void gauss_kronrod_rule(size_t order)
{
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10);
    std::cout << std::fixed;
    auto gauss_nodes = boost::math::legendre_p_zeros<Real>(order);
    auto E = legendre_stieltjes<Real>(order + 1);
    std::vector<Real> gauss_weights(gauss_nodes.size(), std::numeric_limits<Real>::quiet_NaN());
    std::vector<Real> gauss_kronrod_weights(gauss_nodes.size(), std::numeric_limits<Real>::quiet_NaN());
    for (size_t i = 0; i < gauss_nodes.size(); ++i)
    {
        Real node = gauss_nodes[i];
        Real lp = legendre_p_prime<Real>(order, node);
        gauss_weights[i] = 2/( (1-node*node)*lp*lp);
        // P_n(x) = (2n)!/(2^n (n!)^2) pi_n(x), where pi_n is the monic Legendre polynomial.
        gauss_kronrod_weights[i] = gauss_weights[i] + static_cast<Real>(2)/(static_cast<Real>(order+1)*legendre_p_prime(order, node)*E(node));
    }

    std::cout << "static const std::vector<Real> gauss_nodes {\n";
    for (auto const & node : gauss_nodes)
    {
        std::cout << "    boost::lexical_cast<Real>(\"" << node << "\"),\n";
    }
    std::cout << "};\n\n";

    std::cout << "static const std::vector<Real> gauss_weights {\n";
    for (auto const & weight : gauss_weights)
    {
        std::cout << "    boost::lexical_cast<Real>(\"" << weight << "\"),\n";
    }
    std::cout << "};\n\n";

    std::cout << "static const std::vector<Real> gauss_kronrod_weights {\n";
    for (auto const & weight : gauss_kronrod_weights)
    {
        std::cout << "    boost::lexical_cast<Real>(\"" << weight << "\"),\n";
    }
    std::cout << "};\n\n";

    auto kronrod_nodes = E.zeros();
    std::vector<Real> kronrod_weights(kronrod_nodes.size());
    for (size_t i = 0; i < kronrod_weights.size(); ++i)
    {
        Real node = kronrod_nodes[i];
        kronrod_weights[i] = static_cast<Real>(2)/(static_cast<Real>(order+1)*legendre_p(order, node)*E.prime(node));
    }

    std::cout << "static const std::vector<Real> kronrod_nodes {\n";
    for (auto node : kronrod_nodes)
    {
        std::cout << "    boost::lexical_cast<Real>(\"" << node << "\"),\n";
    }
    std::cout << "};\n\n";

    std::cout << "static const std::vector<Real> kronrod_weights {\n";
    for (auto const & weight : kronrod_weights)
    {
        std::cout << "    boost::lexical_cast<Real>(\"" << weight << "\"),\n";
    }
    std::cout << "};\n\n";

}

int main()
{
    //std::cout << "Gauss-Kronrod 7-15 Rule:\n";
    //gauss_kronrod_rule<cpp_dec_float_100>(7);

    //std::cout << "\n\nGauss-Kronrod 10-21 Rule:\n";
    //gauss_kronrod_rule<cpp_dec_float_100>(10);

    std::cout << "\n\nGauss-Kronrod 15-31 Rule:\n";
    gauss_kronrod_rule<cpp_dec_float_100>(15);
    /*
    std::cout << "\n\nGauss-Kronrod 20-41 Rule:\n";
    gauss_kronrod_rule<cpp_dec_float_100>(20);

    std::cout << "\n\nGauss-Kronrod 25-51 Rule:\n";
    gauss_kronrod_rule<cpp_dec_float_100>(25);

    std::cout << "\n\nGauss-Kronrod 30-61 Rule:\n";
    gauss_kronrod_rule<cpp_dec_float_100>(30);

    std::cout << "\n\nGauss-Kronrod 35-71 Rule:\n";
    gauss_kronrod_rule<cpp_dec_float_100>(35);

    std::cout << "\n\nGauss-Kronrod 40-81 Rule:\n";
    gauss_kronrod_rule<cpp_dec_float_100>(40);*/
}
