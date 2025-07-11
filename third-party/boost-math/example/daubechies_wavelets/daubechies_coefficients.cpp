/*
 * Copyright Nick Thompson, 2018
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <complex>
#include <bitset>
#include <boost/math/tools/assert.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/tools/polynomial.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/multiprecision/cpp_complex.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>

typedef boost::multiprecision::float128 float128_t;
#else
typedef boost::multiprecision::cpp_bin_float_quad float128_t;
#endif
//#include <boost/multiprecision/complex128.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>

using std::string;
using boost::math::tools::polynomial;
using boost::math::binomial_coefficient;
using boost::math::tools::schroder_iterate;
using boost::math::tools::halley_iterate;
using boost::math::tools::newton_raphson_iterate;
using boost::math::tools::complex_newton;
using boost::math::constants::half;
using boost::math::constants::root_two;
using boost::math::constants::pi;
using boost::math::quadrature::gauss_kronrod;
using boost::multiprecision::cpp_bin_float_100;
using boost::multiprecision::cpp_complex_100;

template<class Complex>
std::vector<std::pair<Complex, Complex>> find_roots(size_t p)
{
    // Initialize the polynomial; see Mallat, A Wavelet Tour of Signal Processing, equation 7.96
    BOOST_MATH_ASSERT(p>0);
    typedef typename Complex::value_type Real;
    std::vector<Complex> coeffs(p);
    for (size_t k = 0; k < coeffs.size(); ++k)
    {
        coeffs[k] = Complex(binomial_coefficient<Real>(p-1+k, k), 0);
    }

    polynomial<Complex> P(std::move(coeffs));
    polynomial<Complex> Pcopy = P;
    polynomial<Complex> Pcopy_prime = P.prime();
    auto orig = [&](Complex z) { return std::make_pair<Complex, Complex>(Pcopy(z), Pcopy_prime(z)); };

    polynomial<Complex> P_prime = P.prime();

    // Polynomial is of degree p-1.

    std::vector<Complex> roots(p-1, {std::numeric_limits<Real>::quiet_NaN(),std::numeric_limits<Real>::quiet_NaN()});
    size_t i = 0;
    while(P.size() > 1)
    {
        Complex guess = {0.0, 1.0};
        std::cout << std::setprecision(std::numeric_limits<Real>::digits10+3);

        auto f = [&](Complex x)->std::pair<Complex, Complex>
        {
            return std::make_pair<Complex, Complex>(P(x), P_prime(x));
        };

        Complex r = complex_newton(f, guess);
        using std::isnan;
        if(isnan(r.real()))
        {
            int i = 50;
            do {
                // Try a different guess
                guess *= Complex(1.0,-1.0);
                r = complex_newton(f, guess);
                std::cout << "New guess: " << guess << ", result? " << r << std::endl;

            } while (isnan(r.real()) && i-- > 0);

            if (isnan(r.real()))
            {
                std::cout << "Polynomial that killed the process: " << P << std::endl;
                throw std::logic_error("Newton iteration did not converge");
            }
        }
        // Refine r with the original function.
        // We only use the polynomial division to ensure we don't get the same root over and over.
        // However, the division induces error which can grow quickly-or slowly! See Numerical Recipes, section 9.5.1.
        r = complex_newton(orig, r);
        if (isnan(r.real()))
        {
            throw std::logic_error("Found a root for the deflated polynomial which is not a root for the original. Indicative of catastrophic numerical error.");
        }
        // Test the root:
        using std::sqrt;
        Real tol = sqrt(sqrt(std::numeric_limits<Real>::epsilon()));
        if (norm(Pcopy(r)) > tol)
        {
            std::cout << "This is a bad root: P" <<  r << " = " << Pcopy(r) << std::endl;
            std::cout << "Reduced polynomial leading to bad root: " << P << std::endl;
            throw std::logic_error("Donezo.");
        }

        BOOST_MATH_ASSERT(i < roots.size());
        roots[i] = r;
        ++i;
        polynomial<Complex> q{-r, {1,0}};
        // This optimization breaks at p = 11. I have no clue why.
        // Unfortunate, because I expect it to be considerably more stable than
        // repeatedly dividing by the complex root.
        /*polynomial<Complex> q;
        if (r.imag() > sqrt(std::numeric_limits<Real>::epsilon()))
        {
            // Then the complex conjugate is also a root:
            using std::conj;
            using std::norm;
            BOOST_MATH_ASSERT(i < roots.size());
            roots[i] = conj(r);
            ++i;
            q = polynomial<Complex>({{norm(r), 0}, {-2*r.real(),0}, {1,0}});
        }
        else
        {
            // The imaginary part is numerical noise:
            r.imag() = 0;
            q = polynomial<Complex>({-r, {1,0}});
        }*/


        auto PR = quotient_remainder(P, q);
        // I should validate that the remainder is small, but . . .
        //std::cout << "Remainder = " << PR.second<< std::endl;

        P = PR.first;
        P_prime = P.prime();
    }

    std::vector<std::pair<Complex, Complex>> Qroots(p-1);
    for (size_t i = 0; i < Qroots.size(); ++i)
    {
        Complex y = roots[i];
        Complex z1 = static_cast<Complex>(1) - static_cast<Complex>(2)*y + static_cast<Complex>(2)*sqrt(y*(y-static_cast<Complex>(1)));
        Complex z2 = static_cast<Complex>(1) - static_cast<Complex>(2)*y - static_cast<Complex>(2)*sqrt(y*(y-static_cast<Complex>(1)));
        Qroots[i] = {z1, z2};
    }

    return Qroots;
}

template<class Complex>
std::vector<typename Complex::value_type> daubechies_coefficients(std::vector<std::pair<Complex, Complex>> const & Qroots)
{
    typedef typename Complex::value_type Real;
    size_t p = Qroots.size() + 1;
    // Choose the minimum abs root; see Mallat, discussion just after equation 7.98
    std::vector<Complex> chosen_roots(p-1);
    for (size_t i = 0; i < p - 1; ++i)
    {
        if(norm(Qroots[i].first) <= 1)
        {
            chosen_roots[i] = Qroots[i].first;
        }
        else
        {
            BOOST_MATH_ASSERT(norm(Qroots[i].second) <= 1);
            chosen_roots[i] = Qroots[i].second;
        }
    }

    polynomial<Complex> R{1};
    for (size_t i = 0; i < p-1; ++i)
    {
        Complex ak = chosen_roots[i];
        R *= polynomial<Complex>({-ak/(static_cast<Complex>(1)-ak), static_cast<Complex>(1)/(static_cast<Complex>(1)-ak)});
    }
    polynomial<Complex> a{{half<Real>(), 0}, {half<Real>(),0}};
    polynomial<Complex> poly = root_two<Real>()*pow(a, p)*R;
    std::vector<Complex> result = poly.data();
    // If we reverse, we get the Numerical Recipes and Daubechies convention.
    // If we don't reverse, we get the Pywavelets and Mallat convention.
    // I believe this is because of the sign convention on the DFT, which differs between Daubechies and Mallat.
    // You implement a dot product in Daubechies/NR convention, and a convolution in PyWavelets/Mallat convention.
    std::reverse(result.begin(), result.end());
    std::vector<Real> h(result.size());
    for (size_t i = 0; i < result.size(); ++i)
    {
        Complex r = result[i];
        BOOST_MATH_ASSERT(r.imag() < sqrt(std::numeric_limits<Real>::epsilon()));
        h[i] = r.real();
    }

    // Quick sanity check: We could check all vanishing moments, but that sum is horribly ill-conditioned too!
    Real sum = 0;
    Real scale = 0;
    for (size_t i = 0; i < h.size(); ++i)
    {
        sum += h[i];
        scale += h[i]*h[i];
    }
    BOOST_MATH_ASSERT(abs(scale -1) < sqrt(std::numeric_limits<Real>::epsilon()));
    BOOST_MATH_ASSERT(abs(sum - root_two<Real>()) < sqrt(std::numeric_limits<Real>::epsilon()));
    return h;
}

int main()
{
    typedef boost::multiprecision::cpp_complex<500> Complex;
    size_t p_max = 20;
    std::ofstream fs{"daubechies_filters.hpp"};
    fs << "/*\n"
       << " * Copyright Nick Thompson, 2019\n"
       << " * Use, modification and distribution are subject to the\n"
       << " * Boost Software License, Version 1.0. (See accompanying file\n"
       << " * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n"
       << " */\n"
       << "#ifndef BOOST_MATH_FILTERS_DAUBECHIES_HPP\n"
       << "#define BOOST_MATH_FILTERS_DAUBECHIES_HPP\n"
       << "#include <array>\n"
       << "#include <limits>\n"
       << "#include <boost/math/tools/big_constant.hpp>\n\n"
       << "namespace boost::math::filters {\n\n"
       << "template <typename Real, unsigned p>\n"
       << "constexpr std::array<Real, 2*p> daubechies_scaling_filter()\n"
       << "{\n"
       << "    static_assert(p < " << p_max << ", \"Filter coefficients only implemented up to " << p_max - 1 << ".\");\n";

    for(size_t p = 1; p < p_max; ++p)
    {
        fs << std::setprecision(std::numeric_limits<boost::multiprecision::cpp_bin_float_oct>::max_digits10);
        auto roots = find_roots<Complex>(p);
        auto h = daubechies_coefficients(roots);
        fs << "    if constexpr (p == " << p << ") {\n";
        fs << "       return {";
        for (size_t i = 0; i < h.size() - 1; ++i) {
            fs << "BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits, " << h[i] << "), ";
        }
        fs << "BOOST_MATH_BIG_CONSTANT(Real, std::numeric_limits<Real>::digits, " << h[h.size()-1] << ") };\n";
        fs << "    }\n";
    }

    fs << "}\n\n";

    fs << "template<class Real, size_t p>\n";
    fs << "std::array<Real, 2*p> daubechies_wavelet_filter() {\n";
    fs << "    std::array<Real, 2*p> g;\n";
    fs << "    auto h = daubechies_scaling_filter<Real, p>();\n";
    fs << "    for (size_t i = 0; i < g.size(); i += 2)\n";
    fs << "    {\n";
    fs << "        g[i] = h[g.size() - i - 1];\n";
    fs << "        g[i+1] = -h[g.size() - i - 2];\n";
    fs << "    }\n";
    fs << "    return g;\n";
    fs << "}\n\n";
    fs << "} // namespaces\n";
    fs << "#endif\n";
    fs.close();
}
