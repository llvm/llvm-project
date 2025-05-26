/*
 * Copyright Nick Thompson, 2020
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#include <iostream>
#include <boost/core/demangle.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/ext/std/integer_sequence.hpp>

#include <boost/multiprecision/float128.hpp>
#include <boost/math/special_functions/daubechies_wavelet.hpp>
#include <quicksvg/graph_fn.hpp>
#include <quicksvg/ulp_plot.hpp>


using boost::multiprecision::float128;
constexpr const int GRAPH_WIDTH = 700;

template<typename Real, int p>
void plot_psi(int grid_refinements = -1)
{
    auto psi = boost::math::daubechies_wavelet<Real, p>();
    if (grid_refinements >= 0)
    {
        psi = boost::math::daubechies_wavelet<Real, p>(grid_refinements);
    }
    auto [a, b] = psi.support();
    std::string title = "Daubechies " + std::to_string(p) + " wavelet";
    title = "";
    std::string filename = "daubechies_" + std::to_string(p) + "_wavelet.svg";
    int samples = 1024;
    quicksvg::graph_fn daub(a, b, title, filename, samples, GRAPH_WIDTH);
    daub.set_gridlines(8, 2*p-1);
    daub.set_stroke_width(1);
    daub.add_fn(psi);
    daub.write_all();
}

template<typename Real, int p>
void plot_dpsi(int grid_refinements = -1)
{
    auto psi = boost::math::daubechies_wavelet<Real, p>();
    if (grid_refinements >= 0)
    {
        psi = boost::math::daubechies_wavelet<Real, p>(grid_refinements);
    }
    auto [a, b] = psi.support();
    std::string title = "Daubechies " + std::to_string(p) + " wavelet derivative";
    title = "";
    std::string filename = "daubechies_" + std::to_string(p) + "_wavelet_prime.svg";
    int samples = 1024;
    quicksvg::graph_fn daub(a, b, title, filename, samples, GRAPH_WIDTH);
    daub.set_stroke_width(1);
    daub.set_gridlines(8, 2*p-1);
    auto dpsi = [psi](Real x)->Real { return psi.prime(x); };
    daub.add_fn(dpsi);
    daub.write_all();
}

template<typename Real, int p>
void plot_convergence()
{
    auto psi1 = boost::math::daubechies_wavelet<Real, p>(1);
    auto [a, b] = psi1.support();
    std::string title = "Daubechies " + std::to_string(p) + " wavelet at  1 (orange), 2 (red), and 21 (blue) grid refinements";
    title = "";
    std::string filename = "daubechies_" + std::to_string(p) + "_wavelet_convergence.svg";

    quicksvg::graph_fn daub(a, b, title, filename, 1024, GRAPH_WIDTH);
    daub.set_stroke_width(1);
    daub.set_gridlines(8, 2*p-1);

    daub.add_fn(psi1, "orange");
    auto psi2 = boost::math::daubechies_wavelet<Real, p>(2);
    daub.add_fn(psi2, "red");

    auto psi21 = boost::math::daubechies_wavelet<Real, p>(21);
    daub.add_fn(psi21);

    daub.write_all();
}

template<typename Real, int p>
void plot_condition_number()
{
    using std::abs;
    using std::log;
    static_assert(p >= 3, "p = 2 is not differentiable, so condition numbers cannot be effectively evaluated.");
    auto phi = boost::math::daubechies_wavelet<Real, p>();
    Real a = phi.support().first + 1000*std::sqrt(std::numeric_limits<Real>::epsilon());
    Real b = phi.support().second - 1000*std::sqrt(std::numeric_limits<Real>::epsilon());
    std::string title = "log10 of condition number of function evaluation for Daubechies " + std::to_string(p) + " wavelet function.";
    title = "";
    std::string filename = "daubechies_" + std::to_string(p) + "_wavelet_condition_number.svg";


    quicksvg::graph_fn daub(a, b, title, filename, 2048, GRAPH_WIDTH);
    daub.set_stroke_width(1);
    daub.set_gridlines(8, 2*p-1);

    auto cond = [&phi](Real x)
    {
        Real y = phi(x);
        Real dydx = phi.prime(x);
        Real z = abs(x*dydx/y);
        using std::isnan;
        if (z==0)
        {
            return Real(-1);
        }
        if (isnan(z))
        {
            // Graphing libraries don't like nan's:
            return Real(1);
        }
        return log10(z);
    };
    daub.add_fn(cond);
    daub.write_all();
}

template<typename CoarseReal, typename PreciseReal, int p, class PsiPrecise>
void do_ulp(int coarse_refinements, PsiPrecise psi_precise)
{
    auto psi_coarse = boost::math::daubechies_wavelet<CoarseReal, p>(coarse_refinements);

    std::string title = std::to_string(p) + " vanishing moment ULP plot at " + std::to_string(coarse_refinements) + " refinements and " + boost::core::demangle(typeid(CoarseReal).name()) + " precision";
    title = "";

    std::string filename = "daubechies_" + std::to_string(p) + "_wavelet_" + boost::core::demangle(typeid(CoarseReal).name()) + "_" + std::to_string(coarse_refinements) + "_refinements.svg";
    int samples = 20000;
    int clip = 20;
    int horizontal_lines = 8;
    int vertical_lines = 2*p - 1;
    quicksvg::ulp_plot<decltype(psi_coarse), CoarseReal, decltype(psi_precise), PreciseReal>(psi_coarse, psi_precise, CoarseReal(psi_coarse.support().first), psi_coarse.support().second, title, filename, samples, GRAPH_WIDTH, clip, horizontal_lines, vertical_lines);
}


int main()
{
    boost::hana::for_each(std::make_index_sequence<18>(), [&](auto i){ plot_psi<double, i+2>(); });
    boost::hana::for_each(std::make_index_sequence<17>(), [&](auto i){ plot_dpsi<double, i+3>(); });
    boost::hana::for_each(std::make_index_sequence<17>(), [&](auto i){ plot_condition_number<double, i+3>(); });
    boost::hana::for_each(std::make_index_sequence<18>(), [&](auto i){ plot_convergence<double, i+2>(); });

    using PreciseReal = float128;
    using CoarseReal = double;
    int precise_refinements = 22;
    constexpr const int p = 9;
    std::cout << "Computing precise wavelet function in " << boost::core::demangle(typeid(PreciseReal).name()) << " precision.\n";
    auto phi_precise = boost::math::daubechies_wavelet<PreciseReal, p>(precise_refinements);
    std::cout << "Beginning comparison with functions computed in " << boost::core::demangle(typeid(CoarseReal).name()) << " precision.\n";
    for (int i = 7; i <= precise_refinements-1; ++i)
    {
        std::cout << "\tCoarse refinement " << i << "\n";
        do_ulp<CoarseReal, PreciseReal, p>(i, phi_precise);
    }
}
