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
#include <boost/math/special_functions/daubechies_scaling.hpp>
#include <boost/math/tools/ulps_plot.hpp>
#include <quicksvg/graph_fn.hpp>


using boost::multiprecision::float128;
constexpr const int GRAPH_WIDTH = 300;

template<typename Real, int p>
void plot_phi(int grid_refinements = -1)
{
    auto phi = boost::math::daubechies_scaling<Real, p>();
    if (grid_refinements >= 0)
    {
        phi = boost::math::daubechies_scaling<Real, p>(grid_refinements);
    }
    Real a = 0;
    Real b = phi.support().second;
    std::string title = "Daubechies " + std::to_string(p) + " scaling function";
    title = "";
    std::string filename = "daubechies_" + std::to_string(p) + "_scaling.svg";
    int samples = 1024;
    quicksvg::graph_fn daub(a, b, title, filename, samples, GRAPH_WIDTH);
    daub.set_gridlines(8, 2*p-1);
    daub.set_stroke_width(1);
    daub.add_fn(phi);
    daub.write_all();
}

template<typename Real, int p>
void plot_dphi(int grid_refinements = -1)
{
    auto phi = boost::math::daubechies_scaling<Real, p>();
    if (grid_refinements >= 0)
    {
        phi = boost::math::daubechies_scaling<Real, p>(grid_refinements);
    }
    Real a = 0;
    Real b = phi.support().second;
    std::string title = "Daubechies " + std::to_string(p) + " scaling function derivative";
    title = "";
    std::string filename = "daubechies_" + std::to_string(p) + "_scaling_prime.svg";
    int samples = 1024;
    quicksvg::graph_fn daub(a, b, title, filename, samples, GRAPH_WIDTH);
    daub.set_stroke_width(1);
    daub.set_gridlines(8, 2*p-1);
    auto dphi = [phi](Real x)->Real { return phi.prime(x); };
    daub.add_fn(dphi);
    daub.write_all();
}

template<typename Real, int p>
void plot_convergence()
{
    auto phi0 = boost::math::daubechies_scaling<Real, p>(0);
    Real a = 0;
    Real b = phi0.support().second;
    std::string title = "Daubechies " + std::to_string(p) + " scaling at 0 (green), 1 (orange), 2 (red), and 24 (blue) grid refinements";
    title = "";
    std::string filename = "daubechies_" + std::to_string(p) + "_scaling_convergence.svg";

    quicksvg::graph_fn daub(a, b, title, filename, 1024, 900);
    daub.set_stroke_width(1);
    daub.set_gridlines(8, 2*p-1);

    daub.add_fn(phi0, "green");
    auto phi1 = boost::math::daubechies_scaling<Real, p>(1);
    daub.add_fn(phi1, "orange");
    auto phi2 = boost::math::daubechies_scaling<Real, p>(2);
    daub.add_fn(phi2, "red");

    auto phi21 = boost::math::daubechies_scaling<Real, p>(21);
    daub.add_fn(phi21);

    daub.write_all();
}

template<typename Real, int p>
void plot_condition_number()
{
    using std::abs;
    using std::log;
    static_assert(p >= 3, "p = 2 is not differentiable, so condition numbers cannot be effectively evaluated.");
    auto phi = boost::math::daubechies_scaling<Real, p>();
    Real a = std::sqrt(std::numeric_limits<Real>::epsilon());
    Real b = phi.support().second - 1000*std::sqrt(std::numeric_limits<Real>::epsilon());
    std::string title = "log10 of condition number of function evaluation for Daubechies " + std::to_string(p) + " scaling function.";
    title = "";
    std::string filename = "daubechies_" + std::to_string(p) + "_scaling_condition_number.svg";


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

template<typename CoarseReal, typename PreciseReal, int p, class PhiPrecise>
void do_ulp(int coarse_refinements, PhiPrecise phi_precise)
{
    auto phi_coarse = boost::math::daubechies_scaling<CoarseReal, p>(coarse_refinements);

    std::string title = std::to_string(p) + " vanishing moment ULP plot at " + std::to_string(coarse_refinements) + " refinements and " + boost::core::demangle(typeid(CoarseReal).name()) + " precision";
    title = "";

    std::string filename = "daubechies_" + std::to_string(p) + "_" + boost::core::demangle(typeid(CoarseReal).name()) + "_" + std::to_string(coarse_refinements) + "_refinements.svg";
    int samples = 20000;
    int clip = 10;
    int horizontal_lines = 8;
    int vertical_lines = 2*p - 1;
    auto [a, b] = phi_coarse.support();
    auto plot = boost::math::tools::ulps_plot<decltype(phi_precise), PreciseReal, CoarseReal>(phi_precise, a, b, samples);
    plot.clip(clip).width(GRAPH_WIDTH).horizontal_lines(horizontal_lines).vertical_lines(vertical_lines).ulp_envelope(false);

    plot.background_color("white").font_color("black");
    plot.add_fn(phi_coarse);
    plot.write(filename);
}


int main()
{
    boost::hana::for_each(std::make_index_sequence<18>(), [&](auto i){ plot_phi<double, i+2>(); });
    boost::hana::for_each(std::make_index_sequence<17>(), [&](auto i){ plot_dphi<double, i+3>(); });
    boost::hana::for_each(std::make_index_sequence<17>(), [&](auto i){ plot_condition_number<double, i+3>(); });
    boost::hana::for_each(std::make_index_sequence<18>(), [&](auto i){ plot_convergence<double, i+2>(); });

    using PreciseReal = float128;
    using CoarseReal = double;
    int precise_refinements = 23;
    constexpr const int p = 8;
    std::cout << "Computing precise scaling function in " << boost::core::demangle(typeid(PreciseReal).name()) << " precision.\n";
    auto phi_precise = boost::math::daubechies_scaling<PreciseReal, p>(precise_refinements);
    std::cout << "Beginning comparison with functions computed in " << boost::core::demangle(typeid(CoarseReal).name()) << " precision.\n";
    for (int i = 7; i <= precise_refinements-1; ++i)
    {
        std::cout << "\tCoarse refinement " << i << "\n";
        do_ulp<CoarseReal, PreciseReal, p>(i, phi_precise);
    }
}
