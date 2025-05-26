/*
 * Copyright Nick Thompson, John Maddock 2020
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <numeric>
#include <utility>
#include <iomanip>
#include <iostream>
#include <random>
#include <boost/assert.hpp>
#include <boost/core/demangle.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/ext/std/integer_sequence.hpp>
#include <boost/math/tools/condition_numbers.hpp>
#include <boost/math/special_functions/daubechies_scaling.hpp>
#include <boost/math/filters/daubechies.hpp>
#include <boost/math/special_functions/detail/daubechies_scaling_integer_grid.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>
#include <boost/math/special_functions/next.hpp>

#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif

using std::sqrt;
// Mallat, Theorem 7.4, characterization number 3:
// A conjugate mirror filter has p vanishing moments iff h^{(n)}(pi) = 0 for 0 <= n < p.
template<class Real, unsigned p>
void test_daubechies_filters()
{
    using std::sqrt;

    std::cout << "Testing Daubechies filters with " << p << " vanishing moments on type " << boost::core::demangle(typeid(Real).name()) << "\n";
    Real tol = 3*std::numeric_limits<Real>::epsilon();
    using boost::math::filters::daubechies_scaling_filter;
    using boost::math::filters::daubechies_wavelet_filter;

    auto h = daubechies_scaling_filter<Real, p>();
    auto g = daubechies_wavelet_filter<Real, p>();

    auto inner = std::inner_product(h.begin(), h.end(), g.begin(), Real(0));
    CHECK_MOLLIFIED_CLOSE(0, inner, tol);

    // This is implied by Fourier transform of the two-scale dilatation equation;
    // If this doesn't hold, the infinite product for m_0 diverges.
    Real H0 = 0;
    for (size_t j = 0; j < h.size(); ++j)
    {
        H0 += h[j];
    }
    CHECK_MOLLIFIED_CLOSE(sqrt(static_cast<Real>(2)), H0, tol);

    // This is implied if we choose the scaling function to be an orthonormal basis of V0.
    Real scaling = 0;
    for (size_t j = 0; j < h.size(); ++j) {
      scaling += h[j]*h[j];
    }
    CHECK_MOLLIFIED_CLOSE(1, scaling, tol);

    using std::pow;
    // Daubechies wavelet of order p has p vanishing moments.
    // Unfortunately, the condition number of the sum is infinite.
    // Hence we must scale the tolerance by the summation condition number to ensure that we don't get spurious test failures.
    for (size_t k = 1; k < p && k < 9; ++k)
    {
        Real hk = 0;
        Real abs_hk = 0;
        for (size_t n = 0; n < h.size(); ++n)
        {
            Real t = static_cast<Real>(pow(n, k)*h[n]);
            if (n & 1)
            {
                hk -= t;
            }
            else
            {
                hk += t;
            }
            abs_hk += abs(t);
        }
        // Multiply the tolerance by the condition number:
        Real cond = abs(hk) > 0 ? abs_hk/abs(hk) : 1/std::numeric_limits<Real>::epsilon();
        if (!CHECK_MOLLIFIED_CLOSE(0, hk, 2*cond*tol))
        {
            std::cerr << "  The " << k << "th moment of the p = " << p << " filter did not vanish\n";
            std::cerr << "  Condition number = " << abs_hk/abs(hk) << "\n";
        }
    }

    // For the scaling function to be orthonormal to its integer translates,
    // sum h_k h_{k-2l} = \delta_{0,l}.
    // See Theoretical Numerical Analysis, Atkinson, Exercise 4.5.2.
    // This is the last condition we could test to ensure that the filters are correct,
    // but I'm not gonna bother because it's painful!
}

// Test that the filters agree with Daubechies, Ten Lenctures on Wavelets, Table 6.1:
void test_agreement_with_ten_lectures()
{
    std::cout << "Testing agreement with Ten Lectures\n";
    std::array<double, 4> h2 = {0.4829629131445341, 0.8365163037378077, 0.2241438680420134, -0.1294095225512603};
    auto h2_ = boost::math::filters::daubechies_scaling_filter<double, 2>();
    for (size_t i = 0; i < h2.size(); ++i)
    {
        CHECK_ULP_CLOSE(h2[i], h2_[i], 3);
    }

    std::array<double, 6> h3 = {0.3326705529500825, 0.8068915093110924, 0.4598775021184914, -0.1350110200102546, -0.0854412738820267, 0.0352262918857095};
    auto h3_ = boost::math::filters::daubechies_scaling_filter<double, 3>();
    for (size_t i = 0; i < h3.size(); ++i)
    {
        CHECK_ULP_CLOSE(h3[i], h3_[i], 5);
    }

    std::array<double, 8> h4 = {0.2303778133088964, 0.7148465705529154, 0.6308807679298587, -0.0279837694168599, -0.1870348117190931, 0.0308413818355607, 0.0328830116668852 , -0.010597401785069};
    auto h4_ = boost::math::filters::daubechies_scaling_filter<double, 4>();
    for (size_t i = 0; i < h4.size(); ++i)
    {
        if(!CHECK_ULP_CLOSE(h4[i], h4_[i], 18))
        {
            std::cerr << "  Index " << i << " incorrect.\n";
        }
    }

}

template<class Real1, class Real2, size_t p>
void test_filter_ulp_distance()
{
    std::cout << "Testing filters ULP distance between types "
              << boost::core::demangle(typeid(Real1).name()) << "and"
              << boost::core::demangle(typeid(Real2).name()) << "\n";
    using boost::math::filters::daubechies_scaling_filter;
    auto h1 = daubechies_scaling_filter<Real1, p>();
    auto h2 = daubechies_scaling_filter<Real2, p>();

    for (size_t i = 0; i < h1.size(); ++i)
    {
        if(!CHECK_ULP_CLOSE(h1[i], h2[i], 0))
        {
            std::cerr << "  Index " << i << " at order " << p << " failed tolerance check\n";
        }
    }
}


template<class Real, unsigned p, unsigned order>
void test_integer_grid()
{
    std::cout << "Testing integer grid with " << p << " vanishing moments and " << order << " derivative on type " << boost::core::demangle(typeid(Real).name()) << "\n";
    using boost::math::detail::daubechies_scaling_integer_grid;
    using boost::math::tools::summation_condition_number;
    Real unit_roundoff = std::numeric_limits<Real>::epsilon()/2;
    auto grid = daubechies_scaling_integer_grid<Real, p, order>();

    if constexpr (order == 0)
    {
        auto cond = summation_condition_number<Real>(0);
        for (auto & x : grid)
        {
            cond += x;
        }
        CHECK_MOLLIFIED_CLOSE(1, cond.sum(), 6*cond.l1_norm()*unit_roundoff);
    }

    if constexpr (order == 1)
    {
        auto cond = summation_condition_number<Real>(0);
        for (size_t i = 0; i < grid.size(); ++i) {
            cond += i*grid[i];
        }
        CHECK_MOLLIFIED_CLOSE(Real(-1), cond.sum(), 2*cond.l1_norm()*unit_roundoff);

        // Differentiate \sum_{k} \phi(x-k) = 1 to get this:
        cond = summation_condition_number<Real>(0);
        for (size_t i = 0; i < grid.size(); ++i) {
            cond += grid[i];
        }
        CHECK_MOLLIFIED_CLOSE(Real(0), cond.sum(), 2*cond.l1_norm()*unit_roundoff);
    }

    if constexpr (order == 2)
    {
        auto cond = summation_condition_number<Real>(0);
        for (size_t i = 0; i < grid.size(); ++i)
        {
            cond += i*i*grid[i];
        }
        CHECK_MOLLIFIED_CLOSE(Real(2), cond.sum(), 2*cond.l1_norm()*unit_roundoff);

        // Differentiate \sum_{k} \phi(x-k) = 1 to get this:
        cond = summation_condition_number<Real>(0);
        for (size_t i = 0; i < grid.size(); ++i)
        {
            cond += grid[i];
        }
        CHECK_MOLLIFIED_CLOSE(Real(0), cond.sum(), 2*cond.l1_norm()*unit_roundoff);
    }

    if constexpr (order == 3)
    {
        auto cond = summation_condition_number<Real>(0);
        for (size_t i = 0; i < grid.size(); ++i)
        {
            cond += i*i*i*grid[i];
        }
        CHECK_MOLLIFIED_CLOSE(Real(-6), cond.sum(), 2*cond.l1_norm()*unit_roundoff);

        // Differentiate \sum_{k} \phi(x-k) = 1 to get this:
        cond = summation_condition_number<Real>(0);
        for (size_t i = 0; i < grid.size(); ++i)
        {
            cond += grid[i];
        }
        CHECK_MOLLIFIED_CLOSE(Real(0), cond.sum(), 2*cond.l1_norm()*unit_roundoff);
    }

    if constexpr (order == 4)
    {
        auto cond = summation_condition_number<Real>(0);
        for (size_t i = 0; i < grid.size(); ++i)
        {
            cond += i*i*i*i*grid[i];
        }
        CHECK_MOLLIFIED_CLOSE(24, cond.sum(), 2*cond.l1_norm()*unit_roundoff);

        // Differentiate \sum_{k} \phi(x-k) = 1 to get this:
        cond = summation_condition_number<Real>(0);
        for (size_t i = 0; i < grid.size(); ++i)
        {
            cond += grid[i];
        }
        CHECK_MOLLIFIED_CLOSE(Real(0), cond.sum(), 2*cond.l1_norm()*unit_roundoff);
    }
}

template<class Real>
void test_dyadic_grid()
{
    std::cout << "Testing dyadic grid on type " << boost::core::demangle(typeid(Real).name()) << "\n";
    auto f = [&](auto i)
    {
        auto phijk = boost::math::daubechies_scaling_dyadic_grid<Real, i+2, 0>(0);
        auto phik = boost::math::detail::daubechies_scaling_integer_grid<Real, i+2, 0>();
        BOOST_ASSERT(phik.size() == phijk.size());

        for (size_t k = 0; k < phik.size(); ++k)
        {
            CHECK_ULP_CLOSE(phik[k], phijk[k], 0);
        }

        for (uint64_t j = 1; j < 10; ++j)
        {
            phijk = boost::math::daubechies_scaling_dyadic_grid<Real, i+2, 0>(j);
            phik = boost::math::detail::daubechies_scaling_integer_grid<Real, i+2, 0>();
            for (uint64_t l = 0; l < static_cast<uint64_t>(phik.size()); ++l)
            {
                CHECK_ULP_CLOSE(phik[l], phijk[l*(uint64_t(1)<<j)], 0);
            }

            // This test is from Daubechies, Ten Lectures on Wavelets, Ch 7 "More About Compactly Supported Wavelets",
            // page 245: \forall y \in \mathbb{R}, \sum_{n \in \mathbb{Z}} \phi(y+n) = 1
            for (size_t k = 1; k < j; ++k)
            {
                auto cond = boost::math::tools::summation_condition_number<Real>(0);
                for (uint64_t l = 0; l < static_cast<uint64_t>(phik.size()); ++l)
                {
                    uint64_t idx = l*(uint64_t(1)<<j) + k;
                    if (idx < phijk.size())
                    {
                        cond += phijk[idx];
                    }
                }
                CHECK_MOLLIFIED_CLOSE(Real(1), cond.sum(), 10*cond()*std::numeric_limits<Real>::epsilon());
            }
        }
        if constexpr (std::numeric_limits<Real>::digits < 30)
        {
           CHECK_THROW((boost::math::daubechies_scaling_dyadic_grid<Real, i + 2, 0>)(24), std::logic_error);
        }
    };

    boost::hana::for_each(std::make_index_sequence<18>(), f);
}


// Taken from Lin, 2005, doi:10.1016/j.amc.2004.12.038,
// "Direct algorithm for computation of derivatives of the Daubechies basis functions"
void test_first_derivative()
{
#if LDBL_MANT_DIG > 64
   // Limited precision test data means we can't test long double here...
#else
    auto phi1_3 = boost::math::detail::daubechies_scaling_integer_grid<long double, 3, 1>();
    std::array<long double, 6> lin_3{0.0L, 1.638452340884085725014976113635604107L, -2.23275819046313739501774225255380757L,
                                     0.550159358274017614990556164200803310L, 0.044146491305034055012209974717400368L, 0.0L};
    for (size_t i = 0; i < lin_3.size(); ++i)
    {
        if(!CHECK_ULP_CLOSE(lin_3[i], phi1_3[i], 0))
        {
            std::cerr << "  Index " << i << " is incorrect\n";
        }
    }

    auto phi1_4 = boost::math::detail::daubechies_scaling_integer_grid<long double, 4, 1>();
    std::array<long double, 8> lin_4 = {0.0L, 1.776072007522184640093776071522502761L, -2.785349397229543142492784905731245880L, 1.192452536632278174347632339082851360L,
                                       -0.131374515184672958793518896272545740L, -0.053571028220239235953599959390993709L,0.001770396479992522798495350789431024L, 0.0L};

    for (size_t i = 0; i < lin_4.size(); ++i)
    {
        if(!CHECK_ULP_CLOSE(lin_4[i], phi1_4[i], 0))
        {
            std::cerr << "  Index " << i << " is incorrect\n";
        }
    }

    std::array<long double, 10> lin_5 = {0.0L, 1.558326313047001366564379221011472479L, -2.436012783189551921436895932290077033L, 1.235905129801454293947038906779457610L, -0.367437713693886635994756136622838186L,
                                        -0.021780351175646546588845564309594589L,0.032347193508143688858158541500450925L,-0.001335619912770701035229330817898250L,-0.000012168384743544313849705250972915L,0.0L};
    auto phi1_5 = boost::math::detail::daubechies_scaling_integer_grid<long double, 5, 1>();
    for (size_t i = 0; i < lin_5.size(); ++i)
    {
        if(!CHECK_ULP_CLOSE(lin_5[i], phi1_5[i], 0))
        {
            std::cerr << "  Index " << i << " is incorrect\n";
        }
    }
#endif
}

template<typename Real, int p>
void test_quadratures()
{
    std::cout << "Testing " << p << " vanishing moment scaling function quadratures on type " << boost::core::demangle(typeid(Real).name()) << "\n";
    using boost::math::quadrature::trapezoidal;
    if constexpr (p == 2)
    {
        // 2phi is truly bizarre, because two successive trapezoidal estimates are always bitwise equal,
        // whereas the third is way different. I don' t think that's a reasonable thing to optimize for,
        // so one-off it is.
        Real h = Real(1)/Real(256);
        auto phi = boost::math::daubechies_scaling<Real, p>();
        std::cout << "Scaling functor size is " << phi.bytes() << " bytes" << std::endl;
        Real t = 0;
        Real Q = 0;
        while (t < 3) {
            Q += phi(t);
            t += h;
        }
        Q *= h;
        CHECK_ULP_CLOSE(Real(1), Q, 32);

        auto [a, b] = phi.support();
        // Now hit the boundary. Much can go wrong here; this just tests for segfaults:
        int samples = 500;
        Real xlo = a;
        Real xhi = b;
        for (int i = 0; i < samples; ++i)
        {
            CHECK_ULP_CLOSE(Real(0), phi(xlo), 0);
            CHECK_ULP_CLOSE(Real(0), phi(xhi), 0);
            xlo = std::nextafter(xlo, std::numeric_limits<Real>::lowest());
            xhi = std::nextafter(xhi, (std::numeric_limits<Real>::max)());
        }

        xlo = a;
        xhi = b;
        for (int i = 0; i < samples; ++i) {
            BOOST_ASSERT(abs(phi(xlo)) <= 5);
            BOOST_ASSERT(abs(phi(xhi)) <= 5);
            xlo = std::nextafter(xlo, (std::numeric_limits<Real>::max)());
            xhi = std::nextafter(xhi, std::numeric_limits<Real>::lowest());
        }

        return;
    }
    else if constexpr (p > 2)
    {
        auto phi = boost::math::daubechies_scaling<Real, p>();
        std::cout << "Scaling functor size is " << phi.bytes() << " bytes" << std::endl;

        Real tol = std::numeric_limits<Real>::epsilon();
        Real error_estimate = std::numeric_limits<Real>::quiet_NaN();
        Real L1 = std::numeric_limits<Real>::quiet_NaN();
        auto [a, b] = phi.support();
        Real Q = trapezoidal(phi, a, b, tol, 15, &error_estimate, &L1);
        if (!CHECK_MOLLIFIED_CLOSE(Real(1), Q, Real(0.0001)))
        {
            std::cerr << "  Quadrature of " << p << " vanishing moment scaling function is not equal 1.\n";
            std::cerr << "  Error estimate is " << error_estimate << ", L1 norm is " << L1 << "\n";
        }

        auto phi_sq = [phi](Real x) { Real t = phi(x); return t*t; };
        Q = trapezoidal(phi, a, b, tol, 15, &error_estimate, &L1);
        if (!CHECK_MOLLIFIED_CLOSE(Real(1), Q, 20*std::sqrt(std::numeric_limits<Real>::epsilon())/(p*p)))
        {
            std::cerr << "  L2 norm of " << p << " vanishing moment scaling function is not equal 1.\n";
            std::cerr << "  Error estimate is " << error_estimate << ", L1 norm is " << L1 << "\n";
        }

        std::random_device rd;
        Real t = static_cast<Real>(rd())/static_cast<Real>((rd.max)());
        Real S = phi(t);
        Real dS = phi.prime(t);
        while (t < b)
        {
            t += 1;
            S += phi(t);
            dS += phi.prime(t);
        }

        if(!CHECK_ULP_CLOSE(Real(1), S, 64))
        {
            std::cerr << "  Normalizing sum for " << p << " vanishing moment scaling function is incorrect.\n";
        }

        // The p = 3, 4 convergence rate is very slow, making this produce false positives:
        if constexpr(p > 4)
        {
            if(!CHECK_MOLLIFIED_CLOSE(Real(0), dS, 100*std::sqrt(std::numeric_limits<Real>::epsilon())))
            {
                std::cerr << "  Derivative of normalizing sum for " << p << " vanishing moment scaling function doesn't vanish.\n";
            }
        }

        // Test boundary for segfaults:
        int samples = 500;
        Real xlo = a;
        Real xhi = b;
        for (int i = 0; i < samples; ++i)
        {
            CHECK_ULP_CLOSE(Real(0), phi(xlo), 0);
            CHECK_ULP_CLOSE(Real(0), phi(xhi), 0);
            if constexpr (p > 2) {
                BOOST_ASSERT(abs(phi.prime(xlo)) <= 5);
                BOOST_ASSERT(abs(phi.prime(xhi)) <= 5);
                if constexpr (p > 5) {
                     BOOST_ASSERT(abs(phi.double_prime(xlo)) <= 5);
                     BOOST_ASSERT(abs(phi.double_prime(xhi)) <= 5);
                }
            }
            xlo = std::nextafter(xlo, std::numeric_limits<Real>::lowest());
            xhi = std::nextafter(xhi, (std::numeric_limits<Real>::max)());
        }

        xlo = a;
        xhi = b;
        for (int i = 0; i < samples; ++i) {
            BOOST_ASSERT(abs(phi(xlo)) <= 5);
            BOOST_ASSERT(abs(phi(xhi)) <= 5);
            xlo = std::nextafter(xlo, (std::numeric_limits<Real>::max)());
            xhi = std::nextafter(xhi, std::numeric_limits<Real>::lowest());
        }
    }
}

int main()
{
    #ifndef __MINGW32__
    boost::hana::for_each(std::make_index_sequence<18>(), [&](auto i){
      test_quadratures<float, i+2>();
      test_quadratures<double, i+2>();
    });

    test_agreement_with_ten_lectures();

    boost::hana::for_each(std::make_index_sequence<19>(), [&](auto i){
      test_daubechies_filters<float, i+1>();
      test_daubechies_filters<double, i+1>();
      test_daubechies_filters<long double, i+1>();
    });

    test_first_derivative();

    // All scaling functions have a first derivative.
    boost::hana::for_each(std::make_index_sequence<18>(), [&](auto idx){
        test_integer_grid<float, idx+2, 0>();
        test_integer_grid<float, idx+2, 1>();
        test_integer_grid<double, idx+2, 0>();
        test_integer_grid<double, idx+2, 1>();
        test_integer_grid<long double, idx+2, 0>();
        test_integer_grid<long double, idx+2, 1>();
        #ifdef BOOST_HAS_FLOAT128
        test_integer_grid<float128, idx+2, 0>();
        test_integer_grid<float128, idx+2, 1>();
        #endif
    });

    // 4-tap (2 vanishing moment) scaling function does not have a second derivative;
    // all other scaling functions do.
    boost::hana::for_each(std::make_index_sequence<17>(), [&](auto idx){
        test_integer_grid<float, idx+3, 2>();
        test_integer_grid<double, idx+3, 2>();
        test_integer_grid<long double, idx+3, 2>();
        #ifdef BOOST_HAS_FLOAT128
        test_integer_grid<boost::multiprecision::float128, idx+3, 2>();
        #endif
    });

    // 8-tap filter (4 vanishing moments) is the first to have a third derivative.
    boost::hana::for_each(std::make_index_sequence<16>(), [&](auto idx){
        test_integer_grid<float, idx+4, 3>();
        test_integer_grid<double, idx+4, 3>();
        test_integer_grid<long double, idx+4, 3>();
        #ifdef BOOST_HAS_FLOAT128
        test_integer_grid<boost::multiprecision::float128, idx+4, 3>();
        #endif
    });

    // 10-tap filter (5 vanishing moments) is the first to have a fourth derivative.
    boost::hana::for_each(std::make_index_sequence<15>(), [&](auto idx){
        test_integer_grid<float, idx+5, 4>();
        test_integer_grid<double, idx+5, 4>();
        test_integer_grid<long double, idx+5, 4>();
        #ifdef BOOST_HAS_FLOAT128
        test_integer_grid<boost::multiprecision::float128, idx+5, 4>();
        #endif
    });

    test_dyadic_grid<float>();
    test_dyadic_grid<double>();
    test_dyadic_grid<long double>();
    #ifdef BOOST_HAS_FLOAT128
    test_dyadic_grid<float128>();
    #endif


    #ifdef BOOST_HAS_FLOAT128
    boost::hana::for_each(std::make_index_sequence<19>(), [&](auto i){
        test_filter_ulp_distance<float128, long double, i+1>();
        test_filter_ulp_distance<float128, double, i+1>();
        test_filter_ulp_distance<float128, float, i+1>();
    });

    boost::hana::for_each(std::make_index_sequence<19>(), [&](auto i){
        test_daubechies_filters<float128, i+1>();
    });
    #endif
    #endif // compiler guard for CI
    return boost::math::test::report_errors();
}
