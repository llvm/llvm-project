/*
 * Copyright Nick Thompson, 2019
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#define BOOST_TEST_MODULE lanczos_smoothing_test

#include <random>
#include <array>
#include <boost/range.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/differentiation/lanczos_smoothing.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/special_functions/next.hpp> // for float_distance
#include <boost/math/tools/condition_numbers.hpp>

using std::abs;
using std::pow;
using std::sqrt;
using std::sin;
using boost::math::constants::two_pi;
using boost::multiprecision::cpp_bin_float_50;
using boost::multiprecision::cpp_bin_float_100;
using boost::math::differentiation::discrete_lanczos_derivative;
using boost::math::differentiation::detail::discrete_legendre;
using boost::math::differentiation::detail::interior_velocity_filter;
using boost::math::differentiation::detail::boundary_velocity_filter;
using boost::math::tools::summation_condition_number;

template<class Real>
void test_dlp_norms()
{
    std::cout << "Testing Discrete Legendre Polynomial norms on type " << typeid(Real).name() << "\n";
    Real tol = std::numeric_limits<Real>::epsilon();
    auto dlp = discrete_legendre<Real>(1, Real(0));
    BOOST_CHECK_CLOSE_FRACTION(dlp.norm_sq(0), 3, tol);
    BOOST_CHECK_CLOSE_FRACTION(dlp.norm_sq(1), 2, tol);
    dlp = discrete_legendre<Real>(2, Real(0));
    BOOST_CHECK_CLOSE_FRACTION(dlp.norm_sq(0), Real(5)/Real(2), tol);
    BOOST_CHECK_CLOSE_FRACTION(dlp.norm_sq(1), Real(5)/Real(4), tol);
    BOOST_CHECK_CLOSE_FRACTION(dlp.norm_sq(2), Real(3*3*7)/Real(pow(2,6)), 2*tol);
    dlp = discrete_legendre<Real>(200, Real(0));
    for(size_t r = 0; r < 10; ++r)
    {
        Real calc = dlp.norm_sq(r);
        Real expected = Real(2)/Real(2*r+1);
        // As long as r << n, ||q_r||^2 -> 2/(2r+1) as n->infty
        BOOST_CHECK_CLOSE_FRACTION(calc, expected, 0.05);
    }

}

template<class Real>
void test_dlp_evaluation()
{
    std::cout << "Testing evaluation of Discrete Legendre polynomials on type " << typeid(Real).name() << "\n";
    Real tol = std::numeric_limits<Real>::epsilon();
    size_t n = 25;
    Real x = 0.72;
    auto dlp = discrete_legendre<Real>(n, x);
    Real q0 = dlp(x, 0);
    BOOST_TEST(q0 == 1);
    Real q1 = dlp(x, 1);
    BOOST_TEST(q1 == x);
    Real q2 = dlp(x, 2);
    int N = 2*n+1;
    Real expected = 0.5*(3*x*x - Real(N*N - 1)/Real(4*n*n));
    BOOST_CHECK_CLOSE_FRACTION(q2, expected, tol);
    Real q3 = dlp(x, 3);
    expected = (x/3)*(5*expected - (Real(N*N - 4))/(2*n*n));
    BOOST_CHECK_CLOSE_FRACTION(q3, expected, 2*tol);

    // q_r(x) is even for even r, and odd for odd r:
    for (size_t n = 8; n < 22; ++n)
    {
        dlp = discrete_legendre<Real>(n, x);
        for(size_t r = 2; r <= n; ++r)
        {
            if (r & 1)
            {
                Real q1 = dlp(x, r);
                Real q2 = -dlp(-x, r);
                BOOST_CHECK_CLOSE_FRACTION(q1, q2, tol);
            }
            else
            {
                Real q1 = dlp(x, r);
                Real q2 = dlp(-x, r);
                BOOST_CHECK_CLOSE_FRACTION(q1, q2, tol);
            }

            Real l2_sq = 0;
            for (int j = -(int)n; j <= (int) n; ++j)
            {
                Real y = Real(j)/Real(n);
                Real term = dlp(y, r);
                l2_sq += term*term;
            }
            l2_sq /= n;
            Real l2_sq_expected = dlp.norm_sq(r);
            BOOST_CHECK_CLOSE_FRACTION(l2_sq, l2_sq_expected, 20*tol);
        }
    }
}

template<class Real>
void test_dlp_next()
{
    std::cout << "Testing Discrete Legendre polynomial 'next' function on type " << typeid(Real).name() << "\n";
    Real tol = std::numeric_limits<Real>::epsilon();

    for(size_t n = 2; n < 20; ++n)
    {
        for(Real x = -1; x <= 1; x += 0.1)
        {
            auto dlp = discrete_legendre<Real>(n, x);
            for (size_t k = 2; k < n; ++k)
            {
                BOOST_CHECK_CLOSE(dlp.next(), dlp(x, k), tol);
            }

            dlp = discrete_legendre<Real>(n, x);
            for (size_t k = 2; k < n; ++k)
            {
                BOOST_CHECK_CLOSE(dlp.next_prime(), dlp.prime(x, k), tol);
            }
        }
    }
}


template<class Real>
void test_dlp_derivatives()
{
    std::cout << "Testing Discrete Legendre polynomial derivatives on type " << typeid(Real).name() << "\n";
    Real tol = 10*std::numeric_limits<Real>::epsilon();
    int n = 25;
    Real x = 0.72;
    auto dlp = discrete_legendre<Real>(n, x);
    Real q0p = dlp.prime(x, 0);
    BOOST_TEST(q0p == 0);
    Real q1p = dlp.prime(x, 1);
    BOOST_TEST(q1p == 1);
    Real q2p = dlp.prime(x, 2);
    Real expected = 3*x;
    BOOST_CHECK_CLOSE_FRACTION(q2p, expected, tol);
}

template<class Real>
void test_dlp_second_derivative()
{
    std::cout << "Testing Discrete Legendre polynomial derivatives on type " << typeid(Real).name() << "\n";
    int n = 25;
    Real x = Real(1)/Real(3);
    auto dlp = discrete_legendre<Real>(n, x);
    Real q2pp = dlp.next_dbl_prime();
    BOOST_TEST(q2pp == 3);
}


template<class Real>
void test_interior_velocity_filter()
{
    using boost::math::constants::half;
    std::cout << "Testing interior filter on type " << typeid(Real).name() << "\n";
    Real tol = std::numeric_limits<Real>::epsilon();
    for(int n = 1; n < 10; ++n)
    {
        for (int p = 1; p < n; p += 2)
        {
            auto f = interior_velocity_filter<Real>(n,p);
            // Since we only store half the filter coefficients,
            // we need to reindex the moment sums:
            auto cond = summation_condition_number<Real>(0);
            for (size_t j = 0; j < f.size(); ++j)
            {
                cond += j*f[j];
            }
            BOOST_CHECK_CLOSE_FRACTION(cond.sum(), half<Real>(), 2*cond()*tol);

            for (int l = 3; l <= p; l += 2)
            {
                cond = summation_condition_number<Real>(0);
                for (size_t j = 0; j < f.size() - 1; ++j)
                {
                    cond += pow(Real(j), l)*f[j];
                }
                Real expected = -pow(Real(f.size() - 1), l)*f[f.size()-1];
                BOOST_CHECK_CLOSE_FRACTION(expected, cond.sum(), 15*cond()*tol);
            }
            //std::cout << "(n,p) = (" << n  << "," << p << ") = {";
            //for (auto & x : f)
            //{
            //    std::cout << x << ", ";
            //}
            //std::cout << "}\n";
        }
    }
}

template<class Real>
void test_interior_lanczos()
{
    std::cout << "Testing interior Lanczos on type " << typeid(Real).name() << "\n";
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Real> v(500);
    std::fill(v.begin(), v.end(), 7);

    for (size_t n = 1; n < 10; ++n)
    {
        for (size_t p = 2; p < 2*n; p += 2)
        {
            auto dld = discrete_lanczos_derivative(Real(0.1), n, p);
            for (size_t m = n; m < v.size() - n; ++m)
            {
                Real dvdt = dld(v, m);
                BOOST_CHECK_SMALL(dvdt, tol);
            }
            auto dvdt = dld(v);
            for (size_t m = n; m < v.size() - n; ++m)
            {
                BOOST_CHECK_SMALL(dvdt[m], tol);
            }
        }
    }


    for(size_t i = 0; i < v.size(); ++i)
    {
        v[i] = 7*i+8;
    }

    for (size_t n = 1; n < 10; ++n)
    {
        for (size_t p = 2; p < 2*n; p += 2)
        {
            auto dld = discrete_lanczos_derivative(Real(1), n, p);
            for (size_t m = n; m < v.size() - n; ++m)
            {
                Real dvdt = dld(v, m);
                BOOST_CHECK_CLOSE_FRACTION(dvdt, 7, 2000*tol);
            }
            auto dvdt = dld(v);
            for (size_t m = n; m < v.size() - n; ++m)
            {
                BOOST_CHECK_CLOSE_FRACTION(dvdt[m], 7, 2000*tol);
            }
        }
    }

    //std::random_device rd{};
    //auto seed = rd();
    //std::cout << "Seed = " << seed << "\n";
    std::mt19937 gen(4172378669);
    std::normal_distribution<> dis{0, 0.01};
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = 7*i+8 + dis(gen);
    }

    for (size_t n = 1; n < 10; ++n)
    {
        for (size_t p = 2; p < 2*n; p += 2)
        {
            auto dld = discrete_lanczos_derivative(Real(1), n, p);
            for (size_t m = n; m < v.size() - n; ++m)
            {
                BOOST_CHECK_CLOSE_FRACTION(dld(v, m), Real(7), Real(0.0042));
            }
        }
    }


    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = 15*i*i + 7*i+8 + dis(gen);
    }

    for (size_t n = 1; n < 10; ++n)
    {
        for (size_t p = 2; p < 2*n; p += 2)
        {
            auto dld = discrete_lanczos_derivative(Real(1), n, p);
            for (size_t m = n; m < v.size() - n; ++m)
            {
                BOOST_CHECK_CLOSE_FRACTION(dld(v,m), Real(30*m + 7), Real(0.00008));
            }
        }
    }

    std::normal_distribution<> dis1{0, 0.0001};
    Real omega = Real(1)/Real(16);
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = sin(i*omega) + dis1(gen);
    }

    for (size_t n = 10; n < 20; ++n)
    {
        for (size_t p = 3; p < 100 && p < n/2; p += 2)
        {
            auto dld = discrete_lanczos_derivative(Real(1), n, p);

            for (size_t m = n; m < v.size() - n && m < n + 10; ++m)
            {
                BOOST_CHECK_CLOSE_FRACTION(dld(v,m), omega*cos(omega*m), Real(0.03));
            }
        }
    }
}

template<class Real>
void test_boundary_velocity_filters()
{
    std::cout << "Testing boundary filters on type " << typeid(Real).name() << "\n";
    Real tol = std::numeric_limits<Real>::epsilon();
    for(int n = 1; n < 5; ++n)
    {
        for (int p = 1; p < 2*n+1; ++p)
        {
            for (int s = -n; s <= n; ++s)
            {
                auto f = boundary_velocity_filter<Real>(n, p, s);
                // Sum is zero:
                auto cond = summation_condition_number<Real>(0);
                for (size_t i = 0; i < f.size() - 1; ++i)
                {
                    cond += f[i];
                }

                BOOST_CHECK_CLOSE_FRACTION(cond.sum(), -f[f.size()-1], 6*cond()*tol);

                cond = summation_condition_number<Real>(0);
                for (size_t k = 0; k < f.size(); ++k)
                {
                    Real j = Real(k) - Real(n);
                    // note the shifted index here:
                    cond += (j-s)*f[k];
                }
                BOOST_CHECK_CLOSE_FRACTION(cond.sum(), 1, 6*cond()*tol);


                for (int l = 2; l <= p; ++l)
                {
                    cond = summation_condition_number<Real>(0);
                    for (size_t k = 0; k < f.size() - 1; ++k)
                    {
                        Real j = Real(k) - Real(n);
                        // The condition number of this sum is infinite!
                        // No need to get to worked up about the tolerance.
                        cond += pow(j-s, l)*f[k];
                    }

                    Real expected = -pow(Real(f.size()-1) - Real(n) - Real(s), l)*f[f.size()-1];
                    if (expected == 0)
                    {
                        BOOST_CHECK_SMALL(cond.sum(), cond()*tol);
                    }
                    else
                    {
                        BOOST_CHECK_CLOSE_FRACTION(expected, cond.sum(), 200*cond()*tol);
                    }
                }

                //std::cout << "(n,p,s) = ("<< n << ", " << p << "," << s << ") = {";
                //for (auto & x : f)
                //{
                //    std::cout << x << ", ";
                //}
                //std::cout << "}\n";*/
            }
        }
    }
}

template<class Real>
void test_boundary_lanczos()
{
    std::cout << "Testing Lanczos boundary on type " << typeid(Real).name() << "\n";
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Real> v(500, 7);

    for (size_t n = 1; n < 10; ++n)
    {
        for (size_t p = 2; p < 2*n; ++p)
        {
            auto lsd = discrete_lanczos_derivative(Real(0.0125), n, p);
            for (size_t m = 0; m < n; ++m)
            {
                Real dvdt = lsd(v,m);
                BOOST_CHECK_SMALL(dvdt, 4*sqrt(tol));
            }
            for (size_t m = v.size() - n; m < v.size(); ++m)
            {
                Real dvdt = lsd(v,m);
                BOOST_CHECK_SMALL(dvdt, 4*sqrt(tol));
            }
        }
    }

    for(size_t i = 0; i < v.size(); ++i)
    {
        v[i] = 7*i+8;
    }

    for (size_t n = 3; n < 10; ++n)
    {
        for (size_t p = 2; p < 2*n; ++p)
        {
            auto lsd = discrete_lanczos_derivative(Real(1), n, p);
            for (size_t m = 0; m < n; ++m)
            {
                Real dvdt = lsd(v,m);
                BOOST_CHECK_CLOSE_FRACTION(dvdt, 7, sqrt(tol));
            }

            for (size_t m = v.size() - n; m < v.size(); ++m)
            {
                Real dvdt = lsd(v,m);
                BOOST_CHECK_CLOSE_FRACTION(dvdt, 7, 4*sqrt(tol));
            }
        }
    }

    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = 15*i*i + 7*i+8;
    }

    for (size_t n = 1; n < 10; ++n)
    {
        for (size_t p = 2; p < 2*n; ++p)
        {
            auto lsd = discrete_lanczos_derivative(Real(1), n, p);
            for (size_t m = 0; m < v.size(); ++m)
            {
                BOOST_CHECK_CLOSE_FRACTION(lsd(v,m), 30*m+7, 30*sqrt(tol));
            }
        }
    }

    // Demonstrate that the boundary filters are also denoising:
    //std::random_device rd{};
    //auto seed = rd();
    //std::cout << "seed = " << seed << "\n";
    std::mt19937 gen(311354333);
    std::normal_distribution<> dis{0, 0.01};
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] += dis(gen);
    }

    for (size_t n = 1; n < 10; ++n)
    {
        for (size_t p = 2; p < n; ++p)
        {
            auto lsd = discrete_lanczos_derivative(Real(1), n, p);
            for (size_t m = 0; m < v.size(); ++m)
            {
                BOOST_CHECK_CLOSE_FRACTION(lsd(v,m), 30*m+7, 0.005);
            }
            auto dvdt = lsd(v);
            for (size_t m = 0; m < v.size(); ++m)
            {
                BOOST_CHECK_CLOSE_FRACTION(dvdt[m], 30*m+7, 0.005);
            }
        }
    }
}

template<class Real>
void test_acceleration_filters()
{
    Real eps = std::numeric_limits<Real>::epsilon();
    for (size_t n = 1; n < 5; ++n)
    {
        for(size_t p = 3; p <= 2*n; ++p)
        {
            for(int64_t s = -int64_t(n); s <= 0; ++s)
            {
                auto g = boost::math::differentiation::detail::acceleration_filter<long double>(n,p,s);

                std::vector<Real> f(g.size());
                for (size_t i = 0; i < g.size(); ++i)
                {
                    f[i] = static_cast<Real>(g[i]);
                }

                auto cond = summation_condition_number<Real>(0);

                for (size_t i = 0; i < f.size() - 1; ++i)
                {
                    cond += f[i];
                }
                BOOST_CHECK_CLOSE_FRACTION(cond.sum(), -f[f.size()-1], 10*cond()*eps);


                cond = summation_condition_number<Real>(0);
                for (size_t k = 0; k < f.size() -1; ++k)
                {
                    Real j = Real(k) - Real(n);
                    cond += (j-s)*f[k];
                }
                Real expected = -(Real(f.size()-1)- Real(n) - s)*f[f.size()-1];
                BOOST_CHECK_CLOSE_FRACTION(cond.sum(), expected, 10*cond()*eps);

                cond = summation_condition_number<Real>(0);
                for (size_t k = 0; k < f.size(); ++k)
                {
                    Real j = Real(k) - Real(n);
                    cond += (j-s)*(j-s)*f[k];
                }
                BOOST_CHECK_CLOSE_FRACTION(cond.sum(), 2, 100*cond()*eps);
                // See unlabelled equation in McDevitt, 2012, just after equation 26:
                // It appears that there is an off-by-one error in that equation, since p + 1 moments don't vanish, only p.
                // This test is itself suspect; the condition number of the moment sum is infinite.
                // So the *slightest* error in the filter gets amplified by the test; in terms of the
                // behavior of the actual filter, it's not a big deal.
                for (size_t l = 3; l <= p; ++l)
                {
                    cond = summation_condition_number<Real>(0);
                    for (size_t k = 0; k < f.size() - 1; ++k)
                    {
                        Real j = Real(k) - Real(n);
                        cond += pow((j-s), l)*f[k];
                    }
                    Real expected = -pow(Real(f.size()- 1 - n -s), l)*f[f.size()-1];
                    BOOST_CHECK_CLOSE_FRACTION(cond.sum(), expected, 1000*cond()*eps);
                }
            }
        }
    }
}

template<class Real>
void test_lanczos_acceleration()
{
    Real eps = std::numeric_limits<Real>::epsilon();
    std::vector<Real> v(100, 7);
    auto lanczos = discrete_lanczos_derivative<Real, 2>(Real(1), 4, 3);
    for (size_t i = 0; i < v.size(); ++i)
    {
        BOOST_CHECK_SMALL(lanczos(v, i), eps);
    }

    for(size_t i = 0; i < v.size(); ++i)
    {
        v[i] = 7*i + 6;
    }
    for (size_t i = 0; i < v.size(); ++i)
    {
        BOOST_CHECK_SMALL(lanczos(v,i), 200*eps);
    }

    for(size_t i = 0; i < v.size(); ++i)
    {
        v[i] = 7*i*i + 9*i + 6;
    }
    for (size_t i = 0; i < v.size(); ++i)
    {
        BOOST_CHECK_CLOSE_FRACTION(lanczos(v, i), 14, 1500*eps);
    }

    // Now add noise, and kick up the smoothing of the Lanzcos derivative (increase n):
    //std::random_device rd{};
    //auto seed = rd();
    //std::cout << "seed = " << seed << "\n";
    size_t seed = 2507134629;
    std::mt19937 gen(seed);
    Real std_dev = 0.1;
    std::normal_distribution<Real> dis{0, std_dev};
    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] += dis(gen);
    }
    lanczos = discrete_lanczos_derivative<Real, 2>(Real(1), 18, 3);
    auto w = lanczos(v);
    for (size_t i = 0; i < v.size(); ++i)
    {
        BOOST_CHECK_CLOSE_FRACTION(w[i], 14, std_dev/200);
    }
}

template<class Real>
void test_rescaling()
{
    std::cout << "Test rescaling on type " << typeid(Real).name() << "\n";
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Real> v(500);
    for(size_t i = 0; i < v.size(); ++i)
    {
        v[i] = 7*i*i + 9*i + 6;
    }
    std::vector<Real> dvdt1(500);
    std::vector<Real> dvdt2(500);
    auto lanczos1 = discrete_lanczos_derivative(Real(1));
    auto lanczos2 = discrete_lanczos_derivative(Real(2));

    lanczos1(v, dvdt1);
    lanczos2(v, dvdt2);

    for(size_t i = 0; i < v.size(); ++i)
    {
        BOOST_CHECK_CLOSE_FRACTION(dvdt1[i], 2*dvdt2[i], tol);
    }

    auto lanczos3 = discrete_lanczos_derivative<Real, 2>(Real(1));
    auto lanczos4 = discrete_lanczos_derivative<Real, 2>(Real(2));


    std::vector<Real> dv2dt21(500);
    std::vector<Real> dv2dt22(500);

    for(size_t i = 0; i < v.size(); ++i)
    {
        BOOST_CHECK_CLOSE_FRACTION(dv2dt21[i], 4*dv2dt22[i], tol);
    }
}

template<class Real>
void test_data_representations()
{
    std::cout << "Test rescaling on type " << typeid(Real).name() << "\n";
    Real tol = 150*std::numeric_limits<Real>::epsilon();
    std::array<Real, 500> v;
    for(size_t i = 0; i < v.size(); ++i)
    {
        v[i] = 9*i + 6;
    }
    std::array<Real, 500> dvdt;
    auto lanczos = discrete_lanczos_derivative(Real(1));

    lanczos(v, dvdt);

    for(size_t i = 0; i < v.size(); ++i)
    {
        BOOST_CHECK_CLOSE_FRACTION(dvdt[i], 9, tol);
    }

    boost::numeric::ublas::vector<Real> w(500);
    boost::numeric::ublas::vector<Real> dwdt(500);
    for(size_t i = 0; i < w.size(); ++i)
    {
        w[i] = 9*i + 6;
    }

    lanczos(w, dwdt);

    for(size_t i = 0; i < v.size(); ++i)
    {
        BOOST_CHECK_CLOSE_FRACTION(dwdt[i], 9, tol);
    }

    auto v1 = boost::make_iterator_range(v.begin(), v.end());
    auto v2 = boost::make_iterator_range(dvdt.begin(), dvdt.end());
    lanczos(v1, v2);

    for(size_t i = 0; i < v2.size(); ++i)
    {
        BOOST_CHECK_CLOSE_FRACTION(v2[i], 9, tol);
    }

    auto lanczos2 = discrete_lanczos_derivative<Real, 2>(Real(1));

    lanczos2(v1, v2);

    for(size_t i = 0; i < v2.size(); ++i)
    {
        BOOST_CHECK_SMALL(v2[i], 10*tol);
    }

}

BOOST_AUTO_TEST_CASE(lanczos_smoothing_test)
{
    test_dlp_second_derivative<double>();
    test_dlp_norms<double>();
    test_dlp_evaluation<double>();
    test_dlp_derivatives<double>();
    test_dlp_next<double>();

    // Takes too long!
    //test_dlp_norms<cpp_bin_float_50>();
    test_boundary_velocity_filters<double>();
    test_boundary_velocity_filters<long double>();
    // Takes too long!
    //test_boundary_velocity_filters<cpp_bin_float_50>();
    test_boundary_lanczos<double>();
    test_boundary_lanczos<long double>();
    // Takes too long!
    //test_boundary_lanczos<cpp_bin_float_50>();

    test_interior_velocity_filter<double>();
    test_interior_velocity_filter<long double>();
    test_interior_lanczos<double>();

    test_acceleration_filters<double>();

    test_lanczos_acceleration<float>();
    test_lanczos_acceleration<double>();

    test_rescaling<double>();
    test_data_representations<double>();
}
