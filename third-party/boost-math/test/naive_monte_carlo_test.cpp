/*
 * Copyright Nick Thompson, 2017
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#define BOOST_TEST_MODULE naive_monte_carlo_test
#define BOOST_NAIVE_MONTE_CARLO_DEBUG_FAILURES
#include <cmath>
#include <ostream>
#include <boost/math/tools/test_value.hpp>
#include <boost/type_index.hpp>
#include <boost/test/included/unit_test.hpp>

#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/quadrature/naive_monte_carlo.hpp>

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

using std::abs;
using std::vector;
using std::pair;
using boost::math::constants::pi;
using boost::math::quadrature::naive_monte_carlo;


template<class Real>
void test_pi_multithreaded()
{
    std::cout << "Testing pi is calculated correctly (multithreaded) using Monte-Carlo on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    auto g = [](std::vector<Real> const & x)->Real {
        Real r = x[0]*x[0]+x[1]*x[1];
        if (r <= 1) {
          return 4;
        }
        return 0;
    };

    std::vector<std::pair<Real, Real>> bounds{{Real(0), Real(1)}, {Real(0), Real(1)}};
    Real error_goal = 0.0002;
    naive_monte_carlo<Real, decltype(g)> mc(g, bounds, error_goal,
                                          /*singular =*/ false,/* threads = */ 2, /* seed = */ 18012);
    auto task = mc.integrate();
    Real pi_estimated = task.get();
    if (abs(pi_estimated - pi<Real>())/pi<Real>() > 0.005) {
        std::cout << "Error in estimation of pi too high, function calls: " << mc.calls() << "\n";
        std::cout << "Final error estimate : " << mc.current_error_estimate() << "\n";
        std::cout << "Error goal           : " << error_goal << "\n";
        BOOST_CHECK_CLOSE_FRACTION(pi_estimated, pi<Real>(), 0.005);
    }
}

template<class Real>
void test_pi()
{
    std::cout << "Testing pi is calculated correctly using Monte-Carlo on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    auto g = [](std::vector<Real> const & x)->Real
    {
        Real r = x[0]*x[0]+x[1]*x[1];
        if (r <= 1)
        {
            return 4;
        }
        return 0;
    };

    std::vector<std::pair<Real, Real>> bounds{{Real(0), Real(1)}, {Real(0), Real(1)}};
    Real error_goal = (Real) 0.0002;
    naive_monte_carlo<Real, decltype(g)> mc(g, bounds, error_goal,
                                            /*singular =*/ false,/* threads = */ 1, /* seed = */ 128402);
    auto task = mc.integrate();
    Real pi_estimated = task.get();
    if (abs(pi_estimated - pi<Real>())/pi<Real>() > 0.005)
    {
        std::cout << "Error in estimation of pi too high, function calls: " << mc.calls() << "\n";
        std::cout << "Final error estimate : " << mc.current_error_estimate() << "\n";
        std::cout << "Error goal           : " << error_goal << "\n";
        BOOST_CHECK_CLOSE_FRACTION(pi_estimated, pi<Real>(), 0.005);
    }

}

template<class Real>
void test_constant()
{
    std::cout << "Testing constants are integrated correctly using Monte-Carlo on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    auto g = [](std::vector<Real> const &)->Real
    {
      return 1;
    };

    std::vector<std::pair<Real, Real>> bounds{{Real(0), Real(1)}, { Real(0), Real(1)}};
    naive_monte_carlo<Real, decltype(g)> mc(g, bounds, (Real) 0.0001,
                                            /* singular = */ false, /* threads = */ 1, /* seed = */ 87);

    auto task = mc.integrate();
    Real one = task.get();
    BOOST_CHECK_CLOSE_FRACTION(one, 1, 0.001);
    BOOST_CHECK_SMALL(mc.current_error_estimate(), std::numeric_limits<Real>::epsilon());
    BOOST_CHECK(mc.calls() > 1000);
}


template<class Real>
void test_cancel_and_restart()
{
    std::cout << "Testing that cancellation and restarting works on naive Monte-Carlo integration on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real exact = BOOST_MATH_TEST_VALUE(Real, 1.3932039296856768591842462603255);
    constexpr const Real A = 1.0 / (pi<Real>() * pi<Real>() * pi<Real>());
    auto g = [&](std::vector<Real> const & x)->Real
    {
        return A / (1.0 - cos(x[0])*cos(x[1])*cos(x[2]));
    };
    vector<pair<Real, Real>> bounds{{ Real(0), pi<Real>()}, { Real(0), pi<Real>()}, { Real(0), pi<Real>()}};
    naive_monte_carlo<Real, decltype(g)> mc(g, bounds, (Real) 0.05, true, 1, 888889);

    auto task = mc.integrate();
    mc.cancel();
    Real y = task.get();
    // Super low tolerance; because it got canceled so fast:
    BOOST_CHECK_CLOSE_FRACTION(y, exact, static_cast<Real>(1.0));

    mc.update_target_error((Real) 0.01);
    task = mc.integrate();
    y = task.get();
    BOOST_CHECK_CLOSE_FRACTION(y, exact, 0.1);
}

template<class Real>
void test_finite_singular_boundary()
{
    std::cout << "Testing that finite singular boundaries work on naive Monte-Carlo integration on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    using std::pow;
    using std::log;
    using std::log1p;

    auto g = [](std::vector<Real> const & x)->Real
    {
        // The first term is singular at x = 0.
        // The second at x = 1:
        return pow(log(Real(1)/x[0]), Real(2)) + log1p(-x[0]);
    };
    vector<pair<Real, Real>> bounds{{Real(0), Real(1)}};
    naive_monte_carlo<Real, decltype(g)> mc(g, bounds, (Real) 0.01, true, 1, 1922);

    auto task = mc.integrate();

    Real y = task.get();
    BOOST_CHECK_CLOSE_FRACTION(y, Real(1), Real(0.1));
}

template<class Real>
void test_multithreaded_variance()
{
    std::cout << "Testing that variance computed by naive Monte-Carlo integration converges to integral formula on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real exact_variance = (Real) 1/(Real) 12;
    auto g = [&](std::vector<Real> const & x)->Real
    {
        return x[0];
    };
    vector<pair<Real, Real>> bounds{{ Real(0), Real(1)}};
    naive_monte_carlo<Real, decltype(g)> mc(g, bounds, (Real) 0.001, false, 2, 12341);

    auto task = mc.integrate();
    Real y = task.get();
    BOOST_CHECK_CLOSE_FRACTION(y, 0.5, 0.01);
    BOOST_CHECK_CLOSE_FRACTION(mc.variance(), exact_variance, 0.05);
}

template<class Real>
void test_variance()
{
    std::cout << "Testing that variance computed by naive Monte-Carlo integration converges to integral formula on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    Real exact_variance = (Real) 1/(Real) 12;
    auto g = [&](std::vector<Real> const & x)->Real
    {
        return x[0];
    };
    vector<pair<Real, Real>> bounds{{ Real(0), Real(1)}};
    naive_monte_carlo<Real, decltype(g)> mc(g, bounds, (Real) 0.001, false, 1, 12341);

    auto task = mc.integrate();
    Real y = task.get();
    BOOST_CHECK_CLOSE_FRACTION(y, static_cast<Real>(0.5), static_cast<Real>(0.01));
    BOOST_CHECK_CLOSE_FRACTION(mc.variance(), exact_variance, 0.05);
}

template<class Real, uint64_t dimension>
void test_product()
{
    std::cout << "Testing that product functions are integrated correctly by naive Monte-Carlo on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    auto g = [&](std::vector<Real> const & x)->Real
    {
        Real y = 1;
        for (uint64_t i = 0; i < x.size(); ++i)
        {
            y *= 2*x[i];
        }
        return y;
    };

    vector<pair<Real, Real>> bounds(dimension);
    for (uint64_t i = 0; i < dimension; ++i)
    {
        bounds[i] = std::make_pair<Real, Real>(0, 1);
    }
    naive_monte_carlo<Real, decltype(g)> mc(g, bounds, (Real) 0.001, false, 1, 13999);

    auto task = mc.integrate();
    Real y = task.get();
    BOOST_CHECK_CLOSE_FRACTION(y, 1, 0.01);
    using std::pow;
    Real exact_variance = pow(4.0/3.0, dimension) - 1;
    BOOST_CHECK_CLOSE_FRACTION(mc.variance(), exact_variance, 0.1);
}

template<class Real, uint64_t dimension>
void test_alternative_rng_1()
{
    std::cout << "Testing that alternative RNGs work correctly using naive Monte-Carlo on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    auto g = [&](std::vector<Real> const & x)->Real
    {
        Real y = 1;
        for (uint64_t i = 0; i < x.size(); ++i)
        {
            y *= 2*x[i];
        }
        return y;
    };

    vector<pair<Real, Real>> bounds(dimension);
    for (uint64_t i = 0; i < dimension; ++i)
    {
        bounds[i] = std::make_pair<Real, Real>(0, 1);
    }
    std::cout << "Testing std::mt19937" << std::endl;

    naive_monte_carlo<Real, decltype(g), std::mt19937> mc1(g, bounds, (Real) 0.001, false, 1, 1882);

    auto task = mc1.integrate();
    Real y = task.get();
    BOOST_CHECK_CLOSE_FRACTION(y, 1, 0.01);
    using std::pow;
    Real exact_variance = pow(4.0/3.0, dimension) - 1;
    BOOST_CHECK_CLOSE_FRACTION(mc1.variance(), exact_variance, 0.05);

    std::cout << "Testing std::knuth_b" << std::endl;
    naive_monte_carlo<Real, decltype(g), std::knuth_b> mc2(g, bounds, (Real) 0.001, false, 1, 1883);
    task = mc2.integrate();
    y = task.get();
    BOOST_CHECK_CLOSE_FRACTION(y, 1, 0.01);

    std::cout << "Testing std::ranlux48" << std::endl;
    naive_monte_carlo<Real, decltype(g), std::ranlux48> mc3(g, bounds, (Real) 0.001, false, 1, 1884);
    task = mc3.integrate();
    y = task.get();
    BOOST_CHECK_CLOSE_FRACTION(y, 1, 0.01);
}

template<class Real, uint64_t dimension>
void test_alternative_rng_2()
{
    std::cout << "Testing that alternative RNGs work correctly using naive Monte-Carlo on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    auto g = [&](std::vector<Real> const & x)->Real
    {
        Real y = 1;
        for (uint64_t i = 0; i < x.size(); ++i)
        {
            y *= 2*x[i];
        }
        return y;
    };

    vector<pair<Real, Real>> bounds(dimension);
    for (uint64_t i = 0; i < dimension; ++i)
    {
        bounds[i] = std::make_pair<Real, Real>(0, 1);
    }

    std::cout << "Testing std::default_random_engine" << std::endl;
    naive_monte_carlo<Real, decltype(g), std::default_random_engine> mc4(g, bounds, (Real) 0.001, false, 1, 1884);
    auto task = mc4.integrate();
    Real y = task.get();
    BOOST_CHECK_CLOSE_FRACTION(y, 1, 0.01);

    std::cout << "Testing std::minstd_rand" << std::endl;
    naive_monte_carlo<Real, decltype(g), std::minstd_rand> mc5(g, bounds, (Real) 0.001, false, 1, 1887);
    task = mc5.integrate();
    y = task.get();
    BOOST_CHECK_CLOSE_FRACTION(y, 1, 0.01);

    std::cout << "Testing std::minstd_rand0" << std::endl;
    naive_monte_carlo<Real, decltype(g), std::minstd_rand0> mc6(g, bounds, (Real) 0.001, false, 1, 1889);
    task = mc6.integrate();
    y = task.get();
    BOOST_CHECK_CLOSE_FRACTION(y, 1, 0.01);

}

template<class Real>
void test_upper_bound_infinite()
{
    std::cout << "Testing that infinite upper bounds are integrated correctly by naive Monte-Carlo on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    auto g = [](std::vector<Real> const & x)->Real
    {
        return 1.0/(x[0]*x[0] + 1.0);
    };

    vector<pair<Real, Real>> bounds(1);
    for (uint64_t i = 0; i < bounds.size(); ++i)
    {
        bounds[i] = std::make_pair<Real, Real>(0, std::numeric_limits<Real>::infinity());
    }
    naive_monte_carlo<Real, decltype(g)> mc(g, bounds, (Real) 0.001, true, 1, 8765);
    auto task = mc.integrate();
    Real y = task.get();
    BOOST_CHECK_CLOSE_FRACTION(y, boost::math::constants::half_pi<Real>(), 0.01);
}

template<class Real>
void test_lower_bound_infinite()
{
    std::cout << "Testing that infinite lower bounds are integrated correctly by naive Monte-Carlo on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    auto g = [](std::vector<Real> const & x)->Real
    {
        return 1.0/(x[0]*x[0] + 1.0);
    };

    vector<pair<Real, Real>> bounds(1);
    for (uint64_t i = 0; i < bounds.size(); ++i)
    {
        bounds[i] = std::make_pair<Real, Real>(-std::numeric_limits<Real>::infinity(), 0);
    }
    naive_monte_carlo<Real, decltype(g)> mc(g, bounds, (Real) 0.001, true, 1, 1208);

    auto task = mc.integrate();
    Real y = task.get();
    BOOST_CHECK_CLOSE_FRACTION(y, boost::math::constants::half_pi<Real>(), 0.01);
}

template<class Real>
void test_lower_bound_infinite2()
{
    std::cout << "Testing that infinite lower bounds (2) are integrated correctly by naive Monte-Carlo on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    auto g = [](std::vector<Real> const & x)->Real
    {
        // If x[0] = inf, this should blow up:
        return (x[0]*x[0])/(x[0]*x[0]*x[0]*x[0] + 1.0);
    };

    vector<pair<Real, Real>> bounds(1);
    for (uint64_t i = 0; i < bounds.size(); ++i)
    {
        bounds[i] = std::make_pair<Real, Real>(-std::numeric_limits<Real>::infinity(), 0);
    }
    naive_monte_carlo<Real, decltype(g)> mc(g, bounds, (Real) 0.001, true, 1, 1208);
    auto task = mc.integrate();
    Real y = task.get();
    BOOST_CHECK_CLOSE_FRACTION(y, boost::math::constants::half_pi<Real>()/boost::math::constants::root_two<Real>(), 0.01);
}

template<class Real>
void test_double_infinite()
{
    std::cout << "Testing that double infinite bounds are integrated correctly by naive Monte-Carlo on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    auto g = [](std::vector<Real> const & x)->Real
    {
        return 1.0/(x[0]*x[0] + 1.0);
    };

    vector<pair<Real, Real>> bounds(1);
    for (uint64_t i = 0; i < bounds.size(); ++i)
    {
        bounds[i] = std::make_pair<Real, Real>(-std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::infinity());
    }
    naive_monte_carlo<Real, decltype(g)> mc(g, bounds, (Real) 0.001, true, 1, 1776);

    auto task = mc.integrate();
    Real y = task.get();
    BOOST_CHECK_CLOSE_FRACTION(y, boost::math::constants::pi<Real>(), 0.01);
}

template<class Real, uint64_t dimension>
void test_radovic()
{
    // See: Generalized Halton Sequences in 2008: A Comparative Study, function g1:
    std::cout << "Testing that the Radovic function is integrated correctly by naive Monte-Carlo on type " << boost::typeindex::type_id<Real>().pretty_name() << "\n";
    auto g = [](std::vector<Real> const & x)->Real
    {
        using std::abs;
        Real alpha = (Real)0.01;
        Real z = 1;
        for (uint64_t i = 0; i < dimension; ++i)
        {
            z *= (abs(4*x[i]-2) + alpha)/(1+alpha);
        }
        return z;
    };

    vector<pair<Real, Real>> bounds(dimension);
    for (uint64_t i = 0; i < bounds.size(); ++i)
    {
        bounds[i] = std::make_pair<Real, Real>(0, 1);
    }
    Real error_goal = (Real) 0.001;
    naive_monte_carlo<Real, decltype(g)> mc(g, bounds, error_goal, false, 1, 1982);

    auto task = mc.integrate();
    Real y = task.get();
    if (abs(y - 1) > 0.01)
    {
        std::cout << "Error in estimation of Radovic integral too high, function calls: " << mc.calls() << "\n";
        std::cout << "Final error estimate: " << mc.current_error_estimate() << std::endl;
        std::cout << "Error goal          : " << error_goal << std::endl;
        std::cout << "Variance estimate   : " << mc.variance() << std::endl;
        BOOST_CHECK_CLOSE_FRACTION(y, 1, 0.01);
    }
}


BOOST_AUTO_TEST_CASE(naive_monte_carlo_test)
{
   std::cout << "Default hardware concurrency = " << std::thread::hardware_concurrency() << std::endl;
#if !defined(TEST) || TEST == 1

#if defined(__STDCPP_FLOAT32_T__) && defined(__STDCPP_FLOAT64_T__)
    test_finite_singular_boundary<std::float64_t>();
    test_finite_singular_boundary<std::float32_t>();
#else
    test_finite_singular_boundary<double>();
    test_finite_singular_boundary<float>();
#endif

#endif
#if !defined(TEST) || TEST == 2

#if defined(__STDCPP_FLOAT32_T__) && defined(__STDCPP_FLOAT64_T__)
    test_pi<std::float32_t>();
    test_pi<std::float64_t>();
#else
    test_pi<float>();
    test_pi<double>();
#endif

#endif
#if !defined(TEST) || TEST == 3

#ifdef __STDCPP_FLOAT32_T__
    test_pi_multithreaded<std::float32_t>();
    test_constant<std::float32_t>();
#else
    test_pi_multithreaded<float>();
    test_constant<float>();
#endif

#endif
    //test_pi<long double>();
#if !defined(TEST) || TEST == 4

#if defined(__STDCPP_FLOAT32_T__) && defined(__STDCPP_FLOAT64_T__)
    test_constant<std::float64_t>();
    test_cancel_and_restart<std::float32_t>();
#else
    test_constant<double>();
    //test_constant<long double>();
    test_cancel_and_restart<float>();
#endif

#endif
#if !defined(TEST) || TEST == 5

#ifdef __STDCPP_FLOAT32_T__
    test_variance<std::float32_t>();
#else
    test_variance<float>();
#endif

#endif
#if !defined(TEST) || TEST == 6

#ifdef __STDCPP_FLOAT64_T__
    test_variance<std::float64_t>();
    test_multithreaded_variance<std::float64_t>();
#else
    test_variance<double>();
    test_multithreaded_variance<double>();
#endif

#endif
#if !defined(TEST) || TEST == 7

#ifdef __STDCPP_FLOAT32_T__
    test_product<std::float32_t, 1>();
    test_product<std::float32_t, 2>();
#else
    test_product<float, 1>();
    test_product<float, 2>();
#endif

#endif
#if !defined(TEST) || TEST == 8

#ifdef __STDCPP_FLOAT32_T__
    test_product<std::float32_t, 3>();
    test_product<std::float32_t, 4>();
    test_product<std::float32_t, 5>();
#else
    test_product<float, 3>();
    test_product<float, 4>();
    test_product<float, 5>();
#endif

#endif
#if !defined(TEST) || TEST == 9

#if defined(__STDCPP_FLOAT32_T__) && defined(__STDCPP_FLOAT64_T__)
    test_product<std::float32_t, 6>();
    test_product<std::float64_t, 1>();
#else
    test_product<float, 6>();
    test_product<double, 1>();
#endif

#endif
#if !defined(TEST) || TEST == 10

#ifdef __STDCPP_FLOAT64_T__
    test_product<std::float64_t, 2>();
#else
    test_product<double, 2>();
#endif

#endif
#if !defined(TEST) || TEST == 11

#ifdef __STDCPP_FLOAT64_T__
    test_product<std::float64_t, 3>();
    test_product<std::float64_t, 4>();
#else
    test_product<double, 3>();
    test_product<double, 4>();
#endif

#endif
#if !defined(TEST) || TEST == 12

#if defined(__STDCPP_FLOAT32_T__) && defined(__STDCPP_FLOAT64_T__)
    test_upper_bound_infinite<std::float32_t>();
    test_upper_bound_infinite<std::float64_t>();
#else
    test_upper_bound_infinite<float>();
    test_upper_bound_infinite<double>();
#endif

#endif
#if !defined(TEST) || TEST == 13

#if defined(__STDCPP_FLOAT32_T__) && defined(__STDCPP_FLOAT64_T__)
    test_lower_bound_infinite<std::float32_t>();
    test_lower_bound_infinite<std::float64_t>();
#else
    test_lower_bound_infinite<float>();
    test_lower_bound_infinite<double>();
#endif

#endif
#if !defined(TEST) || TEST == 14

#ifdef __STDCPP_FLOAT32_T__
    test_lower_bound_infinite2<std::float32_t>();
#else
    test_lower_bound_infinite2<float>();
#endif

#endif
#if !defined(TEST) || TEST == 15

#if defined(__STDCPP_FLOAT32_T__) && defined(__STDCPP_FLOAT64_T__)
    test_double_infinite<std::float32_t>();
    test_double_infinite<std::float64_t>();
#else
    test_double_infinite<float>();
    test_double_infinite<double>();
#endif

#endif
#if !defined(TEST) || TEST == 16

#ifdef __STDCPP_FLOAT32_T__
    test_radovic<std::float32_t, 1>();
    test_radovic<std::float32_t, 2>();
#else
    test_radovic<float, 1>();
    test_radovic<float, 2>();
#endif

#endif
#if !defined(TEST) || TEST == 17

#if defined(__STDCPP_FLOAT32_T__) && defined(__STDCPP_FLOAT64_T__)
    test_radovic<std::float32_t, 3>();
    test_radovic<std::float64_t, 1>();
#else
    test_radovic<float, 3>();
    test_radovic<double, 1>();
#endif

#endif
#if !defined(TEST) || TEST == 18

#ifdef __STDCPP_FLOAT64_T__
    test_radovic<std::float64_t, 2>();
    test_radovic<std::float64_t, 3>();
#else
    test_radovic<double, 2>();
    test_radovic<double, 3>();
#endif

#endif
#if !defined(TEST) || TEST == 19

#ifdef __STDCPP_FLOAT64_T__
    test_radovic<std::float64_t, 4>();
    test_radovic<std::float64_t, 5>();
#else
    test_radovic<double, 4>();
    test_radovic<double, 5>();
#endif

#endif
#if !defined(TEST) || TEST == 20

#ifdef __STDCPP_FLOAT32_T__
    test_alternative_rng_1<std::float32_t, 3>();
#else
    test_alternative_rng_1<float, 3>();
#endif

#endif
#if !defined(TEST) || TEST == 21

#ifdef __STDCPP_FLOAT64_T__
    test_alternative_rng_1<std::float64_t, 3>();
#else
    test_alternative_rng_1<double, 3>();
#endif

#endif
#if !defined(TEST) || TEST == 22

#ifdef __STDCPP_FLOAT32_T__
    test_alternative_rng_2<std::float32_t, 3>();
#else
    test_alternative_rng_2<float, 3>();
#endif

#endif
#if !defined(TEST) || TEST == 23

#ifdef __STDCPP_FLOAT64_T__
    test_alternative_rng_2<std::float64_t, 3>();
#else
    test_alternative_rng_2<double, 3>();
#endif

#endif

}
