// Copyright 2014 Marco Guazzone (marco.guazzone@gmail.com).
//
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <algorithm>
#include <boost/math/tools/test.hpp>
#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/distributions/complement.hpp>
#include <boost/math/distributions/hyperexponential.hpp>
#include <boost/math/tools/precision.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <cstddef>
#include <iostream>
#include <vector>
#include <tuple>

#define BOOST_MATH_HYPEREXP_CHECK_CLOSE_COLLECTIONS(T, actual, expected, tol) \
    do {                                                                      \
        std::vector<T> x = (actual);                                          \
        std::vector<T> y = (expected);                                        \
        BOOST_CHECK_EQUAL( x.size(), y.size() );                              \
        const std::size_t n = x.size();                                       \
        for (std::size_t i = 0; i < n; ++i)                                   \
        {                                                                     \
            BOOST_CHECK_CLOSE( x[i], y[i], tol );                             \
        }                                                                     \
    } while(false)

#if !defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
using test_types = std::tuple<float, double, long double, boost::math::concepts::real_concept>;
#elif !defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
using test_types = std::tuple<float, double, long double>;
#else
using test_types = std::tuple<float, double>;
#endif

template <typename RealT>
RealT make_tolerance()
{
    // Tolerance is 100eps expressed as a percentage (as required by Boost.Build):
    return boost::math::tools::epsilon<RealT>() * 100 * 100;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(klass, RealT, test_types)
{
    const RealT tol = make_tolerance<RealT>();

    boost::math::hyperexponential_distribution<RealT> dist;
    BOOST_CHECK_EQUAL(dist.num_phases(), 1);
    BOOST_CHECK_CLOSE(dist.probabilities()[0], static_cast<RealT>(1L), tol);
    BOOST_CHECK_CLOSE(dist.rates()[0], static_cast<RealT>(1L), tol);

    const RealT probs[] = { static_cast<RealT>(0.2L), static_cast<RealT>(0.3L), static_cast<RealT>(0.5L) };
    const RealT rates[] = { static_cast<RealT>(0.5L), static_cast<RealT>(1.0L), static_cast<RealT>(1.5L) };
    const std::size_t n = sizeof(probs) / sizeof(RealT);

    boost::math::hyperexponential_distribution<RealT> dist_it(probs, probs+n, rates, rates+n);
    BOOST_CHECK_EQUAL(dist_it.num_phases(), n);
    BOOST_MATH_HYPEREXP_CHECK_CLOSE_COLLECTIONS(RealT, dist_it.probabilities(), std::vector<RealT>(probs, probs+n), tol);
    BOOST_MATH_HYPEREXP_CHECK_CLOSE_COLLECTIONS(RealT, dist_it.rates(), std::vector<RealT>(rates, rates+n), tol);
    
    boost::math::hyperexponential_distribution<RealT> dist_r(probs, rates);
    BOOST_CHECK_EQUAL(dist_r.num_phases(), n);
    BOOST_MATH_HYPEREXP_CHECK_CLOSE_COLLECTIONS(RealT, dist_r.probabilities(), std::vector<RealT>(probs, probs+n), tol);
    BOOST_MATH_HYPEREXP_CHECK_CLOSE_COLLECTIONS(RealT, dist_r.rates(), std::vector<RealT>(rates, rates+n), tol);
    
#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST) && !(defined(BOOST_GCC_VERSION) && (BOOST_GCC_VERSION < 40500))
    boost::math::hyperexponential_distribution<RealT> dist_il = {{static_cast<RealT>(0.2L), static_cast<RealT>(0.3L), static_cast<RealT>(0.5L)}, {static_cast<RealT>(0.5L), static_cast<RealT>(1.0L), static_cast<RealT>(1.5L)}};
    BOOST_CHECK_EQUAL(dist_il.num_phases(), n);
    BOOST_MATH_HYPEREXP_CHECK_CLOSE_COLLECTIONS(RealT, dist_il.probabilities(), std::vector<RealT>(probs, probs+n), tol);
    BOOST_MATH_HYPEREXP_CHECK_CLOSE_COLLECTIONS(RealT, dist_il.rates(), std::vector<RealT>(rates, rates+n), tol);

    boost::math::hyperexponential_distribution<RealT> dist_n_r = { static_cast<RealT>(0.5L), static_cast<RealT>(1.0L), static_cast<RealT>(1.5L) };
    BOOST_CHECK_EQUAL(dist_n_r.num_phases(), n);
    BOOST_MATH_HYPEREXP_CHECK_CLOSE_COLLECTIONS(RealT, dist_n_r.probabilities(), std::vector<RealT>(n, static_cast<RealT>(1.0L / 3.0L)), tol);
    BOOST_MATH_HYPEREXP_CHECK_CLOSE_COLLECTIONS(RealT, dist_n_r.rates(), std::vector<RealT>(rates, rates + n), tol);
#endif // BOOST_NO_CXX11_HDR_INITIALIZER_LIST

    boost::math::hyperexponential_distribution<RealT> dist_n_it(rates, rates+n);
    BOOST_CHECK_EQUAL(dist_n_it.num_phases(), n);
    BOOST_MATH_HYPEREXP_CHECK_CLOSE_COLLECTIONS(RealT, dist_n_it.probabilities(), std::vector<RealT>(n, static_cast<RealT>(1.0L/3.0L)), tol);
    BOOST_MATH_HYPEREXP_CHECK_CLOSE_COLLECTIONS(RealT, dist_n_it.rates(), std::vector<RealT>(rates, rates+n), tol);

    boost::math::hyperexponential_distribution<RealT> dist_n_r2(rates);
    BOOST_CHECK_EQUAL(dist_n_r2.num_phases(), n);
    BOOST_MATH_HYPEREXP_CHECK_CLOSE_COLLECTIONS(RealT, dist_n_r2.probabilities(), std::vector<RealT>(n, static_cast<RealT>(1.0L/3.0L)), tol);
    BOOST_MATH_HYPEREXP_CHECK_CLOSE_COLLECTIONS(RealT, dist_n_r2.rates(), std::vector<RealT>(rates, rates+n), tol);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(range, RealT, test_types)
{
    const RealT tol = make_tolerance<RealT>();

    const RealT probs[] = { static_cast<RealT>(0.2L), static_cast<RealT>(0.3L), static_cast<RealT>(0.5L) };
    const RealT rates[] = { static_cast<RealT>(0.5L), static_cast<RealT>(1.0L), static_cast<RealT>(1.5L) };
    const std::size_t n = sizeof(probs) / sizeof(RealT);

    boost::math::hyperexponential_distribution<RealT> dist(probs, probs+n, rates, rates+n);

    std::pair<RealT,RealT> res;
    res = boost::math::range(dist);

    BOOST_CHECK_CLOSE( res.first, static_cast<RealT>(0), tol );
    if(std::numeric_limits<RealT>::has_infinity)
    {
       BOOST_CHECK_EQUAL(res.second, std::numeric_limits<RealT>::infinity());
    }
    else
    {
       BOOST_CHECK_EQUAL(res.second, boost::math::tools::max_value<RealT>());
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(support, RealT, test_types)
{
    const RealT tol = make_tolerance<RealT>();

    const RealT probs[] = { static_cast<RealT>(0.2L), static_cast<RealT>(0.3L), static_cast<RealT>(0.5L) };
    const RealT rates[] = { static_cast<RealT>(0.5L), static_cast<RealT>(1), static_cast<RealT>(1.5L) };
    const std::size_t n = sizeof(probs)/sizeof(RealT);

    boost::math::hyperexponential_distribution<RealT> dist(probs, probs+n, rates, rates+n);

    std::pair<RealT,RealT> res;
    res = boost::math::support(dist);

    BOOST_CHECK_CLOSE( res.first, boost::math::tools::min_value<RealT>(), tol );
    BOOST_CHECK_CLOSE( res.second, boost::math::tools::max_value<RealT>(), tol );
}

BOOST_AUTO_TEST_CASE_TEMPLATE(pdf, RealT, test_types)
{
    const RealT tol = make_tolerance<RealT>();

    const RealT probs[] = { static_cast<RealT>(0.2L), static_cast<RealT>(0.3L), static_cast<RealT>(0.5L) };
    const RealT rates[] = { static_cast<RealT>(0.5L), static_cast<RealT>(1), static_cast<RealT>(1.5) };
    const std::size_t n = sizeof(probs)/sizeof(RealT);

    boost::math::hyperexponential_distribution<RealT> dist(probs, probs+n, rates, rates+n);

    // Mathematica: Table[N[PDF[HyperexponentialDistribution[{1/5, 3/10, 1/2}, {1/2, 1, 3/2}], x], 35], {x, 0, 4}]
    BOOST_CHECK_CLOSE( boost::math::pdf(dist, static_cast<RealT>(0)), static_cast<RealT>(1.15L), tol );
    BOOST_CHECK_CLOSE( boost::math::pdf(dist, static_cast<RealT>(1)), static_cast<RealT>(0.33836451843401841053899743762056570L), tol );
    BOOST_CHECK_CLOSE( boost::math::pdf(dist, static_cast<RealT>(2)), static_cast<RealT>(0.11472883036402599696225903724543774L), tol );
    BOOST_CHECK_CLOSE( boost::math::pdf(dist, static_cast<RealT>(3)), static_cast<RealT>(0.045580883928883895659238122486617681L), tol );
    BOOST_CHECK_CLOSE( boost::math::pdf(dist, static_cast<RealT>(4)), static_cast<RealT>(0.020887284122781292094799231452333314L), tol );
}

BOOST_AUTO_TEST_CASE_TEMPLATE(cdf, RealT, test_types)
{
    const RealT tol = make_tolerance<RealT>();

    const RealT probs[] = { static_cast<RealT>(0.2L), static_cast<RealT>(0.3L), static_cast<RealT>(0.5L) };
    const RealT rates[] = { static_cast<RealT>(0.5L), static_cast<RealT>(1.0L), static_cast<RealT>(1.5L) };
    const std::size_t n = sizeof(probs)/sizeof(RealT);

    boost::math::hyperexponential_distribution<RealT> dist(probs, probs+n, rates, rates+n);

    // Mathematica: Table[N[CDF[HyperexponentialDistribution[{1/5, 3/10, 1/2}, {1/2, 1, 3/2}], x], 35], {x, 0, 4}]
    BOOST_CHECK_CLOSE( boost::math::cdf(dist, static_cast<RealT>(0)), static_cast<RealT>(0), tol );
    BOOST_CHECK_CLOSE( boost::math::cdf(dist, static_cast<RealT>(1)), static_cast<RealT>(0.65676495563182570433394272657131939L), tol );
    BOOST_CHECK_CLOSE( boost::math::cdf(dist, static_cast<RealT>(2)), static_cast<RealT>(0.86092999261079575662302418965093162L), tol );
    BOOST_CHECK_CLOSE( boost::math::cdf(dist, static_cast<RealT>(3)), static_cast<RealT>(0.93488334919083369807146961400871370L), tol );
    BOOST_CHECK_CLOSE( boost::math::cdf(dist, static_cast<RealT>(4)), static_cast<RealT>(0.96619887559772402832156211090812241L), tol );
}


BOOST_AUTO_TEST_CASE_TEMPLATE(quantile, RealT, test_types)
{
    const RealT tol = make_tolerance<RealT>();

    const RealT probs[] = { static_cast<RealT>(0.2L), static_cast<RealT>(0.3L), static_cast<RealT>(0.5L) };
    const RealT rates[] = { static_cast<RealT>(0.5L), static_cast<RealT>(1.0L), static_cast<RealT>(1.5L) };
    const std::size_t n = sizeof(probs)/sizeof(RealT);

    boost::math::hyperexponential_distribution<RealT> dist(probs, probs+n, rates, rates+n);

    // Mathematica: Table[N[Quantile[HyperexponentialDistribution[{1/5, 3/10, 1/2}, {1/2, 1, 3/2}], p], 35], {p, {0.`35, 0.6567649556318257043339427265713193884067872189124925936717`35, 0.8609299926107957566230241896509316171726985139265620607067`35, 0.9348833491908336980714696140087136988562861627183715044229`35, 0.9661988755977240283215621109081224127091468307592751727719`35}}]
    BOOST_CHECK_CLOSE( boost::math::quantile(dist, static_cast<RealT>(0)), static_cast<RealT>(0), tol );
    BOOST_CHECK_CLOSE( boost::math::quantile(dist, static_cast<RealT>(0.65676495563182570433394272657131939L)), static_cast<RealT>(1), tol );
    BOOST_CHECK_CLOSE( boost::math::quantile(dist, static_cast<RealT>(0.86092999261079575662302418965093162L)), static_cast<RealT>(2), tol );
    BOOST_CHECK_CLOSE( boost::math::quantile(dist, static_cast<RealT>(0.93488334919083369807146961400871370L)), static_cast<RealT>(3), tol );
    BOOST_CHECK_CLOSE( boost::math::quantile(dist, static_cast<RealT>(0.96619887559772402832156211090812241L)), static_cast<RealT>(4), tol );
}

BOOST_AUTO_TEST_CASE_TEMPLATE(ccdf, RealT, test_types)
{
    const RealT tol = make_tolerance<RealT>();

    const RealT probs[] = { static_cast<RealT>(0.2L), static_cast<RealT>(0.3L), static_cast<RealT>(0.5L) };
    const RealT rates[] = { static_cast<RealT>(0.5L), static_cast<RealT>(1.0L), static_cast<RealT>(1.5L) };
    const std::size_t n = sizeof(probs)/sizeof(RealT);

    boost::math::hyperexponential_distribution<RealT> dist(probs, probs+n, rates, rates+n);

    // Mathematica: Table[N[SurvivalFunction[HyperexponentialDistribution[{1/5, 3/10, 1/2}, {1/2, 1, 3/2}], x], 35], {x, 0, 4}]
    BOOST_CHECK_CLOSE( boost::math::cdf(boost::math::complement(dist, static_cast<RealT>(0))), static_cast<RealT>(1), tol );
    BOOST_CHECK_CLOSE( boost::math::cdf(boost::math::complement(dist, static_cast<RealT>(1))), static_cast<RealT>(0.34323504436817429566605727342868061L), tol );
    BOOST_CHECK_CLOSE( boost::math::cdf(boost::math::complement(dist, static_cast<RealT>(2))), static_cast<RealT>(0.13907000738920424337697581034906838L), tol );
    BOOST_CHECK_CLOSE( boost::math::cdf(boost::math::complement(dist, static_cast<RealT>(3))), static_cast<RealT>(0.065116650809166301928530385991286301L), tol );
    BOOST_CHECK_CLOSE( boost::math::cdf(boost::math::complement(dist, static_cast<RealT>(4))), static_cast<RealT>(0.033801124402275971678437889091877587L), tol );
}


BOOST_AUTO_TEST_CASE_TEMPLATE(cquantile, RealT, test_types)
{
    const RealT tol = make_tolerance<RealT>();

    const RealT probs[] = { static_cast<RealT>(0.2L), static_cast<RealT>(0.3L), static_cast<RealT>(0.5L) };
    const RealT rates[] = { static_cast<RealT>(0.5L), static_cast<RealT>(1.0L), static_cast<RealT>(1.5L) };
    const std::size_t n = sizeof(probs) / sizeof(RealT);

    boost::math::hyperexponential_distribution<RealT> dist(probs, probs+n, rates, rates+n);

    // Mathematica: Table[N[InverseSurvivalFunction[HyperexponentialDistribution[{1/5, 3/10, 1/2}, {1/2, 1, 3/2}], p], 35], {p, {1.`35, 0.3432350443681742956660572734286806115932127810875074063283`35, 0.1390700073892042433769758103490683828273014860734379392933`35, 0.0651166508091663019285303859912863011437138372816284955771`35, 0.0338011244022759716784378890918775872908531692407248272281`35}}]
    BOOST_CHECK_CLOSE( boost::math::quantile(boost::math::complement(dist, static_cast<RealT>(1))), static_cast<RealT>(0), tol );
    BOOST_CHECK_CLOSE( boost::math::quantile(boost::math::complement(dist, static_cast<RealT>(0.34323504436817429566605727342868061L))), static_cast<RealT>(1), tol );
    BOOST_CHECK_CLOSE( boost::math::quantile(boost::math::complement(dist, static_cast<RealT>(0.13907000738920424337697581034906838L))), static_cast<RealT>(2), tol );
    BOOST_CHECK_CLOSE( boost::math::quantile(boost::math::complement(dist, static_cast<RealT>(0.065116650809166301928530385991286301L))), static_cast<RealT>(3), tol );
    BOOST_CHECK_CLOSE( boost::math::quantile(boost::math::complement(dist, static_cast<RealT>(0.033801124402275971678437889091877587L))), static_cast<RealT>(4), tol );
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mean, RealT, test_types)
{
    const RealT tol = make_tolerance<RealT>();

    const RealT probs[] = { static_cast<RealT>(0.2L), static_cast<RealT>(0.3L), static_cast<RealT>(0.5L) };
    const RealT rates[] = { static_cast<RealT>(0.5L), static_cast<RealT>(1.0L), static_cast<RealT>(1.5L) };
    const std::size_t n = sizeof(probs) / sizeof(RealT);

    boost::math::hyperexponential_distribution<RealT> dist(probs, probs+n, rates, rates+n);

    // Mathematica: N[Mean[HyperexponentialDistribution[{1/5, 3/10, 1/2}, {1/2, 1, 3/2}]], 35]
    BOOST_CHECK_CLOSE( boost::math::mean(dist), static_cast<RealT>(1.0333333333333333333333333333333333L), tol );
}

BOOST_AUTO_TEST_CASE_TEMPLATE(variance, RealT, test_types)
{
    const RealT tol = make_tolerance<RealT>();

    const RealT probs[] = { static_cast<RealT>(0.2L), static_cast<RealT>(0.3L), static_cast<RealT>(0.5L) };
    const RealT rates[] = { static_cast<RealT>(0.5L), static_cast<RealT>(1.0L), static_cast<RealT>(1.5L) };
    const std::size_t n = sizeof(probs) / sizeof(RealT);

    boost::math::hyperexponential_distribution<RealT> dist(probs, probs+n, rates, rates+n);

    // Mathematica: N[Variance[HyperexponentialDistribution[{1/5, 3/10, 1/2}, {1/2, 1, 3/2}]], 35]
    BOOST_CHECK_CLOSE( boost::math::variance(dist), static_cast<RealT>(1.5766666666666666666666666666666667L), tol );
}

BOOST_AUTO_TEST_CASE_TEMPLATE(kurtosis, RealT, test_types)
{
    const RealT tol = make_tolerance<RealT>();

    const RealT probs[] = { static_cast<RealT>(0.2L), static_cast<RealT>(0.3L), static_cast<RealT>(0.5L) };
    const RealT rates[] = { static_cast<RealT>(0.5L), static_cast<RealT>(1.0L), static_cast<RealT>(1.5L) };
    const std::size_t n = sizeof(probs) / sizeof(RealT);

    boost::math::hyperexponential_distribution<RealT> dist(probs, probs+n, rates, rates+n);

    // Mathematica: N[Kurtosis[HyperexponentialDistribution[{1/5, 3/10, 1/2}, {1/2, 1, 3/2}]], 35]
    BOOST_CHECK_CLOSE( boost::math::kurtosis(dist), static_cast<RealT>(19.750738616808728416968743435138046L), tol );
    // Mathematica: N[Kurtosis[HyperexponentialDistribution[{1/5, 3/10, 1/2}, {1/2, 1, 3/2}] - 3.`35], 35]
    BOOST_CHECK_CLOSE( boost::math::kurtosis_excess(dist), static_cast<RealT>(16.750738616808728416968743435138046L), tol );
}

BOOST_AUTO_TEST_CASE_TEMPLATE(skewness, RealT, test_types)
{
    const RealT tol = make_tolerance<RealT>();

    const RealT probs[] = { static_cast<RealT>(0.2L), static_cast<RealT>(0.3L), static_cast<RealT>(0.5L) };
    const RealT rates[] = { static_cast<RealT>(0.5L), static_cast<RealT>(1.0L), static_cast<RealT>(1.5L) };
    const std::size_t n = sizeof(probs) / sizeof(RealT);

    boost::math::hyperexponential_distribution<RealT> dist(probs, probs+n, rates, rates+n);

    // Mathematica: N[Skewness[HyperexponentialDistribution[{1/5, 3/10, 1/2}, {1/2, 1, 3/2}]], 35]
    BOOST_CHECK_CLOSE( boost::math::skewness(dist), static_cast<RealT>(3.1811387449963809211146099116375685L), tol );
}

BOOST_AUTO_TEST_CASE_TEMPLATE(mode, RealT, test_types)
{
    const RealT tol = make_tolerance<RealT>();

    const RealT probs[] = { static_cast<RealT>(0.2L), static_cast<RealT>(0.3L), static_cast<RealT>(0.5L) };
    const RealT rates[] = { static_cast<RealT>(0.5L), static_cast<RealT>(1.0L), static_cast<RealT>(1.5L) };
    const std::size_t n = sizeof(probs) / sizeof(RealT);

    boost::math::hyperexponential_distribution<RealT> dist(probs, probs+n, rates, rates+n);

    BOOST_CHECK_CLOSE( boost::math::mode(dist), static_cast<RealT>(0), tol );
}

template <class T>
void f(T t)
{
   std::cout << typeid(t).name() << std::endl;
}

BOOST_AUTO_TEST_CASE(construct)
{
   std::array<double, 3> da1 = { { 0.5, 1, 1.5 } };
   std::array<double, 3> da2 = { { 0.25, 0.5, 0.25 } };
   std::vector<double> v1(da1.begin(), da1.end());
   std::vector<double> v2(da2.begin(), da2.end());

   std::vector<double> result_v;
   boost::math::hyperexponential he1(v2, v1);
   result_v = he1.rates();
   BOOST_CHECK_EQUAL_COLLECTIONS(v1.begin(), v1.end(), result_v.begin(), result_v.end());
   result_v = he1.probabilities();
   BOOST_CHECK_EQUAL_COLLECTIONS(v2.begin(), v2.end(), result_v.begin(), result_v.end());

   boost::math::hyperexponential he2(da2, da1);
   result_v = he2.rates();
   BOOST_CHECK_EQUAL_COLLECTIONS(v1.begin(), v1.end(), result_v.begin(), result_v.end());
   result_v = he2.probabilities();
   BOOST_CHECK_EQUAL_COLLECTIONS(v2.begin(), v2.end(), result_v.begin(), result_v.end());

#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST) && !(defined(BOOST_GCC_VERSION) && (BOOST_GCC_VERSION < 40500))
   std::initializer_list<double> il = { 0.25, 0.5, 0.25 };
   std::initializer_list<double> il2 = { 0.5, 1, 1.5 };
   boost::math::hyperexponential he3(il, il2);
   result_v = he3.rates();
   BOOST_CHECK_EQUAL_COLLECTIONS(v1.begin(), v1.end(), result_v.begin(), result_v.end());
   result_v = he3.probabilities();
   BOOST_CHECK_EQUAL_COLLECTIONS(v2.begin(), v2.end(), result_v.begin(), result_v.end());

   boost::math::hyperexponential he4({ 0.25, 0.5, 0.25 }, { 0.5, 1.0, 1.5 });
   result_v = he4.rates();
   BOOST_CHECK_EQUAL_COLLECTIONS(v1.begin(), v1.end(), result_v.begin(), result_v.end());
   result_v = he4.probabilities();
   BOOST_CHECK_EQUAL_COLLECTIONS(v2.begin(), v2.end(), result_v.begin(), result_v.end());
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(special_cases, RealT, test_types)
{
    const RealT tol = make_tolerance<RealT>();

    // When the number of phases is 1, the hyperexponential distribution is an exponential distribution
    const RealT rates1[] = { static_cast<RealT>(0.5L) };
    boost::math::hyperexponential_distribution<RealT> hexp1(rates1);
    boost::math::exponential_distribution<RealT> exp1(rates1[0]);
    BOOST_CHECK_CLOSE(boost::math::pdf(hexp1, static_cast<RealT>(1L)), boost::math::pdf(exp1, static_cast<RealT>(1L)), tol);
    BOOST_CHECK_CLOSE(boost::math::cdf(hexp1, static_cast<RealT>(1L)), boost::math::cdf(exp1, static_cast<RealT>(1L)), tol);
    BOOST_CHECK_CLOSE(boost::math::mean(hexp1), boost::math::mean(exp1), tol);
    BOOST_CHECK_CLOSE(boost::math::variance(hexp1), boost::math::variance(exp1), tol);
    BOOST_CHECK_CLOSE(boost::math::quantile(hexp1, static_cast<RealT>(0.25L)), boost::math::quantile(exp1, static_cast<RealT>(0.25L)), tol);
    BOOST_CHECK_CLOSE(boost::math::median(hexp1), boost::math::median(exp1), tol);
    BOOST_CHECK_CLOSE(boost::math::quantile(hexp1, static_cast<RealT>(0.75L)), boost::math::quantile(exp1, static_cast<RealT>(0.75L)), tol);
    BOOST_CHECK_CLOSE(boost::math::mode(hexp1), boost::math::mode(exp1), tol);

    // When a k-phase hyperexponential distribution has all rates equal to r, the distribution is an exponential distribution with rate r
    const RealT rate2 = static_cast<RealT>(0.5L);
    const RealT rates2[] = { rate2, rate2, rate2 };
    boost::math::hyperexponential_distribution<RealT> hexp2(rates2);
    boost::math::exponential_distribution<RealT> exp2(rate2);
    BOOST_CHECK_CLOSE(boost::math::pdf(hexp2, static_cast<RealT>(1L)), boost::math::pdf(exp2, static_cast<RealT>(1L)), tol);
    BOOST_CHECK_CLOSE(boost::math::cdf(hexp2, static_cast<RealT>(1L)), boost::math::cdf(exp2, static_cast<RealT>(1L)), tol);
    BOOST_CHECK_CLOSE(boost::math::mean(hexp2), boost::math::mean(exp2), tol);
    BOOST_CHECK_CLOSE(boost::math::variance(hexp2), boost::math::variance(exp2), tol);
    BOOST_CHECK_CLOSE(boost::math::quantile(hexp2, static_cast<RealT>(0.25L)), boost::math::quantile(exp2, static_cast<RealT>(0.25L)), tol);
    BOOST_CHECK_CLOSE(boost::math::median(hexp2), boost::math::median(exp2), tol);
    BOOST_CHECK_CLOSE(boost::math::quantile(hexp2, static_cast<RealT>(0.75L)), boost::math::quantile(exp2, static_cast<RealT>(0.75L)), tol);
    BOOST_CHECK_CLOSE(boost::math::mode(hexp2), boost::math::mode(exp2), tol);
}

// Test C++20 ranges (Currently only GCC10 has full support to P0896R4)
#if (__cplusplus > 202000L || _MSVC_LANG > 202000L) && __has_include(<ranges>) && __GNUC__ >= 10
// Support for ranges is broken using gcc 11.1
#if __GNUC__ != 11
#include <ranges>
#include <array>
#endif
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(error_cases, RealT, test_types)
{
   typedef boost::math::hyperexponential_distribution<RealT> dist_t;
   std::array<RealT, 2> probs = { { 1, 2 } };
   std::array<RealT, 3> probs2 = { { 1, 2, 3 } };
   std::array<RealT, 3> rates = { { 1, 2, 3 } };
   BOOST_MATH_CHECK_THROW(dist_t(probs.begin(), probs.end(), rates.begin(), rates.end()), std::domain_error);
   BOOST_MATH_CHECK_THROW(dist_t(probs, rates), std::domain_error);
   rates[1] = 0;
   BOOST_MATH_CHECK_THROW(dist_t(probs2, rates), std::domain_error);
   rates[1] = -1;
   BOOST_MATH_CHECK_THROW(dist_t(probs2, rates), std::domain_error);
   BOOST_MATH_CHECK_THROW(dist_t(probs.begin(), probs.begin(), rates.begin(), rates.begin()), std::domain_error);
   BOOST_MATH_CHECK_THROW(dist_t(rates.begin(), rates.begin()), std::domain_error);

   // Test C++20 ranges (Currently only GCC10 has full support to P0896R4)
   #if (__cplusplus > 202000L || _MSVC_LANG > 202000L) && __has_include(<ranges>) && __GNUC__ >= 10
   // Support for ranges is broken using gcc 11.1
   #if __GNUC__ != 11

   std::array<RealT, 2> probs_array {1,2};
   std::array<RealT, 3> rates_array {1,2,3};
   BOOST_MATH_CHECK_THROW(dist_t(std::ranges::begin(probs_array), std::ranges::end(probs_array), std::ranges::begin(rates_array), std::ranges::end(rates_array)), std::domain_error);

   const auto probs_range = probs_array | std::ranges::views::all;
   const auto rates_range = rates_array | std::ranges::views::all;

   BOOST_MATH_CHECK_THROW(dist_t(probs_range, rates_range), std::domain_error);
   #endif
   #endif
}
