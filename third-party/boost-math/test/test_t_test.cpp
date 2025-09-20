/*
 * Copyright Nick Thompson, 2019
 * Copyright Matt Borland, 2021
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <vector>
#include <random>
#include <utility>
#include <boost/math/tools/config.hpp>
#include <boost/math/statistics/univariate_statistics.hpp>
#include <boost/math/statistics/t_test.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

template<typename Real>
void test_exact_mean()
{
    // Of course this test seems obvious, but just for the sake of comfort, here's the Mathematica to demo this test:
    // data3 = RandomReal[NormalDistribution[], 1024];
    // NumberForm[TTest[data3, Mean[data3], "TestStatistic"], 16]
    // NumberForm[TTest[data3, Mean[data3], "PValue"], 16]
    std::mt19937 gen{5125122};
    std::normal_distribution<Real> dis{0,3};
    std::vector<Real> v(1024);
    for (auto & x : v) {
      x = dis(gen);
    }

    Real mu = boost::math::statistics::mean(v);

    std::pair<Real, Real> temp = boost::math::statistics::one_sample_t_test(v, mu);
    Real computed_statistic = std::get<0>(temp);
    Real computed_pvalue = std::get<1>(temp);

    CHECK_MOLLIFIED_CLOSE(Real(0), computed_statistic, 10*std::numeric_limits<Real>::epsilon());
    CHECK_ULP_CLOSE(Real(1), computed_pvalue, 9);
}

template<typename Real>
void test_multiprecision_exact_mean()
{
    std::mt19937 gen{5125122};
    std::normal_distribution<long double> dis{0,3};
    std::vector<Real> v(1024);
    for (auto & x : v) {
      x = dis(gen);
    }

    Real mu = boost::math::statistics::mean(v);

    std::pair<Real, Real> temp = boost::math::statistics::one_sample_t_test(v, mu);
    Real computed_statistic = std::get<0>(temp);
    Real computed_pvalue = std::get<1>(temp);

    CHECK_MOLLIFIED_CLOSE(Real(0), computed_statistic, 20*std::numeric_limits<Real>::epsilon());
    CHECK_ULP_CLOSE(Real(1), computed_pvalue, 35);
}

template<typename Z>
void test_integer()
{
    // https://www.wolframalpha.com/input/?i=t+test
    std::pair<double, double> temp = boost::math::statistics::one_sample_t_test(Z(12), Z(5*5), Z(25), Z(10));
    double computed_statistic = std::get<0>(temp);
    double computed_pvalue = std::get<1>(temp);
    CHECK_MOLLIFIED_CLOSE(2.0, computed_statistic, 10*std::numeric_limits<double>::epsilon());
    CHECK_MOLLIFIED_CLOSE(0.02847*2, computed_pvalue, 0.00001);
}

void test_agreement_with_mathematica()
{
    // Reproduce via:
    //data = RandomReal[NormalDistribution[], 128];
    //NumberForm[data, 16]
    //NumberForm[TTest[data, 0.0, "TestStatistic"], 16]
    //NumberForm[TTest[data, 0.0, "PValue"], 16]
    std::vector<double> v{1.270498757865948,-0.7097895555907483,1.151445006538434,0.915648732663531,-0.7480131480881454,-0.6837323220203325,
                          2.362877076786142,2.438188734959438,0.1644154283470843,-0.980857299461513,-0.1448627006957758,0.04901437671768214,
                          -0.3895499730435337,1.412356512608596,-0.3865249523080916,-0.6159322168089271,-0.1865107372684944,-0.152509328597876,
                          1.142603106429423,-1.358368106645048,0.2268475747885975,0.4029249376986136,0.1167407378850566,0.05532794835680535,
                          -1.928794899326586,0.6496438708570567,0.269012797381103,-0.908168796067257,-0.6194990582309883,1.606256899489664,
                          -0.903964536847682,-1.375889704354273,0.04906080087803202,0.2039077019578547,-0.4907377045195846,-0.4781929001716083,
                          -0.2289802280011548,-1.339055086640687,-0.3120811524451416,0.06142580393246503,-0.140496390441262,-0.6482824149508374,
                          -0.2944027976542998,1.619416512991051,0.6285648262611375,1.312636016409526,-1.109965359363169,-0.774547681114892,
                          -0.344875897907528,0.816762481553918,0.1500701005574458,0.807790349338737,-0.2052962007348396,1.057657121384678,
                          0.836142529983228,0.3432803448381389,-0.01268905497569333,-1.144036865790547,-0.4530056923174255,-0.3061160863293071,
                          -0.02963689772198411,-1.33671649419749,-0.06052105439831394,0.973282554859066,-1.643904288065807,-1.0293884110541,
                          -0.5291066659852803,-0.3294227039691209,0.002993387508002654,0.2248580674319177,0.574521694409057,1.041337304293327,
                          0.4078548122453237,0.1112225876991191,-0.6448072259486091,-0.3051345048257077,0.089593933234481,0.4611768867673915,
                          0.7644315320471444,-0.341247840010495,0.0326958894744302,-0.05121900335567795,-0.06019531049352196,1.71234441194424,
                          -0.04175157932686885,0.769694813995503,-1.080913235981393,0.5989354496438777,-0.84416230123901,0.03165655009402087,
                          -0.7502374585144876,-2.734748382516766,1.541068679878993,0.1054620771416859,-0.6543692934553028,1.499220114211276,
                          -0.342006571062175,-0.2053132127077213,0.5457125644270833,-0.7956250897267784,0.7320742348115779,0.4674423735122585,
                          -0.3087396963145776,-1.53764162258267,0.2455449906251891,0.3795993803250636,-0.1195480230909131,0.137639511052913,
                          0.931721348902457,0.06704522870668304,-0.03773030445251862,0.3888322348695948,-0.06366757901233728,0.5563758371320388,
                          -0.7918177216642121,-0.7566297580399533,-0.3740377818446702,-0.6065664299451118,-0.2341124269010213,2.028052675971757,
                          0.378550889251416,0.816911727914731,1.162652387697876,-0.3853743867873177,1.196620648443396,0.01265660717000745,
                          1.816698960862263,-0.972941421015463};


    double expected_statistic = 0.4587075249160456;
    double expected_pvalue = 0.6472282548266728;

    std::pair<double, double> temp = boost::math::statistics::one_sample_t_test(v, 0.0);
    double computed_statistic = std::get<0>(temp);
    double computed_pvalue = std::get<1>(temp);

    CHECK_ULP_CLOSE(expected_statistic, computed_statistic, 8);
    CHECK_ULP_CLOSE(expected_pvalue, computed_pvalue, 90);


    {
        std::vector<double> v = {0.7304375676969546,3.227250635039257,1.01821954205186};

        expected_statistic = 2.103013485037935;
        expected_pvalue = 0.1701790440880712;

        std::pair<double, double> temp = boost::math::statistics::one_sample_t_test(v, 0.0);
        double computed_statistic = std::get<0>(temp);
        double computed_pvalue = std::get<1>(temp);
        CHECK_ULP_CLOSE(expected_statistic, computed_statistic, 2);
        CHECK_ULP_CLOSE(expected_pvalue, computed_pvalue, 7);
    }
}

template<typename Real>
void test_two_sample_t()
{
    std::pair<Real, Real> temp = 
        boost::math::statistics::detail::two_sample_t_test_impl<std::pair<Real, Real>>(Real(10.0), Real(1.0), Real(20), Real(5.0), Real(0.25), Real(20));
    Real computed_statistic = std::get<0>(temp);
    Real computed_pvalue = std::get<1>(temp);
    CHECK_ULP_CLOSE(Real(20), computed_statistic, 5);
    CHECK_MOLLIFIED_CLOSE(Real(0), computed_pvalue, 1e-21);

    std::vector<Real> set_1 {301, 298, 295, 297, 304, 305, 309, 298, 291, 299, 293, 304};

    temp = boost::math::statistics::two_sample_t_test(set_1, set_1);
    Real computed_statistic_2 = std::get<0>(temp);
    Real computed_pvalue_2 = std::get<1>(temp);
    CHECK_ULP_CLOSE(Real(0), computed_statistic_2, 5);
    CHECK_ULP_CLOSE(Real(1), computed_pvalue_2, 5);
}

template<typename Z>
void test_integer_two_sample_t()
{
    std::pair<double, double> temp = 
        boost::math::statistics::detail::two_sample_t_test_impl<std::pair<double, double>>(Z(10), Z(4), Z(20), Z(5), Z(1), Z(20));
    double computed_statistic = std::get<0>(temp);
    CHECK_ULP_CLOSE(10.0, computed_statistic, 5);

    std::vector<Z> set_1 {301, 298, 295, 297, 304, 305, 309, 298, 291, 299, 293, 304};

    temp = boost::math::statistics::two_sample_t_test(set_1, set_1);
    double computed_statistic_2 = std::get<0>(temp);
    double computed_pvalue_2 = std::get<1>(temp);
    CHECK_ULP_CLOSE(0.0, computed_statistic_2, 5);
    CHECK_ULP_CLOSE(1.0, computed_pvalue_2, 5);
}

template<typename Real>
void test_welch()
{
    using std::sqrt;
    
    std::pair<Real, Real> temp = 
        boost::math::statistics::detail::welchs_t_test_impl<std::pair<Real, Real>>(Real(10.0), Real(1.0), Real(20), Real(5.0), Real(0.25), Real(20));
    Real computed_statistic = std::get<0>(temp);
    Real computed_pvalue = std::get<1>(temp);
    CHECK_ULP_CLOSE(Real(20), computed_statistic, 5);
    CHECK_MOLLIFIED_CLOSE(Real(0), computed_pvalue, 5e-18);
    
    temp = boost::math::statistics::detail::welchs_t_test_impl<std::pair<Real, Real>>(Real(10.0), Real(0.5), Real(20), Real(10.0), Real(0.5), Real(20));
    Real computed_statistic_2 = std::get<0>(temp);
    Real computed_pvalue_2 = std::get<1>(temp);
    CHECK_ULP_CLOSE(Real(0), computed_statistic_2, 5);
    CHECK_ULP_CLOSE(Real(1), computed_pvalue_2, 5);
}

template<typename Z>
void test_integer_welch()
{
    std::pair<double, double> temp = 
        boost::math::statistics::detail::welchs_t_test_impl<std::pair<double, double>>(10.0, 4.0, 20.0, 5.0, 1.0, 20.0);
    double computed_statistic = std::get<0>(temp);
    CHECK_ULP_CLOSE(10.0, computed_statistic, 5);
    
    temp = boost::math::statistics::detail::welchs_t_test_impl<std::pair<double, double>>(10.0, 0.5, 20.0, 10.0, 0.5, 20.0);
    double computed_statistic_2 = std::get<0>(temp);
    double computed_pvalue_2 = std::get<1>(temp);
    CHECK_ULP_CLOSE(0.0, computed_statistic_2, 5);
    CHECK_ULP_CLOSE(1.0, computed_pvalue_2, 5);
}

template<typename Real>
void test_paired_samples()
{
    std::vector<Real> set_1 {2,4};
    std::vector<Real> set_2 {1,2};

    std::pair<Real, Real> temp = boost::math::statistics::paired_samples_t_test(set_1, set_2);
    Real computed_statistic = std::get<0>(temp);
    CHECK_ULP_CLOSE(Real(3), computed_statistic, 5);
}

template<typename Z>
void test_integer_paired_samples()
{
    std::vector<Z> set_1 {2,4};
    std::vector<Z> set_2 {1,2};

    std::pair<double, double> temp = boost::math::statistics::paired_samples_t_test(set_1, set_2);
    double computed_statistic = std::get<0>(temp);
    CHECK_ULP_CLOSE(3.0, computed_statistic, 5);
}

int main()
{
    test_agreement_with_mathematica();

    test_exact_mean<float>();
    test_exact_mean<double>();
    
    test_integer<int>();
    test_integer<int32_t>();
    test_integer<int64_t>();
    test_integer<uint32_t>();
    
    test_two_sample_t<float>();
    test_two_sample_t<double>();
    
    test_integer_two_sample_t<int>();
    test_integer_two_sample_t<int32_t>();
    test_integer_two_sample_t<int64_t>();
    test_integer_two_sample_t<uint32_t>();
    
    test_welch<float>();
    test_welch<double>();
    
    test_integer_welch<int>();
    test_integer_welch<int32_t>();
    test_integer_welch<int64_t>();
    test_integer_welch<uint32_t>();
    
    test_paired_samples<float>();
    test_paired_samples<double>();
    test_integer_paired_samples<int>();
    test_integer_paired_samples<int32_t>();
    test_integer_paired_samples<int64_t>();
    test_integer_paired_samples<uint32_t>();

    #ifndef BOOST_MATH_NO_MP_TESTS
    using quad = boost::multiprecision::cpp_bin_float_quad;
    test_multiprecision_exact_mean<quad>();
    test_two_sample_t<quad>();
    test_welch<quad>();
    #endif

    return boost::math::test::report_errors();
}
