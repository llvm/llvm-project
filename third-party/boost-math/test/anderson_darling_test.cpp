/*
 * Copyright Nick Thompson, 2019
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include <numeric>
#include <utility>
#include <random>
#include <boost/core/demangle.hpp>
#include <boost/math/statistics/anderson_darling.hpp>
#ifdef BOOST_HAS_FLOAT128
#include <boost/multiprecision/float128.hpp>
using boost::multiprecision::float128;
#endif


using boost::math::statistics::anderson_darling_normality_statistic;

void test_ad_normal_agreement_w_mathematica()
{
    // To reproduce:
    // data = RandomVariate[NormalDistribution[0.0, 1], 32];
    // AndersonDarlingTest[data, NormalDistribution[0.0, 1.0], "TestStatistic"]
    std::vector<double> v = {0.6018671202167422, 0.34935310534248165, -0.2975464874480147, 0.19075371678288966, 0.7871329594028097, 1.153968559452821,
                            -0.22255484635945819, 0.08509539017654276, -1.3495873906132927, 0.5579622886226612, -0.8781637835839865, 1.144016010880056,
                             0.3117638373078617, 1.2774825413798292, 0.24847416220730323, -0.5002468827503855, -0.23124277830160153, 0.1880980838556693,
                            -1.2249409521605903, 2.583585311795, -1.4237305768919715, -0.7613985266437971, -1.089096700141564, -1.2595396780059587,
                            -0.735507006841229, -0.6957024998691735, 0.24671590370045435, 0.24620128485023313, -2.4445908627621833, 0.1882647961008893,
                             1.8527432579288248, -0.1908629038857823, -1.6542624501251562, 1.414542907065601, 0.8188282235404601, -0.5040501225935047,
                            -1.634617885381599, -0.43608956281596023, -1.7347258501594351, 0.8387606547880405, 0.3554282034994122, -0.27872688042721255,
                            -1.6445047421517192, 1.0262620986278987, -0.14517030192098415, -0.15386932114942647, -0.13364750619758223, 0.27194802295714676,
                            -0.6396907621289796, 0.3390884293844603, -0.08298748318375798, -0.7165572035514544, -0.2580817977433051, -1.9250094572058072,
                            -0.07995578394716944, -1.146629910012214, -2.0053073148074665, -1.6745995591714353, 0.2973860474005926, -0.4981763213542575,
                             0.22190763958244972, 0.699111028057888, -1.1872952966557475, 0.6151420566782272};

    double expected = 1.542329830419774;
    std::sort(v.begin(), v.end());
    double ADtest = anderson_darling_normality_statistic(v, 0.0, 1.0);

    CHECK_ULP_CLOSE(expected, ADtest, 250);

    double sigma = 2;
    for (auto & x : v) {
        x *= sigma;
    }
    ADtest = anderson_darling_normality_statistic(v, 0.0, sigma);
    CHECK_ULP_CLOSE(expected, ADtest, 250);

    double mu = 10;
    for (auto & x : v) {
        x += mu;
    }
    ADtest = anderson_darling_normality_statistic(v, mu, sigma);
    CHECK_ULP_CLOSE(expected, ADtest, 250);
}

int main()
{
    test_ad_normal_agreement_w_mathematica();
    return boost::math::test::report_errors();
}
