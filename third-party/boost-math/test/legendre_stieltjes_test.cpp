// Copyright Nick Thompson 2017.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/legendre_stieltjes.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>


using boost::math::legendre_stieltjes;
using boost::math::legendre_p;
using boost::multiprecision::cpp_bin_float_quad;


template<class Real>
void test_legendre_stieltjes()
{
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10);
    using std::sqrt;
    using std::abs;
    using boost::math::constants::third;
    using boost::math::constants::half;

    Real tol = std::numeric_limits<Real>::epsilon();
    legendre_stieltjes<Real> ls1(1);
    legendre_stieltjes<Real> ls2(2);
    legendre_stieltjes<Real> ls3(3);
    legendre_stieltjes<Real> ls4(4);
    legendre_stieltjes<Real> ls5(5);
    legendre_stieltjes<Real> ls8(8);
    Real x = -1;
    while(x <= 1)
    {
        BOOST_CHECK_CLOSE_FRACTION(ls1(x), x, tol);
        BOOST_CHECK_CLOSE_FRACTION(ls1.prime(x), 1, tol);

        Real p2 = legendre_p(2, x);
        BOOST_CHECK_CLOSE_FRACTION(ls2(x), p2 - 2/static_cast<Real>(5), tol);
        BOOST_CHECK_CLOSE_FRACTION(ls2.prime(x), 3*x, tol);

        Real p3 = legendre_p(3, x);
        BOOST_CHECK_CLOSE_FRACTION(ls3(x), p3 - 9*x/static_cast<Real>(14), 600*tol);
        BOOST_CHECK_CLOSE_FRACTION(ls3.prime(x), 15*x*x*half<Real>() -3*half<Real>()-9/static_cast<Real>(14), 100*tol);

        Real p4 = legendre_p(4, x);
        //-20P_2(x)/27 + 14P_0(x)/891
        Real E4 = p4 - 20*p2/static_cast<Real>(27) + 14/static_cast<Real>(891);
        BOOST_CHECK_CLOSE_FRACTION(ls4(x), E4, 250*tol);
        BOOST_CHECK_CLOSE_FRACTION(ls4.prime(x), 35*x*(9*x*x -5)/static_cast<Real>(18), 250*tol);

        Real p5 = legendre_p(5, x);
        Real E5 = p5 - 35*p3/static_cast<Real>(44) + 135*x/static_cast<Real>(12584);
        BOOST_CHECK_CLOSE_FRACTION(ls5(x), E5, 29000*tol);
        Real E5prime = (315*(123 + 143*x*x*(11*x*x-9)))/static_cast<Real>(12584);
        BOOST_CHECK_CLOSE_FRACTION(ls5.prime(x), E5prime, 29000*tol);
        x += 1/static_cast<Real>(1 << 9);
    }

    // Test norm:
    // E_1 = x
    Real expected_norm_sq = 2*third<Real>();
    BOOST_CHECK_CLOSE_FRACTION(expected_norm_sq, ls1.norm_sq(), tol);

    // E_2 = P[sub 2](x) - 2P[sup 0](x)/5
    expected_norm_sq = 2/static_cast<Real>(5) + 8/static_cast<Real>(25);
    BOOST_CHECK_CLOSE_FRACTION(expected_norm_sq, ls2.norm_sq(), tol);

    // E_3 = P[sub 3](x) - 9P[sub 1]/14
    expected_norm_sq = 2/static_cast<Real>(7) + 9*9*2*third<Real>()/static_cast<Real>(14*14);
    BOOST_CHECK_CLOSE_FRACTION(expected_norm_sq, ls3.norm_sq(), tol);

    // E_4 = P[sub 4](x) -20P[sub 2](x)/27 + 14P[sub 0](x)/891
    expected_norm_sq = static_cast<Real>(2)/static_cast<Real>(9) + static_cast<Real>(20*20*2)/static_cast<Real>(27*27*5) + 14*14*2/static_cast<Real>(891*891);
    BOOST_CHECK_CLOSE_FRACTION(expected_norm_sq, ls4.norm_sq(), tol);

    // E_5 = P[sub 5](x) - 35P[sub 3](x)/44 + 135P[sub 1](x)/12584
    expected_norm_sq = 2/static_cast<Real>(11) + (35*35/static_cast<Real>(44*44))*(2/static_cast<Real>(7)) + (135*135/static_cast<Real>(12584*12584))*2*third<Real>();
    BOOST_CHECK_CLOSE_FRACTION(expected_norm_sq, ls5.norm_sq(), tol);

    // Only zero of E1 is 0:
    std::vector<Real> zeros = ls1.zeros();
    BOOST_CHECK(zeros.size() == 1);
    BOOST_CHECK_SMALL(zeros[0], tol);
    BOOST_CHECK_SMALL(ls1(zeros[0]), tol);

    zeros = ls2.zeros();
    BOOST_CHECK(zeros.size() == 1);
    BOOST_CHECK_CLOSE_FRACTION(zeros[0], sqrt(3/static_cast<Real>(5)), tol);
    BOOST_CHECK_SMALL(ls2(zeros[0]), tol);

    zeros = ls3.zeros();
    BOOST_CHECK(zeros.size() == 2);
    BOOST_CHECK_SMALL(zeros[0], tol);
    BOOST_CHECK_CLOSE_FRACTION(zeros[1], sqrt(6/static_cast<Real>(7)), tol);


    zeros = ls4.zeros();
    BOOST_CHECK(zeros.size() == 2);
    Real expected = sqrt( (55 - 2*sqrt(static_cast<Real>(330)))/static_cast<Real>(11) )/static_cast<Real>(3);
    BOOST_CHECK_CLOSE_FRACTION(zeros[0], expected, tol);

    expected = sqrt( (55 + 2*sqrt(static_cast<Real>(330)))/static_cast<Real>(11) )/static_cast<Real>(3);
    BOOST_CHECK_CLOSE_FRACTION(zeros[1], expected, 10*tol);


    zeros = ls5.zeros();
    BOOST_CHECK(zeros.size() == 3);
    BOOST_CHECK_SMALL(zeros[0], tol);

    expected = sqrt( ( 195 - sqrt(static_cast<Real>(6045)) )/static_cast<Real>(286));
    BOOST_CHECK_CLOSE_FRACTION(zeros[1], expected, tol);

    expected = sqrt( ( 195 + sqrt(static_cast<Real>(6045)) )/static_cast<Real>(286));
    BOOST_CHECK_CLOSE_FRACTION(zeros[2], expected, tol);


    for (size_t i = 6; i < 50; ++i)
    {
        legendre_stieltjes<Real> En(i);
        zeros = En.zeros();
        for(auto const & zero : zeros)
        {
            BOOST_CHECK_SMALL(En(zero), 50*tol);
        }
    }
}


BOOST_AUTO_TEST_CASE(LegendreStieltjesZeros)
{
    test_legendre_stieltjes<double>();
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_legendre_stieltjes<long double>();
#endif
    test_legendre_stieltjes<cpp_bin_float_quad>();
    //test_legendre_stieltjes<boost::multiprecision::cpp_bin_float_100>();
}
