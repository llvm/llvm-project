/*
 *  (C) Copyright Nick Thompson 2018.
 *  Use, modification and distribution are subject to the
 *  Boost Software License, Version 1.0. (See accompanying file
 *  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#include <cmath>
#include <vector>
#include <array>
#include <forward_list>
#include <algorithm>
#include <random>
#include <limits>
#include <boost/core/lightweight_test.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/tools/norms.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_complex.hpp>

using std::abs;
using std::pow;
using std::sqrt;
using boost::multiprecision::cpp_bin_float_50;
using boost::multiprecision::cpp_complex_50;
using boost::math::tools::lp_norm;
using boost::math::tools::l1_norm;
using boost::math::tools::l2_norm;
using boost::math::tools::sup_norm;
using boost::math::tools::lp_distance;
using boost::math::tools::l1_distance;
using boost::math::tools::l2_distance;
using boost::math::tools::sup_distance;
using boost::math::tools::total_variation;

/*
 * Test checklist:
 * 1) Does it work with multiprecision?
 * 2) Does it work with .cbegin()/.cend() if the data is not altered?
 * 3) Does it work with ublas and std::array? (Checking Eigen and Armadillo will make the CI system really unhappy.)
 * 4) Does it work with std::forward_list if a forward iterator is all that is required?
 * 5) Does it work with complex data if complex data is sensible?
 */

// To stress test, set global_seed = 0, global_size = huge.
static const constexpr size_t global_seed = 834;
static const constexpr size_t global_size = 64;

template<class T>
std::vector<T> generate_random_vector(size_t size, size_t seed)
{
    if (seed == 0)
    {
        std::random_device rd;
        seed = rd();
    }
    std::vector<T> v(size);

    std::mt19937 gen(seed);

    if constexpr (std::is_floating_point<T>::value)
    {
        std::normal_distribution<T> dis(0, 1);
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = dis(gen);
        }
        return v;
    }
    else if constexpr (std::is_integral<T>::value)
    {
        // Rescaling by larger than 2 is UB!
        std::uniform_int_distribution<T> dis(std::numeric_limits<T>::lowest()/2, (std::numeric_limits<T>::max)()/2);
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = dis(gen);
        }
        return v;
    }
    else if constexpr (boost::is_complex<T>::value)
    {
        std::normal_distribution<typename T::value_type> dis(0, 1);
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = {dis(gen), dis(gen)};
        }
        return v;
    }
    else if constexpr (boost::multiprecision::number_category<T>::value == boost::multiprecision::number_kind_complex)
    {
        std::normal_distribution<long double> dis(0, 1);
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = {dis(gen), dis(gen)};
        }
        return v;
    }
    else if constexpr (boost::multiprecision::number_category<T>::value == boost::multiprecision::number_kind_floating_point)
    {
        std::normal_distribution<long double> dis(0, 1);
        for (size_t i = 0; i < v.size(); ++i)
        {
            v[i] = dis(gen);
        }
        return v;
    }
    else
    {
        BOOST_MATH_ASSERT_MSG(false, "Could not identify type for random vector generation.");
        return v;
    }
}


template<class Real>
void test_lp()
{
    Real tol = 50*std::numeric_limits<Real>::epsilon();

    std::array<Real, 3> u{1,0,0};
    Real l3 = lp_norm(u.begin(), u.end(), 3);
    BOOST_TEST(abs(l3 - 1) < tol);

    u[0] = -8;
    l3 = lp_norm(u.cbegin(), u.cend(), 3);
    BOOST_TEST(abs(l3 - 8) < tol);

    std::vector<Real> v(500);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = 7;
    }
    Real l8 = lp_norm(v, 8);
    Real expected = 7*pow(v.size(), static_cast<Real>(1)/static_cast<Real>(8));
    BOOST_TEST(abs(l8 - expected) < tol*abs(expected));

    // Does it work with ublas vectors?
    // Does it handle the overflow of intermediates?
    boost::numeric::ublas::vector<Real> w(4);
    Real bignum = sqrt((std::numeric_limits<Real>::max)())/256;
    for (size_t i = 0; i < w.size(); ++i)
    {
        w[i] = bignum;
    }
    Real l20 = lp_norm(w.cbegin(), w.cend(), 4);
    expected = bignum*pow(w.size(), static_cast<Real>(1)/static_cast<Real>(4));
    BOOST_TEST(abs(l20 - expected) < tol*expected);

    v = generate_random_vector<Real>(global_size, global_seed);
    Real scale = 8;
    Real l7 = scale*lp_norm(v, 7);
    for (auto & x : v)
    {
        x *= -scale;
    }
    Real l7_ = lp_norm(v, 7);
    BOOST_TEST(abs(l7_ - l7) < tol*l7);
}


template<class Complex>
void test_complex_lp()
{
    typedef typename Complex::value_type Real;
    Real tol = 50*std::numeric_limits<Real>::epsilon();
    std::vector<Complex> v{{1,0}, {0,0}, {0,0}};
    Real l3 = lp_norm(v.cbegin(), v.cend(), 3);
    BOOST_TEST(abs(l3 - 1) < tol);

    l3 = lp_norm(v, 3);
    BOOST_TEST(abs(l3 - 1) < tol);

    v = generate_random_vector<Complex>(global_size, global_seed);
    Real scale = 8;
    Real l7 = scale*lp_norm(v, 7);
    for (auto & x : v)
    {
        x *= -scale;
    }
    Real l7_ = lp_norm(v, 7);
    BOOST_TEST(abs(l7_ - l7) < tol*l7);
}

template<class Z>
void test_integer_lp()
{
    double tol = 100*std::numeric_limits<double>::epsilon();

    std::array<Z, 3> u{1,0,0};
    double l3 = lp_norm(u.begin(), u.end(), 3);
    BOOST_TEST(abs(l3 - 1) < tol);

    auto v = generate_random_vector<Z>(global_size, global_seed);
    Z scale = 2;
    double l7 = scale*lp_norm(v, 7);
    for (auto & x : v)
    {
        x *= scale;
    }
    double l7_ = lp_norm(v, 7);
    BOOST_TEST(abs(l7_ - l7) < tol*l7);
}

template<class Real>
void test_lp_distance()
{
    Real tol = 100*std::numeric_limits<Real>::epsilon();

    std::vector<Real> u{1,0,0};
    std::vector<Real> v{0,0,0};

    Real dist = lp_distance(u,u, 3);
    BOOST_TEST(abs(dist) < tol);

    dist = lp_distance(u,v, 3);
    BOOST_TEST(abs(dist - 1) < tol);

    v = generate_random_vector<Real>(global_size, global_seed);
    u = generate_random_vector<Real>(global_size, global_seed+1);
    Real dist1 = lp_distance(u, v, 7);
    Real dist2 = lp_distance(v, u, 7);

    BOOST_TEST(abs(dist1 - dist2) < tol*dist1);
}

template<class Complex>
void test_complex_lp_distance()
{
    using Real = typename Complex::value_type;
    Real tol = 100*std::numeric_limits<Real>::epsilon();

    std::vector<Complex> u{{1,0},{0,0},{0,0}};
    std::vector<Complex> v{{0,0},{0,0},{0,0}};

    Real dist = boost::math::tools::lp_distance(u,u, 3);
    BOOST_TEST(abs(dist) < tol);

    dist = boost::math::tools::lp_distance(u,v, 3);
    BOOST_TEST(abs(dist - 1) < tol);

    v = generate_random_vector<Complex>(global_size, global_seed);
    u = generate_random_vector<Complex>(global_size, global_seed + 1);
    Real dist1 = lp_distance(u, v, 7);
    Real dist2 = lp_distance(v, u, 7);

    BOOST_TEST(abs(dist1 - dist2) < tol*dist1);
}

template<class Z>
void test_integer_lp_distance()
{
    double tol = 100*std::numeric_limits<double>::epsilon();

    std::array<Z, 3> u{1,0,0};
    std::array<Z, 3> w{0,0,0};
    double l3 = lp_distance(u, w, 3);
    BOOST_TEST(abs(l3 - 1) < tol);

    auto v = generate_random_vector<Z>(global_size, global_seed);
    Z scale = 2;
    for (auto & x : v)
    {
        x *= scale;
    }
    auto s = generate_random_vector<Z>(global_size, global_seed + 1);
    double dist1 = lp_distance(v, s, 7);
    double dist2 = lp_distance(s, v, 7);
    BOOST_TEST(abs(dist1 - dist2) < tol*dist2);
}


template<class Z>
void test_integer_total_variation()
{
    double eps = std::numeric_limits<double>::epsilon();
    std::vector<Z> v{1,1};
    double tv = boost::math::tools::total_variation(v);
    BOOST_TEST_EQ(tv, 0);

    v[1] = 2;
    tv = boost::math::tools::total_variation(v.begin(), v.end());
    BOOST_TEST_EQ(tv, 1);

    v.resize(16);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = i;
    }

    tv = boost::math::tools::total_variation(v);
    BOOST_TEST_EQ(tv, v.size() -1);

    for (size_t i = 0; i < v.size(); ++i)
    {
        v[i] = i*i;
    }

    tv = boost::math::tools::total_variation(v);
    BOOST_TEST_EQ(tv, (v.size() - 1)*(v.size() - 1));

    // Work with std::array?
    std::array<Z, 2> w{1,1};
    tv = boost::math::tools::total_variation(w);
    BOOST_TEST_EQ(tv,0);

    std::array<Z, 4> u{1, 2, 1, 2};
    tv = boost::math::tools::total_variation(u);
    BOOST_TEST_EQ(tv, 3);

    v = generate_random_vector<Z>(global_size, global_seed);
    double tv1 = 2*total_variation(v);
    Z scale = 2;
    for (auto & x : v)
    {
        x *= scale;
    }
    double tv2 = total_variation(v);
    BOOST_TEST(abs(tv1 - tv2) < tv1*eps);
}

template<class Real>
void test_total_variation()
{
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{1,1};
    Real tv = total_variation(v.begin(), v.end());
    BOOST_TEST(tv >= 0 && abs(tv) < tol);

    tv = total_variation(v);
    BOOST_TEST(tv >= 0 && abs(tv) < tol);

    v[1] = 2;
    tv = total_variation(v.begin(), v.end());
    BOOST_TEST(abs(tv - 1) < tol);

    v.resize(50);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = i;
    }

    tv = total_variation(v.begin(), v.end());
    BOOST_TEST(abs(tv - (v.size() -1)) < tol);

    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = i*i;
    }

    tv = total_variation(v.begin(), v.end());
    BOOST_TEST(abs(tv - (v.size() - 1)*(v.size() - 1)) < tol);


    v = generate_random_vector<Real>(global_size, global_seed);
    Real scale = 8;
    Real tv1 = scale*total_variation(v);
    for (auto & x : v)
    {
        x *= -scale;
    }
    Real tv2 = total_variation(v);
    BOOST_TEST(abs(tv1 - tv2) < tol*tv1);
}

template<class Real>
void test_sup_norm()
{
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{-2,1,0};
    Real s = boost::math::tools::sup_norm(v.begin(), v.end());
    BOOST_TEST(abs(s - 2) < tol);

    s = boost::math::tools::sup_norm(v);
    BOOST_TEST(abs(s - 2) < tol);

    // Work with std::array?
    std::array<Real, 3> w{-2,1,0};
    s = boost::math::tools::sup_norm(w);
    BOOST_TEST(abs(s - 2) < tol);

    v = generate_random_vector<Real>(global_size, global_seed);
    Real scale = 8;
    Real sup1 = scale*sup_norm(v);
    for (auto & x : v)
    {
        x *= -scale;
    }
    Real sup2 = sup_norm(v);
    BOOST_TEST(abs(sup1 - sup2) < tol*sup1);
}

template<class Z>
void test_integer_sup_norm()
{
    double eps = std::numeric_limits<double>::epsilon();
    std::vector<Z> v{2,1,0};
    Z s = sup_norm(v.begin(), v.end());
    BOOST_TEST_EQ(s, 2);

    s = sup_norm(v);
    BOOST_TEST_EQ(s,2);

    v = generate_random_vector<Z>(global_size, global_seed);
    double sup1 = 2*sup_norm(v);
    Z scale = 2;
    for (auto & x : v)
    {
        x *= scale;
    }
    double sup2 = sup_norm(v);
    BOOST_TEST(abs(sup1 - sup2) < sup1*eps);
}

template<class Complex>
void test_complex_sup_norm()
{
    typedef typename Complex::value_type Real;
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Complex> w{{0,-8}, {1,1}, {3,2}};
    Real s = sup_norm(w.cbegin(), w.cend());
    BOOST_TEST(abs(s-8) < tol);

    s = sup_norm(w);
    BOOST_TEST(abs(s-8) < tol);

    auto v = generate_random_vector<Complex>(global_size, global_seed);
    Real scale = 8;
    Real sup1 = scale*sup_norm(v);
    for (auto & x : v)
    {
        x *= -scale;
    }
    Real sup2 = sup_norm(v);
    BOOST_TEST(abs(sup1 - sup2) < tol*sup1);
}

template<class Real>
void test_l0_pseudo_norm()
{
    std::vector<Real> v{0,0,1};
    size_t count = boost::math::tools::l0_pseudo_norm(v.begin(), v.end());
    BOOST_TEST_EQ(count, 1);

    // Compiles with cbegin()/cend()?
    count = boost::math::tools::l0_pseudo_norm(v.cbegin(), v.cend());
    BOOST_TEST_EQ(count, 1);

    count = boost::math::tools::l0_pseudo_norm(v);
    BOOST_TEST_EQ(count, 1);

    std::array<Real, 3> w{0,0,1};
    count = boost::math::tools::l0_pseudo_norm(w);
    BOOST_TEST_EQ(count, 1);
}

template<class Complex>
void test_complex_l0_pseudo_norm()
{
    std::vector<Complex> v{{0,0}, {0,0}, {1,0}};
    size_t count = boost::math::tools::l0_pseudo_norm(v.begin(), v.end());
    BOOST_TEST_EQ(count, 1);

    count = boost::math::tools::l0_pseudo_norm(v);
    BOOST_TEST_EQ(count, 1);
}

template<class Z>
void test_hamming_distance()
{
    std::vector<Z> v{1,2,3};
    std::vector<Z> w{1,2,4};
    size_t count = boost::math::tools::hamming_distance(v, w);
    BOOST_TEST_EQ(count, 1);

    count = boost::math::tools::hamming_distance(v, v);
    BOOST_TEST_EQ(count, 0);
}

template<class Real>
void test_l1_norm()
{
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{1,1,1};
    Real l1 = l1_norm(v.begin(), v.end());
    BOOST_TEST(abs(l1 - 3) < tol);

    l1 = l1_norm(v);
    BOOST_TEST(abs(l1 - 3) < tol);

    std::array<Real, 3> w{1,1,1};
    l1 = l1_norm(w);
    BOOST_TEST(abs(l1 - 3) < tol);

    v = generate_random_vector<Real>(global_size, global_seed);
    Real scale = 8;
    Real l1_1 = scale*l1_norm(v);
    for (auto & x : v)
    {
        x *= -scale;
    }
    Real l1_2 = l1_norm(v);
    BOOST_TEST(abs(l1_1 - l1_2) < tol*l1_1);
}

template<class Z>
void test_integer_l1_norm()
{
    double eps = std::numeric_limits<double>::epsilon();
    std::vector<Z> v{1,1,1};
    Z l1 = boost::math::tools::l1_norm(v.begin(), v.end());
    BOOST_TEST_EQ(l1, 3);

    v = generate_random_vector<Z>(global_size, global_seed);
    double l1_1 = 2*l1_norm(v);
    Z scale = 2;
    for (auto & x : v)
    {
        x *= scale;
    }
    double l1_2 = l1_norm(v);
    BOOST_TEST(l1_1 > 0);
    BOOST_TEST(l1_2 > 0);
    if (abs(l1_1 - l1_2) > 2*l1_1*eps)
    {
        std::cout << std::setprecision(std::numeric_limits<double>::digits10);
        std::cout << "L1_1 = " << l1_1 << "\n";
        std::cout << "L1_2 = " << l1_2 << "\n";
        BOOST_TEST(abs(l1_1 - l1_2) < 2*l1_1*eps);
    }
}

template<class Complex>
void test_complex_l1_norm()
{
    typedef typename Complex::value_type Real;
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Complex> v{{1,0}, {0,1},{0,-1}};
    Real l1 = l1_norm(v.begin(), v.end());
    BOOST_TEST(abs(l1 - 3) < tol);

    l1 = l1_norm(v);
    BOOST_TEST(abs(l1 - 3) < tol);

    v = generate_random_vector<Complex>(global_size, global_seed);
    Real scale = 8;
    Real l1_1 = scale*l1_norm(v);
    for (auto & x : v)
    {
        x *= -scale;
    }
    Real l1_2 = l1_norm(v);
    BOOST_TEST(abs(l1_1 - l1_2) < tol*l1_1);
}

template<class Real>
void test_l1_distance()
{
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{1,2,3};
    std::vector<Real> w{1,1,1};
    Real l1 = boost::math::tools::l1_distance(v, v);
    BOOST_TEST(abs(l1) < tol);

    l1 = boost::math::tools::l1_distance(w, v);
    BOOST_TEST(abs(l1 - 3) < tol);

    l1 = boost::math::tools::l1_distance(v, w);
    BOOST_TEST(abs(l1 - 3) < tol);

    v = generate_random_vector<Real>(global_size, global_seed);
    w = generate_random_vector<Real>(global_size, global_seed+1);
    Real dist1 = l1_distance(v, w);
    Real dist2 = l1_distance(w, v);
    BOOST_TEST(abs(dist1 - dist2) < tol*dist1);
}

template<class Z>
void test_integer_l1_distance()
{
    double tol = std::numeric_limits<double>::epsilon();
    std::vector<Z> v{1,2,3};
    std::vector<Z> w{1,1,1};
    double l1 = boost::math::tools::l1_distance(v, v);
    BOOST_TEST(abs(l1) < tol);

    l1 = boost::math::tools::l1_distance(w, v);
    BOOST_TEST(abs(l1 - 3) < tol);

    l1 = boost::math::tools::l1_distance(v, w);
    BOOST_TEST(abs(l1 - 3) < tol);

    v = generate_random_vector<Z>(global_size, global_seed);
    w = generate_random_vector<Z>(global_size, global_seed + 1);
    double dist1 = l1_distance(v, w);
    double dist2 = l1_distance(w, v);
    BOOST_TEST(abs(dist1 - dist2) < tol*dist1);
}

template<class Complex>
void test_complex_l1_distance()
{
    typedef typename Complex::value_type Real;
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Complex> v{{1,0}, {0,1},{0,-1}};
    Real l1 = boost::math::tools::l1_distance(v, v);
    BOOST_TEST(abs(l1) < tol);

    std::vector<Complex> w{{2,0}, {0,1},{0,-1}};
    l1 = boost::math::tools::l1_distance(v.cbegin(), v.cend(), w.cbegin());
    BOOST_TEST(abs(l1 - 1) < tol);

    v = generate_random_vector<Complex>(global_size, global_seed);
    w = generate_random_vector<Complex>(global_size, global_seed + 1);
    Real dist1 = l1_distance(v, w);
    Real dist2 = l1_distance(w, v);
    BOOST_TEST(abs(dist1 - dist2) < tol*dist1);
}


template<class Real>
void test_l2_norm()
{
    using std::sqrt;
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{1,1,1,1};
    Real l2 = boost::math::tools::l2_norm(v.begin(), v.end());
    BOOST_TEST(abs(l2 - 2) < tol);

    l2 = boost::math::tools::l2_norm(v);
    BOOST_TEST(abs(l2 - 2) < tol);

    std::array<Real, 4> w{1,1,1,1};
    l2 = boost::math::tools::l2_norm(w);
    BOOST_TEST(abs(l2 - 2) < tol);

    Real bignum = 4*sqrt((std::numeric_limits<Real>::max)());
    v[0] = bignum;
    v[1] = 0;
    v[2] = 0;
    v[3] = 0;
    l2 = boost::math::tools::l2_norm(v.begin(), v.end());
    BOOST_TEST(abs(l2 - bignum) < tol*l2);

    v = generate_random_vector<Real>(global_size, global_seed);
    Real scale = 8;
    Real l2_1 = scale*l2_norm(v);
    for (auto & x : v)
    {
        x *= -scale;
    }
    Real l2_2 = l2_norm(v);
    BOOST_TEST(l2_1 > 0);
    BOOST_TEST(l2_2 > 0);
    BOOST_TEST(abs(l2_1 - l2_2) < tol*l2_1);
}

template<class Z>
void test_integer_l2_norm()
{
    double tol = 100*std::numeric_limits<double>::epsilon();
    std::vector<Z> v{1,1,1,1};
    double l2 = boost::math::tools::l2_norm(v.begin(), v.end());
    BOOST_TEST(abs(l2 - 2) < tol);

    v = generate_random_vector<Z>(global_size, global_seed);
    Z scale = 2;
    double l2_1 = scale*l2_norm(v);
    for (auto & x : v)
    {
        x *= scale;
    }
    double l2_2 = l2_norm(v);
    BOOST_TEST(l2_1 > 0);
    BOOST_TEST(l2_2 > 0);
    BOOST_TEST(abs(l2_1 - l2_2) < tol*l2_1);
}

template<class Complex>
void test_complex_l2_norm()
{
    typedef typename Complex::value_type Real;
    Real tol = 100*std::numeric_limits<Real>::epsilon();
    std::vector<Complex> v{{1,0}, {0,1},{0,-1}, {1,0}};
    Real l2 = boost::math::tools::l2_norm(v.begin(), v.end());
    BOOST_TEST(abs(l2 - 2) < tol);

    l2 = boost::math::tools::l2_norm(v);
    BOOST_TEST(abs(l2 - 2) < tol);

    v = generate_random_vector<Complex>(global_size, global_seed);
    Real scale = 8;
    Real l2_1 = scale*l2_norm(v);
    for (auto & x : v)
    {
        x *= -scale;
    }
    Real l2_2 = l2_norm(v);
    BOOST_TEST(abs(l2_1 - l2_2) < tol*l2_1);
}

template<class Real>
void test_l2_distance()
{
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{1,1,1,1};
    Real l2 = boost::math::tools::l2_distance(v, v);
    BOOST_TEST(abs(l2) < tol);

    v = generate_random_vector<Real>(global_size, global_seed);
    auto w = generate_random_vector<Real>(global_size, global_seed + 1);
    Real dist1 = l2_distance(v, w);
    Real dist2 = l2_distance(w, v);
    BOOST_TEST(abs(dist1 - dist2) < tol*dist1);
}


template<class Z>
void test_integer_l2_distance()
{
    double tol = std::numeric_limits<double>::epsilon();
    std::vector<Z> v{1,1,1,1};
    double l2 = boost::math::tools::l2_distance(v, v);
    BOOST_TEST(abs(l2) < tol);

    v = generate_random_vector<Z>(global_size, global_seed);
    auto w = generate_random_vector<Z>(global_size, global_seed + 1);
    double dist1 = l2_distance(v, w);
    double dist2 = l2_distance(w, v);
    BOOST_TEST(abs(dist1 - dist2) < tol*dist1);
}

template<class Complex>
void test_complex_l2_distance()
{
    typedef typename Complex::value_type Real;
    Real tol = 100*std::numeric_limits<Real>::epsilon();
    std::vector<Complex> v{{1,0}, {0,1},{0,-1}, {1,0}};
    Real l2 = boost::math::tools::l2_distance(v, v);
    BOOST_TEST(abs(l2) < tol);

    v = generate_random_vector<Complex>(global_size, global_seed);
    auto w = generate_random_vector<Complex>(global_size, global_seed + 1);
    Real dist1 = l2_distance(v, w);
    Real dist2 = l2_distance(w, v);
    BOOST_TEST(abs(dist1 - dist2) < tol*dist1);
}

template<class Real>
void test_sup_distance()
{
    Real tol = std::numeric_limits<Real>::epsilon();
    std::vector<Real> v{1,1,1,1};
    std::vector<Real> w{0,0,0,0};
    Real sup = boost::math::tools::sup_distance(v, v);
    BOOST_TEST(abs(sup) < tol);
    sup = boost::math::tools::sup_distance(v, w);
    BOOST_TEST(abs(sup -1) < tol);

    v = generate_random_vector<Real>(global_size, global_seed);
    w = generate_random_vector<Real>(global_size, global_seed + 1);
    Real dist1 = sup_distance(v, w);
    Real dist2 = sup_distance(w, v);
    BOOST_TEST(abs(dist1 - dist2) < tol*dist1);
}


template<class Z>
void test_integer_sup_distance()
{
    double tol = std::numeric_limits<double>::epsilon();
    std::vector<Z> v{1,1,1,1};
    std::vector<Z> w{0,0,0,0};
    double sup = boost::math::tools::sup_distance(v, v);
    BOOST_TEST(abs(sup) < tol);

    sup = boost::math::tools::sup_distance(v, w);
    BOOST_TEST(abs(sup -1) < tol);

    v = generate_random_vector<Z>(global_size, global_seed);
    w = generate_random_vector<Z>(global_size, global_seed + 1);
    double dist1 = sup_distance(v, w);
    double dist2 = sup_distance(w, v);
    BOOST_TEST(abs(dist1 - dist2) < tol*dist1);
}

template<class Complex>
void test_complex_sup_distance()
{
    typedef typename Complex::value_type Real;
    Real tol = 100*std::numeric_limits<Real>::epsilon();
    std::vector<Complex> v{{1,0}, {0,1},{0,-1}, {1,0}};
    Real sup = boost::math::tools::sup_distance(v, v);
    BOOST_TEST(abs(sup) < tol);

    v = generate_random_vector<Complex>(global_size, global_seed);
    auto w = generate_random_vector<Complex>(global_size, global_seed + 1);
    Real dist1 = sup_distance(v, w);
    Real dist2 = sup_distance(w, v);
    BOOST_TEST(abs(dist1 - dist2) < tol*dist1);
}

int main()
{
    test_l0_pseudo_norm<unsigned>();
    test_l0_pseudo_norm<int>();
    test_l0_pseudo_norm<float>();
    test_l0_pseudo_norm<double>();
    test_l0_pseudo_norm<long double>();
    test_l0_pseudo_norm<cpp_bin_float_50>();

    test_complex_l0_pseudo_norm<std::complex<float>>();
    test_complex_l0_pseudo_norm<std::complex<double>>();
    test_complex_l0_pseudo_norm<std::complex<long double>>();
    test_complex_l0_pseudo_norm<cpp_complex_50>();

    test_hamming_distance<int>();
    test_hamming_distance<unsigned>();

    test_l1_norm<float>();
    test_l1_norm<double>();
    test_l1_norm<long double>();
    test_l1_norm<cpp_bin_float_50>();

    test_integer_l1_norm<int>();
    test_integer_l1_norm<unsigned>();

    test_complex_l1_norm<std::complex<float>>();
    test_complex_l1_norm<std::complex<double>>();
    test_complex_l1_norm<std::complex<long double>>();
    test_complex_l1_norm<cpp_complex_50>();

    test_l1_distance<float>();
    test_l1_distance<cpp_bin_float_50>();

    test_integer_l1_distance<int>();
    test_integer_l1_distance<unsigned>();

    test_complex_l1_distance<std::complex<float>>();
    test_complex_l1_distance<cpp_complex_50>();

    test_complex_l2_norm<std::complex<float>>();
    test_complex_l2_norm<std::complex<double>>();
    test_complex_l2_norm<std::complex<long double>>();
    test_complex_l2_norm<cpp_complex_50>();

    test_l2_norm<float>();
    test_l2_norm<double>();
    test_l2_norm<long double>();
    test_l2_norm<cpp_bin_float_50>();

    test_integer_l2_norm<int>();
    test_integer_l2_norm<unsigned>();

    test_l2_distance<double>();
    test_l2_distance<cpp_bin_float_50>();

    test_integer_l2_distance<int>();
    test_integer_l2_distance<unsigned>();

    test_complex_l2_distance<std::complex<double>>();
    test_complex_l2_distance<cpp_complex_50>();

    test_lp<float>();
    test_lp<double>();
    test_lp<long double>();
    test_lp<cpp_bin_float_50>();

    test_complex_lp<std::complex<float>>();
    test_complex_lp<std::complex<double>>();
    test_complex_lp<std::complex<long double>>();
    test_complex_lp<cpp_complex_50>();

    test_integer_lp<int>();
    test_integer_lp<unsigned>();

    test_lp_distance<double>();
    test_lp_distance<cpp_bin_float_50>();

    test_complex_lp_distance<std::complex<double>>();
    test_complex_lp_distance<cpp_complex_50>();

    test_integer_lp_distance<int>();
    test_integer_lp_distance<unsigned>();

    test_sup_norm<float>();
    test_sup_norm<double>();
    test_sup_norm<long double>();
    test_sup_norm<cpp_bin_float_50>();

    test_integer_sup_norm<int>();
    test_integer_sup_norm<unsigned>();

    test_complex_sup_norm<std::complex<float>>();
    test_complex_sup_norm<std::complex<double>>();
    test_complex_sup_norm<std::complex<long double>>();
    test_complex_sup_norm<cpp_complex_50>();

    test_sup_distance<double>();
    test_sup_distance<cpp_bin_float_50>();

    test_integer_sup_distance<int>();
    test_integer_sup_distance<unsigned>();

    test_complex_sup_distance<std::complex<double>>();
    test_complex_sup_distance<cpp_complex_50>();

    test_total_variation<float>();
    test_total_variation<double>();
    test_total_variation<long double>();
    test_total_variation<cpp_bin_float_50>();

    test_integer_total_variation<uint32_t>();
    test_integer_total_variation<int>();

    return boost::report_errors();
}
