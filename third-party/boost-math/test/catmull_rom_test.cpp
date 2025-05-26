/*
 * Copyright Nick Thompson, 2017
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#define BOOST_TEST_MODULE catmull_rom_test

#include <array>
#include <random>
#include <boost/cstdfloat.hpp>
#include <boost/type_index.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/interpolators/catmull_rom.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/numeric/ublas/vector.hpp>

#if __has_include(<stdfloat>)
#  include <stdfloat>
#endif

using std::abs;
using boost::multiprecision::cpp_bin_float_50;
using boost::math::catmull_rom;

template<class Real>
void test_alpha_distance()
{
    Real tol = std::numeric_limits<Real>::epsilon();
    std::array<Real, 3> v1 = {0,0,0};
    std::array<Real, 3> v2 = {1,0,0};
    Real alpha = 0.5;
    Real d = boost::math::detail::alpha_distance<std::array<Real, 3>>(v1, v2, alpha);
    BOOST_CHECK_CLOSE_FRACTION(d, 1, tol);

    d = boost::math::detail::alpha_distance<std::array<Real, 3>>(v1, v2, 0.0);
    BOOST_CHECK_CLOSE_FRACTION(d, 1, tol);

    d = boost::math::detail::alpha_distance<std::array<Real, 3>>(v1, v2, 1.0);
    BOOST_CHECK_CLOSE_FRACTION(d, 1, tol);

    v2[0] = 2;
    d = boost::math::detail::alpha_distance<std::array<Real, 3>>(v1, v2, alpha);
    BOOST_CHECK_CLOSE_FRACTION(d, pow(2, (Real)1/ (Real) 2), tol);

    d = boost::math::detail::alpha_distance<std::array<Real, 3>>(v1, v2, 0.0);
    BOOST_CHECK_CLOSE_FRACTION(d, 1, tol);

    d = boost::math::detail::alpha_distance<std::array<Real, 3>>(v1, v2, 1.0);
    BOOST_CHECK_CLOSE_FRACTION(d, 2, tol);
}


template<class Real>
void test_linear()
{
    std::cout << "Testing that the Catmull-Rom spline interpolates linear functions correctly on type "
              << boost::typeindex::type_id<Real>().pretty_name() << "\n";

    Real tol = 10*std::numeric_limits<Real>::epsilon();
    std::vector<std::array<Real, 3>> v(4);
    v[0] = {0,0,0};
    v[1] = {1,0,0};
    v[2] = {2,0,0};
    v[3] = {3,0,0};
    catmull_rom<std::array<Real, 3>> cr(std::move(v));

    // Test that the interpolation condition is obeyed:
    BOOST_CHECK_CLOSE_FRACTION(cr.max_parameter(), 3, tol);
    auto p0 = cr(0.0);
    BOOST_CHECK_SMALL(p0[0], tol);
    BOOST_CHECK_SMALL(p0[1], tol);
    BOOST_CHECK_SMALL(p0[2], tol);
    auto p1 = cr(1.0);
    BOOST_CHECK_CLOSE_FRACTION(p1[0], 1, tol);
    BOOST_CHECK_SMALL(p1[1], tol);
    BOOST_CHECK_SMALL(p1[2], tol);

    auto p2 = cr(2.0);
    BOOST_CHECK_CLOSE_FRACTION(p2[0], 2, tol);
    BOOST_CHECK_SMALL(p2[1], tol);
    BOOST_CHECK_SMALL(p2[2], tol);


    auto p3 = cr(3.0);
    BOOST_CHECK_CLOSE_FRACTION(p3[0], 3, tol);
    BOOST_CHECK_SMALL(p3[1], tol);
    BOOST_CHECK_SMALL(p3[2], tol);

    Real s = cr.parameter_at_point(0);
    BOOST_CHECK_SMALL(s, tol);

    s = cr.parameter_at_point(1);
    BOOST_CHECK_CLOSE_FRACTION(s, 1, tol);

    s = cr.parameter_at_point(2);
    BOOST_CHECK_CLOSE_FRACTION(s, 2, tol);

    s = cr.parameter_at_point(3);
    BOOST_CHECK_CLOSE_FRACTION(s, 3, tol);

    // Test that the function is linear on the interval [1,2]:
    for (double s = 1; s < 2; s += 0.01)
    {
        auto p = cr(s);
        BOOST_CHECK_CLOSE_FRACTION(p[0], s, tol);
        BOOST_CHECK_SMALL(p[1], tol);
        BOOST_CHECK_SMALL(p[2], tol);

        auto tangent = cr.prime(s);
        BOOST_CHECK_CLOSE_FRACTION(tangent[0], 1, tol);
        BOOST_CHECK_SMALL(tangent[1], tol);
        BOOST_CHECK_SMALL(tangent[2], tol);
    }

}

template<class Real>
void test_circle()
{
    using boost::math::constants::pi;
    using std::cos;
    using std::sin;

    std::cout << "Testing that the Catmull-Rom spline interpolates circles correctly on type "
              << boost::typeindex::type_id<Real>().pretty_name() << "\n";

    Real tol = 10*std::numeric_limits<Real>::epsilon();
    std::vector<std::array<Real, 2>> v(20*sizeof(Real));
    std::vector<std::array<Real, 2>> u(20*sizeof(Real));
    for (size_t i = 0; i < v.size(); ++i)
    {
        Real theta = ((Real) i/ (Real) v.size())*2*pi<Real>();
        v[i] = {cos(theta), sin(theta)};
        u[i] = v[i];
    }
    catmull_rom<std::array<Real, 2>> circle(std::move(v), true);

    // Interpolation condition:
    for (size_t i = 0; i < v.size(); ++i)
    {
        Real s = circle.parameter_at_point(i);
        auto p = circle(s);
        Real x = p[0];
        Real y = p[1];
        if (abs(x) < std::numeric_limits<Real>::epsilon())
        {
            BOOST_CHECK_SMALL(u[i][0], tol);
        }
        if (abs(y) < std::numeric_limits<Real>::epsilon())
        {
            BOOST_CHECK_SMALL(u[i][1], tol);
        }
        else
        {
            BOOST_CHECK_CLOSE_FRACTION(x, u[i][0], tol);
            BOOST_CHECK_CLOSE_FRACTION(y, u[i][1], tol);
        }
    }

    Real max_s = circle.max_parameter();
    for(Real s = 0; s < max_s; s += Real(0.01))
    {
        auto p = circle(s);
        Real x = p[0];
        Real y = p[1];
        BOOST_CHECK_CLOSE_FRACTION(x*x+y*y, 1, 0.001);
    }
}


template<class Real, size_t dimension>
void test_affine_invariance()
{
    std::cout << "Testing that the Catmull-Rom spline is affine invariant in dimension "
              << dimension << " on type "
              << boost::typeindex::type_id<Real>().pretty_name() << "\n";

    Real tol = 1000*std::numeric_limits<Real>::epsilon();
    std::vector<std::array<Real, dimension>> v(100);
    std::vector<std::array<Real, dimension>> u(100);
    std::mt19937_64 gen(438232);
    Real inv_denom = (Real) 100/( (Real) (gen.max)() + (Real) 2);
    for(size_t j = 0; j < dimension; ++j)
    {
        v[0][j] = gen()*inv_denom;
        u[0][j] = v[0][j];
    }

    for (size_t i = 1; i < v.size(); ++i)
    {
        for(size_t j = 0; j < dimension; ++j)
        {
            v[i][j] = v[i-1][j] + gen()*inv_denom;
            u[i][j] = v[i][j];
        }
    }
    std::array<Real, dimension> affine_shift;
    for (size_t j = 0; j < dimension; ++j)
    {
        affine_shift[j] = gen()*inv_denom;
    }

    catmull_rom<std::array<Real, dimension>> cr1(std::move(v));

    for(size_t i = 0; i< u.size(); ++i)
    {
        for(size_t j = 0; j < dimension; ++j)
        {
            u[i][j] += affine_shift[j];
        }
    }

    catmull_rom<std::array<Real, dimension>> cr2(std::move(u));

    BOOST_CHECK_CLOSE_FRACTION(cr1.max_parameter(), cr2.max_parameter(), tol);

    Real ds = cr1.max_parameter()/1024;
    for (Real s = 0; s < cr1.max_parameter(); s += ds)
    {
        auto p0 = cr1(s);
        auto p1 = cr2(s);
        auto tangent0 = cr1.prime(s);
        auto tangent1 = cr2.prime(s);
        for (size_t j = 0; j < dimension; ++j)
        {
            BOOST_CHECK_CLOSE_FRACTION(p0[j] + affine_shift[j], p1[j], tol);
            if (abs(tangent0[j]) > 5000*tol)
            {
                BOOST_CHECK_CLOSE_FRACTION(tangent0[j], tangent1[j], 5000*tol);
            }
        }
    }
}

template<class Real>
void test_helix()
{
    using boost::math::constants::pi;
    std::cout << "Testing that the Catmull-Rom spline interpolates helices correctly on type "
              << boost::typeindex::type_id<Real>().pretty_name() << "\n";

    Real tol = 0.001;
    std::vector<std::array<Real, 3>> v(400*sizeof(Real));
    for (size_t i = 0; i < v.size(); ++i)
    {
        Real theta = ((Real) i/ (Real) v.size())*2*pi<Real>();
        v[i] = {cos(theta), sin(theta), theta};
    }
    catmull_rom<std::array<Real, 3>> helix(std::move(v));

    // Interpolation condition:
    for (size_t i = 0; i < v.size(); ++i)
    {
        Real s = helix.parameter_at_point(i);
        auto p = helix(s);
        Real t = p[2];

        Real x = p[0];
        Real y = p[1];
        if (abs(x) < tol)
        {
            BOOST_CHECK_SMALL(Real(cos(t)), tol);
        }
        if (abs(y) < tol)
        {
            BOOST_CHECK_SMALL(Real(sin(t)), tol);
        }
        else
        {
            BOOST_CHECK_CLOSE_FRACTION(x, cos(t), tol);
            BOOST_CHECK_CLOSE_FRACTION(y, sin(t), tol);
        }
    }

    Real max_s = helix.max_parameter();
    for(Real s = helix.parameter_at_point(1); s < max_s; s += 0.01)
    {
        auto p = helix(s);
        Real x = p[0];
        Real y = p[1];
        Real t = p[2];
        BOOST_CHECK_CLOSE_FRACTION(x*x+y*y, (Real) 1, (Real) 0.01);
        if (abs(x) < 0.01)
        {
            BOOST_CHECK_SMALL(Real(cos(t)),  (Real) 0.05);
        }
        if (abs(y) < 0.01)
        {
            BOOST_CHECK_SMALL(Real(sin(t)), (Real) 0.05);
        }
        else
        {
            BOOST_CHECK_CLOSE_FRACTION(x, Real(cos(t)), (Real) 0.05);
            BOOST_CHECK_CLOSE_FRACTION(y, Real(sin(t)), (Real) 0.05);
        }
    }
}


template<class Real>
class mypoint3d
{
public:
    // Must define a value_type:
    typedef Real value_type;

    // Regular constructor:
    mypoint3d(Real x, Real y, Real z)
    {
        m_vec[0] = x;
        m_vec[1] = y;
        m_vec[2] = z;
    }

    // Must define a default constructor:
    mypoint3d() {}

    // Must define array access:
    Real operator[](size_t i) const
    {
        return m_vec[i];
    }

    // Array element assignment:
    Real& operator[](size_t i)
    {
        return m_vec[i];
    }


private:
    std::array<Real, 3>  m_vec;
};


// Must define the free function "size()":
template<class Real>
constexpr std::size_t size(const mypoint3d<Real>&)
{
    return 3;
}

template<class Real>
void test_data_representations()
{
    std::cout << "Testing that the Catmull-Rom spline works with multiple data representations.\n";
    mypoint3d<Real> p0(Real(0.1), Real(0.2), Real(0.3));
    mypoint3d<Real> p1(Real(0.2), Real(0.3), Real(0.4));
    mypoint3d<Real> p2(Real(0.3), Real(0.4), Real(0.5));
    mypoint3d<Real> p3(Real(0.4), Real(0.5), Real(0.6));
    mypoint3d<Real> p4(Real(0.5), Real(0.6), Real(0.7));
    mypoint3d<Real> p5(Real(0.6), Real(0.7), Real(0.8));


    // Tests initializer_list:
    catmull_rom<mypoint3d<Real>> cat({p0, p1, p2, p3, p4, p5});

    Real tol = Real(0.001);
    auto p = cat(cat.parameter_at_point(0));
    BOOST_CHECK_CLOSE_FRACTION(p[0], p0[0], tol);
    BOOST_CHECK_CLOSE_FRACTION(p[1], p0[1], tol);
    BOOST_CHECK_CLOSE_FRACTION(p[2], p0[2], tol);
    p = cat(cat.parameter_at_point(1));
    BOOST_CHECK_CLOSE_FRACTION(p[0], p1[0], tol);
    BOOST_CHECK_CLOSE_FRACTION(p[1], p1[1], tol);
    BOOST_CHECK_CLOSE_FRACTION(p[2], p1[2], tol);
}

template<class Real>
void test_random_access_container()
{
    std::cout << "Testing that the Catmull-Rom spline works with multiple data representations.\n";
    mypoint3d<Real> p0(0.1, 0.2, 0.3);
    mypoint3d<Real> p1(0.2, 0.3, 0.4);
    mypoint3d<Real> p2(0.3, 0.4, 0.5);
    mypoint3d<Real> p3(0.4, 0.5, 0.6);
    mypoint3d<Real> p4(0.5, 0.6, 0.7);
    mypoint3d<Real> p5(0.6, 0.7, 0.8);

    boost::numeric::ublas::vector<mypoint3d<Real>> u(6);
    u[0] = p0;
    u[1] = p1;
    u[2] = p2;
    u[3] = p3;
    u[4] = p4;
    u[5] = p5;

    // Tests initializer_list:
    catmull_rom<mypoint3d<Real>, decltype(u)> cat(std::move(u));

    Real tol = Real(0.001);
    auto p = cat(cat.parameter_at_point(0));
    BOOST_CHECK_CLOSE_FRACTION(p[0], p0[0], tol);
    BOOST_CHECK_CLOSE_FRACTION(p[1], p0[1], tol);
    BOOST_CHECK_CLOSE_FRACTION(p[2], p0[2], tol);
    p = cat(cat.parameter_at_point(1));
    BOOST_CHECK_CLOSE_FRACTION(p[0], p1[0], tol);
    BOOST_CHECK_CLOSE_FRACTION(p[1], p1[1], tol);
    BOOST_CHECK_CLOSE_FRACTION(p[2], p1[2], tol);
}

BOOST_AUTO_TEST_CASE(catmull_rom_test)
{
#if !defined(TEST) || (TEST == 1)

    #ifdef __STDCPP_FLOAT32_T__
    test_circle<std::float32_t>();
    test_data_representations<std::float32_t>();
    #else
    test_circle<float>();
    test_data_representations<float>();
    #endif

    #ifdef __STDCPP_FLOAT64_T__
    test_alpha_distance<std::float64_t>();
    test_linear<std::float64_t>();
    test_circle<std::float64_t>();
    #else
    test_alpha_distance<double>();
    test_linear<double>();
    test_circle<double>();
    #endif
    
    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_linear<long double>();
    #endif

#endif

#if !defined(TEST) || (TEST == 2)

    #ifdef __STDCPP_FLOAT64_T__
    test_helix<std::float64_t>();
    test_affine_invariance<std::float64_t, 1>();
    test_affine_invariance<std::float64_t, 2>();
    test_affine_invariance<std::float64_t, 3>();
    test_affine_invariance<std::float64_t, 4>();
    test_random_access_container<std::float64_t>();
    #else
    test_helix<double>();
    test_affine_invariance<double, 1>();
    test_affine_invariance<double, 2>();
    test_affine_invariance<double, 3>();
    test_affine_invariance<double, 4>();
    test_random_access_container<double>();
    #endif

#endif

#if !defined(TEST) || (TEST == 3)
    test_affine_invariance<cpp_bin_float_50, 4>();
#endif
}
