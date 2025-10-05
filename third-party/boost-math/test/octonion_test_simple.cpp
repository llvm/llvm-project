// Copyright Hubert Holin 2001.
// Copyright Christopher Kormanyos 2024
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <boost/mpl/list.hpp>
#include <boost/math/octonion.hpp>
#include <boost/core/lightweight_test.hpp>

#include <cstdint>
#include <random>

// test file for octonion.hpp

namespace local
{
  std::mt19937 eng(static_cast<typename std::mt19937::result_type>(UINT8_C(42)));
  std::uniform_int_distribution<int> dst_one(1, 1);

  template<typename NumericType>
  auto is_close_fraction(const NumericType& a,
                         const NumericType& b,
                         const NumericType& tol) noexcept -> bool
  {
    using std::fabs;

    auto result_is_ok = bool { };

    if(b == static_cast<NumericType>(0))
    {
      result_is_ok = (fabs(a - b) < tol);
    }
    else
    {
      const auto delta = fabs(1 - (a / b));

      result_is_ok = (delta < tol);
    }

    return result_is_ok;
  }

  template<typename T>
  auto is_close_fraction(const ::boost::math::octonion<T>& a,
                         const ::boost::math::octonion<T>& b,
                         const T& tol) noexcept -> bool
  {
    using std::fabs;

    bool result_is_ok { true };

    result_is_ok = (is_close_fraction(a.R_component_1(), b.R_component_1(), tol) && result_is_ok);
    result_is_ok = (is_close_fraction(a.R_component_2(), b.R_component_2(), tol) && result_is_ok);
    result_is_ok = (is_close_fraction(a.R_component_3(), b.R_component_3(), tol) && result_is_ok);
    result_is_ok = (is_close_fraction(a.R_component_4(), b.R_component_4(), tol) && result_is_ok);
    result_is_ok = (is_close_fraction(a.R_component_5(), b.R_component_5(), tol) && result_is_ok);
    result_is_ok = (is_close_fraction(a.R_component_6(), b.R_component_6(), tol) && result_is_ok);
    result_is_ok = (is_close_fraction(a.R_component_7(), b.R_component_7(), tol) && result_is_ok);
    result_is_ok = (is_close_fraction(a.R_component_8(), b.R_component_8(), tol) && result_is_ok);

    return result_is_ok;
  }

  template<typename T>
  ::boost::math::octonion<T> index_i_element(int idx)
  {
      return(
          ::boost::math::octonion<T>(
                      (idx == 0) ?
                          static_cast<T>(1) :
                          static_cast<T>(0),
                      (idx == 1) ?
                          static_cast<T>(1) :
                          static_cast<T>(0),
                      (idx == 2) ?
                          static_cast<T>(1) :
                          static_cast<T>(0),
                      (idx == 3) ?
                          static_cast<T>(1) :
                          static_cast<T>(0),
                      (idx == 4) ?
                          static_cast<T>(1) :
                          static_cast<T>(0),
                      (idx == 5) ?
                          static_cast<T>(1) :
                          static_cast<T>(0),
                      (idx == 6) ?
                          static_cast<T>(1) :
                          static_cast<T>(0),
                      (idx == 7) ?
                          static_cast<T>(1) :
                          static_cast<T>(0)
          ));
  }
}

template<class T>
void multiplication_test()
{
    using ::std::numeric_limits;

    using ::boost::math::abs;

    // Testing multiplication.

    const auto one_by_one = ::boost::math::octonion<T>(1,0,0,0,0,0,0,0) * ::boost::math::octonion<T>(1,0,0,0,0,0,0,0);

    const T delta { abs(one_by_one - static_cast<T>(1)) };

    const auto result_mul_one_is_ok = (delta < numeric_limits<T>::epsilon());

    BOOST_TEST(result_mul_one_is_ok);

    for    (int idx = 1; idx < 8; ++idx)
    {
        ::boost::math::octonion<T> toto = local::index_i_element<T>(idx);

        const T tabs { abs(toto*toto+static_cast<T>(1)) };

        const auto result_mul_toto_is_ok = (tabs < numeric_limits<T>::epsilon());

        BOOST_TEST(result_mul_toto_is_ok);
    }

    {
        const boost::math::octonion<T> lhs(T(1),T(2),T(3),T(4),T(5),T(6),T(7),T(8));
        const boost::math::octonion<T> rhs(T(8),T(7),T(6),T(5),T(4),T(3),T(2),T(1));

        const boost::math::octonion<T> prod = lhs * rhs;

        const boost::math::octonion<T> ctrl(T(-104), T(14), T(12), T(10), T(152), T(42), T(4), T(74));

        BOOST_TEST(prod == ctrl);
    }

    for(auto i = static_cast<unsigned>(UINT8_C(0)); i < static_cast<unsigned>(UINT8_C(16)); ++i)
    {
        const boost::math::octonion<T> lhs(T(1),T(2),T(3),T(4),T(5),T(6),T(7),T(8));

        const boost::math::octonion<T> rhs =
              boost::math::octonion<T> { T(1),T(1),T(1),T(1),T(1),T(1),T(1),T(1) }
            * static_cast<T>(local::dst_one(local::eng));

        const boost::math::octonion<T> quot = lhs / rhs;

        const boost::math::octonion<T> ctrl(T(4.5), T(0.25), T(0.5), T(0.75), T(-1), T(0.75), T(1.5), T(0.75));

        BOOST_TEST(quot == ctrl);
    }
}

template<class T>
void division_test()
{
    {
        const boost::math::octonion<T> lhs(T(1),T(2),T(3),T(4),T(5),T(6),T(7),T(8));
        const boost::math::octonion<T> rhs(T(1),T(1),T(1),T(1),T(1),T(1),T(1),T(1));

        const boost::math::octonion<T> quot = lhs / rhs;

        const boost::math::octonion<T> ctrl(T(4.5), T(0.25), T(0.5), T(0.75), T(-1), T(0.75), T(1.5), T(0.75));

        BOOST_TEST(quot == ctrl);
    }

    {
        const std::complex<T> one_one(T(1), T(1));

        const boost::math::octonion<T> lhs(T(1),T(2),T(3),T(4),T(5),T(6),T(7),T(8));
        const boost::math::octonion<T> rhs(one_one, one_one, one_one, one_one);

        const boost::math::octonion<T> quot = lhs / rhs;

        const boost::math::octonion<T> ctrl(T(4.5), T(0.25), T(0.5), T(0.75), T(-1), T(0.75), T(1.5), T(0.75));

        BOOST_TEST(quot == ctrl);
    }
}

void octonion_original_manual_test()
{
    // tests for evaluation by humans

    // using default constructor
    ::boost::math::octonion<float>            o0;

    ::boost::math::octonion<float>            oa[2];

    // using constructor "O seen as R^8"
    ::boost::math::octonion<float>            o1(1,2,3,4,5,6,7,8);

    ::std::complex<double>                    c0(9,10);

    // using constructor "O seen as C^4"
    ::boost::math::octonion<double>            o2(c0);

    ::boost::math::quaternion<long double>    q0(11,12,13,14);

    // using constructor "O seen as H^2"
    ::boost::math::octonion<long double>      o3(q0);

    // using UNtemplated copy constructor
    ::boost::math::octonion<float>            o4(o1);

    // using templated copy constructor
    ::boost::math::octonion<long double>      o5(o2);

    // using UNtemplated assignment operator
    o5 = o3;
    oa[0] = o0;

    // using templated assignment operator
    o5 = o2;
    oa[1] = o5;

    float                                     f0(15);

    // using converting assignment operator
    o0 = f0;

    // using converting assignment operator
    o2 = c0;

    // using converting assignment operator
    o5 = q0;

    // using += (const T &)
    o4 += f0;

    // using == (const octonion<T> &,const octonion<T> &)
    BOOST_TEST(o0 != o4);

    // using += (const ::std::complex<T> &)
    o2 += c0;

    // using == (const ::boost::math::quaternion<T> &, const octonion<T> &)
    BOOST_TEST(q0 == o3);

    // using += (const ::boost::math::quaternion<T> &)
    o3 += q0;

    BOOST_TEST(2 * q0 == o3);

    // using += (const quaternion<X> &)
    o5 += o4;

    // using -= (const T &)
    o1 -= f0;

    // using -= (const ::std::complex<T> &)
    o2 -= c0;

    // using -= (const ::boost::math::quaternion<T> &)
    o5 -= q0;

    // using -= (const octonion<X> &)
    o3 -= o4;

    // using == (const ::std::complex<T> &, const octonion<T> &)
    BOOST_TEST(c0 == o2);

    // using == (const octonion<T> &, const ::std::complex<T> &)
    BOOST_TEST(o2 == c0);

    double                                    d0(16);
    ::std::complex<double>                    c1(17,18);
    ::boost::math::quaternion<double>         q1(19,20,21,22);

    // using *= (const T &)
    o2 *= d0;

    // using *= (const ::std::complex<T> &)
    o2 *= c1;

    // using *= (const ::boost::math::quaternion<T> &)
    o2 *= q1;

    // using *= (const octonion<X> &)
    o2 *= o4;

    long double                               l0(23);
    ::std::complex<long double>               c2(24,25);

    // using /= (const T &)
    o5 /= l0;

    // using /= (const ::std::complex<T> &)
    o5 /= c2;

    // using /= (const quaternion<X> &)
    o5 /= q0;

    // using /= (const octonion<X> &)
    o5 /= o5;

    // using + (const T &, const octonion<T> &)
    ::boost::math::octonion<float>            o6 = f0+o0;

    // using + (const octonion<T> &, const T &)
    ::boost::math::octonion<float>            o7 = o0+f0;

    // using + (const ::std::complex<T> &, const quaternion<T> &)
    ::boost::math::octonion<double>           o8 = c0+o2;

    // using + (const octonion<T> &, const ::std::complex<T> &)
    ::boost::math::octonion<double>           o9 = o2+c0;

    // using + (const ::boost::math::quaternion<T>, const octonion<T> &)
    ::boost::math::octonion<long double>      o10 = q0+o3;

    // using + (const octonion<T> &, const ::boost::math::quaternion<T> &)
    ::boost::math::octonion<long double>      o11 = o3+q0;

    // using + (const quaternion<T> &,const quaternion<T> &)
    ::boost::math::octonion<float>            o12 = o0+o4;

    // using - (const T &, const octonion<T> &)
    o6 = f0-o0;

    // using - (const octonion<T> &, const T &)
    o7 = o0-f0;

    // using - (const ::std::complex<T> &, const octonion<T> &)
    o8 = c0-o2;

    // using - (const octonion<T> &, const ::std::complex<T> &)
    o9 = o2-c0;

    // using - (const quaternion<T> &,const octonion<T> &)
    o10 = q0-o3;

    // using - (const octonion<T> &,const quaternion<T> &)
    o11 = o3-q0;

    // using - (const octonion<T> &,const octonion<T> &)
    o12 = o0-o4;

    // using * (const T &, const octonion<T> &)
    o6 = f0*o0;

    // using * (const octonion<T> &, const T &)
    o7 = o0*f0;

    // using * (const ::std::complex<T> &, const octonion<T> &)
    o8 = c0*o2;

    // using * (const octonion<T> &, const ::std::complex<T> &)
    o9 = o2*c0;

    // using * (const quaternion<T> &,const octonion<T> &)
    o10 = q0*o3;

    // using * (const octonion<T> &,const quaternion<T> &)
    o11 = o3*q0;

    // using * (const octonion<T> &,const octonion<T> &)
    o12 = o0*o4;

    // using / (const T &, const octonion<T> &)
    o6 = f0/o0;

    // using / (const octonion<T> &, const T &)
    o7 = o0/f0;

    // using / (const ::std::complex<T> &, const octonion<T> &)
    o8 = c0/o2;

    // using / (const octonion<T> &, const ::std::complex<T> &)
    o9 = o2/c0;

    // using / (const ::boost::math::quaternion<T> &, const octonion<T> &)
    o10 = q0/o3;

    // using / (const octonion<T> &, const ::boost::math::quaternion<T> &)
    o11 = o3/q0;

    // using / (const octonion<T> &,const octonion<T> &)
    o12 = o0/o4;

    // using + (const octonion<T> &)
    o4 = +o0;

    // using == (const T &, const octonion<T> &)
    BOOST_TEST(f0 == o0);

    // using == (const octonion<T> &, const T &)
    BOOST_TEST(o0 == f0);

    // using - (const octonion<T> &)
    o0 = -o4;

    // using != (const T &, const octonion<T> &)
    BOOST_TEST(f0 != o0);

    // using != (const octonion<T> &, const T &)
    BOOST_TEST(o0 != f0);

    // using != (const ::std::complex<T> &, const octonion<T> &)
    BOOST_TEST(c0 != o2);

    // using != (const octonion<T> &, const ::std::complex<T> &)
    BOOST_TEST(o2 != c0);

    // using != (const ::boost::math::quaternion<T> &, const octonion<T> &)
    BOOST_TEST(q0 != o3);

    // using != (const octonion<T> &, const ::boost::math::quaternion<T> &)
    BOOST_TEST(o3 != q0);

    // using != (const octonion<T> &,const octonion<T> &)
    BOOST_TEST(o0 != o4);
}

template <class T>
void elem_func_test()
{
    using ::std::numeric_limits;

    using ::std::atan;

    using ::boost::math::abs;

    // Testing exp.

    for(int idx = 1; idx < 8; ++idx)
    {
        ::boost::math::octonion<T> toto =
            static_cast<T>(4)*atan(static_cast<T>(1)) * local::index_i_element<T>(idx);

        const T tabs { abs(exp(toto)+static_cast<T>(1)) };

        const auto result_exp_toto_is_ok = (tabs < 2*numeric_limits<T>::epsilon());

        BOOST_TEST(result_exp_toto_is_ok);
    }

    {
        const std::complex<T> one_sixteenth_one_over_32 { T { 1 } / 16, T { 1 } / 32 };
        const ::boost::math::octonion<T>
            octo_small
            {
                          one_sixteenth_one_over_32,
                T { 2 } * one_sixteenth_one_over_32,
                T { 3 } * one_sixteenth_one_over_32,
                T { 4 } * one_sixteenth_one_over_32
            };

        const auto octo_small_exp = exp(octo_small);

        const auto r0 = octo_small_exp.real();

        using std::exp;

        BOOST_TEST(r0 < exp(one_sixteenth_one_over_32.real()));

        {
            auto octo_small_exp_inv = ::boost::math::octonion<T>(1);
            octo_small_exp_inv /= octo_small_exp;

            const auto value_cosh      = cosh(octo_small);
            const auto value_cosh_ctrl = (octo_small_exp + octo_small_exp_inv) / T(2);

            const auto result_cosh_is_ok = local::is_close_fraction(value_cosh, value_cosh_ctrl, std::numeric_limits<T>::epsilon() * 64);

            BOOST_TEST(result_cosh_is_ok);
        }

        {
            auto octo_small_exp_inv = ::boost::math::octonion<T>(1);
            octo_small_exp_inv /= octo_small_exp;

            const auto value_sinh      = sinh(octo_small);
            const auto value_sinh_ctrl = (octo_small_exp - octo_small_exp_inv) / T(2);

            const auto result_sinh_is_ok = local::is_close_fraction(value_sinh, value_sinh_ctrl, std::numeric_limits<T>::epsilon() * 64);

            BOOST_TEST(result_sinh_is_ok);
        }

        {
            const auto value_sinh      = sinh(octo_small);
            const auto value_cosh      = cosh(octo_small);
            const auto value_tanh      = tanh(octo_small);
            const auto value_tanh_ctrl = value_sinh / value_cosh;

            const auto result_tanh_is_ok = local::is_close_fraction(value_tanh, value_tanh_ctrl, std::numeric_limits<T>::epsilon() * 64);

            BOOST_TEST(result_tanh_is_ok);
        }
    }

    {
        const std::complex<T> one_sixteenth_one_over_32 { T { 1 } / 16, T { 1 } / 32 };
        const ::boost::math::octonion<T>
            octo_small
            {
                          one_sixteenth_one_over_32,
                T { 2 } * one_sixteenth_one_over_32,
                T { 3 } * one_sixteenth_one_over_32,
                T { 4 } * one_sixteenth_one_over_32
            };

        const auto octo_small_sin = sin(octo_small);

        const auto r0 = octo_small_sin.real();

        using std::sin;

        BOOST_TEST(r0 > sin(one_sixteenth_one_over_32.real()));
    }

    {
        const std::complex<T> one_sixteenth_one_over_32 { T { 1 } / 16, T { 1 } / 32 };
        const ::boost::math::octonion<T>
            octo_small
            {
                          one_sixteenth_one_over_32,
                T { 2 } * one_sixteenth_one_over_32,
                T { 3 } * one_sixteenth_one_over_32,
                T { 4 } * one_sixteenth_one_over_32
            };

        const auto octo_small_cos = cos(octo_small);

        const auto r0 = octo_small_cos.real();

        using std::cos;

        BOOST_TEST(r0 > cos(one_sixteenth_one_over_32.real()));
    }

    {
        const std::complex<T> one_sixteenth_one_over_32 { T { 1 } / 16, T { 1 } / 32 };
        const ::boost::math::octonion<T>
            octo_small
            {
                          one_sixteenth_one_over_32,
                T { 2 } * one_sixteenth_one_over_32,
                T { 3 } * one_sixteenth_one_over_32,
                T { 4 } * one_sixteenth_one_over_32
            };

        const auto octo_small_sin = sin(octo_small);
        const auto octo_small_cos = cos(octo_small);
        const auto octo_small_tan = tan(octo_small);

        const auto octo_small_tan_ctrl = octo_small_sin / octo_small_cos;

        const auto result_tan_is_ok = local::is_close_fraction(octo_small_tan, octo_small_tan_ctrl, std::numeric_limits<T>::epsilon() * 64);

        BOOST_TEST(result_tan_is_ok);
    }

    for(auto i = static_cast<unsigned>(UINT8_C(0)); i < static_cast<unsigned>(UINT8_C(8)); ++i)
    {
        static_cast<void>(i);

        ::boost::math::octonion<T> b { T {1}, T {2}, T {3}, T {4}, T {5}, T {6}, T {7}, T {8} };

        b *= static_cast<T>(local::dst_one(local::eng));

        {
            ::boost::math::octonion<T> bp0      = pow(b, 0);
            ::boost::math::octonion<T> bp0_ctrl { T { 1 } };

            const auto result_b0_is_ok = local::is_close_fraction(bp0, bp0_ctrl, std::numeric_limits<T>::epsilon() * 64);
        }

        {
            ::boost::math::octonion<T> bp1      = pow(b, 1);
            ::boost::math::octonion<T> bp1_ctrl = b;

            const auto result_b1_is_ok = local::is_close_fraction(bp1, bp1_ctrl, std::numeric_limits<T>::epsilon() * 64);
        }

        {
            ::boost::math::octonion<T> bp2      = pow(b, 2);
            ::boost::math::octonion<T> bp2_ctrl = b * b;

            const auto result_b2_is_ok = local::is_close_fraction(bp2, bp2_ctrl, std::numeric_limits<T>::epsilon() * 64);
        }

        {
            ::boost::math::octonion<T> bp3      = pow(b, 3);
            ::boost::math::octonion<T> bp3_ctrl = (b * b) * b;

            const auto result_b3_is_ok = local::is_close_fraction(bp3, bp3_ctrl, std::numeric_limits<T>::epsilon() * 64);
        }

        {
            ::boost::math::octonion<T> bp4      = pow(b, 4);
            ::boost::math::octonion<T> bp2_ctrl = b * b;
            ::boost::math::octonion<T> bp4_ctrl = bp2_ctrl * bp2_ctrl;

            const auto result_b3_is_ok = local::is_close_fraction(bp4, bp4_ctrl, std::numeric_limits<T>::epsilon() * 64);
        }
    }
}

auto main() -> int
{
  multiplication_test<float>();
  multiplication_test<double>();

  division_test<float>();

  elem_func_test<float>();
  elem_func_test<double>();

  octonion_original_manual_test();

  const auto result_is_ok = (boost::report_errors() == 0);

  return (result_is_ok ? 0 : -1);
}
