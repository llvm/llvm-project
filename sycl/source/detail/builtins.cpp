//==----------- builtins.cpp - SYCL built-in functions ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/generic_type_traits.hpp>

#include <CL/sycl/exception.hpp>
#include <CL/sycl/pointers.hpp>
#include <CL/sycl/types.hpp>

#include <cmath>

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

// TODO Remove when half type will supported by SYCL Runtime
#define NO_HALF_ENABLED

namespace s = cl::sycl;
namespace d = s::detail;

#define __MAKE_1V(Fun, Call, N, Ret, Arg1)                                     \
  Ret##N Fun __NOEXC(Arg1##N x) {                                              \
    Ret##N r;                                                                  \
    using base_t = typename Arg1##N::element_type;                             \
    detail::helper<N - 1>().run_1v(                                            \
        r, [](base_t x) { return cl::__host_std::Call(x); }, x);               \
    return r;                                                                  \
  }

#define __MAKE_1V_2V(Fun, Call, N, Ret, Arg1, Arg2)                            \
  Ret##N Fun __NOEXC(Arg1##N x, Arg2##N y) {                                   \
    Ret##N r;                                                                  \
    using base1_t = typename Arg1##N::element_type;                            \
    using base2_t = typename Arg2##N::element_type;                            \
    detail::helper<N - 1>().run_1v_2v(                                         \
        r, [](base1_t x, base2_t y) { return cl::__host_std::Call(x, y); }, x, \
        y);                                                                    \
    return r;                                                                  \
  }

#define __MAKE_1V_2V_RS(Fun, Call, N, Ret, Arg1, Arg2)                           \
  Ret Fun __NOEXC(Arg1##N x, Arg2##N y) {                                        \
    Ret r = Ret();                                                               \
    using base1_t = typename Arg1##N::element_type;                              \
    using base2_t = typename Arg2##N::element_type;                              \
    detail::helper<N - 1>().run_1v_2v_rs(r,                                      \
                                         [](Ret &r, base1_t x, base2_t y) {      \
                                           return cl::__host_std::Call(r, x, y); \
                                         },                                      \
                                         x, y);                                  \
    return r;                                                                    \
  }

#define __MAKE_1V_RS(Fun, Call, N, Ret, Arg1)                                  \
  Ret Fun __NOEXC(Arg1##N x) {                                                 \
    Ret r = Ret();                                                             \
    using base1_t = typename Arg1##N::element_type;                            \
    detail::helper<N - 1>().run_1v_rs(                                         \
        r, [](Ret &r, base1_t x) { return cl::__host_std::Call(r, x); }, x);   \
    return r;                                                                  \
  }

#define __MAKE_1V_2V_3V(Fun, Call, N, Ret, Arg1, Arg2, Arg3)                     \
  Ret##N Fun __NOEXC(Arg1##N x, Arg2##N y, Arg3##N z) {                          \
    Ret##N r;                                                                    \
    using base1_t = typename Arg1##N::element_type;                              \
    using base2_t = typename Arg2##N::element_type;                              \
    using base3_t = typename Arg3##N::element_type;                              \
    detail::helper<N - 1>().run_1v_2v_3v(r,                                      \
                                         [](base1_t x, base2_t y, base3_t z) {   \
                                           return cl::__host_std::Call(x, y, z); \
                                         },                                      \
                                         x, y, z);                               \
    return r;                                                                    \
  }

#define __MAKE_1V_2S_3S(Fun, N, Ret, Arg1, Arg2, Arg3)                          \
  Ret##N Fun __NOEXC(Arg1##N x, Arg2 y, Arg3 z) {                               \
    Ret##N r;                                                                   \
    using base1_t = typename Arg1##N::element_type;                             \
    detail::helper<N - 1>().run_1v_2s_3s(r,                                     \
                                         [](base1_t x, Arg2 y, Arg3 z) {        \
                                           return cl::__host_std::Fun(x, y, z); \
                                         },                                     \
                                         x, y, z);                              \
    return r;                                                                   \
  }

#define __MAKE_1V_2S(Fun, N, Ret, Arg1, Arg2)                                  \
  Ret##N Fun __NOEXC(Arg1##N x, Arg2 y) {                                      \
    Ret##N r;                                                                  \
    using base1_t = typename Arg1##N::element_type;                            \
    detail::helper<N - 1>().run_1v_2s(                                         \
        r, [](base1_t x, Arg2 y) { return cl::__host_std::Fun(x, y); }, x, y); \
    return r;                                                                  \
  }

#define __MAKE_SR_1V_AND(Fun, Call, N, Ret, Arg1)                              \
  Ret Fun __NOEXC(Arg1##N x) {                                                 \
    Ret r;                                                                     \
    using base_t = typename Arg1##N::element_type;                             \
    detail::helper<N - 1>().run_1v_sr_and(                                     \
        r, [](base_t x) { return cl::__host_std::Call(x); }, x);               \
    return r;                                                                  \
  }

#define __MAKE_SR_1V_OR(Fun, Call, N, Ret, Arg1)                               \
  Ret Fun __NOEXC(Arg1##N x) {                                                 \
    Ret r;                                                                     \
    using base_t = typename Arg1##N::element_type;                             \
    detail::helper<N - 1>().run_1v_sr_or(                                      \
        r, [](base_t x) { return cl::__host_std::Call(x); }, x);               \
    return r;                                                                  \
  }

#define __MAKE_1V_2P(Fun, N, Ret, Arg1, Arg2)                                  \
  Ret##N Fun __NOEXC(Arg1##N x, Arg2##N *y) {                                  \
    Ret##N r;                                                                  \
    using base1_t = typename Arg1##N::element_type;                            \
    using base2_t = typename Arg2##N::element_type;                            \
    detail::helper<N - 1>().run_1v_2p(                                         \
        r, [](base1_t x, base2_t *y) { return cl::__host_std::Fun(x, y); }, x, \
        y);                                                                    \
    return r;                                                                  \
  }

#define __MAKE_1V_2V_3P(Fun, N, Ret, Arg1, Arg2, Arg3)                         \
  Ret##N Fun __NOEXC(Arg1##N x, Arg2##N y, Arg3##N *z) {                       \
    Ret##N r;                                                                  \
    using base1_t = typename Arg1##N::element_type;                            \
    using base2_t = typename Arg2##N::element_type;                            \
    using base3_t = typename Arg3##N::element_type;                            \
    detail::helper<N - 1>().run_1v_2v_3p(                                      \
        r, [](base1_t x, base2_t y,                                            \
              base3_t *z) { return cl::__host_std::Fun(x, y, z); },            \
        x, y, z);                                                              \
    return r;                                                                  \
  }

#define MAKE_1V(Fun, Ret, Arg1) MAKE_1V_FUNC(Fun, Fun, Ret, Arg1)

#define MAKE_1V_FUNC(Fun, Call, Ret, Arg1)                                     \
  __MAKE_1V(Fun, Call, 2, Ret, Arg1) __MAKE_1V(Fun, Call, 3, Ret, Arg1)        \
      __MAKE_1V(Fun, Call, 4, Ret, Arg1) __MAKE_1V(Fun, Call, 8, Ret, Arg1)    \
      __MAKE_1V(Fun, Call, 16, Ret, Arg1)

#define MAKE_1V_2V(Fun, Ret, Arg1, Arg2)                                       \
  MAKE_1V_2V_FUNC(Fun, Fun, Ret, Arg1, Arg2)

#define MAKE_1V_2V_FUNC(Fun, Call, Ret, Arg1, Arg2)                            \
  __MAKE_1V_2V(Fun, Call, 2, Ret, Arg1, Arg2)                                  \
      __MAKE_1V_2V(Fun, Call, 3, Ret, Arg1, Arg2)                              \
      __MAKE_1V_2V(Fun, Call, 4, Ret, Arg1, Arg2)                              \
      __MAKE_1V_2V(Fun, Call, 8, Ret, Arg1, Arg2)                              \
      __MAKE_1V_2V(Fun, Call, 16, Ret, Arg1, Arg2)

#define MAKE_1V_2V_3V(Fun, Ret, Arg1, Arg2, Arg3)                              \
  MAKE_1V_2V_3V_FUNC(Fun, Fun, Ret, Arg1, Arg2, Arg3)

#define MAKE_1V_2V_3V_FUNC(Fun, Call, Ret, Arg1, Arg2, Arg3)                   \
  __MAKE_1V_2V_3V(Fun, Call, 2, Ret, Arg1, Arg2, Arg3)                         \
      __MAKE_1V_2V_3V(Fun, Call, 3, Ret, Arg1, Arg2, Arg3)                     \
      __MAKE_1V_2V_3V(Fun, Call, 4, Ret, Arg1, Arg2, Arg3)                     \
      __MAKE_1V_2V_3V(Fun, Call, 8, Ret, Arg1, Arg2, Arg3)                     \
      __MAKE_1V_2V_3V(Fun, Call, 16, Ret, Arg1, Arg2, Arg3)

#define MAKE_SC_1V_2V_3V(Fun, Ret, Arg1, Arg2, Arg3)                           \
  MAKE_SC_3ARG(Fun, Ret, Arg1, Arg2, Arg3)                                     \
      MAKE_1V_2V_3V_FUNC(Fun, Fun, Ret, Arg1, Arg2, Arg3)

#define MAKE_SC_FSC_1V_2V_3V_FV(FunSc, FunV, Ret, Arg1, Arg2, Arg3)            \
  MAKE_SC_3ARG(FunSc, Ret, Arg1, Arg2, Arg3)                                   \
      MAKE_1V_2V_3V_FUNC(FunSc, FunV, Ret, Arg1, Arg2, Arg3)

#define MAKE_SC_3ARG(Fun, Ret, Arg1, Arg2, Arg3)                               \
  Ret Fun __NOEXC(Arg1 x, Arg2 y, Arg3 z) { return (Ret)__##Fun(x, y, z); }

#define MAKE_1V_2S(Fun, Ret, Arg1, Arg2)                                       \
  __MAKE_1V_2S(Fun, 2, Ret, Arg1, Arg2) __MAKE_1V_2S(Fun, 3, Ret, Arg1, Arg2)  \
      __MAKE_1V_2S(Fun, 4, Ret, Arg1, Arg2)                                    \
      __MAKE_1V_2S(Fun, 8, Ret, Arg1, Arg2)                                    \
      __MAKE_1V_2S(Fun, 16, Ret, Arg1, Arg2)

#define MAKE_1V_2S_3S(Fun, Ret, Arg1, Arg2, Arg3)                              \
  __MAKE_1V_2S_3S(Fun, 2, Ret, Arg1, Arg2, Arg3)                               \
      __MAKE_1V_2S_3S(Fun, 3, Ret, Arg1, Arg2, Arg3)                           \
      __MAKE_1V_2S_3S(Fun, 4, Ret, Arg1, Arg2, Arg3)                           \
      __MAKE_1V_2S_3S(Fun, 8, Ret, Arg1, Arg2, Arg3)                           \
      __MAKE_1V_2S_3S(Fun, 16, Ret, Arg1, Arg2, Arg3)

#define MAKE_SR_1V_AND(Fun, Call, Ret, Arg1)                                   \
  __MAKE_SR_1V_AND(Fun, Call, 2, Ret, Arg1)                                    \
      __MAKE_SR_1V_AND(Fun, Call, 3, Ret, Arg1)                                \
      __MAKE_SR_1V_AND(Fun, Call, 4, Ret, Arg1)                                \
      __MAKE_SR_1V_AND(Fun, Call, 8, Ret, Arg1)                                \
      __MAKE_SR_1V_AND(Fun, Call, 16, Ret, Arg1)

#define MAKE_SR_1V_OR(Fun, Call, Ret, Arg1)                                    \
  __MAKE_SR_1V_OR(Fun, Call, 2, Ret, Arg1)                                     \
      __MAKE_SR_1V_OR(Fun, Call, 3, Ret, Arg1)                                 \
      __MAKE_SR_1V_OR(Fun, Call, 4, Ret, Arg1)                                 \
      __MAKE_SR_1V_OR(Fun, Call, 8, Ret, Arg1)                                 \
      __MAKE_SR_1V_OR(Fun, Call, 16, Ret, Arg1)

#define MAKE_1V_2P(Fun, Ret, Arg1, Arg2)                                       \
  __MAKE_1V_2P(Fun, 2, Ret, Arg1, Arg2) __MAKE_1V_2P(Fun, 3, Ret, Arg1, Arg2)  \
      __MAKE_1V_2P(Fun, 4, Ret, Arg1, Arg2)                                    \
      __MAKE_1V_2P(Fun, 8, Ret, Arg1, Arg2)                                    \
      __MAKE_1V_2P(Fun, 16, Ret, Arg1, Arg2)

#define MAKE_GEO_1V_2V_RS(Fun, Call, Ret, Arg1, Arg2)                          \
  __MAKE_1V_2V_RS(Fun, Call, 2, Ret, Arg1, Arg2)                               \
      __MAKE_1V_2V_RS(Fun, Call, 3, Ret, Arg1, Arg2)                           \
      __MAKE_1V_2V_RS(Fun, Call, 4, Ret, Arg1, Arg2)

#define MAKE_1V_2V_3P(Fun, Ret, Arg1, Arg2, Arg3)                              \
  __MAKE_1V_2V_3P(Fun, 2, Ret, Arg1, Arg2, Arg3)                               \
      __MAKE_1V_2V_3P(Fun, 3, Ret, Arg1, Arg2, Arg3)                           \
      __MAKE_1V_2V_3P(Fun, 4, Ret, Arg1, Arg2, Arg3)                           \
      __MAKE_1V_2V_3P(Fun, 8, Ret, Arg1, Arg2, Arg3)                           \
      __MAKE_1V_2V_3P(Fun, 16, Ret, Arg1, Arg2, Arg3)

namespace cl {
namespace __host_std {
namespace detail {

template <int N> struct helper {
  template <typename Res, typename Op, typename T1>
  inline void run_1v(Res &r, Op op, T1 x) {
    helper<N - 1>().run_1v(r, op, x);
    r.template swizzle<N>() = op(x.template swizzle<N>());
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2v(Res &r, Op op, T1 x, T2 y) {
    helper<N - 1>().run_1v_2v(r, op, x, y);
    r.template swizzle<N>() =
        op(x.template swizzle<N>(), y.template swizzle<N>());
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2s(Res &r, Op op, T1 x, T2 y) {
    helper<N - 1>().run_1v_2s(r, op, x, y);
    r.template swizzle<N>() = op(x.template swizzle<N>(), y);
  }

  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  inline void run_1v_2s_3s(Res &r, Op op, T1 x, T2 y, T3 z) {
    helper<N - 1>().run_1v_2s_3s(r, op, x, y, z);
    r.template swizzle<N>() = op(x.template swizzle<N>(), y, z);
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2v_rs(Res &r, Op op, T1 x, T2 y) {
    helper<N - 1>().run_1v_2v_rs(r, op, x, y);
    op(r, x.template swizzle<N>(), y.template swizzle<N>());
  }

  template <typename Res, typename Op, typename T1>
  inline void run_1v_rs(Res &r, Op op, T1 x) {
    helper<N - 1>().run_1v_rs(r, op, x);
    op(r, x.template swizzle<N>());
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2p(Res &r, Op op, T1 x, T2 y) {
    helper<N - 1>().run_1v_2p(r, op, x, y);
    // TODO avoid creating a temporary variable
    typename std::remove_pointer<T2>::type::element_type temp;
    r.template swizzle<N>() = op(x.template swizzle<N>(), &temp);
    y->template swizzle<N>() = temp;
  }

  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  inline void run_1v_2v_3p(Res &r, Op op, T1 x, T2 y, T3 z) {
    helper<N - 1>().run_1v_2v_3p(r, op, x, y, z);
    // TODO avoid creating a temporary variable
    typename std::remove_pointer<T3>::type::element_type temp;
    r.template swizzle<N>() =
        op(x.template swizzle<N>(), y.template swizzle<N>(), &temp);
    z->template swizzle<N>() = temp;
  }

  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  inline void run_1v_2v_3v(Res &r, Op op, T1 x, T2 y, T3 z) {
    helper<N - 1>().run_1v_2v_3v(r, op, x, y, z);
    r.template swizzle<N>() =
        op(x.template swizzle<N>(), y.template swizzle<N>(),
           z.template swizzle<N>());
  }

  template <typename Res, typename Op, typename T1>
  inline void run_1v_sr_or(Res &r, Op op, T1 x) {
    helper<N - 1>().run_1v_sr_or(r, op, x);
    r = (op(x.template swizzle<N>()) || r);
  }

  template <typename Res, typename Op, typename T1>
  inline void run_1v_sr_and(Res &r, Op op, T1 x) {
    helper<N - 1>().run_1v_sr_and(r, op, x);
    r = (op(x.template swizzle<N>()) && r);
  }
};

template <> struct helper<0> {
  template <typename Res, typename Op, typename T1>
  inline void run_1v(Res &r, Op op, T1 x) {
    r.template swizzle<0>() = op(x.template swizzle<0>());
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2v(Res &r, Op op, T1 x, T2 y) {
    r.template swizzle<0>() =
        op(x.template swizzle<0>(), y.template swizzle<0>());
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2s(Res &r, Op op, T1 x, T2 y) {
    r.template swizzle<0>() = op(x.template swizzle<0>(), y);
  }

  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  inline void run_1v_2s_3s(Res &r, Op op, T1 x, T2 y, T3 z) {
    r.template swizzle<0>() = op(x.template swizzle<0>(), y, z);
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2v_rs(Res &r, Op op, T1 x, T2 y) {
    op(r, x.template swizzle<0>(), y.template swizzle<0>());
  }

  template <typename Res, typename Op, typename T1>
  inline void run_1v_rs(Res &r, Op op, T1 x) {
    op(r, x.template swizzle<0>());
  }

  template <typename Res, typename Op, typename T1, typename T2>
  inline void run_1v_2p(Res &r, Op op, T1 x, T2 y) {
    // TODO avoid creating a temporary variable
    typename std::remove_pointer<T2>::type::element_type temp;
    r.template swizzle<0>() = op(x.template swizzle<0>(), &temp);
    y->template swizzle<0>() = temp;
  }

  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  inline void run_1v_2v_3p(Res &r, Op op, T1 x, T2 y, T3 z) {
    // TODO avoid creating a temporary variable
    typename std::remove_pointer<T3>::type::element_type temp;
    r.template swizzle<0>() =
        op(x.template swizzle<0>(), y.template swizzle<0>(), &temp);
    z->template swizzle<0>() = temp;
  }

  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  inline void run_1v_2v_3v(Res &r, Op op, T1 x, T2 y, T3 z) {
    r.template swizzle<0>() =
        op(x.template swizzle<0>(), y.template swizzle<0>(),
           z.template swizzle<0>());
  }

  template <typename Res, typename Op, typename T1>
  inline void run_1v_sr_or(Res &r, Op op, T1 x) {
    r = op(x.template swizzle<0>());
  }

  template <typename Res, typename Op, typename T1>
  inline void run_1v_sr_and(Res &r, Op op, T1 x) {
    r = op(x.template swizzle<0>());
  }
};
} // namespace detail

s::cl_float OpDot(s::cl_float2, s::cl_float2);
s::cl_float OpDot(s::cl_float3, s::cl_float3);
s::cl_float OpDot(s::cl_float4, s::cl_float4);
s::cl_double OpDot(s::cl_double2, s::cl_double2);
s::cl_double OpDot(s::cl_double3, s::cl_double3);
s::cl_double OpDot(s::cl_double4, s::cl_double4);
#ifndef NO_HALF_ENABLED
s::cl_half OpDot(s::cl_half2, s::cl_half2);
s::cl_half OpDot(s::cl_half3, s::cl_half3);
s::cl_half OpDot(s::cl_half4, s::cl_half4);
#endif

s::cl_int OpAll(s::cl_int2);
s::cl_int OpAll(s::cl_int3);
s::cl_int OpAll(s::cl_int4);

namespace {
template <typename T> inline T __acospi(T x) { return std::acos(x) / M_PI; }

template <typename T> inline T __asinpi(T x) { return std::asin(x) / M_PI; }

template <typename T> inline T __atanpi(T x) { return std::atan(x) / M_PI; }

template <typename T> inline T __atan2pi(T x, T y) {
  return std::atan2(x, y) / M_PI;
}

template <typename T> inline T __cospi(T x) { return std::cos(M_PI * x); }

template <typename T> T inline __fract(T x, T *iptr) {
  T f = std::floor(x);
  *(iptr) = f;
  return std::fmin(x - f, nextafter(T(1.0), T(0.0)));
}

template <typename T> inline T __mad(T a, T b, T c) { return (a * b) + c; }

template <typename T> inline T __maxmag(T x, T y) {
  if (std::fabs(x) > std::fabs(y))
    return x;
  if (std::fabs(y) > std::fabs(x))
    return y;
  return std::fmax(x, y);
}

template <typename T> inline T __minmag(T x, T y) {
  if (std::fabs(x) < std::fabs(y))
    return x;
  if (std::fabs(y) < std::fabs(x))
    return y;
  return std::fmin(x, y);
}

template <typename T> inline T __powr(T x, T y) {
  return (x >= T(0)) ? T(std::pow(x, y)) : x;
}

template <typename T> inline T __rootn(T x, s::cl_int y) {
  return std::pow(x, T(1.0) / y);
}

template <typename T> inline T __rsqrt(T x) { return T(1.0) / std::sqrt(x); }

template <typename T> inline T __sincos(T x, T *cosval) {
  (*cosval) = std::cos(x);
  return std::sin(x);
}

template <typename T> inline T __sinpi(T x) { return std::sin(M_PI * x); }

template <typename T> inline T __tanpi(T x) { return std::tan(M_PI * x); }

template <typename T> inline T __abs_diff(T x, T y) { return std::abs(x - y); }

template <typename T> inline T __u_add_sat(T x, T y) {
  return (x < (d::max_v<T>() - y) ? x + y : d::max_v<T>());
}

template <typename T> inline T __s_add_sat(T x, T y) {
  if (x > 0 && y > 0)
    return (x < (d::max_v<T>() - y) ? (x + y) : d::max_v<T>());
  if (x < 0 && y < 0)
    return (x > (d::min_v<T>() - y) ? (x + y) : d::min_v<T>());
  return x + y;
}

template <typename T> inline T __hadd(T x, T y) {
  const T one = 1;
  return (x >> one) + (y >> one) + ((y & x) & one);
}

template <typename T> inline T __rhadd(T x, T y) {
  const T one = 1;
  return (x >> one) + (y >> one) + ((y | x) & one);
}

template <typename T> inline T __clamp(T x, T minval, T maxval) {
  return std::min(std::max(x, minval), maxval);
}

template <typename T> inline constexpr T __clz_impl(T x, T m, T n = 0) {
  return (x & m) ? n : __clz_impl(x, T(m >> 1), ++n);
}

template <typename T> inline constexpr T __clz(T x) {
  using UT = typename std::make_unsigned<T>::type;
  return (x == T(0)) ? sizeof(T) * 8 : __clz_impl<UT>(x, d::msbMask<UT>(x));
}

template <typename T> T __mul_hi(T a, T b) {
  using UPT = typename d::make_upper<T>::type;
  UPT a_s = a;
  UPT b_s = b;
  UPT mul = a_s * b_s;
  return (mul >> (sizeof(T) * 8));
}

// T is minimum of 64 bits- long or longlong
template <typename T> inline T __long_mul_hi(T a, T b) {
  int halfsize = (sizeof(T) * 8) / 2;
  T a1 = a >> halfsize;
  T a0 = (a << halfsize) >> halfsize;
  T b1 = b >> halfsize;
  T b0 = (b << halfsize) >> halfsize;

  // a1b1 - for bits - [64-128)
  // a1b0 a0b1 for bits - [32-96)
  // a0b0 for bits - [0-64)
  T a1b1 = a1 * b1;
  T a0b1 = a0 * b1;
  T a1b0 = a1 * b0;
  T a0b0 = a0 * b0;

  // To get the upper 64 bits:
  // 64 bits from a1b1, upper 32 bits from [a1b0 + (a0b1 + a0b0>>32 (carry bit
  // in 33rd bit))] with carry bit on 64th bit - use of hadd. Add the a1b1 to
  // the above 32 bit result.
  T result =
      a1b1 + (__hadd(a1b0, (a0b1 + (a0b0 >> halfsize))) >> (halfsize - 1));
  return result;
}

template <typename T> inline T __mad_hi(T a, T b, T c) {
  return __mul_hi(a, b) + c;
}

template <typename T> inline T __long_mad_hi(T a, T b, T c) {
  return __long_mul_hi(a, b) + c;
}

template <typename T> inline T __s_mad_sat(T a, T b, T c) {
  using UPT = typename d::make_upper<T>::type;
  UPT mul = UPT(a) * UPT(b);
  const UPT max = d::max_v<T>();
  const UPT min = d::min_v<T>();
  mul = std::min(std::max(mul, min), max);
  return __s_add_sat(T(mul), c);
}

template <typename T> inline T __s_long_mad_sat(T a, T b, T c) {
  bool neg_prod = (a < 0) ^ (b < 0);
  T mulhi = __long_mul_hi(a, b);

  // check mul_hi. If it is any value != 0.
  // if prod is +ve, any value in mulhi means we need to saturate.
  // if prod is -ve, any value in mulhi besides -1 means we need to saturate.
  if (!neg_prod && mulhi != 0)
    return d::max_v<T>();
  if (neg_prod && mulhi != -1)
    return d::max_v<T>(); // essentially some other negative value.
  return __s_add_sat(T(a * b), c);
}

template <typename T> inline T __u_mad_sat(T a, T b, T c) {
  using UPT = typename d::make_upper<T>::type;
  UPT mul = UPT(a) * UPT(b);
  const UPT min = d::min_v<T>();
  const UPT max = d::max_v<T>();
  mul = std::min(std::max(mul, min), max);
  return __u_add_sat(T(mul), c);
}

template <typename T> inline T __u_long_mad_sat(T a, T b, T c) {
  T mulhi = __long_mul_hi(a, b);
  // check mul_hi. If it is any value != 0.
  if (mulhi != 0)
    return d::max_v<T>();
  return __u_add_sat(T(a * b), c);
}

template <typename T> inline T __rotate(T x, T n) {
  using UT = typename std::make_unsigned<T>::type;
  return (x << n) | (UT(x) >> ((sizeof(x) * 8) - n));
}

template <typename T> inline T __u_sub_sat(T x, T y) {
  return (y < (x - d::min_v<T>())) ? (x - y) : d::min_v<T>();
}

template <typename T> inline T __s_sub_sat(T x, T y) {
  if (y > 0)
    return (y < (x - d::min_v<T>()) ? x - y : d::min_v<T>());
  if (y < 0)
    return (y > (x - d::max_v<T>()) ? x - y : d::max_v<T>());
  return x;
}

template <typename T1, typename T2>
typename d::make_upper<T1>::type inline __upsample(T1 hi, T2 lo) {
  using UT = typename d::make_upper<T1>::type;
  return (UT(hi) << (sizeof(T1) * 8)) | lo;
}

template <typename T> inline constexpr T __popcount_impl(T x, size_t n = 0) {
  return (x == T(0)) ? n : __popcount_impl(x >> 1, ((x & T(1)) ? ++n : n));
}

template <typename T> inline constexpr T __popcount(T x) {
  using UT = typename d::make_unsigned<T>::type;
  return __popcount_impl(UT(x));
}

template <typename T> inline T __mad24(T x, T y, T z) { return (x * y) + z; }

template <typename T> inline T __mul24(T x, T y) { return (x * y); }

template <typename T> inline T __fclamp(T x, T minval, T maxval) {
  return std::fmin(std::fmax(x, minval), maxval);
}

template <typename T> inline T __degrees(T radians) {
  return (180 / M_PI) * radians;
}

template <typename T> inline T __mix(T x, T y, T a) { return x + (y - x) * a; }

template <typename T> inline T __radians(T degrees) {
  return (M_PI / 180) * degrees;
}

template <typename T> inline T __step(T edge, T x) {
  return (x < edge) ? 0.0 : 1.0;
}

template <typename T> inline T __smoothstep(T edge0, T edge1, T x) {
  cl_float t;
  t = __fclamp((x - edge0) / (edge1 - edge0), T(0), T(1));
  return t * t * (3 - 2 * t);
}

template <typename T> inline T __sign(T x) {
  if (std::isnan(x))
    return T(0.0);
  if (x > 0)
    return T(1.0);
  if (x < 0)
    return T(-1.0);
  /* x is +0.0 or -0.0 */
  return x;
}

template <typename T> inline T __cross(T p0, T p1) {
  T result(0);
  result.x() = p0.y() * p1.z() - p0.z() * p1.y();
  result.y() = p0.z() * p1.x() - p0.x() * p1.z();
  result.z() = p0.x() * p1.y() - p0.y() * p1.x();
  return result;
}

template <typename T> inline void __OpFMul_impl(T &r, T p0, T p1) {
  r += p0 * p1;
}

template <typename T> inline T __OpFMul(T p0, T p1) {
  T result = 0;
  __OpFMul_impl(result, p0, p1);
  return result;
}

template <typename T>
inline typename std::enable_if<d::is_sgengeo<T>::value, T>::type __length(T t) {
  return std::sqrt(__OpFMul(t, t));
}

template <typename T>
inline typename std::enable_if<d::is_vgengeo<T>::value,
                               typename T::element_type>::type
__length(T t) {
  return std::sqrt(OpDot(t, t));
}

template <typename T>
inline typename std::enable_if<d::is_sgengeo<T>::value, T>::type
__normalize(T t) {
  T r = __length(t);
  return t / T(r);
}

template <typename T>
inline typename std::enable_if<d::is_vgengeo<T>::value, T>::type
__normalize(T t) {
  typename T::element_type r = __length(t);
  return t / T(r);
}

template <typename T>
inline typename std::enable_if<d::is_sgengeo<T>::value, T>::type
__fast_length(T t) {
  return std::sqrt(__OpFMul(t, t));
}

template <typename T>
inline typename std::enable_if<d::is_vgengeo<T>::value,
                               typename T::element_type>::type
__fast_length(T t) {
  return std::sqrt(OpDot(t, t));
}

template <typename T>
inline typename std::enable_if<d::is_vgengeo<T>::value, T>::type
__fast_normalize(T t) {
  if (OpAll(t == T(0.0f)))
    return t;
  typename T::element_type r = std::sqrt(OpDot(t, t));
  return t / T(r);
}

template <typename T> inline T __vOpFOrdEqual(T x, T y) { return -(x == y); }

template <typename T> inline T __sOpFOrdEqual(T x, T y) { return x == y; }

template <typename T> inline T __vOpFUnordNotEqual(T x, T y) {
  return -(x != y);
}

template <typename T> inline T __sOpFUnordNotEqual(T x, T y) { return x != y; }

template <typename T> inline T __vOpFOrdGreaterThan(T x, T y) {
  return -(x > y);
}

template <typename T> inline T __sOpFOrdGreaterThan(T x, T y) { return x > y; }

template <typename T> inline T __vOpFOrdGreaterThanEqual(T x, T y) {
  return -(x >= y);
}

template <typename T> inline T __sOpFOrdGreaterThanEqual(T x, T y) {
  return x >= y;
}

template <typename T> inline T __vOpFOrdLessThanEqual(T x, T y) {
  return -(x <= y);
}

template <typename T> inline T __sOpFOrdLessThanEqual(T x, T y) {
  return x <= y;
}

template <typename T> inline T __vOpLessOrGreater(T x, T y) {
  return -((x < y) || (x > y));
}

template <typename T> inline T __sOpLessOrGreater(T x, T y) {
  return ((x < y) || (x > y));
}

template <typename T> cl_int inline __OpAny(T x) { return d::msbIsSet(x); }
template <typename T> cl_int inline __OpAll(T x) { return d::msbIsSet(x); }

template <typename T> inline T __vOpOrdered(T x, T y) {
  return -(!(std::isunordered(x, y)));
}

template <typename T> inline T __sOpOrdered(T x, T y) {
  return !(std::isunordered(x, y));
}

template <typename T> inline T __vOpUnordered(T x, T y) {
  return -(std::isunordered(x, y));
}

template <typename T> inline T __sOpUnordered(T x, T y) {
  return std::isunordered(x, y);
}

template <typename T>
inline typename std::enable_if<d::is_sgeninteger<T>::value, T>::type
__bitselect(T a, T b, T c) {
  return (a & ~c) | (b & c);
}

template <typename T> union databitset;
// float
template <> union databitset<float> {
  static_assert(sizeof(uint32_t) == sizeof(float),
                "size of float is not equal to 32 bits.");
  float f;
  uint32_t i;
};

// double
template <> union databitset<double> {
  static_assert(sizeof(uint64_t) == sizeof(double),
                "size of double is not equal to 64 bits.");
  double f;
  uint64_t i;
};

#ifndef NO_HALF_ENABLED
// Half
template <> union databitset<cl_half> {
  static_assert(sizeof(uint16_t) == sizeof(cl_half),
                "size of half is not equal to 16 bits.");
  cl_half f;
  uint16_t i;
};
#endif

template <typename T>
typename std::enable_if<d::is_sgenfloat<T>::value, T>::type inline __bitselect(
    T a, T b, T c) {
  databitset<T> ba;
  ba.f = a;
  databitset<T> bb;
  bb.f = b;
  databitset<T> bc;
  bc.f = c;
  databitset<T> br;
  br.f = 0;
  br.i = ((ba.i & ~bc.i) | (bb.i & bc.i));
  return br.f;
}

template <typename T, typename T2> inline T2 __OpSelect(T c, T2 b, T2 a) {
  return (c ? b : a);
}

template <typename T, typename T2> inline T2 __vOpSelect(T c, T2 b, T2 a) {
  return d::msbIsSet(c) ? b : a;
}
}

/* ----------------- 4.13.3 Math functions. Host version --------------------*/
// acos
cl_float acos(s::cl_float x) __NOEXC { return std::acos(x); }
cl_double acos(s::cl_double x) __NOEXC { return std::acos(x); }
#ifndef NO_HALF_ENABLED
cl_half acos(s::cl_half x) __NOEXC { return std::acos(x); }
#endif
MAKE_1V(acos, s::cl_float, s::cl_float)
MAKE_1V(acos, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(acos, s::cl_half, s::cl_half)
#endif

// acosh
cl_float acosh(s::cl_float x) __NOEXC { return std::acosh(x); }
cl_double acosh(s::cl_double x) __NOEXC { return std::acosh(x); }
#ifndef NO_HALF_ENABLED
cl_half acosh(s::cl_half x) __NOEXC { return std::acosh(x); }
#endif
MAKE_1V(acosh, s::cl_float, s::cl_float)
MAKE_1V(acosh, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(acosh, s::cl_half, s::cl_half)
#endif

// acospi
cl_float acospi(s::cl_float x) __NOEXC { return __acospi(x); }
cl_double acospi(s::cl_double x) __NOEXC { return __acospi(x); }
#ifndef NO_HALF_ENABLED
cl_half acospi(s::cl_half x) __NOEXC { return __acospi(x); }
#endif
MAKE_1V(acospi, s::cl_float, s::cl_float)
MAKE_1V(acospi, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(acospi, s::cl_half, s::cl_half)
#endif

// asin
cl_float asin(s::cl_float x) __NOEXC { return std::asin(x); }
cl_double asin(s::cl_double x) __NOEXC { return std::asin(x); }
#ifndef NO_HALF_ENABLED
cl_half asin(s::cl_half x) __NOEXC { return std::asin(x); }
#endif
MAKE_1V(asin, s::cl_float, s::cl_float)
MAKE_1V(asin, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(asin, s::cl_half, s::cl_half)
#endif

// asinh
cl_float asinh(s::cl_float x) __NOEXC { return std::asinh(x); }
cl_double asinh(s::cl_double x) __NOEXC { return std::asinh(x); }
#ifndef NO_HALF_ENABLED
cl_half asinh(s::cl_half x) __NOEXC { return std::asinh(x); }
#endif
MAKE_1V(asinh, s::cl_float, s::cl_float)
MAKE_1V(asinh, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(asinh, s::cl_half, s::cl_half)
#endif

// asinpi
cl_float asinpi(s::cl_float x) __NOEXC { return __asinpi(x); }
cl_double asinpi(s::cl_double x) __NOEXC { return __asinpi(x); }
#ifndef NO_HALF_ENABLED
cl_half asinpi(s::cl_half x) __NOEXC { return __asinpi(x); }
#endif
MAKE_1V(asinpi, s::cl_float, s::cl_float)
MAKE_1V(asinpi, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(asinpi, s::cl_half, s::cl_half)
#endif

// atan
cl_float atan(s::cl_float x) __NOEXC { return std::atan(x); }
cl_double atan(s::cl_double x) __NOEXC { return std::atan(x); }
#ifndef NO_HALF_ENABLED
cl_half atan(s::cl_half x) __NOEXC { return std::atan(x); }
#endif
MAKE_1V(atan, s::cl_float, s::cl_float)
MAKE_1V(atan, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(atan, s::cl_half, s::cl_half)
#endif

// atan2
cl_float atan2(s::cl_float x, s::cl_float y) __NOEXC {
  return std::atan2(x, y);
}
cl_double atan2(s::cl_double x, s::cl_double y) __NOEXC {
  return std::atan2(x, y);
}
#ifndef NO_HALF_ENABLED
cl_half atan2(s::cl_half x, s::cl_half y) __NOEXC { return std::atan2(x, y); }
#endif
MAKE_1V_2V(atan2, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(atan2, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(atan2, s::cl_half, s::cl_half, s::cl_half)
#endif

// atanh
cl_float atanh(s::cl_float x) __NOEXC { return std::atanh(x); }
cl_double atanh(s::cl_double x) __NOEXC { return std::atanh(x); }
#ifndef NO_HALF_ENABLED
cl_half atanh(s::cl_half x) __NOEXC { return std::atanh(x); }
#endif
MAKE_1V(atanh, s::cl_float, s::cl_float)
MAKE_1V(atanh, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(atanh, s::cl_half, s::cl_half)
#endif

// atanpi
cl_float atanpi(s::cl_float x) __NOEXC { return __atanpi(x); }
cl_double atanpi(s::cl_double x) __NOEXC { return __atanpi(x); }
#ifndef NO_HALF_ENABLED
cl_half atanpi(s::cl_half x) __NOEXC { return __atanpi(x); }
#endif
MAKE_1V(atanpi, s::cl_float, s::cl_float)
MAKE_1V(atanpi, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(atanpi, s::cl_half, s::cl_half)
#endif

// atan2pi
cl_float atan2pi(s::cl_float x, s::cl_float y) __NOEXC {
  return __atan2pi(x, y);
}
cl_double atan2pi(s::cl_double x, s::cl_double y) __NOEXC {
  return __atan2pi(x, y);
}
#ifndef NO_HALF_ENABLED
cl_half atan2pi(s::cl_half x, s::cl_half y) __NOEXC { return __atan2pi(x, y); }
#endif
MAKE_1V_2V(atan2pi, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(atan2pi, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(atan2pi, s::cl_half, s::cl_half, s::cl_half)
#endif

// cbrt
cl_float cbrt(s::cl_float x) __NOEXC { return std::cbrt(x); }
cl_double cbrt(s::cl_double x) __NOEXC { return std::cbrt(x); }
#ifndef NO_HALF_ENABLED
cl_half cbrt(s::cl_half x) __NOEXC { return std::cbrt(x); }
#endif
MAKE_1V(cbrt, s::cl_float, s::cl_float)
MAKE_1V(cbrt, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(cbrt, s::cl_half, s::cl_half)
#endif

// ceil
cl_float ceil(s::cl_float x) __NOEXC { return std::ceil(x); }
cl_double ceil(s::cl_double x) __NOEXC { return std::ceil(x); }
#ifndef NO_HALF_ENABLED
cl_half ceil(s::cl_half x) __NOEXC { return std::ceil(x); }
#endif
MAKE_1V(ceil, s::cl_float, s::cl_float)
MAKE_1V(ceil, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(ceil, s::cl_half, s::cl_half)
#endif

// copysign
cl_float copysign(s::cl_float x, s::cl_float y) __NOEXC {
  return std::copysign(x, y);
}
cl_double copysign(s::cl_double x, s::cl_double y) __NOEXC {
  return std::copysign(x, y);
}
#ifndef NO_HALF_ENABLED
cl_half copysign(s::cl_half x, s::cl_half y) __NOEXC {
  return std::copysign(x, y);
}
#endif
MAKE_1V_2V(copysign, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(copysign, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(copysign, s::cl_half, s::cl_half, s::cl_half)
#endif

// cos
cl_float cos(s::cl_float x) __NOEXC { return std::cos(x); }
cl_double cos(s::cl_double x) __NOEXC { return std::cos(x); }
#ifndef NO_HALF_ENABLED
cl_half cos(s::cl_half x) __NOEXC { return std::cos(x); }
#endif
MAKE_1V(cos, s::cl_float, s::cl_float)
MAKE_1V(cos, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(cos, s::cl_half, s::cl_half)
#endif

// cosh
cl_float cosh(s::cl_float x) __NOEXC { return std::cosh(x); }
cl_double cosh(s::cl_double x) __NOEXC { return std::cosh(x); }
#ifndef NO_HALF_ENABLED
cl_half cosh(s::cl_half x) __NOEXC { return std::cosh(x); }
#endif
MAKE_1V(cosh, s::cl_float, s::cl_float)
MAKE_1V(cosh, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(cosh, s::cl_half, s::cl_half)
#endif

// cospi
cl_float cospi(s::cl_float x) __NOEXC { return __cospi(x); }
cl_double cospi(s::cl_double x) __NOEXC { return __cospi(x); }
#ifndef NO_HALF_ENABLED
cl_half cospi(s::cl_half x) __NOEXC { return __cospi(x); }
#endif
MAKE_1V(cospi, s::cl_float, s::cl_float)
MAKE_1V(cospi, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(cospi, s::cl_half, s::cl_half)
#endif

// erfc
cl_float erfc(s::cl_float x) __NOEXC { return std::erfc(x); }
cl_double erfc(s::cl_double x) __NOEXC { return std::erfc(x); }
#ifndef NO_HALF_ENABLED
cl_half erfc(s::cl_half x) __NOEXC { return std::erfc(x); }
#endif
MAKE_1V(erfc, s::cl_float, s::cl_float)
MAKE_1V(erfc, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(erfc, s::cl_half, s::cl_half)
#endif

// erf
cl_float erf(s::cl_float x) __NOEXC { return std::erf(x); }
cl_double erf(s::cl_double x) __NOEXC { return std::erf(x); }
#ifndef NO_HALF_ENABLED
cl_half erf(s::cl_half x) __NOEXC { return std::erf(x); }
#endif
MAKE_1V(erf, s::cl_float, s::cl_float)
MAKE_1V(erf, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(erf, s::cl_half, s::cl_half)
#endif

// exp
cl_float exp(s::cl_float x) __NOEXC { return std::exp(x); }
cl_double exp(s::cl_double x) __NOEXC { return std::exp(x); }
#ifndef NO_HALF_ENABLED
cl_half exp(s::cl_half x) __NOEXC { return std::exp(x); }
#endif
MAKE_1V(exp, s::cl_float, s::cl_float)
MAKE_1V(exp, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(exp, s::cl_half, s::cl_half)
#endif

// exp2
cl_float exp2(s::cl_float x) __NOEXC { return std::exp2(x); }
cl_double exp2(s::cl_double x) __NOEXC { return std::exp2(x); }
#ifndef NO_HALF_ENABLED
cl_half exp2(s::cl_half x) __NOEXC { return std::exp2(x); }
#endif
MAKE_1V(exp2, s::cl_float, s::cl_float)
MAKE_1V(exp2, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(exp2, s::cl_half, s::cl_half)
#endif

// exp10
cl_float exp10(s::cl_float x) __NOEXC { return std::pow(10, x); }
cl_double exp10(s::cl_double x) __NOEXC { return std::pow(10, x); }
#ifndef NO_HALF_ENABLED
cl_half exp10(s::cl_half x) __NOEXC { return std::pow(10, x); }
#endif
MAKE_1V(exp10, s::cl_float, s::cl_float)
MAKE_1V(exp10, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(exp10, s::cl_half, s::cl_half)
#endif

// expm1
cl_float expm1(s::cl_float x) __NOEXC { return std::expm1(x); }
cl_double expm1(s::cl_double x) __NOEXC { return std::expm1(x); }
#ifndef NO_HALF_ENABLED
cl_half expm1(s::cl_half x) __NOEXC { return std::expm1(x); }
#endif
MAKE_1V(expm1, s::cl_float, s::cl_float)
MAKE_1V(expm1, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(expm1, s::cl_half, s::cl_half)
#endif

// fabs
cl_float fabs(s::cl_float x) __NOEXC { return std::fabs(x); }
cl_double fabs(s::cl_double x) __NOEXC { return std::fabs(x); }
#ifndef NO_HALF_ENABLED
cl_half fabs(s::cl_half x) __NOEXC { return std::fabs(x); }
#endif
MAKE_1V(fabs, s::cl_float, s::cl_float)
MAKE_1V(fabs, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(fabs, s::cl_half, s::cl_half)
#endif

// fdim
cl_float fdim(s::cl_float x, s::cl_float y) __NOEXC { return std::fdim(x, y); }
cl_double fdim(s::cl_double x, s::cl_double y) __NOEXC {
  return std::fdim(x, y);
}
#ifndef NO_HALF_ENABLED
cl_half fdim(s::cl_half x, s::cl_half y) __NOEXC { return std::fdim(x, y); }
#endif
MAKE_1V_2V(fdim, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(fdim, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(fdim, s::cl_half, s::cl_half, s::cl_half)
#endif

// floor
cl_float floor(s::cl_float x) __NOEXC { return std::floor(x); }
cl_double floor(s::cl_double x) __NOEXC { return std::floor(x); }
#ifndef NO_HALF_ENABLED
cl_half floor(s::cl_half x) __NOEXC { return std::floor(x); }
#endif
MAKE_1V(floor, s::cl_float, s::cl_float)
MAKE_1V(floor, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(floor, s::cl_half, s::cl_half)
#endif

// fma
cl_float fma(s::cl_float a, s::cl_float b, s::cl_float c) __NOEXC {
  return std::fma(a, b, c);
}
cl_double fma(s::cl_double a, s::cl_double b, s::cl_double c) __NOEXC {
  return std::fma(a, b, c);
}
#ifndef NO_HALF_ENABLED
cl_half fma(s::cl_half a, s::cl_half b, s::cl_half c) __NOEXC {
  return std::fma(a, b, c);
}
#endif
MAKE_1V_2V_3V(fma, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V_3V(fma, s::cl_double, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_3V(fma, s::cl_half, s::cl_half, s::cl_half, s::cl_half)
#endif

// fmax
cl_float fmax(s::cl_float x, s::cl_float y) __NOEXC { return std::fmax(x, y); }
cl_double fmax(s::cl_double x, s::cl_double y) __NOEXC {
  return std::fmax(x, y);
}
#ifndef NO_HALF_ENABLED
cl_half fmax(s::cl_half x, s::cl_half y) __NOEXC { return std::fmax(x, y); }
#endif
MAKE_1V_2V(fmax, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(fmax, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(fmax, s::cl_half, s::cl_half, s::cl_half)
#endif

// fmin
cl_float fmin(s::cl_float x, s::cl_float y) __NOEXC { return std::fmin(x, y); }
cl_double fmin(s::cl_double x, s::cl_double y) __NOEXC {
  return std::fmin(x, y);
}
#ifndef NO_HALF_ENABLED
cl_half fmin(s::cl_half x, s::cl_half y) __NOEXC { return std::fmin(x, y); }
#endif
MAKE_1V_2V(fmin, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(fmin, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(fmin, s::cl_half, s::cl_half, s::cl_half)
#endif

// fmod
cl_float fmod(s::cl_float x, s::cl_float y) __NOEXC { return std::fmod(x, y); }
cl_double fmod(s::cl_double x, s::cl_double y) __NOEXC {
  return std::fmod(x, y);
}
#ifndef NO_HALF_ENABLED
cl_half fmod(s::cl_half x, s::cl_half y) __NOEXC { return std::fmod(x, y); }
#endif
MAKE_1V_2V(fmod, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(fmod, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(fmod, s::cl_half, s::cl_half, s::cl_half)
#endif

// nextafter
cl_float nextafter(s::cl_float x, s::cl_float y) __NOEXC {
  return std::nextafter(x, y);
}
cl_double nextafter(s::cl_double x, s::cl_double y) __NOEXC {
  return std::nextafter(x, y);
}
#ifndef __HALF_NO_ENABLED
cl_half nextafter(s::cl_half x, s::cl_half y) __NOEXC {
  return std::nextafter(x, y);
}
#endif
MAKE_1V_2V(nextafter, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(nextafter, s::cl_double, s::cl_double, s::cl_double)
#ifndef __HALF_NO_ENABLED
MAKE_1V_2V(nextafter, s::cl_half, s::cl_half, s::cl_half)
#endif

// fract
cl_float fract(s::cl_float x, s::cl_float *iptr) __NOEXC {
  return __fract(x, iptr);
}
cl_double fract(s::cl_double x, s::cl_double *iptr) __NOEXC {
  return __fract(x, iptr);
}
#ifndef __HALF_NO_ENABLED
cl_half fract(s::cl_half x, s::cl_half *iptr) __NOEXC {
  return __fract(x, iptr);
}
#endif
MAKE_1V_2P(fract, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2P(fract, s::cl_double, s::cl_double, s::cl_double)
#ifndef __HALF_NO_ENABLED
MAKE_1V_2P(fract, s::cl_half, s::cl_half, s::cl_half)
#endif

// frexp
cl_float frexp(s::cl_float x, s::cl_int *exp) __NOEXC {
  return std::frexp(x, exp);
}
cl_double frexp(s::cl_double x, s::cl_int *exp) __NOEXC {
  return std::frexp(x, exp);
}
#ifndef __HALF_NO_ENABLED
cl_half frexp(s::cl_half x, s::cl_int *exp) __NOEXC {
  return std::frexp(x, exp);
}
#endif
MAKE_1V_2P(frexp, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2P(frexp, s::cl_double, s::cl_double, s::cl_int)
#ifndef __HALF_NO_ENABLED
MAKE_1V_2P(frexp, s::cl_half, s::cl_half, s::cl_int)
#endif

// hypot
cl_float hypot(s::cl_float x, s::cl_float y) __NOEXC {
  return std::hypot(x, y);
}
cl_double hypot(s::cl_double x, s::cl_double y) __NOEXC {
  return std::hypot(x, y);
}
#ifndef NO_HALF_ENABLED
cl_half hypot(s::cl_half x, s::cl_half y) __NOEXC { return std::hypot(x, y); }
#endif
MAKE_1V_2V(hypot, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(hypot, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(hypot, s::cl_half, s::cl_half, s::cl_half)
#endif

// ilogb
cl_int ilogb(s::cl_float x) __NOEXC { return std::ilogb(x); }
cl_int ilogb(s::cl_double x) __NOEXC { return std::ilogb(x); }
#ifndef NO_HALF_ENABLED
cl_int ilogb(s::cl_half x) __NOEXC { return std::ilogb(x); }
#endif
MAKE_1V(ilogb, s::cl_int, s::cl_float)
MAKE_1V(ilogb, s::cl_int, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(ilogb, s::cl_int, s::cl_half)
#endif

// ldexp
cl_float ldexp(s::cl_float x, s::cl_int k) __NOEXC { return std::ldexp(x, k); }
cl_double ldexp(s::cl_double x, s::cl_int k) __NOEXC {
  return std::ldexp(x, k);
}
#ifndef NO_HALF_ENABLED
cl_half ldexp(s::cl_half x, s::cl_int k) __NOEXC { return std::ldexp(x, k); }
#endif
MAKE_1V_2V(ldexp, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2V(ldexp, s::cl_double, s::cl_double, s::cl_int)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(ldexp, s::cl_half, s::cl_half, s::cl_int)
#endif

// lgamma
cl_float lgamma(s::cl_float x) __NOEXC { return std::lgamma(x); }
cl_double lgamma(s::cl_double x) __NOEXC { return std::lgamma(x); }
#ifndef NO_HALF_ENABLED
cl_half lgamma(s::cl_half x) __NOEXC { return std::lgamma(x); }
#endif
MAKE_1V(lgamma, s::cl_float, s::cl_float)
MAKE_1V(lgamma, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(lgamma, s::cl_half, s::cl_half)
#endif

// lgamma_r
cl_float lgamma_r(s::cl_float x, s::cl_int *signp) __NOEXC {
  return ::lgamma_r(x, signp);
}
cl_double lgamma_r(s::cl_double x, s::cl_int *signp) __NOEXC {
  return ::lgamma_r(x, signp);
}
#ifndef __HALF_NO_ENABLED
cl_half lgamma_r(s::cl_half x, s::cl_int *signp) __NOEXC {
  return ::lgamma_r(x, signp);
}
#endif
MAKE_1V_2P(lgamma_r, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2P(lgamma_r, s::cl_double, s::cl_double, s::cl_int)
#ifndef __HALF_NO_ENABLED
MAKE_1V_2P(lgamma_r, s::cl_half, s::cl_half, s::cl_int)
#endif

// log
cl_float log(s::cl_float x) __NOEXC { return std::log(x); }
cl_double log(s::cl_double x) __NOEXC { return std::log(x); }
#ifndef NO_HALF_ENABLED
cl_half log(s::cl_half x) __NOEXC { return std::log(x); }
#endif
MAKE_1V(log, s::cl_float, s::cl_float)
MAKE_1V(log, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(log, s::cl_half, s::cl_half)
#endif

// log2
cl_float log2(s::cl_float x) __NOEXC { return std::log2(x); }
cl_double log2(s::cl_double x) __NOEXC { return std::log2(x); }
#ifndef NO_HALF_ENABLED
cl_half log2(s::cl_half x) __NOEXC { return std::log2(x); }
#endif
MAKE_1V(log2, s::cl_float, s::cl_float)
MAKE_1V(log2, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(log2, s::cl_half, s::cl_half)
#endif

// log10
cl_float log10(s::cl_float x) __NOEXC { return std::log10(x); }
cl_double log10(s::cl_double x) __NOEXC { return std::log10(x); }
#ifndef NO_HALF_ENABLED
cl_half log10(s::cl_half x) __NOEXC { return std::log10(x); }
#endif
MAKE_1V(log10, s::cl_float, s::cl_float)
MAKE_1V(log10, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(log10, s::cl_half, s::cl_half)
#endif

// log1p
cl_float log1p(s::cl_float x) __NOEXC { return std::log1p(x); }
cl_double log1p(s::cl_double x) __NOEXC { return std::log1p(x); }
#ifndef NO_HALF_ENABLED
cl_half log1p(s::cl_half x) __NOEXC { return std::log1p(x); }
#endif
MAKE_1V(log1p, s::cl_float, s::cl_float)
MAKE_1V(log1p, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(log1p, s::cl_half, s::cl_half)
#endif

// logb
cl_float logb(s::cl_float x) __NOEXC { return std::logb(x); }
cl_double logb(s::cl_double x) __NOEXC { return std::logb(x); }
#ifndef NO_HALF_ENABLED
cl_half logb(s::cl_half x) __NOEXC { return std::logb(x); }
#endif
MAKE_1V(logb, s::cl_float, s::cl_float)
MAKE_1V(logb, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(logb, s::cl_half, s::cl_half)
#endif

// mad
cl_float mad(s::cl_float a, s::cl_float b, s::cl_float c) __NOEXC {
  return __mad(a, b, c);
}
cl_double mad(s::cl_double a, s::cl_double b, s::cl_double c) __NOEXC {
  return __mad(a, b, c);
}
#ifndef NO_HALF_ENABLED
cl_half mad(s::cl_half a, s::cl_half b, s::cl_half c) __NOEXC {
  return __mad(a, b, c);
}
#endif
MAKE_1V_2V_3V(mad, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V_3V(mad, s::cl_double, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_3V(mad, s::cl_half, s::cl_half, s::cl_half, s::cl_half)
#endif

// maxmag
cl_float maxmag(s::cl_float x, s::cl_float y) __NOEXC { return __maxmag(x, y); }
cl_double maxmag(s::cl_double x, s::cl_double y) __NOEXC {
  return __maxmag(x, y);
}
#ifndef NO_HALF_ENABLED
cl_half maxmag(s::cl_half x, s::cl_half y) __NOEXC { return __maxmag(x, y); }
#endif
MAKE_1V_2V(maxmag, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(maxmag, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(maxmag, s::cl_half, s::cl_half, s::cl_half)
#endif

// minmag
cl_float minmag(s::cl_float x, s::cl_float y) __NOEXC { return __minmag(x, y); }
cl_double minmag(s::cl_double x, s::cl_double y) __NOEXC {
  return __minmag(x, y);
}
#ifndef NO_HALF_ENABLED
cl_half minmag(s::cl_half x, s::cl_half y) __NOEXC { return __minmag(x, y); }
#endif
MAKE_1V_2V(minmag, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(minmag, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(minmag, s::cl_half, s::cl_half, s::cl_half)
#endif

// modf
cl_float modf(s::cl_float x, s::cl_float *iptr) __NOEXC {
  return std::modf(x, iptr);
}
cl_double modf(s::cl_double x, s::cl_double *iptr) __NOEXC {
  return std::modf(x, iptr);
}
#ifndef __HALF_NO_ENABLED
cl_half modf(s::cl_half x, s::cl_half *iptr) __NOEXC {
  return std::modf(x, reinterpret_cast<s::cl_float *>(iptr));
}
#endif
MAKE_1V_2P(modf, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2P(modf, s::cl_double, s::cl_double, s::cl_double)
#ifndef __HALF_NO_ENABLED
MAKE_1V_2P(modf, s::cl_half, s::cl_half, s::cl_half)
#endif

// nan
cl_float nan(s::cl_uint nancode) __NOEXC { return d::quiet_NaN<float>(); }
cl_double nan(s::cl_ulong nancode) __NOEXC { return d::quiet_NaN<double>(); }
cl_double nan(s::ulonglong nancode) __NOEXC { return d::quiet_NaN<double>(); }
#ifndef __HALF_NO_ENABLED
cl_half nan(s::cl_ushort nancode) __NOEXC {
  return s::cl_half(d::quiet_NaN<float>());
}
#endif
MAKE_1V(nan, s::cl_float, s::cl_uint)
MAKE_1V(nan, s::cl_double, s::cl_ulong)
MAKE_1V(nan, s::cl_double, s::ulonglong)
#ifndef __HALF_NO_ENABLED
MAKE_1V(nan, s::cl_half, s::cl_ushort)
#endif

// pow
cl_float pow(s::cl_float x, s::cl_float y) __NOEXC { return std::pow(x, y); }
cl_double pow(s::cl_double x, s::cl_double y) __NOEXC { return std::pow(x, y); }
#ifndef __HALF_NO_ENABLED
cl_half pow(s::cl_half x, s::cl_half y) __NOEXC { return std::pow(x, y); }
#endif
MAKE_1V_2V(pow, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(pow, s::cl_double, s::cl_double, s::cl_double)
#ifndef __HALF_NO_ENABLED
MAKE_1V_2V(pow, s::cl_half, s::cl_half, s::cl_half)
#endif

// pown
cl_float pown(s::cl_float x, s::cl_int y) __NOEXC { return std::pow(x, y); }
cl_double pown(s::cl_double x, s::cl_int y) __NOEXC { return std::pow(x, y); }
#ifndef __HALF_NO_ENABLED
cl_half pown(s::cl_half x, s::cl_int y) __NOEXC { return std::pow(x, y); }
#endif
MAKE_1V_2V(pown, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2V(pown, s::cl_double, s::cl_double, s::cl_int)
#ifndef __HALF_NO_ENABLED
MAKE_1V_2V(pown, s::cl_half, s::cl_half, s::cl_int)
#endif

// powr
cl_float powr(s::cl_float x, s::cl_float y) __NOEXC { return __powr(x, y); }
cl_double powr(s::cl_double x, s::cl_double y) __NOEXC { return __powr(x, y); }
#ifndef __HALF_NO_ENABLED
cl_half powr(s::cl_half x, s::cl_half y) __NOEXC { return __powr(x, y); }
#endif
MAKE_1V_2V(powr, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(powr, s::cl_double, s::cl_double, s::cl_double)
#ifndef __HALF_NO_ENABLED
MAKE_1V_2V(powr, s::cl_half, s::cl_half, s::cl_half)
#endif

// remainder
cl_float remainder(s::cl_float x, s::cl_float y) __NOEXC {
  return std::remainder(x, y);
}
cl_double remainder(s::cl_double x, s::cl_double y) __NOEXC {
  return std::remainder(x, y);
}
#ifndef __HALF_NO_ENABLED
cl_half remainder(s::cl_half x, s::cl_half y) __NOEXC {
  return std::remainder(x, y);
}
#endif
MAKE_1V_2V(remainder, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(remainder, s::cl_double, s::cl_double, s::cl_double)
#ifndef __HALF_NO_ENABLED
MAKE_1V_2V(remainder, s::cl_half, s::cl_half, s::cl_half)
#endif

// remquo
cl_float remquo(s::cl_float x, s::cl_float y, s::cl_int *quo) __NOEXC {
  return std::remquo(x, y, quo);
}
cl_double remquo(s::cl_double x, s::cl_double y, s::cl_int *quo) __NOEXC {
  return std::remquo(x, y, quo);
}
#ifndef __HALF_NO_ENABLED
cl_half remquo(s::cl_half x, s::cl_half y, s::cl_int *quo) __NOEXC {
  return std::remquo(x, y, quo);
}
#endif
MAKE_1V_2V_3P(remquo, s::cl_float, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2V_3P(remquo, s::cl_double, s::cl_double, s::cl_double, s::cl_int)
#ifndef __HALF_NO_ENABLED
MAKE_1V_2V_3P(remquo, s::cl_half, s::cl_half, s::cl_half, s::cl_int)
#endif

// rint
cl_float rint(s::cl_float x) __NOEXC { return std::rint(x); }
cl_double rint(s::cl_double x) __NOEXC { return std::rint(x); }
#ifndef NO_HALF_ENABLED
cl_half rint(s::cl_half x) __NOEXC { return std::rint(x); }
#endif
MAKE_1V(rint, s::cl_float, s::cl_float)
MAKE_1V(rint, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(rint, s::cl_half, s::cl_half)
#endif

// rootn
cl_float rootn(s::cl_float x, s::cl_int y) __NOEXC { return __rootn(x, y); }
cl_double rootn(s::cl_double x, s::cl_int y) __NOEXC { return __rootn(x, y); }
#ifndef NO_HALF_ENABLED
cl_half rootn(s::cl_half x, s::cl_int y) __NOEXC { return __rootn(x, y); }
#endif
MAKE_1V_2V(rootn, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2V(rootn, s::cl_double, s::cl_double, s::cl_int)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(rootn, s::cl_half, s::cl_half, s::cl_int)
#endif

// round
cl_float round(s::cl_float x) __NOEXC { return std::round(x); }
cl_double round(s::cl_double x) __NOEXC { return std::round(x); }
#ifndef NO_HALF_ENABLED
cl_half round(s::cl_half x) __NOEXC { return std::round(x); }
#endif
MAKE_1V(round, s::cl_float, s::cl_float)
MAKE_1V(round, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(round, s::cl_half, s::cl_half)
#endif

// rsqrt
cl_float rsqrt(s::cl_float x) __NOEXC { return __rsqrt(x); }
cl_double rsqrt(s::cl_double x) __NOEXC { return __rsqrt(x); }
#ifndef NO_HALF_ENABLED
cl_half rsqrt(s::cl_half x) __NOEXC { return __rsqrt(x); }
#endif
MAKE_1V(rsqrt, s::cl_float, s::cl_float)
MAKE_1V(rsqrt, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(rsqrt, s::cl_half, s::cl_half)
#endif

// sin
cl_float sin(s::cl_float x) __NOEXC { return std::sin(x); }
cl_double sin(s::cl_double x) __NOEXC { return std::sin(x); }
#ifndef NO_HALF_ENABLED
cl_half sin(s::cl_half x) __NOEXC { return std::sin(x); }
#endif
MAKE_1V(sin, s::cl_float, s::cl_float)
MAKE_1V(sin, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(sin, s::cl_half, s::cl_half)
#endif

// sincos
cl_float sincos(s::cl_float x, s::cl_float *cosval) __NOEXC {
  return __sincos(x, cosval);
}
cl_double sincos(s::cl_double x, s::cl_double *cosval) __NOEXC {
  return __sincos(x, cosval);
}
#ifndef NO_HALF_ENABLED
cl_half sincos(s::cl_half x, s::cl_half *cosval) __NOEXC {
  return __sincos(x, cosval);
}
#endif
MAKE_1V_2P(sincos, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2P(sincos, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2P(sincos, s::cl_half, s::cl_half, s::cl_half)
#endif

// sinh
cl_float sinh(s::cl_float x) __NOEXC { return std::sinh(x); }
cl_double sinh(s::cl_double x) __NOEXC { return std::sinh(x); }
#ifndef NO_HALF_ENABLED
cl_half sinh(s::cl_half x) __NOEXC { return std::sinh(x); }
#endif
MAKE_1V(sinh, s::cl_float, s::cl_float)
MAKE_1V(sinh, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(sinh, s::cl_half, s::cl_half)
#endif

// sinpi
cl_float sinpi(s::cl_float x) __NOEXC { return __sinpi(x); }
cl_double sinpi(s::cl_double x) __NOEXC { return __sinpi(x); }
#ifndef NO_HALF_ENABLED
cl_half sinpi(s::cl_half x) __NOEXC { return __sinpi(x); }
#endif
MAKE_1V(sinpi, s::cl_float, s::cl_float)
MAKE_1V(sinpi, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(sinpi, s::cl_half, s::cl_half)
#endif

// sqrt
cl_float sqrt(s::cl_float x) __NOEXC { return std::sqrt(x); }
cl_double sqrt(s::cl_double x) __NOEXC { return std::sqrt(x); }
#ifndef NO_HALF_ENABLED
cl_half sqrt(s::cl_half x) __NOEXC { return std::sqrt(x); }
#endif
MAKE_1V(sqrt, s::cl_float, s::cl_float)
MAKE_1V(sqrt, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(sqrt, s::cl_half, s::cl_half)
#endif

// tan
cl_float tan(s::cl_float x) __NOEXC { return std::tan(x); }
cl_double tan(s::cl_double x) __NOEXC { return std::tan(x); }
#ifndef NO_HALF_ENABLED
cl_half tan(s::cl_half x) __NOEXC { return std::tan(x); }
#endif
MAKE_1V(tan, s::cl_float, s::cl_float)
MAKE_1V(tan, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(tan, s::cl_half, s::cl_half)
#endif

// tanh
cl_float tanh(s::cl_float x) __NOEXC { return std::tanh(x); }
cl_double tanh(s::cl_double x) __NOEXC { return std::tanh(x); }
#ifndef NO_HALF_ENABLED
cl_half tanh(s::cl_half x) __NOEXC { return std::tanh(x); }
#endif
MAKE_1V(tanh, s::cl_float, s::cl_float)
MAKE_1V(tanh, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(tanh, s::cl_half, s::cl_half)
#endif

// tanpi
cl_float tanpi(s::cl_float x) __NOEXC { return __tanpi(x); }
cl_double tanpi(s::cl_double x) __NOEXC { return __tanpi(x); }
#ifndef NO_HALF_ENABLED
cl_half tanpi(s::cl_half x) __NOEXC { return __tanpi(x); }
#endif
MAKE_1V(tanpi, s::cl_float, s::cl_float)
MAKE_1V(tanpi, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(tanpi, s::cl_half, s::cl_half)
#endif

// tgamma
cl_float tgamma(s::cl_float x) __NOEXC { return std::tgamma(x); }
cl_double tgamma(s::cl_double x) __NOEXC { return std::tgamma(x); }
#ifndef NO_HALF_ENABLED
cl_half tgamma(s::cl_half x) __NOEXC { return std::tgamma(x); }
#endif
MAKE_1V(tgamma, s::cl_float, s::cl_float)
MAKE_1V(tgamma, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(tgamma, s::cl_half, s::cl_half)
#endif

// trunc
cl_float trunc(s::cl_float x) __NOEXC { return std::trunc(x); }
cl_double trunc(s::cl_double x) __NOEXC { return std::trunc(x); }
#ifndef NO_HALF_ENABLED
cl_half trunc(s::cl_half x) __NOEXC { return std::trunc(x); }
#endif
MAKE_1V(trunc, s::cl_float, s::cl_float)
MAKE_1V(trunc, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(trunc, s::cl_half, s::cl_half)
#endif

/* --------------- 4.13.4 Integer functions. Host version -------------------*/
// u_abs
cl_uchar u_abs(s::cl_uchar x) __NOEXC { return x; }
cl_ushort u_abs(s::cl_ushort x) __NOEXC { return x; }
cl_uint u_abs(s::cl_uint x) __NOEXC { return x; }
cl_ulong u_abs(s::cl_ulong x) __NOEXC { return x; }
s::ulonglong u_abs(s::ulonglong x) __NOEXC { return x; }
MAKE_1V(u_abs, s::cl_uchar, s::cl_uchar)
MAKE_1V(u_abs, s::cl_ushort, s::cl_ushort)
MAKE_1V(u_abs, s::cl_uint, s::cl_uint)
MAKE_1V(u_abs, s::cl_ulong, s::cl_ulong)
MAKE_1V(u_abs, s::ulonglong, s::ulonglong)

// s_abs
cl_uchar s_abs(s::cl_char x) __NOEXC { return std::abs(x); }
cl_ushort s_abs(s::cl_short x) __NOEXC { return std::abs(x); }
cl_uint s_abs(s::cl_int x) __NOEXC { return std::abs(x); }
cl_ulong s_abs(s::cl_long x) __NOEXC { return std::abs(x); }
s::ulonglong s_abs(s::longlong x) __NOEXC { return std::abs(x); }
MAKE_1V(s_abs, s::cl_uchar, s::cl_char)
MAKE_1V(s_abs, s::cl_ushort, s::cl_short)
MAKE_1V(s_abs, s::cl_uint, s::cl_int)
MAKE_1V(s_abs, s::cl_ulong, s::cl_long)
MAKE_1V(s_abs, s::ulonglong, s::longlong)

// u_abs_diff
cl_uchar u_abs_diff(s::cl_uchar x, s::cl_uchar y) __NOEXC { return x - y; }
cl_ushort u_abs_diff(s::cl_ushort x, s::cl_ushort y) __NOEXC { return x - y; }
cl_uint u_abs_diff(s::cl_uint x, s::cl_uint y) __NOEXC { return x - y; }
cl_ulong u_abs_diff(s::cl_ulong x, s::cl_ulong y) __NOEXC { return x - y; }
s::ulonglong u_abs_diff(s::ulonglong x, s::ulonglong y) __NOEXC {
  return x - y;
}
MAKE_1V_2V(u_abs_diff, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_abs_diff, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_abs_diff, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_abs_diff, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2V(u_abs_diff, s::ulonglong, s::ulonglong, s::ulonglong)

// s_abs_diff
cl_uchar s_abs_diff(s::cl_char x, s::cl_char y) __NOEXC {
  return __abs_diff(x, y);
}
cl_ushort s_abs_diff(s::cl_short x, s::cl_short y) __NOEXC {
  return __abs_diff(x, y);
}
cl_uint s_abs_diff(s::cl_int x, s::cl_int y) __NOEXC {
  return __abs_diff(x, y);
}
cl_ulong s_abs_diff(s::cl_long x, s::cl_long y) __NOEXC {
  return __abs_diff(x, y);
}
s::ulonglong s_abs_diff(s::longlong x, s::longlong y) __NOEXC {
  return __abs_diff(x, y);
}
MAKE_1V_2V(s_abs_diff, s::cl_uchar, s::cl_char, s::cl_char)
MAKE_1V_2V(s_abs_diff, s::cl_ushort, s::cl_short, s::cl_short)
MAKE_1V_2V(s_abs_diff, s::cl_uint, s::cl_int, s::cl_int)
MAKE_1V_2V(s_abs_diff, s::cl_ulong, s::cl_long, s::cl_long)
MAKE_1V_2V(s_abs_diff, s::ulonglong, s::longlong, s::longlong)

// u_add_sat
cl_uchar u_add_sat(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return __u_add_sat(x, y);
}
cl_ushort u_add_sat(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __u_add_sat(x, y);
}
cl_uint u_add_sat(s::cl_uint x, s::cl_uint y) __NOEXC {
  return __u_add_sat(x, y);
}
cl_ulong u_add_sat(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return __u_add_sat(x, y);
}
s::ulonglong u_add_sat(s::ulonglong x, s::ulonglong y) __NOEXC {
  return __u_add_sat(x, y);
}
MAKE_1V_2V(u_add_sat, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_add_sat, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_add_sat, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_add_sat, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2V(u_add_sat, s::ulonglong, s::ulonglong, s::ulonglong)

// s_add_sat
cl_char s_add_sat(s::cl_char x, s::cl_char y) __NOEXC {
  return __s_add_sat(x, y);
}
cl_short s_add_sat(s::cl_short x, s::cl_short y) __NOEXC {
  return __s_add_sat(x, y);
}
cl_int s_add_sat(s::cl_int x, s::cl_int y) __NOEXC { return __s_add_sat(x, y); }
cl_long s_add_sat(s::cl_long x, s::cl_long y) __NOEXC {
  return __s_add_sat(x, y);
}
s::longlong s_add_sat(s::longlong x, s::longlong y) __NOEXC {
  return __s_add_sat(x, y);
}
MAKE_1V_2V(s_add_sat, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_add_sat, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_add_sat, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_add_sat, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2V(s_add_sat, s::longlong, s::longlong, s::longlong)

// u_hadd
cl_uchar u_hadd(s::cl_uchar x, s::cl_uchar y) __NOEXC { return __hadd(x, y); }
cl_ushort u_hadd(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __hadd(x, y);
}
cl_uint u_hadd(s::cl_uint x, s::cl_uint y) __NOEXC { return __hadd(x, y); }
cl_ulong u_hadd(s::cl_ulong x, s::cl_ulong y) __NOEXC { return __hadd(x, y); }
s::ulonglong u_hadd(s::ulonglong x, s::ulonglong y) __NOEXC {
  return __hadd(x, y);
}
MAKE_1V_2V(u_hadd, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_hadd, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_hadd, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_hadd, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2V(u_hadd, s::ulonglong, s::ulonglong, s::ulonglong)

// s_hadd
cl_char s_hadd(s::cl_char x, s::cl_char y) __NOEXC { return __hadd(x, y); }
cl_short s_hadd(s::cl_short x, s::cl_short y) __NOEXC { return __hadd(x, y); }
cl_int s_hadd(s::cl_int x, s::cl_int y) __NOEXC { return __hadd(x, y); }
cl_long s_hadd(s::cl_long x, s::cl_long y) __NOEXC { return __hadd(x, y); }
s::longlong s_hadd(s::longlong x, s::longlong y) __NOEXC {
  return __hadd(x, y);
}
MAKE_1V_2V(s_hadd, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_hadd, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_hadd, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_hadd, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2V(s_hadd, s::longlong, s::longlong, s::longlong)

// u_rhadd
cl_uchar u_rhadd(s::cl_uchar x, s::cl_uchar y) __NOEXC { return __rhadd(x, y); }
cl_ushort u_rhadd(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __rhadd(x, y);
}
cl_uint u_rhadd(s::cl_uint x, s::cl_uint y) __NOEXC { return __rhadd(x, y); }
cl_ulong u_rhadd(s::cl_ulong x, s::cl_ulong y) __NOEXC { return __rhadd(x, y); }
s::ulonglong u_rhadd(s::ulonglong x, s::ulonglong y) __NOEXC {
  return __rhadd(x, y);
}
MAKE_1V_2V(u_rhadd, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_rhadd, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_rhadd, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_rhadd, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2V(u_rhadd, s::ulonglong, s::ulonglong, s::ulonglong)

// s_rhadd
cl_char s_rhadd(s::cl_char x, s::cl_char y) __NOEXC { return __rhadd(x, y); }
cl_short s_rhadd(s::cl_short x, s::cl_short y) __NOEXC { return __rhadd(x, y); }
cl_int s_rhadd(s::cl_int x, s::cl_int y) __NOEXC { return __rhadd(x, y); }
cl_long s_rhadd(s::cl_long x, s::cl_long y) __NOEXC { return __rhadd(x, y); }
s::longlong s_rhadd(s::longlong x, s::longlong y) __NOEXC {
  return __rhadd(x, y);
}
MAKE_1V_2V(s_rhadd, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_rhadd, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_rhadd, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_rhadd, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2V(s_rhadd, s::longlong, s::longlong, s::longlong)

// u_clamp
cl_uchar u_clamp(s::cl_uchar x, s::cl_uchar minval,
                 s::cl_uchar maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
cl_ushort u_clamp(s::cl_ushort x, s::cl_ushort minval,
                  s::cl_ushort maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
cl_uint u_clamp(s::cl_uint x, s::cl_uint minval, s::cl_uint maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
cl_ulong u_clamp(s::cl_ulong x, s::cl_ulong minval,
                 s::cl_ulong maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
s::ulonglong u_clamp(s::ulonglong x, s::ulonglong minval,
                     s::ulonglong maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
MAKE_1V_2V_3V(u_clamp, s::cl_uchar, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V_3V(u_clamp, s::cl_ushort, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V_3V(u_clamp, s::cl_uint, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V_3V(u_clamp, s::cl_ulong, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2V_3V(u_clamp, s::ulonglong, s::ulonglong, s::ulonglong, s::ulonglong)
MAKE_1V_2S_3S(u_clamp, s::cl_uchar, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2S_3S(u_clamp, s::cl_ushort, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2S_3S(u_clamp, s::cl_uint, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2S_3S(u_clamp, s::cl_ulong, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2S_3S(u_clamp, s::ulonglong, s::ulonglong, s::ulonglong, s::ulonglong)

// s_clamp
cl_char s_clamp(s::cl_char x, s::cl_char minval, s::cl_char maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
cl_short s_clamp(s::cl_short x, s::cl_short minval,
                 s::cl_short maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
cl_int s_clamp(s::cl_int x, s::cl_int minval, s::cl_int maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
cl_long s_clamp(s::cl_long x, s::cl_long minval, s::cl_long maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
s::longlong s_clamp(s::longlong x, s::longlong minval,
                    s::longlong maxval) __NOEXC {
  return __clamp(x, minval, maxval);
}
MAKE_1V_2V_3V(s_clamp, s::cl_char, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V_3V(s_clamp, s::cl_short, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V_3V(s_clamp, s::cl_int, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V_3V(s_clamp, s::cl_long, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2V_3V(s_clamp, s::longlong, s::longlong, s::longlong, s::longlong)
MAKE_1V_2S_3S(s_clamp, s::cl_char, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2S_3S(s_clamp, s::cl_short, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2S_3S(s_clamp, s::cl_int, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2S_3S(s_clamp, s::cl_long, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2S_3S(s_clamp, s::longlong, s::longlong, s::longlong, s::longlong)

// clz
cl_uchar clz(s::cl_uchar x) __NOEXC { return __clz(x); }
cl_char clz(s::cl_char x) __NOEXC { return __clz(x); }
cl_ushort clz(s::cl_ushort x) __NOEXC { return __clz(x); }
cl_short clz(s::cl_short x) __NOEXC { return __clz(x); }
cl_uint clz(s::cl_uint x) __NOEXC { return __clz(x); }
cl_int clz(s::cl_int x) __NOEXC { return __clz(x); }
cl_ulong clz(s::cl_ulong x) __NOEXC { return __clz(x); }
cl_long clz(s::cl_long x) __NOEXC { return __clz(x); }
s::ulonglong clz(s::ulonglong x) __NOEXC { return __clz(x); }
s::longlong clz(s::longlong x) __NOEXC { return __clz(x); }
MAKE_1V(clz, s::cl_uchar, s::cl_uchar)
MAKE_1V(clz, s::cl_char, s::cl_char)
MAKE_1V(clz, s::cl_ushort, s::cl_ushort)
MAKE_1V(clz, s::cl_short, s::cl_short)
MAKE_1V(clz, s::cl_uint, s::cl_uint)
MAKE_1V(clz, s::cl_int, s::cl_int)
MAKE_1V(clz, s::cl_ulong, s::cl_ulong)
MAKE_1V(clz, s::cl_long, s::cl_long)
MAKE_1V(clz, s::longlong, s::longlong)
MAKE_1V(clz, s::ulonglong, s::ulonglong)

// s_mul_hi
cl_char s_mul_hi(cl_char a, cl_char b) { return __mul_hi(a, b); }
cl_short s_mul_hi(cl_short a, cl_short b) { return __mul_hi(a, b); }
cl_int s_mul_hi(cl_int a, cl_int b) { return __mul_hi(a, b); }
cl_long s_mul_hi(s::cl_long x, s::cl_long y) __NOEXC {
  return __long_mul_hi(x, y);
}
s::longlong s_mul_hi(s::longlong x, s::longlong y) __NOEXC {
  return __long_mul_hi(x, y);
}
MAKE_1V_2V(s_mul_hi, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_mul_hi, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_mul_hi, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_mul_hi, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2V(s_mul_hi, s::longlong, s::longlong, s::longlong)

// u_mul_hi
cl_uchar u_mul_hi(cl_uchar a, cl_uchar b) { return __mul_hi(a, b); }
cl_ushort u_mul_hi(cl_ushort a, cl_ushort b) { return __mul_hi(a, b); }
cl_uint u_mul_hi(cl_uint a, cl_uint b) { return __mul_hi(a, b); }
cl_ulong u_mul_hi(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return __long_mul_hi(x, y);
}
s::ulonglong u_mul_hi(s::ulonglong x, s::ulonglong y) __NOEXC {
  return __long_mul_hi(x, y);
}
MAKE_1V_2V(u_mul_hi, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_mul_hi, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_mul_hi, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_mul_hi, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2V(u_mul_hi, s::ulonglong, s::ulonglong, s::ulonglong)

// s_mad_hi
cl_char s_mad_hi(s::cl_char x, s::cl_char minval, s::cl_char maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
cl_short s_mad_hi(s::cl_short x, s::cl_short minval,
                  s::cl_short maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
cl_int s_mad_hi(s::cl_int x, s::cl_int minval, s::cl_int maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
cl_long s_mad_hi(s::cl_long x, s::cl_long minval, s::cl_long maxval) __NOEXC {
  return __long_mad_hi(x, minval, maxval);
}
s::longlong s_mad_hi(s::longlong x, s::longlong minval,
                     s::longlong maxval) __NOEXC {
  return __long_mad_hi(x, minval, maxval);
}
MAKE_1V_2V_3V(s_mad_hi, s::cl_char, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V_3V(s_mad_hi, s::cl_short, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V_3V(s_mad_hi, s::cl_int, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V_3V(s_mad_hi, s::cl_long, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2V_3V(s_mad_hi, s::longlong, s::longlong, s::longlong, s::longlong)

// u_mad_hi
cl_uchar u_mad_hi(s::cl_uchar x, s::cl_uchar minval,
                  s::cl_uchar maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
cl_ushort u_mad_hi(s::cl_ushort x, s::cl_ushort minval,
                   s::cl_ushort maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
cl_uint u_mad_hi(s::cl_uint x, s::cl_uint minval, s::cl_uint maxval) __NOEXC {
  return __mad_hi(x, minval, maxval);
}
cl_ulong u_mad_hi(s::cl_ulong x, s::cl_ulong minval,
                  s::cl_ulong maxval) __NOEXC {
  return __long_mad_hi(x, minval, maxval);
}
s::ulonglong u_mad_hi(s::ulonglong x, s::ulonglong minval,
                      s::ulonglong maxval) __NOEXC {
  return __long_mad_hi(x, minval, maxval);
}
MAKE_1V_2V_3V(u_mad_hi, s::cl_uchar, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V_3V(u_mad_hi, s::cl_ushort, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V_3V(u_mad_hi, s::cl_uint, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V_3V(u_mad_hi, s::cl_ulong, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2V_3V(u_mad_hi, s::ulonglong, s::ulonglong, s::ulonglong, s::ulonglong)

// s_mad_sat
cl_char s_mad_sat(s::cl_char a, s::cl_char b, s::cl_char c) __NOEXC {
  return __s_mad_sat(a, b, c);
}
cl_short s_mad_sat(s::cl_short a, s::cl_short b, s::cl_short c) __NOEXC {
  return __s_mad_sat(a, b, c);
}
cl_int s_mad_sat(s::cl_int a, s::cl_int b, s::cl_int c) __NOEXC {
  return __s_mad_sat(a, b, c);
}
cl_long s_mad_sat(s::cl_long a, s::cl_long b, s::cl_long c) __NOEXC {
  return __s_long_mad_sat(a, b, c);
}
s::longlong s_mad_sat(s::longlong a, s::longlong b, s::longlong c) __NOEXC {
  return __s_long_mad_sat(a, b, c);
}
MAKE_1V_2V_3V(s_mad_sat, s::cl_char, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V_3V(s_mad_sat, s::cl_short, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V_3V(s_mad_sat, s::cl_int, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V_3V(s_mad_sat, s::cl_long, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2V_3V(s_mad_sat, s::longlong, s::longlong, s::longlong, s::longlong)

// u_mad_sat
cl_uchar u_mad_sat(s::cl_uchar a, s::cl_uchar b, s::cl_uchar c) __NOEXC {
  return __u_mad_sat(a, b, c);
}
cl_ushort u_mad_sat(s::cl_ushort a, s::cl_ushort b, s::cl_ushort c) __NOEXC {
  return __u_mad_sat(a, b, c);
}
cl_uint u_mad_sat(s::cl_uint a, s::cl_uint b, s::cl_uint c) __NOEXC {
  return __u_mad_sat(a, b, c);
}
cl_ulong u_mad_sat(s::cl_ulong a, s::cl_ulong b, s::cl_ulong c) __NOEXC {
  return __u_long_mad_sat(a, b, c);
}
s::ulonglong u_mad_sat(s::ulonglong a, s::ulonglong b, s::ulonglong c) __NOEXC {
  return __u_long_mad_sat(a, b, c);
}
MAKE_1V_2V_3V(u_mad_sat, s::cl_uchar, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V_3V(u_mad_sat, s::cl_ushort, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V_3V(u_mad_sat, s::cl_uint, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V_3V(u_mad_sat, s::cl_ulong, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2V_3V(u_mad_sat, s::ulonglong, s::ulonglong, s::ulonglong, s::ulonglong)

// s_max
cl_char s_max(s::cl_char x, s::cl_char y) __NOEXC { return std::max(x, y); }
cl_short s_max(s::cl_short x, s::cl_short y) __NOEXC { return std::max(x, y); }
cl_int s_max(s::cl_int x, s::cl_int y) __NOEXC { return std::max(x, y); }
cl_long s_max(s::cl_long x, s::cl_long y) __NOEXC { return std::max(x, y); }
s::longlong s_max(s::longlong x, s::longlong y) __NOEXC {
  return std::max(x, y);
}
MAKE_1V_2V(s_max, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_max, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_max, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_max, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2V(s_max, s::longlong, s::longlong, s::longlong)
MAKE_1V_2S(s_max, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2S(s_max, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2S(s_max, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2S(s_max, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2S(s_max, s::longlong, s::longlong, s::longlong)

// u_max
cl_uchar u_max(s::cl_uchar x, s::cl_uchar y) __NOEXC { return std::max(x, y); }
cl_ushort u_max(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return std::max(x, y);
}
cl_uint u_max(s::cl_uint x, s::cl_uint y) __NOEXC { return std::max(x, y); }
cl_ulong u_max(s::cl_ulong x, s::cl_ulong y) __NOEXC { return std::max(x, y); }
s::ulonglong u_max(s::ulonglong x, s::ulonglong y) __NOEXC {
  return std::max(x, y);
}
MAKE_1V_2V(u_max, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_max, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_max, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_max, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2V(u_max, s::ulonglong, s::ulonglong, s::ulonglong)
MAKE_1V_2S(u_max, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2S(u_max, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2S(u_max, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2S(u_max, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2S(u_max, s::ulonglong, s::ulonglong, s::ulonglong)

// s_min
cl_char s_min(s::cl_char x, s::cl_char y) __NOEXC { return std::min(x, y); }
cl_short s_min(s::cl_short x, s::cl_short y) __NOEXC { return std::min(x, y); }
cl_int s_min(s::cl_int x, s::cl_int y) __NOEXC { return std::min(x, y); }
cl_long s_min(s::cl_long x, s::cl_long y) __NOEXC { return std::min(x, y); }
s::longlong s_min(s::longlong x, s::longlong y) __NOEXC {
  return std::min(x, y);
}
MAKE_1V_2V(s_min, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_min, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_min, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_min, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2V(s_min, s::longlong, s::longlong, s::longlong)
MAKE_1V_2S(s_min, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2S(s_min, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2S(s_min, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2S(s_min, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2S(s_min, s::longlong, s::longlong, s::longlong)

// u_min
cl_uchar u_min(s::cl_uchar x, s::cl_uchar y) __NOEXC { return std::min(x, y); }
cl_ushort u_min(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return std::min(x, y);
}
cl_uint u_min(s::cl_uint x, s::cl_uint y) __NOEXC { return std::min(x, y); }
cl_ulong u_min(s::cl_ulong x, s::cl_ulong y) __NOEXC { return std::min(x, y); }
s::ulonglong u_min(s::ulonglong x, s::ulonglong y) __NOEXC {
  return std::min(x, y);
}
MAKE_1V_2V(u_min, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_min, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_min, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_min, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2V(u_min, s::ulonglong, s::ulonglong, s::ulonglong)
MAKE_1V_2S(u_min, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2S(u_min, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2S(u_min, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2S(u_min, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2S(u_min, s::ulonglong, s::ulonglong, s::ulonglong)

// rotate
cl_uchar rotate(s::cl_uchar x, s::cl_uchar y) __NOEXC { return __rotate(x, y); }
cl_ushort rotate(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __rotate(x, y);
}
cl_uint rotate(s::cl_uint x, s::cl_uint y) __NOEXC { return __rotate(x, y); }
cl_ulong rotate(s::cl_ulong x, s::cl_ulong y) __NOEXC { return __rotate(x, y); }
s::ulonglong rotate(s::ulonglong x, s::ulonglong y) __NOEXC {
  return __rotate(x, y);
}
cl_char rotate(s::cl_char x, s::cl_char y) __NOEXC { return __rotate(x, y); }
cl_short rotate(s::cl_short x, s::cl_short y) __NOEXC { return __rotate(x, y); }
cl_int rotate(s::cl_int x, s::cl_int y) __NOEXC { return __rotate(x, y); }
cl_long rotate(s::cl_long x, s::cl_long y) __NOEXC { return __rotate(x, y); }
s::longlong rotate(s::longlong x, s::longlong y) __NOEXC {
  return __rotate(x, y);
}
MAKE_1V_2V(rotate, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(rotate, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(rotate, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(rotate, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2V(rotate, s::ulonglong, s::ulonglong, s::ulonglong)
MAKE_1V_2V(rotate, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(rotate, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(rotate, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(rotate, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2V(rotate, s::longlong, s::longlong, s::longlong)

// u_sub_sat
cl_uchar u_sub_sat(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return __u_sub_sat(x, y);
}
cl_ushort u_sub_sat(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __u_sub_sat(x, y);
}
cl_uint u_sub_sat(s::cl_uint x, s::cl_uint y) __NOEXC {
  return __u_sub_sat(x, y);
}
cl_ulong u_sub_sat(s::cl_ulong x, s::cl_ulong y) __NOEXC {
  return __u_sub_sat(x, y);
}
s::ulonglong u_sub_sat(s::ulonglong x, s::ulonglong y) __NOEXC {
  return __u_sub_sat(x, y);
}
MAKE_1V_2V(u_sub_sat, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_sub_sat, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_sub_sat, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_sub_sat, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2V(u_sub_sat, s::ulonglong, s::ulonglong, s::ulonglong)

// s_sub_sat
cl_char s_sub_sat(s::cl_char x, s::cl_char y) __NOEXC {
  return __s_sub_sat(x, y);
}
cl_short s_sub_sat(s::cl_short x, s::cl_short y) __NOEXC {
  return __s_sub_sat(x, y);
}
cl_int s_sub_sat(s::cl_int x, s::cl_int y) __NOEXC { return __s_sub_sat(x, y); }
cl_long s_sub_sat(s::cl_long x, s::cl_long y) __NOEXC {
  return __s_sub_sat(x, y);
}
s::longlong s_sub_sat(s::longlong x, s::longlong y) __NOEXC {
  return __s_sub_sat(x, y);
}
MAKE_1V_2V(s_sub_sat, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_sub_sat, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_sub_sat, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_sub_sat, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2V(s_sub_sat, s::longlong, s::longlong, s::longlong)

// u_upsample
cl_ushort u_upsample(s::cl_uchar x, s::cl_uchar y) __NOEXC {
  return __upsample(x, y);
}
cl_uint u_upsample(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return __upsample(x, y);
}
cl_ulong u_upsample(s::cl_uint x, s::cl_uint y) __NOEXC {
  return __upsample(x, y);
}
MAKE_1V_2V(u_upsample, s::cl_ushort, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_upsample, s::cl_uint, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_upsample, s::cl_ulong, s::cl_uint, s::cl_uint)

// TODO delete when Intel CPU OpenCL runtime will be fixed
// OpExtInst ... s_upsample -> _Z8upsampleij (now _Z8upsampleii)
#define s_upsample u_upsample

cl_short s_upsample(s::cl_char x, s::cl_uchar y) __NOEXC {
  return __upsample(x, y);
}
cl_int s_upsample(s::cl_short x, s::cl_ushort y) __NOEXC {
  return __upsample(x, y);
}
cl_long s_upsample(s::cl_int x, s::cl_uint y) __NOEXC {
  return __upsample(x, y);
}
MAKE_1V_2V(s_upsample, s::cl_short, s::cl_char, s::cl_uchar)
MAKE_1V_2V(s_upsample, s::cl_int, s::cl_short, s::cl_ushort)
MAKE_1V_2V(s_upsample, s::cl_long, s::cl_int, s::cl_uint)

#undef s_upsample

// popcount
cl_uchar popcount(s::cl_uchar x) __NOEXC { return __popcount(x); }
cl_ushort popcount(s::cl_ushort x) __NOEXC { return __popcount(x); }
cl_uint popcount(s::cl_uint x) __NOEXC { return __popcount(x); }
cl_ulong popcount(s::cl_ulong x) __NOEXC { return __popcount(x); }
s::ulonglong popcount(s::ulonglong x) __NOEXC { return __popcount(x); }
MAKE_1V(popcount, s::cl_uchar, s::cl_uchar)
MAKE_1V(popcount, s::cl_ushort, s::cl_ushort)
MAKE_1V(popcount, s::cl_uint, s::cl_uint)
MAKE_1V(popcount, s::cl_ulong, s::cl_ulong)
MAKE_1V(popcount, s::ulonglong, s::ulonglong)

cl_char popcount(s::cl_char x) __NOEXC { return __popcount(x); }
cl_short popcount(s::cl_short x) __NOEXC { return __popcount(x); }
cl_int popcount(s::cl_int x) __NOEXC { return __popcount(x); }
cl_long popcount(s::cl_long x) __NOEXC { return __popcount(x); }
s::longlong popcount(s::longlong x) __NOEXC { return __popcount(x); }
MAKE_1V(popcount, s::cl_char, s::cl_char)
MAKE_1V(popcount, s::cl_short, s::cl_short)
MAKE_1V(popcount, s::cl_int, s::cl_int)
MAKE_1V(popcount, s::cl_long, s::cl_long)
MAKE_1V(popcount, s::longlong, s::longlong)

// u_mad24
cl_uint u_mad24(s::cl_uint x, s::cl_uint y, s::cl_uint z) __NOEXC {
  return __mad24(x, y, z);
}
MAKE_1V_2V_3V(u_mad24, s::cl_uint, s::cl_uint, s::cl_uint, s::cl_uint)

// s_mad24
cl_int s_mad24(s::cl_int x, s::cl_int y, s::cl_int z) __NOEXC {
  return __mad24(x, y, z);
}
MAKE_1V_2V_3V(s_mad24, s::cl_int, s::cl_int, s::cl_int, s::cl_int)

// u_mul24
cl_uint u_mul24(s::cl_uint x, s::cl_uint y) __NOEXC { return __mul24(x, y); }
MAKE_1V_2V(u_mul24, s::cl_uint, s::cl_uint, s::cl_uint)

// s_mul24
cl_int s_mul24(s::cl_int x, s::cl_int y) __NOEXC { return __mul24(x, y); }
MAKE_1V_2V(s_mul24, s::cl_int, s::cl_int, s::cl_int)

/* --------------- 4.13.5 Common functions. Host version --------------------*/
// fclamp
cl_float fclamp(s::cl_float x, s::cl_float minval, s::cl_float maxval) __NOEXC {
  return __fclamp(x, minval, maxval);
}
cl_double fclamp(s::cl_double x, s::cl_double minval,
                 s::cl_double maxval) __NOEXC {
  return __fclamp(x, minval, maxval);
}
#ifndef NO_HALF_ENABLED
cl_half fclamp(s::cl_half x, s::cl_half minval, s::cl_half maxval) __NOEXC {
  return __fclamp(x, minval, maxval);
}
#endif
MAKE_1V_2V_3V(fclamp, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V_3V(fclamp, s::cl_double, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_3V(fclamp, s::cl_half, s::cl_half, s::cl_half, s::cl_half)
#endif

// degrees
cl_float degrees(s::cl_float radians) __NOEXC { return __degrees(radians); }
cl_double degrees(s::cl_double radians) __NOEXC { return __degrees(radians); }
#ifndef NO_HALF_ENABLED
cl_half degrees(s::cl_half radians) __NOEXC { return __degrees(radians); }
#endif
MAKE_1V(degrees, s::cl_float, s::cl_float)
MAKE_1V(degrees, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(degrees, s::cl_half, s::cl_half)
#endif

// fmin_common
cl_float fmin_common(s::cl_float x, s::cl_float y) __NOEXC {
  return std::fmin(x, y);
}
cl_double fmin_common(s::cl_double x, s::cl_double y) __NOEXC {
  return std::fmin(x, y);
}
#ifndef NO_HALF_ENABLED
cl_half fmin_common(s::cl_half x, s::cl_half y) __NOEXC {
  return std::fmin(x, y);
}
#endif
MAKE_1V_2V(fmin_common, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(fmin_common, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(fmin_common, s::cl_half, s::cl_half, s::cl_half)
#endif

// fmax_common
cl_float fmax_common(s::cl_float x, s::cl_float y) __NOEXC {
  return std::fmax(x, y);
}
cl_double fmax_common(s::cl_double x, s::cl_double y) __NOEXC {
  return std::fmax(x, y);
}
#ifndef NO_HALF_ENABLED
cl_half fmax_common(s::cl_half x, s::cl_half y) __NOEXC {
  return std::fmax(x, y);
}
#endif
MAKE_1V_2V(fmax_common, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(fmax_common, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(fmax_common, s::cl_half, s::cl_half, s::cl_half)
#endif

// mix
cl_float mix(s::cl_float x, s::cl_float y, s::cl_float a) __NOEXC {
  return __mix(x, y, a);
}
cl_double mix(s::cl_double x, s::cl_double y, s::cl_double a) __NOEXC {
  return __mix(x, y, a);
}
#ifndef NO_HALF_ENABLED
cl_half mix(s::cl_half x, s::cl_half y, s::cl_half a) __NOEXC {
  return __mix(x, y, a);
}
#endif
MAKE_1V_2V_3V(mix, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V_3V(mix, s::cl_double, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_3V(mix, s::cl_half, s::cl_half, s::cl_half, s::cl_half)
#endif

// radians
cl_float radians(s::cl_float degrees) __NOEXC { return __radians(degrees); }
cl_double radians(s::cl_double degrees) __NOEXC { return __radians(degrees); }
#ifndef NO_HALF_ENABLED
cl_half radians(s::cl_half degrees) __NOEXC { return __radians(degrees); }
#endif
MAKE_1V(radians, s::cl_float, s::cl_float)
MAKE_1V(radians, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(radians, s::cl_half, s::cl_half)
#endif

// step
cl_float step(s::cl_float edge, s::cl_float x) __NOEXC {
  return __step(edge, x);
}
cl_double step(s::cl_double edge, s::cl_double x) __NOEXC {
  return __step(edge, x);
}
#ifndef NO_HALF_ENABLED
cl_half step(s::cl_half edge, s::cl_half x) __NOEXC { return __step(edge, x); }
#endif
MAKE_1V_2V(step, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(step, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(step, s::cl_half, s::cl_half, s::cl_half)
#endif

// fma
cl_float smoothstep(s::cl_float edge0, s::cl_float edge1,
                    s::cl_float x) __NOEXC {
  return __smoothstep(edge0, edge1, x);
}
cl_double smoothstep(s::cl_double edge0, s::cl_double edge1,
                     s::cl_double x) __NOEXC {
  return __smoothstep(edge0, edge1, x);
}
#ifndef NO_HALF_ENABLED
cl_half smoothstep(s::cl_half edge0, s::cl_half edge1, s::cl_half x) __NOEXC {
  return __smoothstep(edge0, edge1, x);
}
#endif
MAKE_1V_2V_3V(smoothstep, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V_3V(smoothstep, s::cl_double, s::cl_double, s::cl_double,
              s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_3V(smoothstep, s::cl_half, s::cl_half, s::cl_half, s::cl_half)
#endif

// sign
cl_float sign(s::cl_float x) __NOEXC { return __sign(x); }
cl_double sign(s::cl_double x) __NOEXC { return __sign(x); }
#ifndef NO_HALF_ENABLED
cl_half sign(s::cl_half x) __NOEXC { return __sign(x); }
#endif
MAKE_1V(sign, s::cl_float, s::cl_float)
MAKE_1V(sign, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(sign, s::cl_half, s::cl_half)
#endif

/* --------------- 4.13.6 Geometric Functions. Host version -----------------*/
// cross
s::cl_float3 cross(s::cl_float3 p0, s::cl_float3 p1) __NOEXC {
  return __cross(p0, p1);
}
s::cl_float4 cross(s::cl_float4 p0, s::cl_float4 p1) __NOEXC {
  return __cross(p0, p1);
}
s::cl_double3 cross(s::cl_double3 p0, s::cl_double3 p1) __NOEXC {
  return __cross(p0, p1);
}
s::cl_double4 cross(s::cl_double4 p0, s::cl_double4 p1) __NOEXC {
  return __cross(p0, p1);
}
#ifndef NO_HALF_ENABLED
s::cl_half3 cross(s::cl_half3 p0, s::cl_half3 p1) __NOEXC {
  return __cross(p0, p1);
}
s::cl_half4 cross(s::cl_half4 p0, s::cl_half4 p1) __NOEXC {
  return __cross(p0, p1);
}
#endif

// OpFMul
cl_float OpFMul(s::cl_float p0, s::cl_float p1) { return __OpFMul(p0, p1); }
cl_double OpFMul(s::cl_double p0, s::cl_double p1) { return __OpFMul(p0, p1); }
#ifndef NO_HALF_ENABLED
cl_float OpFMul(s::cl_half p0, s::cl_half p1) { return __OpFMul(p0, p1); }
#endif

// OpDot
MAKE_GEO_1V_2V_RS(OpDot, __OpFMul_impl, s::cl_float, s::cl_float, s::cl_float)
MAKE_GEO_1V_2V_RS(OpDot, __OpFMul_impl, s::cl_double, s::cl_double,
                  s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_GEO_1V_2V_RS(OpDot, __OpFMul_impl, s::cl_half, s::cl_half, s::cl_half)
#endif

// length
cl_float length(s::cl_float p) { return __length(p); }
cl_double length(s::cl_double p) { return __length(p); }
#ifndef NO_HALF_ENABLED
cl_half length(s::cl_half p) { return __length(p); }
#endif
cl_float length(s::cl_float2 p) { return __length(p); }
cl_float length(s::cl_float3 p) { return __length(p); }
cl_float length(s::cl_float4 p) { return __length(p); }
cl_double length(s::cl_double2 p) { return __length(p); }
cl_double length(s::cl_double3 p) { return __length(p); }
cl_double length(s::cl_double4 p) { return __length(p); }
#ifndef NO_HALF_ENABLED
cl_half length(s::cl_half2 p) { return __length(p); }
cl_half length(s::cl_half3 p) { return __length(p); }
cl_half length(s::cl_half4 p) { return __length(p); }
#endif

// distance
cl_float distance(s::cl_float p0, s::cl_float p1) { return length(p0 - p1); }
cl_float distance(s::cl_float2 p0, s::cl_float2 p1) { return length(p0 - p1); }
cl_float distance(s::cl_float3 p0, s::cl_float3 p1) { return length(p0 - p1); }
cl_float distance(s::cl_float4 p0, s::cl_float4 p1) { return length(p0 - p1); }
cl_double distance(s::cl_double p0, s::cl_double p1) { return length(p0 - p1); }
cl_double distance(s::cl_double2 p0, s::cl_double2 p1) {
  return length(p0 - p1);
}
cl_double distance(s::cl_double3 p0, s::cl_double3 p1) {
  return length(p0 - p1);
}
cl_double distance(s::cl_double4 p0, s::cl_double4 p1) {
  return length(p0 - p1);
}
#ifndef NO_HALF_ENABLED
cl_half distance(s::cl_half p0, s::cl_half p1) { return length(p0 - p1); }
cl_half distance(s::cl_half2 p0, s::cl_half2 p1) { return length(p0 - p1); }
cl_half distance(s::cl_half3 p0, s::cl_half3 p1) { return length(p0 - p1); }
cl_half distance(s::cl_half4 p0, s::cl_half4 p1) { return length(p0 - p1); }
#endif

// normalize
s::cl_float normalize(s::cl_float p) { return __normalize(p); }
s::cl_float2 normalize(s::cl_float2 p) { return __normalize(p); }
s::cl_float3 normalize(s::cl_float3 p) { return __normalize(p); }
s::cl_float4 normalize(s::cl_float4 p) { return __normalize(p); }
s::cl_double normalize(s::cl_double p) { return __normalize(p); }
s::cl_double2 normalize(s::cl_double2 p) { return __normalize(p); }
s::cl_double3 normalize(s::cl_double3 p) { return __normalize(p); }
s::cl_double4 normalize(s::cl_double4 p) { return __normalize(p); }
#ifndef NO_HALF_ENABLED
s::cl_half normalize(s::cl_half p) { return __normalize(p); }
s::cl_half2 normalize(s::cl_half2 p) { return __normalize(p); }
s::cl_half3 normalize(s::cl_half3 p) { return __normalize(p); }
s::cl_half4 normalize(s::cl_half4 p) { return __normalize(p); }
#endif

// fast_length
cl_float fast_length(s::cl_float p) { return __fast_length(p); }
cl_float fast_length(s::cl_float2 p) { return __fast_length(p); }
cl_float fast_length(s::cl_float3 p) { return __fast_length(p); }
cl_float fast_length(s::cl_float4 p) { return __fast_length(p); }

// fast_normalize
s::cl_float fast_normalize(s::cl_float p) {
  if (p == 0.0f)
    return p;
  s::cl_float r = std::sqrt(OpFMul(p, p));
  return p / r;
}
s::cl_float2 fast_normalize(s::cl_float2 p) { return __fast_normalize(p); }
s::cl_float3 fast_normalize(s::cl_float3 p) { return __fast_normalize(p); }
s::cl_float4 fast_normalize(s::cl_float4 p) { return __fast_normalize(p); }

// fast_distance
cl_float fast_distance(s::cl_float p0, s::cl_float p1) {
  return fast_length(p0 - p1);
}
cl_float fast_distance(s::cl_float2 p0, s::cl_float2 p1) {
  return fast_length(p0 - p1);
}
cl_float fast_distance(s::cl_float3 p0, s::cl_float3 p1) {
  return fast_length(p0 - p1);
}
cl_float fast_distance(s::cl_float4 p0, s::cl_float4 p1) {
  return fast_length(p0 - p1);
}

/* --------------- 4.13.7 Relational functions. Host version --------------*/
// OpFOrdEqual-isequal
cl_int OpFOrdEqual(s::cl_float x, s::cl_float y) __NOEXC {
  return __sOpFOrdEqual(x, y);
}
cl_int OpFOrdEqual(s::cl_double x, s::cl_double y) __NOEXC {
  return __sOpFOrdEqual(x, y);
}
#ifndef NO_HALF_ENABLED
cl_int OpFOrdEqual(s::cl_half x, s::cl_half y) __NOEXC {
  return __sOpFOrdEqual(x, y);
}
#endif
MAKE_1V_2V_FUNC(OpFOrdEqual, __vOpFOrdEqual, s::cl_int, s::cl_float,
                s::cl_float)
MAKE_1V_2V_FUNC(OpFOrdEqual, __vOpFOrdEqual, s::cl_long, s::cl_double,
                s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_FUNC(OpFOrdEqual, __vOpFOrdEqual, s::cl_short, s::cl_half,
                s::cl_half)
#endif

// OpFUnordNotEqual-isnotequal
cl_int OpFUnordNotEqual(s::cl_float x, s::cl_float y) __NOEXC {
  return __sOpFUnordNotEqual(x, y);
}
cl_int OpFUnordNotEqual(s::cl_double x, s::cl_double y) __NOEXC {
  return __sOpFUnordNotEqual(x, y);
}
#ifndef NO_HALF_ENABLED
cl_int OpFUnordNotEqual(s::cl_half x, s::cl_half y) __NOEXC {
  return __sOpFUnordNotEqual(x, y);
}
#endif
MAKE_1V_2V_FUNC(OpFUnordNotEqual, __vOpFUnordNotEqual, s::cl_int, s::cl_float,
                s::cl_float)
MAKE_1V_2V_FUNC(OpFUnordNotEqual, __vOpFUnordNotEqual, s::cl_long, s::cl_double,
                s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_FUNC(OpFUnordNotEqual, __vOpFUnordNotEqual, s::cl_short, s::cl_half,
                s::cl_half)
#endif

// (OpFOrdGreaterThan)      // isgreater
cl_int OpFOrdGreaterThan(s::cl_float x, s::cl_float y) __NOEXC {
  return __sOpFOrdGreaterThan(x, y);
}
cl_int OpFOrdGreaterThan(s::cl_double x, s::cl_double y) __NOEXC {
  return __sOpFOrdGreaterThan(x, y);
}
#ifndef NO_HALF_ENABLED
cl_int OpFOrdGreaterThan(s::cl_half x, s::cl_half y) __NOEXC {
  return __sOpFOrdGreaterThan(x, y);
}
#endif
MAKE_1V_2V_FUNC(OpFOrdGreaterThan, __vOpFOrdGreaterThan, s::cl_int, s::cl_float,
                s::cl_float)
MAKE_1V_2V_FUNC(OpFOrdGreaterThan, __vOpFOrdGreaterThan, s::cl_long,
                s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_FUNC(OpFOrdGreaterThan, __vOpFOrdGreaterThan, s::cl_short,
                s::cl_half, s::cl_half)
#endif

// (OpFOrdGreaterThanEqual) // isgreaterequal
cl_int OpFOrdGreaterThanEqual(s::cl_float x, s::cl_float y) __NOEXC {
  return __sOpFOrdGreaterThanEqual(x, y);
}
cl_int OpFOrdGreaterThanEqual(s::cl_double x, s::cl_double y) __NOEXC {
  return __sOpFOrdGreaterThanEqual(x, y);
}
#ifndef NO_HALF_ENABLED
cl_int OpFOrdGreaterThanEqual(s::cl_half x, s::cl_half y) __NOEXC {
  return __sOpFOrdGreaterThanEqual(x, y);
}
#endif
MAKE_1V_2V_FUNC(OpFOrdGreaterThanEqual, __vOpFOrdGreaterThanEqual, s::cl_int,
                s::cl_float, s::cl_float)
MAKE_1V_2V_FUNC(OpFOrdGreaterThanEqual, __vOpFOrdGreaterThanEqual, s::cl_long,
                s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_FUNC(OpFOrdGreaterThanEqual, __vOpFOrdGreaterThanEqual, s::cl_short,
                s::cl_half, s::cl_half)
#endif

// (OpFOrdLessThan)         // isless
cl_int OpFOrdLessThan(s::cl_float x, s::cl_float y) __NOEXC { return (x < y); }
cl_int OpFOrdLessThan(s::cl_double x, s::cl_double y) __NOEXC {
  return (x < y);
}
cl_int __vOpFOrdLessThan(s::cl_float x, s::cl_float y) __NOEXC {
  return -(x < y);
}
cl_long __vOpFOrdLessThan(s::cl_double x, s::cl_double y) __NOEXC {
  return -(x < y);
}
#ifndef NO_HALF_ENABLED
cl_int OpFOrdLessThan(s::cl_half x, s::cl_half y) __NOEXC { return (x < y); }
cl_short __vOpFOrdLessThan(s::cl_half x, s::cl_half y) __NOEXC {
  return -(x < y);
}
#endif
MAKE_1V_2V_FUNC(OpFOrdLessThan, __vOpFOrdLessThan, s::cl_int, s::cl_float,
                s::cl_float)
MAKE_1V_2V_FUNC(OpFOrdLessThan, __vOpFOrdLessThan, s::cl_long, s::cl_double,
                s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_FUNC(OpFOrdLessThan, __vOpFOrdLessThan, s::cl_short, s::cl_half,
                s::cl_half)
#endif

// (OpFOrdLessThanEqual)    // islessequal
cl_int OpFOrdLessThanEqual(s::cl_float x, s::cl_float y) __NOEXC {
  return __sOpFOrdLessThanEqual(x, y);
}
cl_int OpFOrdLessThanEqual(s::cl_double x, s::cl_double y) __NOEXC {
  return __sOpFOrdLessThanEqual(x, y);
}
#ifndef NO_HALF_ENABLED
cl_int OpFOrdLessThanEqual(s::cl_half x, s::cl_half y) __NOEXC {
  return __sOpFOrdLessThanEqual(x, y);
}
#endif
MAKE_1V_2V_FUNC(OpFOrdLessThanEqual, __vOpFOrdLessThanEqual, s::cl_int,
                s::cl_float, s::cl_float)
MAKE_1V_2V_FUNC(OpFOrdLessThanEqual, __vOpFOrdLessThanEqual, s::cl_long,
                s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_FUNC(OpFOrdLessThanEqual, __vOpFOrdLessThanEqual, s::cl_short,
                s::cl_half, s::cl_half)
#endif

// (OpLessOrGreater)        // islessgreater
cl_int OpLessOrGreater(s::cl_float x, s::cl_float y) __NOEXC {
  return __sOpLessOrGreater(x, y);
}
cl_int OpLessOrGreater(s::cl_double x, s::cl_double y) __NOEXC {
  return __sOpLessOrGreater(x, y);
}
#ifndef NO_HALF_ENABLED
cl_int OpLessOrGreater(s::cl_half x, s::cl_half y) __NOEXC {
  return __sOpLessOrGreater(x, y);
}
#endif
MAKE_1V_2V_FUNC(OpLessOrGreater, __vOpLessOrGreater, s::cl_int, s::cl_float,
                s::cl_float)
MAKE_1V_2V_FUNC(OpLessOrGreater, __vOpLessOrGreater, s::cl_long, s::cl_double,
                s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_FUNC(OpLessOrGreater, __vOpLessOrGreater, s::cl_short, s::cl_half,
                s::cl_half)
#endif

// (OpIsFinite)             // isfinite
cl_int OpIsFinite(s::cl_float x) __NOEXC { return std::isfinite(x); }
cl_int OpIsFinite(s::cl_double x) __NOEXC { return std::isfinite(x); }
cl_int __vOpIsFinite(s::cl_float x) __NOEXC { return -(std::isfinite(x)); }
cl_long __vOpIsFinite(s::cl_double x) __NOEXC { return -(std::isfinite(x)); }
#ifndef NO_HALF_ENABLED
cl_int OpIsFinite(s::cl_half x) __NOEXC { return std::isfinite(x); }
cl_short __vOpIsFinite(s::cl_half x) __NOEXC { return -(std::isfinite(x)); }
#endif
MAKE_1V_FUNC(OpIsFinite, __vOpIsFinite, s::cl_int, s::cl_float)
MAKE_1V_FUNC(OpIsFinite, __vOpIsFinite, s::cl_long, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_FUNC(OpIsFinite, __vOpIsFinite, s::cl_short, s::cl_half)
#endif

// (OpIsInf)                // isinf
cl_int OpIsInf(s::cl_float x) __NOEXC { return std::isinf(x); }
cl_int OpIsInf(s::cl_double x) __NOEXC { return std::isinf(x); }
cl_int __vOpIsInf(s::cl_float x) __NOEXC { return -(std::isinf(x)); }
cl_long __vOpIsInf(s::cl_double x) __NOEXC { return -(std::isinf(x)); }
#ifndef NO_HALF_ENABLED
cl_int OpIsInf(s::cl_half x) __NOEXC { return std::isinf(x); }
cl_short __vOpIsInf(s::cl_half x) __NOEXC { return -(std::isinf(x)); }
#endif
MAKE_1V_FUNC(OpIsInf, __vOpIsInf, s::cl_int, s::cl_float)
MAKE_1V_FUNC(OpIsInf, __vOpIsInf, s::cl_long, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_FUNC(OpIsInf, __vOpIsInf, s::cl_short, s::cl_half)
#endif

// (OpIsNan)                // isnan
cl_int OpIsNan(s::cl_float x) __NOEXC { return std::isnan(x); }
cl_int OpIsNan(s::cl_double x) __NOEXC { return std::isnan(x); }
cl_int __vOpIsNan(s::cl_float x) __NOEXC { return -(std::isnan(x)); }
cl_long __vOpIsNan(s::cl_double x) __NOEXC { return -(std::isnan(x)); }

#ifndef NO_HALF_ENABLED
cl_int OpIsNan(s::cl_half x) __NOEXC { return std::isnan(x); }
cl_short __vOpIsNan(s::cl_half x) __NOEXC { return -(std::isnan(x)); }
#endif
MAKE_1V_FUNC(OpIsNan, __vOpIsNan, s::cl_int, s::cl_float)
MAKE_1V_FUNC(OpIsNan, __vOpIsNan, s::cl_long, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_FUNC(OpIsNan, __vOpIsNan, s::cl_short, s::cl_half)
#endif

// (OpIsNormal)             // isnormal
cl_int OpIsNormal(s::cl_float x) __NOEXC { return std::isnormal(x); }
cl_int OpIsNormal(s::cl_double x) __NOEXC { return std::isnormal(x); }
cl_int __vOpIsNormal(s::cl_float x) __NOEXC { return -(std::isnormal(x)); }
cl_long __vOpIsNormal(s::cl_double x) __NOEXC { return -(std::isnormal(x)); }
#ifndef NO_HALF_ENABLED
cl_int OpIsNormal(s::cl_half x) __NOEXC { return std::isnormal(x); }
cl_short __vOpIsNormal(s::cl_half x) __NOEXC { return -(std::isnormal(x)); }
#endif
MAKE_1V_FUNC(OpIsNormal, __vOpIsNormal, s::cl_int, s::cl_float)
MAKE_1V_FUNC(OpIsNormal, __vOpIsNormal, s::cl_long, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_FUNC(OpIsNormal, __vOpIsNormal, s::cl_short, s::cl_half)
#endif

// (OpOrdered)              // isordered
cl_int OpOrdered(s::cl_float x, s::cl_float y) __NOEXC {
  return __vOpOrdered(x, y);
}
cl_int OpOrdered(s::cl_double x, s::cl_double y) __NOEXC {
  return __vOpOrdered(x, y);
}
#ifndef NO_HALF_ENABLED
cl_int OpOrdered(s::cl_half x, s::cl_half y) __NOEXC {
  return __vOpOrdered(x, y);
}
#endif
MAKE_1V_2V_FUNC(OpOrdered, __vOpOrdered, s::cl_int, s::cl_float, s::cl_float)
MAKE_1V_2V_FUNC(OpOrdered, __vOpOrdered, s::cl_long, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_FUNC(OpOrdered, __vOpOrdered, s::cl_short, s::cl_half, s::cl_half)
#endif

// (OpUnordered)            // isunordered
cl_int OpUnordered(s::cl_float x, s::cl_float y) __NOEXC {
  return __sOpUnordered(x, y);
}
cl_int OpUnordered(s::cl_double x, s::cl_double y) __NOEXC {
  return __sOpUnordered(x, y);
}
#ifndef NO_HALF_ENABLED
cl_int OpUnordered(s::cl_half x, s::cl_half y) __NOEXC {
  return __sOpUnordered(x, y);
}
#endif
MAKE_1V_2V_FUNC(OpUnordered, __vOpUnordered, s::cl_int, s::cl_float,
                s::cl_float)
MAKE_1V_2V_FUNC(OpUnordered, __vOpUnordered, s::cl_long, s::cl_double,
                s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_FUNC(OpUnordered, __vOpUnordered, s::cl_short, s::cl_half,
                s::cl_half)
#endif

// (OpSignBitSet)           // signbit
cl_int OpSignBitSet(s::cl_float x) __NOEXC { return std::signbit(x); }
cl_int OpSignBitSet(s::cl_double x) __NOEXC { return std::signbit(x); }
cl_int __vOpSignBitSet(s::cl_float x) __NOEXC { return -(std::signbit(x)); }
cl_long __vOpSignBitSet(s::cl_double x) __NOEXC { return -(std::signbit(x)); }
#ifndef NO_HALF_ENABLED
cl_int OpSignBitSet(s::cl_half x) __NOEXC { return std::signbit(x); }
cl_short __vOpSignBitSet(s::cl_half x) __NOEXC { return -(std::signbit(x)); }
#endif
MAKE_1V_FUNC(OpSignBitSet, __vOpSignBitSet, s::cl_int, s::cl_float)
MAKE_1V_FUNC(OpSignBitSet, __vOpSignBitSet, s::cl_long, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_FUNC(OpSignBitSet, __vOpSignBitSet, s::cl_short, s::cl_half)
#endif

// (OpAny)                  // any

MAKE_SR_1V_OR(OpAny, __OpAny, s::cl_int, s::cl_char)
MAKE_SR_1V_OR(OpAny, __OpAny, s::cl_int, s::cl_short)
MAKE_SR_1V_OR(OpAny, __OpAny, s::cl_int, s::cl_int)
MAKE_SR_1V_OR(OpAny, __OpAny, s::cl_int, s::cl_long)
MAKE_SR_1V_OR(OpAny, __OpAny, s::cl_int, s::longlong)

// (OpAll)                  // all

MAKE_SR_1V_AND(OpAll, __OpAll, s::cl_int, s::cl_char)
MAKE_SR_1V_AND(OpAll, __OpAll, s::cl_int, s::cl_short)
MAKE_SR_1V_AND(OpAll, __OpAll, s::cl_int, s::cl_int)
MAKE_SR_1V_AND(OpAll, __OpAll, s::cl_int, s::cl_long)
MAKE_SR_1V_AND(OpAll, __OpAll, s::cl_int, s::longlong)

// (bitselect)
// Instantiate functions for the scalar types and vector types.
MAKE_SC_1V_2V_3V(bitselect, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_SC_1V_2V_3V(bitselect, s::cl_double, s::cl_double, s::cl_double,
                 s::cl_double)
MAKE_SC_1V_2V_3V(bitselect, s::cl_char, s::cl_char, s::cl_char, s::cl_char)
MAKE_SC_1V_2V_3V(bitselect, s::cl_uchar, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_SC_1V_2V_3V(bitselect, s::cl_short, s::cl_short, s::cl_short, s::cl_short)
MAKE_SC_1V_2V_3V(bitselect, s::cl_ushort, s::cl_ushort, s::cl_ushort,
                 s::cl_ushort)
MAKE_SC_1V_2V_3V(bitselect, s::cl_int, s::cl_int, s::cl_int, s::cl_int)
MAKE_SC_1V_2V_3V(bitselect, s::cl_uint, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_SC_1V_2V_3V(bitselect, s::cl_long, s::cl_long, s::cl_long, s::cl_long)
MAKE_SC_1V_2V_3V(bitselect, s::cl_ulong, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_SC_1V_2V_3V(bitselect, s::longlong, s::longlong, s::longlong, s::longlong)
MAKE_SC_1V_2V_3V(bitselect, s::ulonglong, s::ulonglong, s::ulonglong,
                 s::ulonglong)
#ifndef NO_HALF_ENABLED
MAKE_SC_1V_2V_3V(bitselect, s::cl_half, s::cl_half, s::cl_half, s::cl_half)
#endif

// (OpSelect) // select
// for scalar: result = c ? b : a.
// for vector: result[i] = (MSB of c[i] is set)? b[i] : a[i]
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_float, s::cl_int,
                        s::cl_float, s::cl_float)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_float, s::cl_uint,
                        s::cl_float, s::cl_float)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_double, s::cl_long,
                        s::cl_double, s::cl_double)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_double, s::cl_ulong,
                        s::cl_double, s::cl_double)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_double, s::longlong,
                        s::cl_double, s::cl_double)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_double, s::ulonglong,
                        s::cl_double, s::cl_double)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_char, s::cl_char,
                        s::cl_char, s::cl_char)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_char, s::cl_uchar,
                        s::cl_char, s::cl_char)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_uchar, s::cl_char,
                        s::cl_uchar, s::cl_uchar)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_uchar, s::cl_uchar,
                        s::cl_uchar, s::cl_uchar)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_short, s::cl_short,
                        s::cl_short, s::cl_short)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_short, s::cl_ushort,
                        s::cl_short, s::cl_short)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_ushort, s::cl_short,
                        s::cl_ushort, s::cl_ushort)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_ushort, s::cl_ushort,
                        s::cl_ushort, s::cl_ushort)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_int, s::cl_int, s::cl_int,
                        s::cl_int)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_int, s::cl_uint, s::cl_int,
                        s::cl_int)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_uint, s::cl_int,
                        s::cl_uint, s::cl_uint)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_uint, s::cl_uint,
                        s::cl_uint, s::cl_uint)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_long, s::cl_long,
                        s::cl_long, s::cl_long)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_long, s::cl_ulong,
                        s::cl_long, s::cl_long)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_ulong, s::cl_long,
                        s::cl_ulong, s::cl_ulong)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_ulong, s::cl_ulong,
                        s::cl_ulong, s::cl_ulong)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::longlong, s::longlong,
                        s::longlong, s::longlong)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::longlong, s::ulonglong,
                        s::longlong, s::longlong)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::ulonglong, s::ulonglong,
                        s::ulonglong, s::ulonglong)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::ulonglong, s::longlong,
                        s::ulonglong, s::ulonglong)
#ifndef NO_HALF_ENABLED
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_half, s::cl_short,
                        s::cl_half, s::cl_half)
MAKE_SC_FSC_1V_2V_3V_FV(OpSelect, __vOpSelect, s::cl_half, s::cl_ushort,
                        s::cl_half, s::cl_half)
#endif

/* --------------- 4.13.3 Native Math functions. Host version ---------------*/
// native_cos
cl_float native_cos(s::cl_float x) __NOEXC { return std::cos(x); }
MAKE_1V(native_cos, s::cl_float, s::cl_float)

// native_divide
cl_float native_divide(s::cl_float x, s::cl_float y) __NOEXC { return x / y; }
MAKE_1V_2V(native_divide, s::cl_float, s::cl_float, s::cl_float)

// native_exp
cl_float native_exp(s::cl_float x) __NOEXC { return std::exp(x); }
MAKE_1V(native_exp, s::cl_float, s::cl_float)

// native_exp2
cl_float native_exp2(s::cl_float x) __NOEXC { return std::exp2(x); }
MAKE_1V(native_exp2, s::cl_float, s::cl_float)

// native_exp10
cl_float native_exp10(s::cl_float x) __NOEXC { return std::pow(10, x); }
MAKE_1V(native_exp10, s::cl_float, s::cl_float)

// native_log
cl_float native_log(s::cl_float x) __NOEXC { return std::log(x); }
MAKE_1V(native_log, s::cl_float, s::cl_float)

// native_log2
cl_float native_log2(s::cl_float x) __NOEXC { return std::log2(x); }
MAKE_1V(native_log2, s::cl_float, s::cl_float)

// native_log10
cl_float native_log10(s::cl_float x) __NOEXC { return std::log10(x); }
MAKE_1V(native_log10, s::cl_float, s::cl_float)

// native_powr
cl_float native_powr(s::cl_float x, s::cl_float y) __NOEXC {
  return (x >= 0 ? std::pow(x, y) : x);
}
MAKE_1V_2V(native_powr, s::cl_float, s::cl_float, s::cl_float)

// native_recip
cl_float native_recip(s::cl_float x) __NOEXC { return 1.0 / x; }
MAKE_1V(native_recip, s::cl_float, s::cl_float)

// native_rsqrt
cl_float native_rsqrt(s::cl_float x) __NOEXC { return 1.0 / std::sqrt(x); }
MAKE_1V(native_rsqrt, s::cl_float, s::cl_float)

// native_sin
cl_float native_sin(s::cl_float x) __NOEXC { return std::sin(x); }
MAKE_1V(native_sin, s::cl_float, s::cl_float)

// native_sqrt
cl_float native_sqrt(s::cl_float x) __NOEXC { return std::sqrt(x); }
MAKE_1V(native_sqrt, s::cl_float, s::cl_float)

// native_tan
cl_float native_tan(s::cl_float x) __NOEXC { return std::tan(x); }
MAKE_1V(native_tan, s::cl_float, s::cl_float)

/* ----------------- 4.13.3 Half Precision Math functions. Host version -----*/
// half_cos
cl_float half_cos(s::cl_float x) __NOEXC { return std::cos(x); }
MAKE_1V(half_cos, s::cl_float, s::cl_float)

// half_divide
cl_float half_divide(s::cl_float x, s::cl_float y) __NOEXC { return x / y; }
MAKE_1V_2V(half_divide, s::cl_float, s::cl_float, s::cl_float)

// half_exp
cl_float half_exp(s::cl_float x) __NOEXC { return std::exp(x); }
MAKE_1V(half_exp, s::cl_float, s::cl_float)
// half_exp2
cl_float half_exp2(s::cl_float x) __NOEXC { return std::exp2(x); }
MAKE_1V(half_exp2, s::cl_float, s::cl_float)

// half_exp10
cl_float half_exp10(s::cl_float x) __NOEXC { return std::pow(10, x); }
MAKE_1V(half_exp10, s::cl_float, s::cl_float)
// half_log
cl_float half_log(s::cl_float x) __NOEXC { return std::log(x); }
MAKE_1V(half_log, s::cl_float, s::cl_float)

// half_log2
cl_float half_log2(s::cl_float x) __NOEXC { return std::log2(x); }
MAKE_1V(half_log2, s::cl_float, s::cl_float)

// half_log10
cl_float half_log10(s::cl_float x) __NOEXC { return std::log10(x); }
MAKE_1V(half_log10, s::cl_float, s::cl_float)

// half_powr
cl_float half_powr(s::cl_float x, s::cl_float y) __NOEXC {
  return (x >= 0 ? std::pow(x, y) : x);
}
MAKE_1V_2V(half_powr, s::cl_float, s::cl_float, s::cl_float)

// half_recip
cl_float half_recip(s::cl_float x) __NOEXC { return 1.0 / x; }
MAKE_1V(half_recip, s::cl_float, s::cl_float)

// half_rsqrt
cl_float half_rsqrt(s::cl_float x) __NOEXC { return 1.0 / std::sqrt(x); }
MAKE_1V(half_rsqrt, s::cl_float, s::cl_float)

// half_sin
cl_float half_sin(s::cl_float x) __NOEXC { return std::sin(x); }
MAKE_1V(half_sin, s::cl_float, s::cl_float)

// half_sqrt
cl_float half_sqrt(s::cl_float x) __NOEXC { return std::sqrt(x); }
MAKE_1V(half_sqrt, s::cl_float, s::cl_float)

// half_tan
cl_float half_tan(s::cl_float x) __NOEXC { return std::tan(x); }
MAKE_1V(half_tan, s::cl_float, s::cl_float)
} // namespace __host_std
} // namespace cl

#undef __NOEXC
#undef NO_HALF_ENABLED
#undef __MAKE_1V
#undef __MAKE_1V_2V
#undef __MAKE_1V_2V_RS
#undef __MAKE_1V_RS
#undef __MAKE_1V_2V_3V
#undef __MAKE_1V_2S
#undef __MAKE_SR_1V_AND
#undef __MAKE_SR_1V_OR
#undef __MAKE_1V_2P
#undef __MAKE_1V_2V_3P
#undef MAKE_1V
#undef MAKE_1V_FUNC
#undef MAKE_1V_2V
#undef MAKE_1V_2V_FUNC
#undef MAKE_1V_2V_3V
#undef MAKE_1V_2V_3V_FUNC
#undef MAKE_SC_1V_2V_3V
#undef MAKE_SC_FSC_1V_2V_3V_FV
#undef MAKE_SC_3ARG
#undef MAKE_1V_2S
#undef MAKE_SR_1V_AND
#undef MAKE_SR_1V_OR
#undef MAKE_1V_2P
#undef MAKE_GEO_1V_2V_RS
#undef MAKE_1V_2V_3P
