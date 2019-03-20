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
#include <limits>

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

// TODO Remove when half type will supported by SYCL Runtime
#define NO_HALF_ENABLED

namespace s = cl::sycl;

#define __MAKE_1V(Fun, N, Ret, Arg1)                                           \
  Ret##N Fun __NOEXC(Arg1##N x) {                                              \
    Ret##N r;                                                                  \
    using base_t = typename Arg1##N::element_type;                             \
    detail::helper<N - 1>().run_1v(                                            \
        r, [](base_t x) { return cl::__host_std::Fun(x); }, x);                \
    return r;                                                                  \
  }

#define __MAKE_1V_2V(Fun, N, Ret, Arg1, Arg2)                                  \
  Ret##N Fun __NOEXC(Arg1##N x, Arg2##N y) {                                   \
    Ret##N r;                                                                  \
    using base1_t = typename Arg1##N::element_type;                            \
    using base2_t = typename Arg2##N::element_type;                            \
    detail::helper<N - 1>().run_1v_2v(                                         \
        r, [](base1_t x, base2_t y) { return cl::__host_std::Fun(x, y); }, x,  \
        y);                                                                    \
    return r;                                                                  \
  }

#define __MAKE_1V_2V_RS(Fun, Call, N, Ret, Arg1, Arg2)                         \
  Ret Fun __NOEXC(Arg1##N x, Arg2##N y) {                                      \
    Ret r = Ret();                                                             \
    using base1_t = typename Arg1##N::element_type;                            \
    using base2_t = typename Arg2##N::element_type;                            \
    detail::helper<N - 1>().run_1v_2v_rs(                                      \
        r,                                                                     \
        [](Ret &r, base1_t x, base2_t y) {                                     \
          return cl::__host_std::Call(r, x, y);                                \
        },                                                                     \
        x, y);                                                                 \
    return r;                                                                  \
  }

#define __MAKE_1V_2V_3V(Fun, N, Ret, Arg1, Arg2, Arg3)                         \
  Ret##N Fun __NOEXC(Arg1##N x, Arg2##N y, Arg3##N z) {                        \
    Ret##N r;                                                                  \
    using base1_t = typename Arg1##N::element_type;                            \
    using base2_t = typename Arg2##N::element_type;                            \
    using base3_t = typename Arg3##N::element_type;                            \
    detail::helper<N - 1>().run_1v_2v_3v(                                      \
        r,                                                                     \
        [](base1_t x, base2_t y, base3_t z) {                                  \
          return cl::__host_std::Fun(x, y, z);                                 \
        },                                                                     \
        x, y, z);                                                              \
    return r;                                                                  \
  }

#define __MAKE_1V_2S(Fun, N, Ret, Arg1, Arg2)                                  \
  Ret##N Fun __NOEXC(Arg1##N x, Arg2 y) {                                      \
    Ret##N r;                                                                  \
    using base1_t = typename Arg1##N::element_type;                            \
    detail::helper<N - 1>().run_1v_2s(                                         \
        r, [](base1_t x, Arg2 y) { return cl::__host_std::Fun(x, y); }, x, y); \
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
        r,                                                                     \
        [](base1_t x, base2_t y, base3_t *z) {                                 \
          return cl::__host_std::Fun(x, y, z);                                 \
        },                                                                     \
        x, y, z);                                                              \
    return r;                                                                  \
  }

#define MAKE_1V(Fun, Ret, Arg1)                                                \
  __MAKE_1V(Fun, 2, Ret, Arg1)                                                 \
  __MAKE_1V(Fun, 3, Ret, Arg1)                                                 \
  __MAKE_1V(Fun, 4, Ret, Arg1)                                                 \
  __MAKE_1V(Fun, 8, Ret, Arg1)                                                 \
  __MAKE_1V(Fun, 16, Ret, Arg1)

#define MAKE_1V_2V(Fun, Ret, Arg1, Arg2)                                       \
  __MAKE_1V_2V(Fun, 2, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2V(Fun, 3, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2V(Fun, 4, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2V(Fun, 8, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2V(Fun, 16, Ret, Arg1, Arg2)

#define MAKE_1V_2V_3V(Fun, Ret, Arg1, Arg2, Arg3)                              \
  __MAKE_1V_2V_3V(Fun, 2, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2V_3V(Fun, 3, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2V_3V(Fun, 4, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2V_3V(Fun, 8, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2V_3V(Fun, 16, Ret, Arg1, Arg2, Arg3)

#define MAKE_1V_2S(Fun, Ret, Arg1, Arg2)                                       \
  __MAKE_1V_2S(Fun, 2, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2S(Fun, 3, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2S(Fun, 4, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2S(Fun, 8, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2S(Fun, 16, Ret, Arg1, Arg2)

#define MAKE_1V_2P(Fun, Ret, Arg1, Arg2)                                       \
  __MAKE_1V_2P(Fun, 2, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2P(Fun, 3, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2P(Fun, 4, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2P(Fun, 8, Ret, Arg1, Arg2)                                        \
  __MAKE_1V_2P(Fun, 16, Ret, Arg1, Arg2)

#define MAKE_1V_2V_RS(Fun, Call, Ret, Arg1, Arg2)                              \
  __MAKE_1V_2V_RS(Fun, Call, 2, Ret, Arg1, Arg2)                               \
  __MAKE_1V_2V_RS(Fun, Call, 3, Ret, Arg1, Arg2)                               \
  __MAKE_1V_2V_RS(Fun, Call, 4, Ret, Arg1, Arg2)                               \
  __MAKE_1V_2V_RS(Fun, Call, 8, Ret, Arg1, Arg2)                               \
  __MAKE_1V_2V_RS(Fun, Call, 16, Ret, Arg1, Arg2)

#define MAKE_1V_2V_3P(Fun, Ret, Arg1, Arg2, Arg3)                              \
  __MAKE_1V_2V_3P(Fun, 2, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2V_3P(Fun, 3, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2V_3P(Fun, 4, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2V_3P(Fun, 8, Ret, Arg1, Arg2, Arg3)                               \
  __MAKE_1V_2V_3P(Fun, 16, Ret, Arg1, Arg2, Arg3)

namespace cl {
namespace __host_std {
namespace detail {

template <int N> struct helper {
  template <typename Res, typename Op, typename T1>
  void run_1v(Res &r, Op op, T1 x) {
    helper<N - 1>().run_1v(r, op, x);
    r.template swizzle<N>() = op(x.template swizzle<N>());
  }
  template <typename Res, typename Op, typename T1, typename T2>
  void run_1v_2v(Res &r, Op op, T1 x, T2 y) {
    helper<N - 1>().run_1v_2v(r, op, x, y);
    r.template swizzle<N>() =
        op(x.template swizzle<N>(), y.template swizzle<N>());
  }
  template <typename Res, typename Op, typename T1, typename T2>
  void run_1v_2s(Res &r, Op op, T1 x, T2 y) {
    helper<N - 1>().run_1v_2s(r, op, x, y);
    r.template swizzle<N>() = op(x.template swizzle<N>(), y);
  }

  template <typename Res, typename Op, typename T1, typename T2>
  void run_1v_2v_rs(Res &r, Op op, T1 x, T2 y) {
    helper<N - 1>().run_1v_2v_rs(r, op, x, y);
    op(r, x.template swizzle<N>(), y.template swizzle<N>());
  }

  template <typename Res, typename Op, typename T1, typename T2>
  void run_1v_2p(Res &r, Op op, T1 x, T2 y) {
    helper<N - 1>().run_1v_2p(r, op, x, y);
    // TODO avoid creating a temporary variable
    typename std::remove_pointer<T2>::type::element_type temp;
    r.template swizzle<N>() = op(x.template swizzle<N>(), &temp);
    y->template swizzle<N>() = temp;
  }

  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  void run_1v_2v_3p(Res &r, Op op, T1 x, T2 y, T3 z) {
    helper<N - 1>().run_1v_2v_3p(r, op, x, y, z);
    // TODO avoid creating a temporary variable
    typename std::remove_pointer<T3>::type::element_type temp;
    r.template swizzle<N>() =
        op(x.template swizzle<N>(), y.template swizzle<N>(), &temp);
    z->template swizzle<N>() = temp;
  }

  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  void run_1v_2v_3v(Res &r, Op op, T1 x, T2 y, T3 z) {
    helper<N - 1>().run_1v_2v_3v(r, op, x, y, z);
    r.template swizzle<N>() =
        op(x.template swizzle<N>(), y.template swizzle<N>(),
           z.template swizzle<N>());
  }
};

template <> struct helper<0> {
  template <typename Res, typename Op, typename T1>
  void run_1v(Res &r, Op op, T1 x) {
    r.template swizzle<0>() = op(x.template swizzle<0>());
  }
  template <typename Res, typename Op, typename T1, typename T2>
  void run_1v_2v(Res &r, Op op, T1 x, T2 y) {
    r.template swizzle<0>() =
        op(x.template swizzle<0>(), y.template swizzle<0>());
  }
  template <typename Res, typename Op, typename T1, typename T2>
  void run_1v_2s(Res &r, Op op, T1 x, T2 y) {
    r.template swizzle<0>() = op(x.template swizzle<0>(), y);
  }
  template <typename Res, typename Op, typename T1, typename T2>
  void run_1v_2v_rs(Res &r, Op op, T1 x, T2 y) {
    op(r, x.template swizzle<0>(), y.template swizzle<0>());
  }
  template <typename Res, typename Op, typename T1, typename T2>
  void run_1v_2p(Res &r, Op op, T1 x, T2 y) {
    // TODO avoid creating a temporary variable
    typename std::remove_pointer<T2>::type::element_type temp;
    r.template swizzle<0>() = op(x.template swizzle<0>(), &temp);
    y->template swizzle<0>() = temp;
  }
  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  void run_1v_2v_3p(Res &r, Op op, T1 x, T2 y, T3 z) {
    // TODO avoid creating a temporary variable
    typename std::remove_pointer<T3>::type::element_type temp;
    r.template swizzle<0>() =
        op(x.template swizzle<0>(), y.template swizzle<0>(), &temp);
    z->template swizzle<0>() = temp;
  }
  template <typename Res, typename Op, typename T1, typename T2, typename T3>
  void run_1v_2v_3v(Res &r, Op op, T1 x, T2 y, T3 z) {
    r.template swizzle<0>() =
        op(x.template swizzle<0>(), y.template swizzle<0>(),
           z.template swizzle<0>());
  }
};
} // namespace detail
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
cl_float acospi(s::cl_float x) __NOEXC { return std::acos(x) / M_PI; }
cl_double acospi(s::cl_double x) __NOEXC { return std::acos(x) / M_PI; }
#ifndef NO_HALF_ENABLED
cl_half acospi(s::cl_half x) __NOEXC { return std::acos(x) / M_PI; }
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
cl_float asinpi(s::cl_float x) __NOEXC { return std::asin(x) / M_PI; }
cl_double asinpi(s::cl_double x) __NOEXC { return std::asin(x) / M_PI; }
#ifndef NO_HALF_ENABLED
cl_half asinpi(s::cl_half x) __NOEXC { return std::asin(x) / M_PI; }
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
cl_float atanpi(s::cl_float x) __NOEXC { return std::atan(x) / M_PI; }
cl_double atanpi(s::cl_double x) __NOEXC { return std::atan(x) / M_PI; }
#ifndef NO_HALF_ENABLED
cl_half atanpi(s::cl_half x) __NOEXC { return std::atan(x) / M_PI; }
#endif
MAKE_1V(atanpi, s::cl_float, s::cl_float)
MAKE_1V(atanpi, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(atanpi, s::cl_half, s::cl_half)
#endif

// atan2pi
cl_float atan2pi(s::cl_float x, s::cl_float y) __NOEXC {
  return std::atan2(x, y) / M_PI;
}
cl_double atan2pi(s::cl_double x, s::cl_double y) __NOEXC {
  return std::atan2(x, y) / M_PI;
}
#ifndef NO_HALF_ENABLED
cl_half atan2pi(s::cl_half x, s::cl_half y) __NOEXC {
  return std::atan2(x, y) / M_PI;
}
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
cl_float cospi(s::cl_float x) __NOEXC { return std::cos(M_PI * x); }
cl_double cospi(s::cl_double x) __NOEXC { return std::cos(M_PI * x); }
#ifndef NO_HALF_ENABLED
cl_half cospi(s::cl_half x) __NOEXC { return std::cos(M_PI * x); }
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
/* fract - disabled until proper C++11 compatible implementation
cl_float fract(s::cl_float x, s::cl_float *iptr) __NOEXC {
  decltype(x) f = std::floor(x);
  iptr[0] = f;
  return std::fmin(x - f, 0x1.fffffep-1f);
}
cl_double fract(s::cl_double x, s::cl_double *iptr) __NOEXC {
  decltype(x) f = std::floor(x);
  iptr[0] = f;
  return std::fmin(x - f, 0x1.fffffep-1f);
}
#ifdef __HAFL_ENABLED
cl_half fract(s::cl_half x, s::cl_half *iptr) __NOEXC {
  decltype(x) f = std::floor(x);
  iptr[0] = f;
  return std::fmin(x - f, 0x1.fffffep-1f);
}
#endif
MAKE_1V_2P(fract, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2P(fract, s::cl_double, s::cl_double, s::cl_double)
#ifdef __HAFL_ENABLED
MAKE_1V_2P(fract, s::cl_half, s::cl_half, s::cl_half)
#endif
*/
// frexp
cl_float frexp(s::cl_float x, s::cl_int *exp) __NOEXC {
  return std::frexp(x, exp);
}
cl_double frexp(s::cl_double x, s::cl_int *exp) __NOEXC {
  return std::frexp(x, exp);
}
#ifdef __HAFL_ENABLED
cl_half frexp(s::cl_half x, s::cl_int *exp) __NOEXC {
  return std::frexp(x, exp);
}
#endif
MAKE_1V_2P(frexp, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2P(frexp, s::cl_double, s::cl_double, s::cl_int)
#ifdef __HAFL_ENABLED
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
#ifdef __HAFL_ENABLED
cl_half lgamma_r(s::cl_half x, s::cl_int *signp) __NOEXC {
  return ::lgamma_r(x, signp);
}
#endif
MAKE_1V_2P(lgamma_r, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2P(lgamma_r, s::cl_double, s::cl_double, s::cl_int)
#ifdef __HAFL_ENABLED
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
  return (a * b) + c;
}
cl_double mad(s::cl_double a, s::cl_double b, s::cl_double c) __NOEXC {
  return (a * b) + c;
}
#ifndef NO_HALF_ENABLED
cl_half mad(s::cl_half a, s::cl_half b, s::cl_half c) __NOEXC {
  return (a * b) + c;
}
#endif
MAKE_1V_2V_3V(mad, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V_3V(mad, s::cl_double, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_3V(mad, s::cl_half, s::cl_half, s::cl_half, s::cl_half)
#endif

// maxmag
cl_float maxmag(s::cl_float x, s::cl_float y) __NOEXC {
  if (std::fabs(x) > std::fabs(y)) {
    return x;
  } else if (std::fabs(y) > std::fabs(x)) {
    return y;
  } else {
    return std::fmax(x, y);
  }
}
cl_double maxmag(s::cl_double x, s::cl_double y) __NOEXC {
  if (std::fabs(x) > std::fabs(y)) {
    return x;
  } else if (std::fabs(y) > std::fabs(x)) {
    return y;
  } else {
    return std::fmax(x, y);
  }
}
#ifndef NO_HALF_ENABLED
cl_half maxmag(s::cl_half x, s::cl_half y) __NOEXC {
  if (std::fabs(x) > std::fabs(y)) {
    return x;
  } else if (std::fabs(y) > std::fabs(x)) {
    return y;
  } else {
    return std::fmax(x, y);
  }
}
#endif
MAKE_1V_2V(maxmag, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(maxmag, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(maxmag, s::cl_half, s::cl_half, s::cl_half)
#endif

// minmag
cl_float minmag(s::cl_float x, s::cl_float y) __NOEXC {
  if (std::fabs(x) < std::fabs(y)) {
    return x;
  } else if (std::fabs(y) < std::fabs(x)) {
    return y;
  } else {
    return std::fmin(x, y);
  }
}
cl_double minmag(s::cl_double x, s::cl_double y) __NOEXC {
  if (std::fabs(x) < std::fabs(y)) {
    return x;
  } else if (std::fabs(y) < std::fabs(x)) {
    return y;
  } else {
    return std::fmin(x, y);
  }
}
#ifndef NO_HALF_ENABLED
cl_half minmag(s::cl_half x, s::cl_half y) __NOEXC {
  if (std::fabs(x) < std::fabs(y)) {
    return x;
  } else if (std::fabs(y) < std::fabs(x)) {
    return y;
  } else {
    return std::fmin(x, y);
  }
}
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
#ifdef __HAFL_ENABLED
cl_half modf(s::cl_half x, s::cl_half *iptr) __NOEXC {
  return std::modf(x, iptr);
}
#endif
MAKE_1V_2P(modf, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2P(modf, s::cl_double, s::cl_double, s::cl_double)
#ifdef __HAFL_ENABLED
MAKE_1V_2P(modf, s::cl_half, s::cl_half, s::cl_half)
#endif

// nan
cl_float nan(s::cl_uint nancode) __NOEXC {
  return std::numeric_limits<float>::quiet_NaN();
}
cl_double nan(s::cl_ulong nancode) __NOEXC {
  return std::numeric_limits<double>::quiet_NaN();
}
cl_double nan(s::ulonglong nancode) __NOEXC {
  return std::numeric_limits<double>::quiet_NaN();
}
#ifdef __HAFL_ENABLED
cl_half nan(s::cl_ushort nancode) __NOEXC { return NAN; }
#endif
MAKE_1V(nan, s::cl_float, s::cl_uint)
MAKE_1V(nan, s::cl_double, s::cl_ulong)
MAKE_1V(nan, s::cl_double, s::ulonglong)
#ifdef __HAFL_ENABLED
MAKE_1V(nan, s::cl_half, s::cl_ushort)
#endif

// nextafter
cl_float nextafter(s::cl_float x, s::cl_float y) __NOEXC {
  return std::nextafter(x, y);
}
cl_double nextafter(s::cl_double x, s::cl_double y) __NOEXC {
  return std::nextafter(x, y);
}
#ifdef __HAFL_ENABLED
cl_half nextafter(s::cl_half x, s::cl_half y) __NOEXC {
  return std::nextafter(x, y);
}
#endif
MAKE_1V_2V(nextafter, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(nextafter, s::cl_double, s::cl_double, s::cl_double)
#ifdef __HAFL_ENABLED
MAKE_1V_2V(nextafter, s::cl_half, s::cl_half, s::cl_half)
#endif

// pow
cl_float pow(s::cl_float x, s::cl_float y) __NOEXC { return std::pow(x, y); }
cl_double pow(s::cl_double x, s::cl_double y) __NOEXC { return std::pow(x, y); }
#ifdef __HAFL_ENABLED
cl_half pow(s::cl_half x, s::cl_half y) __NOEXC { return std::pow(x, y); }
#endif
MAKE_1V_2V(pow, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(pow, s::cl_double, s::cl_double, s::cl_double)
#ifdef __HAFL_ENABLED
MAKE_1V_2V(pow, s::cl_half, s::cl_half, s::cl_half)
#endif

// pown
cl_float pown(s::cl_float x, s::cl_int y) __NOEXC { return std::pow(x, y); }
cl_double pown(s::cl_double x, s::cl_int y) __NOEXC { return std::pow(x, y); }
#ifdef __HAFL_ENABLED
cl_half pown(s::cl_half x, s::cl_int y) __NOEXC { return std::pow(x, y); }
#endif
MAKE_1V_2V(pown, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2V(pown, s::cl_double, s::cl_double, s::cl_int)
#ifdef __HAFL_ENABLED
MAKE_1V_2V(pown, s::cl_half, s::cl_half, s::cl_int)
#endif

// powr
cl_float powr(s::cl_float x, s::cl_float y) __NOEXC {
  return (x >= 0 ? std::pow(x, y) : x);
}
cl_double powr(s::cl_double x, s::cl_double y) __NOEXC {
  return (x >= 0 ? std::pow(x, y) : x);
}
#ifdef __HAFL_ENABLED
cl_half powr(s::cl_half x, s::cl_half y) __NOEXC {
  return (x >= 0 ? std::pow(x, y) : x);
}
#endif
MAKE_1V_2V(powr, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(powr, s::cl_double, s::cl_double, s::cl_double)
#ifdef __HAFL_ENABLED
MAKE_1V_2V(powr, s::cl_half, s::cl_half, s::cl_half)
#endif

// remainder
cl_float remainder(s::cl_float x, s::cl_float y) __NOEXC {
  return std::remainder(x, y);
}
cl_double remainder(s::cl_double x, s::cl_double y) __NOEXC {
  return std::remainder(x, y);
}
#ifdef __HAFL_ENABLED
cl_half remainder(s::cl_half x, s::cl_half y) __NOEXC {
  return std::remainder(x, y);
}
#endif
MAKE_1V_2V(remainder, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(remainder, s::cl_double, s::cl_double, s::cl_double)
#ifdef __HAFL_ENABLED
MAKE_1V_2V(remainder, s::cl_half, s::cl_half, s::cl_half)
#endif

// remquo
cl_float remquo(s::cl_float x, s::cl_float y, s::cl_int *quo) __NOEXC {
  return std::remquo(x, y, quo);
}
cl_double remquo(s::cl_double x, s::cl_double y, s::cl_int *quo) __NOEXC {
  return std::remquo(x, y, quo);
}
#ifdef __HAFL_ENABLED
cl_half remquo(s::cl_half x, s::cl_half y, s::cl_int *quo) __NOEXC {
  return std::remquo(x, y, quo);
}
#endif
MAKE_1V_2V_3P(remquo, s::cl_float, s::cl_float, s::cl_float, s::cl_int)
MAKE_1V_2V_3P(remquo, s::cl_double, s::cl_double, s::cl_double, s::cl_int)
#ifdef __HAFL_ENABLED
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
cl_float rootn(s::cl_float x, s::cl_int y) __NOEXC {
  return std::pow(x, 1.0 / y);
}
cl_double rootn(s::cl_double x, s::cl_int y) __NOEXC {
  return std::pow(x, 1.0 / y);
}
#ifndef NO_HALF_ENABLED
cl_half rootn(s::cl_half x, s::cl_int y) __NOEXC {
  return std::pow(x, 1.0 / y);
}
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
cl_float rsqrt(s::cl_float x) __NOEXC { return 1.0 / std::sqrt(x); }
cl_double rsqrt(s::cl_double x) __NOEXC { return 1.0 / std::sqrt(x); }
#ifndef NO_HALF_ENABLED
cl_half rsqrt(s::cl_half x) __NOEXC { return 1.0 / std::sqrt(x); }
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
  cosval[0] = std::cos(x);
  return std::sin(x);
}
cl_double sincos(s::cl_double x, s::cl_double *cosval) __NOEXC {
  cosval[0] = std::cos(x);
  return std::sin(x);
}
#ifndef NO_HALF_ENABLED
cl_half sincos(s::cl_half x, s::cl_half *cosval) __NOEXC {
  cosval[0] = std::cos(x);
  return std::sin(x);
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
cl_float sinpi(s::cl_float x) __NOEXC { return std::sin(M_PI * x); }
cl_double sinpi(s::cl_double x) __NOEXC { return std::sin(M_PI * x); }
#ifndef NO_HALF_ENABLED
cl_half sinpi(s::cl_half x) __NOEXC { return std::sin(M_PI * x); }
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
cl_float tanpi(s::cl_float x) __NOEXC { return std::tan(M_PI * x); }
cl_double tanpi(s::cl_double x) __NOEXC { return std::tan(M_PI * x); }
#ifndef NO_HALF_ENABLED
cl_half tanpi(s::cl_half x) __NOEXC { return std::tan(M_PI * x); }
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
// s_max
cl_char s_max(s::cl_char x, s::cl_char y) __NOEXC { return std::max(x, y); }
cl_short s_max(s::cl_short x, s::cl_short y) __NOEXC { return std::max(x, y); }
cl_int s_max(s::cl_int x, s::cl_int y) __NOEXC { return std::max(x, y); }
cl_long s_max(s::cl_long x, s::cl_long y) __NOEXC { return std::max(x, y); }
MAKE_1V_2V(s_max, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_max, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_max, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_max, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2S(s_max, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2S(s_max, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2S(s_max, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2S(s_max, s::cl_long, s::cl_long, s::cl_long)

// u_max
cl_uchar u_max(s::cl_uchar x, s::cl_uchar y) __NOEXC { return std::max(x, y); }
cl_ushort u_max(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return std::max(x, y);
}
cl_uint u_max(s::cl_uint x, s::cl_uint y) __NOEXC { return std::max(x, y); }
cl_ulong u_max(s::cl_ulong x, s::cl_ulong y) __NOEXC { return std::max(x, y); }
MAKE_1V_2V(u_max, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_max, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_max, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_max, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2S(u_max, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2S(u_max, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2S(u_max, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2S(u_max, s::cl_ulong, s::cl_ulong, s::cl_ulong)

// s_min
cl_char s_min(s::cl_char x, s::cl_char y) __NOEXC { return std::min(x, y); }
cl_short s_min(s::cl_short x, s::cl_short y) __NOEXC { return std::min(x, y); }
cl_int s_min(s::cl_int x, s::cl_int y) __NOEXC { return std::min(x, y); }
cl_long s_min(s::cl_long x, s::cl_long y) __NOEXC { return std::min(x, y); }
MAKE_1V_2V(s_min, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2V(s_min, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2V(s_min, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2V(s_min, s::cl_long, s::cl_long, s::cl_long)
MAKE_1V_2S(s_min, s::cl_char, s::cl_char, s::cl_char)
MAKE_1V_2S(s_min, s::cl_short, s::cl_short, s::cl_short)
MAKE_1V_2S(s_min, s::cl_int, s::cl_int, s::cl_int)
MAKE_1V_2S(s_min, s::cl_long, s::cl_long, s::cl_long)

// u_min
cl_uchar u_min(s::cl_uchar x, s::cl_uchar y) __NOEXC { return std::min(x, y); }
cl_ushort u_min(s::cl_ushort x, s::cl_ushort y) __NOEXC {
  return std::min(x, y);
}
cl_uint u_min(s::cl_uint x, s::cl_uint y) __NOEXC { return std::min(x, y); }
cl_ulong u_min(s::cl_ulong x, s::cl_ulong y) __NOEXC { return std::min(x, y); }
MAKE_1V_2V(u_min, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2V(u_min, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2V(u_min, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2V(u_min, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_1V_2S(u_min, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_1V_2S(u_min, s::cl_ushort, s::cl_ushort, s::cl_ushort)
MAKE_1V_2S(u_min, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_1V_2S(u_min, s::cl_ulong, s::cl_ulong, s::cl_ulong)

/* --------------- 4.13.5 Common functions. Host version --------------------*/
// fclamp
cl_float fclamp(s::cl_float x, s::cl_float minval, s::cl_float maxval) __NOEXC {
  return std::fmin(std::fmax(x, minval), maxval);
}
cl_double fclamp(s::cl_double x, s::cl_double minval,
                 s::cl_double maxval) __NOEXC {
  return std::fmin(std::fmax(x, minval), maxval);
}
#ifndef NO_HALF_ENABLED
cl_half fclamp(s::cl_half x, s::cl_half minval, s::cl_half maxval) __NOEXC {
  return std::fmin(std::fmax(x, minval), maxval);
}
#endif
MAKE_1V_2V_3V(fclamp, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V_3V(fclamp, s::cl_double, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_3V(fclamp, s::cl_half, s::cl_half, s::cl_half, s::cl_half)
#endif

// degrees
cl_float degrees(s::cl_float radians) __NOEXC { return (180 / M_PI) * radians; }
cl_double degrees(s::cl_double radians) __NOEXC {
  return (180 / M_PI) * radians;
}
#ifndef NO_HALF_ENABLED
cl_half degrees(s::cl_half radians) __NOEXC { return (180 / M_PI) * radians; }
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
  return x + (y - x) * a;
}
cl_double mix(s::cl_double x, s::cl_double y, s::cl_double a) __NOEXC {
  return x + (y - x) * a;
}
#ifndef NO_HALF_ENABLED
cl_half mix(s::cl_half x, s::cl_half y, s::cl_half a) __NOEXC {
  return x + (y - x) * a;
}
#endif
MAKE_1V_2V_3V(mix, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V_3V(mix, s::cl_double, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_3V(mix, s::cl_half, s::cl_half, s::cl_half, s::cl_half)
#endif

// radians
cl_float radians(s::cl_float degrees) __NOEXC { return (M_PI / 180) * degrees; }
cl_double radians(s::cl_double degrees) __NOEXC {
  return (M_PI / 180) * degrees;
}
#ifndef NO_HALF_ENABLED
cl_half radians(s::cl_half degrees) __NOEXC { return (M_PI / 180) * degrees; }
#endif
MAKE_1V(radians, s::cl_float, s::cl_float)
MAKE_1V(radians, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(radians, s::cl_half, s::cl_half)
#endif

// step
cl_float step(s::cl_float edge, s::cl_float x) __NOEXC {
  return (x < edge) ? 0.0 : 1.0;
}
cl_double step(s::cl_double edge, s::cl_double x) __NOEXC {
  return (x < edge) ? 0.0 : 1.0;
}
#ifndef NO_HALF_ENABLED
cl_half step(s::cl_half edge, s::cl_half x) __NOEXC {
  return (x < edge) ? 0.0 : 1.0;
}
#endif
MAKE_1V_2V(step, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V(step, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V(step, s::cl_half, s::cl_half, s::cl_half)
#endif

// fma
cl_float smoothstep(s::cl_float edge0, s::cl_float edge1,
                    s::cl_float x) __NOEXC {
  cl_float t;
  t = fclamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}
cl_double smoothstep(s::cl_double edge0, s::cl_double edge1,
                     s::cl_double x) __NOEXC {
  cl_float t;
  t = fclamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}
#ifndef NO_HALF_ENABLED
cl_half smoothstep(s::cl_half edge0, s::cl_half edge1, s::cl_half x) __NOEXC {
  cl_float t;
  t = fclamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}
#endif
MAKE_1V_2V_3V(smoothstep, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V_3V(smoothstep, s::cl_double, s::cl_double, s::cl_double,
              s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_3V(smoothstep, s::cl_half, s::cl_half, s::cl_half, s::cl_half)
#endif

// sign
cl_float sign(s::cl_float x) __NOEXC {
  if (std::isnan(x)) {
    return 0.0;
  } else if (x > 0) {
    return 1.0;
  } else if (x < 0) {
    return -1.0;
  } else /* x is +0.0 or -0.0} */ {
    return x;
  }
}
cl_double sign(s::cl_double x) __NOEXC {
  if (std::isnan(x)) {
    return 0.0;
  } else if (x > 0) {
    return 1.0;
  } else if (x < 0) {
    return -1.0;
  } else /* x is +0.0 or -0.0} */ {
    return x;
  }
}
#ifndef NO_HALF_ENABLED
cl_half sign(s::cl_half x) __NOEXC {
  if (std::isnan(x)) {
    return 0.0;
  } else if (x > 0) {
    return 1.0;
  } else if (x < 0) {
    return -1.0;
  } else /* x is +0.0 or -0.0} */ {
    return x;
  }
}
#endif
MAKE_1V(sign, s::cl_float, s::cl_float)
MAKE_1V(sign, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V(sign, s::cl_half, s::cl_half)
#endif

/* --------------- 4.13.6 Geometric Functions. Host version -----------------*/
// cross
#define MAKE_CROSS(r, p0, p1)                                                  \
  r.x() = p0.y() * p1.z() - p0.z() * p1.y();                                   \
  r.y() = p0.z() * p1.x() - p0.x() * p1.z();                                   \
  r.z() = p0.x() * p1.y() - p0.y() * p1.x();

s::cl_float3 cross(s::cl_float3 p0, s::cl_float3 p1) __NOEXC {
  s::cl_float3 r;
  MAKE_CROSS(r, p0, p1) return r;
}
s::cl_float4 cross(s::cl_float4 p0, s::cl_float4 p1) __NOEXC {
  s::cl_float4 r;
  MAKE_CROSS(r, p0, p1) r.w() = 0;
  return r;
}
s::cl_double3 cross(s::cl_double3 p0, s::cl_double3 p1) __NOEXC {
  s::cl_double3 r;
  MAKE_CROSS(r, p0, p1) return r;
}
s::cl_double4 cross(s::cl_double4 p0, s::cl_double4 p1) __NOEXC {
  s::cl_double4 r;
  MAKE_CROSS(r, p0, p1) r.w() = 0;
  return r;
}
#ifndef NO_HALF_ENABLED
s::cl_half3 cross(s::cl_half3 p0, s::cl_half3 p1) __NOEXC {
  s::cl_half3 r;
  MAKE_CROSS(r, p0, p1) return r;
}
s::cl_half4 cross(s::cl_half4 p0, s::cl_half4 p1) __NOEXC {
  s::cl_half4 r;
  MAKE_CROSS(r, p0, p1) r.w() = 0;
  return r;
}
#endif
#undef MAKE_CROSS

// OpFMul
template <typename T>
typename std::enable_if<sycl::detail::is_sgenfloat<T>::value, void>::type
__OpFMul(T &r, T p0, T p1) {
  r += p0 * p1;
}

cl_float OpFMul(s::cl_float p0, s::cl_float p1) {
  s::cl_float r = 0;
  __OpFMul(r, p0, p1);
  return r;
}
cl_double OpFMul(s::cl_double p0, s::cl_double p1) {
  s::cl_double r = 0;
  __OpFMul(r, p0, p1);
  return r;
}
#ifndef NO_HALF_ENABLED
cl_float OpFMul(s::cl_half p0, s::cl_half p1) {
  s::cl_half r = 0;
  __OpFMul(r, p0, p1);
  return r;
}
#endif
// OpDot
MAKE_1V_2V_RS(OpDot, __OpFMul, s::cl_float, s::cl_float, s::cl_float)
MAKE_1V_2V_RS(OpDot, __OpFMul, s::cl_double, s::cl_double, s::cl_double)
#ifndef NO_HALF_ENABLED
MAKE_1V_2V_RS(OpDot, __OpFMul, s::cl_half, s::cl_half, s::cl_half)
#endif

/* --------------- 4.13.7 Relational functions. Host version ----------------*/

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

#undef __MAKE_1V
#undef __MAKE_1V_2V
#undef __MAKE_1V_2S
#undef __MAKE_1V_2P
#undef __MAKE_1V_2V_3V
#undef __MAKE_1V_2V_3P
#undef MAKE_1V
#undef MAKE_1V_2V
#undef MAKE_1V_2S
#undef MAKE_1V_2P
#undef MAKE_1V_2V_3V
#undef MAKE_1V_2V_3P

#undef __NOEXC
