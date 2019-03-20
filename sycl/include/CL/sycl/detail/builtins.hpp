//==----------- builtins.hpp - SYCL built-in functions ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>

#include <type_traits>
// TODO Delete this include after solving the problems in the test
// infrastructure.
#include <cmath>

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

namespace cl {
namespace sycl {
namespace detail {

// Try to get pointer_t, otherwise T
template <typename T> class TryToGetPointerT {
  static T check(...);
  template <typename A> static typename A::pointer_t check(const A &);

public:
  using type = decltype(check(T()));
  static constexpr bool value = !std::is_same<T, type>::value;
};

// Try to get element_type, otherwise T
template <typename T> class TryToGetElementType {
  static T check(...);
  template <typename A> static typename A::element_type check(const A &);

public:
  using type = decltype(check(T()));
  static constexpr bool value = !std::is_same<T, type>::value;
};

// Try to get vector_t, otherwise T
template <typename T> class TryToGetVectorT {
  static T check(...);
  template <typename A> static typename A::vector_t check(const A &);

public:
  using type = decltype(check(T()));
  static constexpr bool value = !std::is_same<T, type>::value;
};

// Try to get pointer_t (if pointer_t indicates on the type with vector_t
// creates a pointer type on vector_t), otherwise T
template <typename T> class TryToGetPointerVecT {
  static T check(...);
  template <typename A>
  static typename PtrValueType<
      typename TryToGetVectorT<typename TryToGetElementType<A>::type>::type,
      A::address_space>::type *
  check(const A &);

public:
  using type = decltype(check(T()));
};

template <typename T, typename = typename std::enable_if<
                          TryToGetPointerT<T>::value, std::true_type>::type>
typename TryToGetPointerVecT<T>::type TryToGetPointer(T &t) {
  // TODO find the better way to get the pointer to underlying data from vec
  // class
  return reinterpret_cast<typename TryToGetPointerVecT<T>::type>(t.get());
}

template <typename T, typename = typename std::enable_if<
                          !TryToGetPointerT<T>::value, std::false_type>::type>
T TryToGetPointer(T &t) {
  return t;
}

// Converts T to OpenCL friendly
template <typename T>
using ConvertToOpenCLType = std::conditional<
    TryToGetVectorT<T>::value, typename TryToGetVectorT<T>::type,
    typename std::conditional<TryToGetPointerT<T>::value,
                              typename TryToGetPointerVecT<T>::type, T>::type>;

} // namespace detail
} // namespace sycl
} // namespace cl

#define MAKE_CALL_ARG1(call)                                                   \
  template <typename R, typename T1>                                           \
  ALWAYS_INLINE                                                                \
      typename cl::sycl::detail::ConvertToOpenCLType<R>::type __invoke_##call( \
          T1 t1) __NOEXC {                                                     \
    using Ret = typename cl::sycl::detail::ConvertToOpenCLType<R>::type;       \
    using Arg1 = typename cl::sycl::detail::ConvertToOpenCLType<T1>::type;     \
    extern Ret call(Arg1);                                                     \
    return call(cl::sycl::detail::TryToGetPointer(t1));                        \
  }

#define MAKE_CALL_ARG2(call)                                                   \
  template <typename R, typename T1, typename T2>                              \
  ALWAYS_INLINE                                                                \
      typename cl::sycl::detail::ConvertToOpenCLType<R>::type __invoke_##call( \
          T1 t1, T2 t2) __NOEXC {                                              \
    using Ret = typename cl::sycl::detail::ConvertToOpenCLType<R>::type;       \
    using Arg1 = typename cl::sycl::detail::ConvertToOpenCLType<T1>::type;     \
    using Arg2 = typename cl::sycl::detail::ConvertToOpenCLType<T2>::type;     \
    extern Ret call(Arg1, Arg2);                                               \
    return call(cl::sycl::detail::TryToGetPointer(t1),                         \
                cl::sycl::detail::TryToGetPointer(t2));                        \
  }

#define MAKE_CALL_ARG3(call)                                                   \
  template <typename R, typename T1, typename T2, typename T3>                 \
  ALWAYS_INLINE                                                                \
      typename cl::sycl::detail::ConvertToOpenCLType<R>::type __invoke_##call( \
          T1 t1, T2 t2, T3 t3) __NOEXC {                                       \
    using Ret = typename cl::sycl::detail::ConvertToOpenCLType<R>::type;       \
    using Arg1 = typename cl::sycl::detail::ConvertToOpenCLType<T1>::type;     \
    using Arg2 = typename cl::sycl::detail::ConvertToOpenCLType<T2>::type;     \
    using Arg3 = typename cl::sycl::detail::ConvertToOpenCLType<T3>::type;     \
    extern Ret call(Arg1, Arg2, Arg3);                                         \
    return call(cl::sycl::detail::TryToGetPointer(t1),                         \
                cl::sycl::detail::TryToGetPointer(t2),                         \
                cl::sycl::detail::TryToGetPointer(t3));                        \
  }

namespace cl {
#ifdef __SYCL_DEVICE_ONLY__
namespace __spirv {
#else
namespace __host_std {
#endif // __SYCL_DEVICE_ONLY__
/* ----------------- 4.13.3 Math functions. ---------------------------------*/
MAKE_CALL_ARG1(acos)
MAKE_CALL_ARG1(acosh)
MAKE_CALL_ARG1(acospi)
MAKE_CALL_ARG1(asin)
MAKE_CALL_ARG1(asinh)
MAKE_CALL_ARG1(asinpi)
MAKE_CALL_ARG1(atan)
MAKE_CALL_ARG2(atan2)
MAKE_CALL_ARG1(atanh)
MAKE_CALL_ARG1(atanpi)
MAKE_CALL_ARG2(atan2pi)
MAKE_CALL_ARG1(cbrt)
MAKE_CALL_ARG1(ceil)
MAKE_CALL_ARG2(copysign)
MAKE_CALL_ARG1(cos)
MAKE_CALL_ARG1(cosh)
MAKE_CALL_ARG1(cospi)
MAKE_CALL_ARG1(erfc)
MAKE_CALL_ARG1(erf)
MAKE_CALL_ARG1(exp)
MAKE_CALL_ARG1(exp2)
MAKE_CALL_ARG1(exp10)
MAKE_CALL_ARG1(expm1)
MAKE_CALL_ARG1(fabs)
MAKE_CALL_ARG2(fdim)
MAKE_CALL_ARG1(floor)
MAKE_CALL_ARG3(fma)
MAKE_CALL_ARG2(fmax)
MAKE_CALL_ARG2(fmin)
MAKE_CALL_ARG2(fmod)
MAKE_CALL_ARG2(fract)
MAKE_CALL_ARG2(frexp)
MAKE_CALL_ARG2(hypot)
MAKE_CALL_ARG1(ilogb)
MAKE_CALL_ARG2(ldexp)
MAKE_CALL_ARG1(lgamma)
MAKE_CALL_ARG2(lgamma_r)
MAKE_CALL_ARG1(log)
MAKE_CALL_ARG1(log2)
MAKE_CALL_ARG1(log10)
MAKE_CALL_ARG1(log1p)
MAKE_CALL_ARG1(logb)
MAKE_CALL_ARG3(mad)
MAKE_CALL_ARG2(maxmag)
MAKE_CALL_ARG2(minmag)
MAKE_CALL_ARG2(modf)
MAKE_CALL_ARG1(nan)
MAKE_CALL_ARG2(nextafter)
MAKE_CALL_ARG2(pow)
MAKE_CALL_ARG2(pown)
MAKE_CALL_ARG2(powr)
MAKE_CALL_ARG2(remainder)
MAKE_CALL_ARG3(remquo)
MAKE_CALL_ARG1(rint)
MAKE_CALL_ARG2(rootn)
MAKE_CALL_ARG1(round)
MAKE_CALL_ARG1(rsqrt)
MAKE_CALL_ARG1(sin)
MAKE_CALL_ARG2(sincos)
MAKE_CALL_ARG1(sinh)
MAKE_CALL_ARG1(sinpi)
MAKE_CALL_ARG1(sqrt)
MAKE_CALL_ARG1(tan)
MAKE_CALL_ARG1(tanh)
MAKE_CALL_ARG1(tanpi)
MAKE_CALL_ARG1(tgamma)
MAKE_CALL_ARG1(trunc)
MAKE_CALL_ARG1(native_cos)
MAKE_CALL_ARG2(native_divide)
MAKE_CALL_ARG1(native_exp)
MAKE_CALL_ARG1(native_exp2)
MAKE_CALL_ARG1(native_exp10)
MAKE_CALL_ARG1(native_log)
MAKE_CALL_ARG1(native_log2)
MAKE_CALL_ARG1(native_log10)
MAKE_CALL_ARG2(native_powr)
MAKE_CALL_ARG1(native_recip)
MAKE_CALL_ARG1(native_rsqrt)
MAKE_CALL_ARG1(native_sin)
MAKE_CALL_ARG1(native_sqrt)
MAKE_CALL_ARG1(native_tan)
MAKE_CALL_ARG1(half_cos)
MAKE_CALL_ARG2(half_divide)
MAKE_CALL_ARG1(half_exp)
MAKE_CALL_ARG1(half_exp2)
MAKE_CALL_ARG1(half_exp10)
MAKE_CALL_ARG1(half_log)
MAKE_CALL_ARG1(half_log2)
MAKE_CALL_ARG1(half_log10)
MAKE_CALL_ARG2(half_powr)
MAKE_CALL_ARG1(half_recip)
MAKE_CALL_ARG1(half_rsqrt)
MAKE_CALL_ARG1(half_sin)
MAKE_CALL_ARG1(half_sqrt)
MAKE_CALL_ARG1(half_tan)
/* --------------- 4.13.4 Integer functions. --------------------------------*/
MAKE_CALL_ARG1(s_abs)
MAKE_CALL_ARG1(u_abs)
MAKE_CALL_ARG2(s_abs_diff)
MAKE_CALL_ARG2(u_abs_diff)
MAKE_CALL_ARG2(s_add_sat)
MAKE_CALL_ARG2(u_add_sat)
MAKE_CALL_ARG2(s_hadd)
MAKE_CALL_ARG2(u_hadd)
MAKE_CALL_ARG2(s_rhadd)
MAKE_CALL_ARG2(u_rhadd)
MAKE_CALL_ARG3(s_clamp)
MAKE_CALL_ARG3(u_clamp)
MAKE_CALL_ARG1(clz)
MAKE_CALL_ARG3(s_mad_hi)
MAKE_CALL_ARG3(u_mad_hi)
MAKE_CALL_ARG3(u_mad_sat)
MAKE_CALL_ARG3(s_mad_sat)
MAKE_CALL_ARG2(s_max)
MAKE_CALL_ARG2(u_max)
MAKE_CALL_ARG2(s_min)
MAKE_CALL_ARG2(u_min)
MAKE_CALL_ARG2(s_mul_hi)
MAKE_CALL_ARG2(u_mul_hi)
MAKE_CALL_ARG2(rotate)
MAKE_CALL_ARG2(s_sub_sat)
MAKE_CALL_ARG2(u_sub_sat)
MAKE_CALL_ARG2(u_upsample)
MAKE_CALL_ARG2(s_upsample)
MAKE_CALL_ARG1(popcount)
MAKE_CALL_ARG3(s_mad24)
MAKE_CALL_ARG3(u_mad24)
MAKE_CALL_ARG3(s_mul24)
MAKE_CALL_ARG3(u_mul24)
/* --------------- 4.13.5 Common functions. ---------------------------------*/
MAKE_CALL_ARG3(fclamp)
MAKE_CALL_ARG1(degrees)
MAKE_CALL_ARG2(fmax_common)
MAKE_CALL_ARG2(fmin_common)
MAKE_CALL_ARG3(mix)
MAKE_CALL_ARG1(radians)
MAKE_CALL_ARG2(step)
MAKE_CALL_ARG3(smoothstep)
MAKE_CALL_ARG1(sign)
/* --------------- 4.13.6 Geometric Functions. ------------------------------*/
MAKE_CALL_ARG2(cross)
MAKE_CALL_ARG2(OpDot)  // dot
MAKE_CALL_ARG2(OpFMul) // dot
MAKE_CALL_ARG2(distance)
MAKE_CALL_ARG1(length)
MAKE_CALL_ARG1(normalize)
MAKE_CALL_ARG2(fast_distance)
MAKE_CALL_ARG1(fast_length)
MAKE_CALL_ARG1(fast_normalize)
/* --------------- 4.13.7 Relational functions. -----------------------------*/
MAKE_CALL_ARG2(OpFOrdEqual)            // isequal
MAKE_CALL_ARG2(OpFUnordNotEqual)       // isnotequal
MAKE_CALL_ARG2(OpFOrdGreaterThan)      // isgreater
MAKE_CALL_ARG2(OpFOrdGreaterThanEqual) // isgreaterequal
MAKE_CALL_ARG2(OpFOrdLessThan)         // isless
MAKE_CALL_ARG2(OpFOrdLessThanEqual)    // islessequal
MAKE_CALL_ARG2(OpLessOrGreater)        // islessgreater
MAKE_CALL_ARG1(OpIsFinite)             // isfinite
MAKE_CALL_ARG1(OpIsInf)                // isinf
MAKE_CALL_ARG1(OpIsNan)                // isnan
MAKE_CALL_ARG1(OpIsNormal)             // isnormal
MAKE_CALL_ARG2(OpOrdered)              // isordered
MAKE_CALL_ARG2(OpUnordered)            // isunordered
MAKE_CALL_ARG1(OpSignBitSet)           // signbit
MAKE_CALL_ARG1(OpAny)                  // any
MAKE_CALL_ARG1(OpAll)                  // all
MAKE_CALL_ARG3(bitselect)
MAKE_CALL_ARG3(OpSelect) // select
} // namespace __spirv or __host_std
} // namespace cl

#undef __NOEXC
#undef MAKE_CALL_ARG1
#undef MAKE_CALL_ARG2
#undef MAKE_CALL_ARG3
