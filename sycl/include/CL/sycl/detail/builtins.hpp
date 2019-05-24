//==----------- builtins.hpp - SYCL built-in functions ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/generic_type_traits.hpp>

#include <type_traits>

// TODO Decide whether to mark functions with this attribute.
#define __NOEXC /*noexcept*/

#ifdef __SYCL_DEVICE_ONLY__
#define __FUNC_PREFIX_OCL  __spirv_ocl_
#define __FUNC_PREFIX_CORE  __spirv_
#else
#define __FUNC_PREFIX_OCL
#define __FUNC_PREFIX_CORE
#endif

#define PPCAT_NX(A, B) A ## B
#define PPCAT(A, B) PPCAT_NX(A, B)

#define MAKE_CALL_ARG1(call, prefix)                                           \
  template <typename R, typename T1>                                           \
  ALWAYS_INLINE                                                                \
      typename cl::sycl::detail::ConvertToOpenCLType<R>::type __invoke_##call( \
          T1 t1) __NOEXC {                                                     \
    using Ret = typename cl::sycl::detail::ConvertToOpenCLType<R>::type;       \
    using Arg1 = typename cl::sycl::detail::ConvertToOpenCLType<T1>::type;     \
    extern Ret PPCAT(prefix, call)(Arg1);                                      \
    return PPCAT(prefix, call)(cl::sycl::detail::TryToGetPointer(t1));         \
  }

#define MAKE_CALL_ARG2(call, prefix)                                           \
  template <typename R, typename T1, typename T2>                              \
  ALWAYS_INLINE                                                                \
      typename cl::sycl::detail::ConvertToOpenCLType<R>::type __invoke_##call( \
          T1 t1, T2 t2) __NOEXC {                                              \
    using Ret = typename cl::sycl::detail::ConvertToOpenCLType<R>::type;       \
    using Arg1 = typename cl::sycl::detail::ConvertToOpenCLType<T1>::type;     \
    using Arg2 = typename cl::sycl::detail::ConvertToOpenCLType<T2>::type;     \
    extern Ret PPCAT(prefix, call)(Arg1, Arg2);                                \
    return PPCAT(prefix, call)(cl::sycl::detail::TryToGetPointer(t1),          \
                               cl::sycl::detail::TryToGetPointer(t2));         \
  }

#define MAKE_CALL_ARG3(call, prefix)                                           \
  template <typename R, typename T1, typename T2, typename T3>                 \
  ALWAYS_INLINE                                                                \
      typename cl::sycl::detail::ConvertToOpenCLType<R>::type __invoke_##call( \
          T1 t1, T2 t2, T3 t3) __NOEXC {                                       \
    using Ret = typename cl::sycl::detail::ConvertToOpenCLType<R>::type;       \
    using Arg1 = typename cl::sycl::detail::ConvertToOpenCLType<T1>::type;     \
    using Arg2 = typename cl::sycl::detail::ConvertToOpenCLType<T2>::type;     \
    using Arg3 = typename cl::sycl::detail::ConvertToOpenCLType<T3>::type;     \
    extern Ret PPCAT(prefix, call)(Arg1, Arg2, Arg3);                          \
    return PPCAT(prefix, call)(cl::sycl::detail::TryToGetPointer(t1),          \
                               cl::sycl::detail::TryToGetPointer(t2),          \
                               cl::sycl::detail::TryToGetPointer(t3));         \
  }

#ifndef __SYCL_DEVICE_ONLY__
namespace cl {
namespace __host_std {
#endif // __SYCL_DEVICE_ONLY__
/* ----------------- 4.13.3 Math functions. ---------------------------------*/
MAKE_CALL_ARG1(acos, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(acosh, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(acospi, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(asin, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(asinh, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(asinpi, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(atan, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(atan2, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(atanh, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(atanpi, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(atan2pi, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(cbrt, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(ceil, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(copysign, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(cos, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(cosh, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(cospi, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(erfc, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(erf, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(exp, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(exp2, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(exp10, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(expm1, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(fabs, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(fdim, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(floor, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG3(fma, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(fmax, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(fmin, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(fmod, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(fract, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(frexp, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(hypot, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(ilogb, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(ldexp, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(lgamma, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(lgamma_r, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(log, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(log2, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(log10, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(log1p, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(logb, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG3(mad, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(maxmag, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(minmag, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(modf, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(nan, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(nextafter, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(pow, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(pown, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(powr, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(remainder, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG3(remquo, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(rint, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(rootn, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(round, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(rsqrt, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(sin, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(sincos, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(sinh, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(sinpi, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(sqrt, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(tan, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(tanh, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(tanpi, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(tgamma, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(trunc, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(native_cos, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(native_divide, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(native_exp, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(native_exp2, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(native_exp10, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(native_log, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(native_log2, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(native_log10, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(native_powr, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(native_recip, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(native_rsqrt, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(native_sin, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(native_sqrt, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(native_tan, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(half_cos, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(half_divide, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(half_exp, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(half_exp2, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(half_exp10, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(half_log, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(half_log2, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(half_log10, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(half_powr, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(half_recip, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(half_rsqrt, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(half_sin, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(half_sqrt, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(half_tan, __FUNC_PREFIX_OCL)
/* --------------- 4.13.4 Integer functions. --------------------------------*/
MAKE_CALL_ARG1(s_abs, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(u_abs, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(s_abs_diff, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(u_abs_diff, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(s_add_sat, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(u_add_sat, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(s_hadd, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(u_hadd, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(s_rhadd, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(u_rhadd, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG3(s_clamp, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG3(u_clamp, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(clz, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG3(s_mad_hi, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG3(u_mad_hi, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG3(u_mad_sat, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG3(s_mad_sat, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(s_max, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(u_max, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(s_min, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(u_min, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(s_mul_hi, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(u_mul_hi, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(rotate, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(s_sub_sat, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(u_sub_sat, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(u_upsample, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(s_upsample, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(popcount, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG3(s_mad24, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG3(u_mad24, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(s_mul24, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(u_mul24, __FUNC_PREFIX_OCL)
/* --------------- 4.13.5 Common functions. ---------------------------------*/
MAKE_CALL_ARG3(fclamp, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(degrees, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(fmax_common, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(fmin_common, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG3(mix, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(radians, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(step, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG3(smoothstep, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(sign, __FUNC_PREFIX_OCL)
/* --------------- 4.13.6 Geometric Functions. ------------------------------*/
MAKE_CALL_ARG2(cross, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(Dot, __FUNC_PREFIX_CORE)  // dot
MAKE_CALL_ARG2(FMul, __FUNC_PREFIX_CORE) // dot
MAKE_CALL_ARG2(distance, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(length, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(normalize, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG2(fast_distance, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(fast_length, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG1(fast_normalize, __FUNC_PREFIX_OCL)
/* --------------- 4.13.7 Relational functions. -----------------------------*/
MAKE_CALL_ARG2(FOrdEqual, __FUNC_PREFIX_CORE)            // isequal
MAKE_CALL_ARG2(FUnordNotEqual, __FUNC_PREFIX_CORE)       // isnotequal
MAKE_CALL_ARG2(FOrdGreaterThan, __FUNC_PREFIX_CORE)      // isgreater
MAKE_CALL_ARG2(FOrdGreaterThanEqual, __FUNC_PREFIX_CORE) // isgreaterequal
MAKE_CALL_ARG2(FOrdLessThan, __FUNC_PREFIX_CORE)         // isless
MAKE_CALL_ARG2(FOrdLessThanEqual, __FUNC_PREFIX_CORE)    // islessequal
MAKE_CALL_ARG2(LessOrGreater, __FUNC_PREFIX_CORE)        // islessgreater
MAKE_CALL_ARG1(IsFinite, __FUNC_PREFIX_CORE)             // isfinite
MAKE_CALL_ARG1(IsInf, __FUNC_PREFIX_CORE)                // isinf
MAKE_CALL_ARG1(IsNan, __FUNC_PREFIX_CORE)                // isnan
MAKE_CALL_ARG1(IsNormal, __FUNC_PREFIX_CORE)             // isnormal
MAKE_CALL_ARG2(Ordered, __FUNC_PREFIX_CORE)              // isordered
MAKE_CALL_ARG2(Unordered, __FUNC_PREFIX_CORE)            // isunordered
MAKE_CALL_ARG1(SignBitSet, __FUNC_PREFIX_CORE)           // signbit
MAKE_CALL_ARG1(Any, __FUNC_PREFIX_CORE)                  // any
MAKE_CALL_ARG1(All, __FUNC_PREFIX_CORE)                  // all
MAKE_CALL_ARG3(bitselect, __FUNC_PREFIX_OCL)
MAKE_CALL_ARG3(Select, __FUNC_PREFIX_CORE) // select
#ifndef __SYCL_DEVICE_ONLY__
} // namespace __host_std
} // namespace cl
#endif

#undef __NOEXC
#undef MAKE_CALL_ARG1
#undef MAKE_CALL_ARG2
#undef MAKE_CALL_ARG3
#undef PPCAT_NX
#undef PPCAT
#undef __FUNC_PREFIX_OCL
#undef __FUNC_PREFIX_CORE
