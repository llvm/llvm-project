//==----------- math.hpp - SYCL math functions ------------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/types.hpp>

#include <cmath>

#ifdef __SYCL_DEVICE_ONLY__

#define CONCAT_HELP(a, b) a##b
#define CONCAT(a, b) CONCAT_HELP(a, b)

#define SCALAR(type) CONCAT(CONCAT(__, type), _t)
#define VECTOR(type, len) CONCAT(CONCAT(CONCAT(__, type), len), _vec_t)

#define MAKE_FUN_OF_1_ARG(name, ret_ty, arg_1_ty) ret_ty name(arg_1_ty);

#define MAKE_FUN_OF_2_ARG(name, ret_ty, arg_1_ty, arg_2_ty)                    \
  ret_ty name(arg_1_ty, arg_2_ty);

#define MAKE_FUN_OF_3_ARG(name, ret_ty, arg_1_ty, arg_2_ty, arg_3_ty)          \
  ret_ty name(arg_1_ty, arg_2_ty, arg_3_ty);

#define GEN_FUNC_OF_ONE_ARG_V(name, ret_ty, arg_1_ty)                          \
  MAKE_FUN_OF_1_ARG(name, VECTOR(ret_ty, 2), VECTOR(arg_1_ty, 2))              \
  MAKE_FUN_OF_1_ARG(name, VECTOR(ret_ty, 3), VECTOR(arg_1_ty, 3))              \
  MAKE_FUN_OF_1_ARG(name, VECTOR(ret_ty, 4), VECTOR(arg_1_ty, 4))              \
  MAKE_FUN_OF_1_ARG(name, VECTOR(ret_ty, 8), VECTOR(arg_1_ty, 8))              \
  MAKE_FUN_OF_1_ARG(name, VECTOR(ret_ty, 16), VECTOR(arg_1_ty, 16))

#define GEN_FUNC_OF_TWO_ARG_V(name, ret_ty, arg_1_ty, arg_2_ty)                \
  MAKE_FUN_OF_2_ARG(name, VECTOR(ret_ty, 2), VECTOR(arg_1_ty, 2),              \
                    VECTOR(arg_2_ty, 2))                                       \
  MAKE_FUN_OF_2_ARG(name, VECTOR(ret_ty, 3), VECTOR(arg_1_ty, 3),              \
                    VECTOR(arg_2_ty, 3))                                       \
  MAKE_FUN_OF_2_ARG(name, VECTOR(ret_ty, 4), VECTOR(arg_1_ty, 4),              \
                    VECTOR(arg_2_ty, 4))                                       \
  MAKE_FUN_OF_2_ARG(name, VECTOR(ret_ty, 8), VECTOR(arg_1_ty, 8),              \
                    VECTOR(arg_2_ty, 8))                                       \
  MAKE_FUN_OF_2_ARG(name, VECTOR(ret_ty, 16), VECTOR(arg_1_ty, 16),            \
                    VECTOR(arg_2_ty, 16))

#define GEN_FUNC_OF_THREE_ARG_V(name, ret_ty, arg_1_ty, arg_2_ty, arg_3_ty)    \
  MAKE_FUN_OF_3_ARG(name, VECTOR(ret_ty, 2), VECTOR(arg_1_ty, 2),              \
                    VECTOR(arg_2_ty, 2), VECTOR(arg_3_ty, 2))                  \
  MAKE_FUN_OF_3_ARG(name, VECTOR(ret_ty, 3), VECTOR(arg_1_ty, 3),              \
                    VECTOR(arg_2_ty, 3), VECTOR(arg_3_ty, 3))                  \
  MAKE_FUN_OF_3_ARG(name, VECTOR(ret_ty, 4), VECTOR(arg_1_ty, 4),              \
                    VECTOR(arg_2_ty, 4), VECTOR(arg_3_ty, 4))                  \
  MAKE_FUN_OF_3_ARG(name, VECTOR(ret_ty, 8), VECTOR(arg_1_ty, 8),              \
                    VECTOR(arg_2_ty, 8), VECTOR(arg_3_ty, 8))                  \
  MAKE_FUN_OF_3_ARG(name, VECTOR(ret_ty, 16), VECTOR(arg_1_ty, 16),            \
                    VECTOR(arg_2_ty, 16), VECTOR(arg_3_ty, 16))

#define GEN_FUNC_OF_ONE_ARG_S(name, ret_ty, arg_1_ty)                          \
  MAKE_FUN_OF_1_ARG(name, SCALAR(ret_ty), SCALAR(arg_1_ty))

#define GEN_FUNC_OF_TWO_ARG_S(name, ret_ty, arg_1_ty, arg_2_ty)                \
  MAKE_FUN_OF_2_ARG(name, SCALAR(ret_ty), SCALAR(arg_1_ty), SCALAR(arg_2_ty))

#define GEN_FUNC_OF_THREE_ARG_S(name, ret_ty, arg_1_ty, arg_2_ty, arg_3_ty)    \
  MAKE_FUN_OF_3_ARG(name, SCALAR(ret_ty), SCALAR(arg_1_ty), SCALAR(arg_2_ty),  \
                    SCALAR(arg_3_ty))

#define GEN_FUNC_OF_ONE_ARG(name, ret_ty, arg_1_ty)                            \
  GEN_FUNC_OF_ONE_ARG_S(name, ret_ty, arg_1_ty)                                \
  GEN_FUNC_OF_ONE_ARG_V(name, ret_ty, arg_1_ty)

#define GEN_FUNC_OF_TWO_ARG(name, ret_ty, arg_1_ty, arg_2_ty)                  \
  GEN_FUNC_OF_TWO_ARG_S(name, ret_ty, arg_1_ty, arg_2_ty)                      \
  GEN_FUNC_OF_TWO_ARG_V(name, ret_ty, arg_1_ty, arg_2_ty)

#define GEN_FUNC_OF_THREE_ARG(name, ret_ty, arg_1_ty, arg_2_ty, arg_3_ty)      \
  GEN_FUNC_OF_THREE_ARG_S(name, ret_ty, arg_1_ty, arg_2_ty, arg_3_ty)          \
  GEN_FUNC_OF_THREE_ARG_V(name, ret_ty, arg_1_ty, arg_2_ty, arg_3_ty)

#define GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(name, ret_ty, arg_1_ty, arg_2_ty)   \
  MAKE_FUN_OF_2_ARG(name, VECTOR(ret_ty, 2), VECTOR(arg_1_ty, 2),              \
                    SCALAR(arg_2_ty))                                          \
  MAKE_FUN_OF_2_ARG(name, VECTOR(ret_ty, 3), VECTOR(arg_1_ty, 3),              \
                    SCALAR(arg_2_ty))                                          \
  MAKE_FUN_OF_2_ARG(name, VECTOR(ret_ty, 4), VECTOR(arg_1_ty, 4),              \
                    SCALAR(arg_2_ty))                                          \
  MAKE_FUN_OF_2_ARG(name, VECTOR(ret_ty, 8), VECTOR(arg_1_ty, 8),              \
                    SCALAR(arg_2_ty))                                          \
  MAKE_FUN_OF_2_ARG(name, VECTOR(ret_ty, 16), VECTOR(arg_1_ty, 16),            \
                    SCALAR(arg_2_ty))

#define GEN_FUNC_OF_TWO_ARG_S_SECOND_ARG_S(name, ret_ty, arg_1_ty, arg_2_ty)   \
  MAKE_FUN_OF_2_ARG(name, SCALAR(ret_ty), SCALAR(arg_1_ty), SCALAR(arg_2_ty))

#define GEN_FUNC_OF_TWO_ARG_SECOND_ARG_S(name, ret_ty, arg_1_ty, arg_2_ty)     \
  GEN_FUNC_OF_TWO_ARG_S_SECOND_ARG_S(name, ret_ty, arg_1_ty)                   \
  GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(name, ret_ty, arg_1_ty)

#define GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(name, ret_ty, arg_1_ty)  \
  GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(name, ret_ty, arg_1_ty, char)             \
  GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(name, ret_ty, arg_1_ty, uchar)            \
  GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(name, ret_ty, arg_1_ty, short)            \
  GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(name, ret_ty, arg_1_ty, ushort)           \
  GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(name, ret_ty, arg_1_ty, int)              \
  GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(name, ret_ty, arg_1_ty, uint)             \
  GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(name, ret_ty, arg_1_ty, long)             \
  GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(name, ret_ty, arg_1_ty, ulong)

namespace cl {
namespace __spirv {
/* ----------------- 4.13.3 Math functions. Device version ------------------*/
// TODO: Enable built-in functions with 'half' parameters once 'half' data type
/// is supported by the clang
// genfloat exp (genfloat x )
GEN_FUNC_OF_ONE_ARG(exp, float, float)
GEN_FUNC_OF_ONE_ARG(exp, double, double)
// GEN_FUNC_OF_ONE_ARG(exp, half, half)

// genfloat fmax (genfloat x, genfloat y)
GEN_FUNC_OF_TWO_ARG(fmax, float, float, float)
GEN_FUNC_OF_TWO_ARG(fmax, double, double, double)
// GEN_FUNC_OF_TWO_ARG(fmax, half, half, half)

// genfloat fmax (genfloat x, sgenfloat y)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmax, float, float, float)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmax, double, double, float)
// GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmax, half, half, float)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmax, float, float, double)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmax, double, double, double)
// GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmax, half, half, double)
// GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmax, float, float, half)
// GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmax, double, double, half)
// GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmax, half, half, half)

// genfloat fmin (genfloat x, genfloat y)
GEN_FUNC_OF_TWO_ARG(fmin, float, float, float)
GEN_FUNC_OF_TWO_ARG(fmin, double, double, double)
// GEN_FUNC_OF_TWO_ARG(fmin, half, half, half)

// genfloat fmin (genfloat x, sgenfloat y)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmin, float, float, float)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmin, double, double, float)
// GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmin, half, half, float)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmin, float, float, double)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmin, double, double, double)
// GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmin, half, half, double)
// GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmin, float, float, half)
// GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmin, double, double, half)
// GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S(fmin, half, half, half)

// genfloat sqrt (genfloat x)
GEN_FUNC_OF_ONE_ARG(sqrt, float, float)
GEN_FUNC_OF_ONE_ARG(sqrt, double, double)
// GEN_FUNC_OF_ONE_ARG(sqrt, half, half)

// genfloatf log (genfloatf x)
GEN_FUNC_OF_ONE_ARG(log, float, float)

// genfloatf sin (genfloatf x)
GEN_FUNC_OF_ONE_ARG(sin, float, float)

// genfloatf cos (genfloatf x)
GEN_FUNC_OF_ONE_ARG(cos, float, float)

// genfloat mad (genfloat a, genfloat b, genfloat c)
GEN_FUNC_OF_THREE_ARG(mad, float, float, float, float)
GEN_FUNC_OF_THREE_ARG(mad, double, double, double, double)
// GEN_FUNC_OF_THREE_ARG_V(mad, half, half, half, half)

// genfloatf exp (genfloatf x)
GEN_FUNC_OF_ONE_ARG(native_exp, float, float)

// genfloatf fabs (genfloatf x)
GEN_FUNC_OF_ONE_ARG(fabs, float, float)
GEN_FUNC_OF_ONE_ARG(fabs, double, double)
// GEN_FUNC_OF_ONE_ARG(fabs, half, half)

/* --------------- 4.13.4 Integer functions. Device version -----------------*/
// geninteger max (geninteger x, geninteger y)
GEN_FUNC_OF_TWO_ARG(max, char, char, char)
GEN_FUNC_OF_TWO_ARG(max, uchar, uchar, uchar)
GEN_FUNC_OF_TWO_ARG(max, short, short, short)
GEN_FUNC_OF_TWO_ARG(max, ushort, ushort, ushort)
GEN_FUNC_OF_TWO_ARG(max, int, int, int)
GEN_FUNC_OF_TWO_ARG(max, uint, uint, uint)
GEN_FUNC_OF_TWO_ARG(max, long, long, long)
GEN_FUNC_OF_TWO_ARG(max, ulong, ulong, ulong)

// geninteger max (geninteger x, sgeninteger y)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(max, char, char)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(max, uchar, uchar)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(max, short, short)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(max, ushort, ushort)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(max, int, int)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(max, uint, uint)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(max, long, long)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(max, ulong, ulong)

// geninteger min (geninteger x, geninteger y)
GEN_FUNC_OF_TWO_ARG(min, char, char, char)
GEN_FUNC_OF_TWO_ARG(min, uchar, uchar, uchar)
GEN_FUNC_OF_TWO_ARG(min, short, short, short)
GEN_FUNC_OF_TWO_ARG(min, ushort, ushort, ushort)
GEN_FUNC_OF_TWO_ARG(min, int, int, int)
GEN_FUNC_OF_TWO_ARG(min, uint, uint, uint)
GEN_FUNC_OF_TWO_ARG(min, long, long, long)
GEN_FUNC_OF_TWO_ARG(min, ulong, ulong, ulong)

// geninteger min (geninteger x, sgeninteger y)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(min, char, char)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(min, uchar, uchar)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(min, short, short)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(min, ushort, ushort)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(min, int, int)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(min, uint, uint)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(min, long, long)
GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER(min, ulong, ulong)
/* --------------- 4.13.5 Common functions. Device version ------------------*/
/* --------------- 4.13.6 Geometric Functions. Device version ---------------*/
/* --------------- 4.13.7 Relational functions. Device version --------------*/
} // namespace __spirv
} // namespace cl

#undef CONCAT_HELP
#undef CONCAT
#undef SCALAR
#undef VECTOR
#undef MAKE_FUN_OF_1_ARG
#undef MAKE_FUN_OF_2_ARG
#undef MAKE_FUN_OF_3_ARG
#undef GEN_FUNC_OF_ONE_ARG_V
#undef GEN_FUNC_OF_TWO_ARG_V
#undef GEN_FUNC_OF_THREE_ARG_V
#undef GEN_FUNC_OF_ONE_ARG_S
#undef GEN_FUNC_OF_TWO_ARG_S
#undef GEN_FUNC_OF_THREE_ARG_S
#undef GEN_FUNC_OF_ONE_ARG
#undef GEN_FUNC_OF_TWO_ARG
#undef GEN_FUNC_OF_THREE_ARG
#undef GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S
#undef GEN_FUNC_OF_TWO_ARG_S_SECOND_ARG_S
#undef GEN_FUNC_OF_TWO_ARG_SECOND_ARG_S
#undef GEN_FUNC_OF_TWO_ARG_V_SECOND_ARG_S_GENINTEGER
#endif // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__
namespace __sycl_std = cl::__spirv;
#else
namespace __sycl_std = std;
#endif

namespace cl {
namespace sycl {
template <typename T> T cos(T x) {
  return __sycl_std::cos(x);
}
template <typename T> T exp(T x) {
  return __sycl_std::exp(x);
}
template <typename T1, typename T2> T1 fmax(T1 x, T2 y) {
  return __sycl_std::fmax(x, y);
}
template <typename T1, typename T2> T1 fmin(T1 x, T2 y) {
  return __sycl_std::fmin(x, y);
}
template <typename T> T log(T x) {
  return __sycl_std::log(x);
}
template <typename T> T mad(T a, T b, T c) {
#ifdef __SYCL_DEVICE_ONLY__
  return __sycl_std::mad(a, b, c);
#else
  return (a * b) + c;
#endif
}
template <typename T1, typename T2> T1 max(T1 x, T2 y) {
  return __sycl_std::max(x, y);
}
template <typename T1, typename T2> T1 min(T1 x, T2 y) {
  return __sycl_std::min(x, y);
}
template <typename T> T sin(T x) {
  return __sycl_std::sin(x);
}
template <typename T> T sqrt(T x) {
  return __sycl_std::sqrt(x);
}
template <typename T> T fabs(T x) {
  return __sycl_std::fabs(x);
}
namespace native {
template <typename T> T exp(T x) {
#ifdef __SYCL_DEVICE_ONLY__
  return __sycl_std::native_exp(x);
#else
  return __sycl_std::exp(x);
#endif
}
} // namespace native
namespace half_precision {} // namespace half_precision
} // namespace sycl
} // namespace cl
