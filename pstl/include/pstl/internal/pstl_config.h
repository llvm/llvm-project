// -*- C++ -*-
//===-- pstl_config.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __PSTL_config_H
#define __PSTL_config_H

#define PSTL_VERSION 203
#define PSTL_VERSION_MAJOR (PSTL_VERSION / 100)
#define PSTL_VERSION_MINOR (PSTL_VERSION - PSTL_VERSION_MAJOR * 100)

// Check the user-defined macro for parallel policies
#if defined(PSTL_USE_PARALLEL_POLICIES)
#    undef __PSTL_USE_PAR_POLICIES
#    define __PSTL_USE_PAR_POLICIES PSTL_USE_PARALLEL_POLICIES
// Check the internal macro for parallel policies
#elif !defined(__PSTL_USE_PAR_POLICIES)
#    define __PSTL_USE_PAR_POLICIES 1
#endif

#if __PSTL_USE_PAR_POLICIES
#    if !defined(__PSTL_PAR_BACKEND_TBB)
#        define __PSTL_PAR_BACKEND_TBB 1
#    endif
#else
#    undef __PSTL_PAR_BACKEND_TBB
#endif

// Check the user-defined macro for warnings
#if defined(PSTL_USAGE_WARNINGS)
#    undef __PSTL_USAGE_WARNINGS
#    define __PSTL_USAGE_WARNINGS PSTL_USAGE_WARNINGS
// Check the internal macro for warnings
#elif !defined(__PSTL_USAGE_WARNINGS)
#    define __PSTL_USAGE_WARNINGS 0
#endif

// Portability "#pragma" definition
#ifdef _MSC_VER
#    define __PSTL_PRAGMA(x) __pragma(x)
#else
#    define __PSTL_PRAGMA(x) _Pragma(#    x)
#endif

#define __PSTL_STRING_AUX(x) #x
#define __PSTL_STRING(x) __PSTL_STRING_AUX(x)
#define __PSTL_STRING_CONCAT(x, y) x #y

// note that when ICC or Clang is in use, __PSTL_GCC_VERSION might not fully match
// the actual GCC version on the system.
#define __PSTL_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#if __clang__
// according to clang documentation, version can be vendor specific
#    define __PSTL_CLANG_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
#endif

// Enable SIMD for compilers that support OpenMP 4.0
#if (_OPENMP >= 201307) || (__INTEL_COMPILER >= 1600) || (!defined(__INTEL_COMPILER) && __PSTL_GCC_VERSION >= 40900)
#    define __PSTL_PRAGMA_SIMD __PSTL_PRAGMA(omp simd)
#    define __PSTL_PRAGMA_DECLARE_SIMD __PSTL_PRAGMA(omp declare simd)
#    define __PSTL_PRAGMA_SIMD_REDUCTION(PRM) __PSTL_PRAGMA(omp simd reduction(PRM))
#elif !defined(_MSC_VER) //#pragma simd
#    define __PSTL_PRAGMA_SIMD __PSTL_PRAGMA(simd)
#    define __PSTL_PRAGMA_DECLARE_SIMD
#    define __PSTL_PRAGMA_SIMD_REDUCTION(PRM) __PSTL_PRAGMA(simd reduction(PRM))
#else //no simd
#    define __PSTL_PRAGMA_SIMD
#    define __PSTL_PRAGMA_DECLARE_SIMD
#    define __PSTL_PRAGMA_SIMD_REDUCTION(PRM)
#endif //Enable SIMD

#if (__INTEL_COMPILER)
#    define __PSTL_PRAGMA_FORCEINLINE __PSTL_PRAGMA(forceinline)
#else
#    define __PSTL_PRAGMA_FORCEINLINE
#endif

#if (__INTEL_COMPILER >= 1900)
#    define __PSTL_PRAGMA_SIMD_SCAN(PRM) __PSTL_PRAGMA(omp simd reduction(inscan, PRM))
#    define __PSTL_PRAGMA_SIMD_INCLUSIVE_SCAN(PRM) __PSTL_PRAGMA(omp scan inclusive(PRM))
#    define __PSTL_PRAGMA_SIMD_EXCLUSIVE_SCAN(PRM) __PSTL_PRAGMA(omp scan exclusive(PRM))
#else
#    define __PSTL_PRAGMA_SIMD_SCAN(PRM)
#    define __PSTL_PRAGMA_SIMD_INCLUSIVE_SCAN(PRM)
#    define __PSTL_PRAGMA_SIMD_EXCLUSIVE_SCAN(PRM)
#endif

// Should be defined to 1 for environments with a vendor implementation of C++17 execution policies
#define __PSTL_CPP17_EXECUTION_POLICIES_PRESENT (_MSC_VER >= 1912)

#define __PSTL_CPP14_2RANGE_MISMATCH_EQUAL_PRESENT                                                                     \
    (_MSC_VER >= 1900 || __cplusplus >= 201300L || __cpp_lib_robust_nonmodifying_seq_ops == 201304)
#define __PSTL_CPP14_MAKE_REVERSE_ITERATOR_PRESENT                                                                     \
    (_MSC_VER >= 1900 || __cplusplus >= 201402L || __cpp_lib_make_reverse_iterator == 201402)
#define __PSTL_CPP14_INTEGER_SEQUENCE_PRESENT (_MSC_VER >= 1900 || __cplusplus >= 201402L)
#define __PSTL_CPP14_VARIABLE_TEMPLATES_PRESENT                                                                        \
    (!__INTEL_COMPILER || __INTEL_COMPILER >= 1700) && (_MSC_FULL_VER >= 190023918 || __cplusplus >= 201402L)

#define __PSTL_EARLYEXIT_PRESENT (__INTEL_COMPILER >= 1800)
#define __PSTL_MONOTONIC_PRESENT (__INTEL_COMPILER >= 1800)

#if (__INTEL_COMPILER >= 1900 || !defined(__INTEL_COMPILER) && __PSTL_GCC_VERSION >= 40900 || _OPENMP >= 201307)
#    define __PSTL_UDR_PRESENT 1
#else
#    define __PSTL_UDR_PRESENT 0
#endif

#define __PSTL_UDS_PRESENT (__INTEL_COMPILER >= 1900 && __INTEL_COMPILER_BUILD_DATE >= 20180626)

#if __PSTL_EARLYEXIT_PRESENT
#    define __PSTL_PRAGMA_SIMD_EARLYEXIT __PSTL_PRAGMA(omp simd early_exit)
#else
#    define __PSTL_PRAGMA_SIMD_EARLYEXIT
#endif

#if __PSTL_MONOTONIC_PRESENT
#    define __PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC(PRM) __PSTL_PRAGMA(omp ordered simd monotonic(PRM))
#    define __PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC_2ARGS(PRM1, PRM2) __PSTL_PRAGMA(omp ordered simd monotonic(PRM1, PRM2))
#else
#    define __PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC(PRM)
#    define __PSTL_PRAGMA_SIMD_ORDERED_MONOTONIC_2ARGS(PRM1, PRM2)
#endif

// Declaration of reduction functor, where
// NAME - the name of the functor
// OP - type of the callable object with the reduction operation
// omp_in - refers to the local partial result
// omp_out - refers to the final value of the combiner operator
// omp_priv - refers to the private copy of the initial value
// omp_orig - refers to the original variable to be reduced
#define __PSTL_PRAGMA_DECLARE_REDUCTION(NAME, OP)                                                                      \
    __PSTL_PRAGMA(omp declare reduction(NAME:OP : omp_out(omp_in)) initializer(omp_priv = omp_orig))

#if (__INTEL_COMPILER >= 1600)
#    define __PSTL_PRAGMA_VECTOR_UNALIGNED __PSTL_PRAGMA(vector unaligned)
#else
#    define __PSTL_PRAGMA_VECTOR_UNALIGNED
#endif

// Check the user-defined macro to use non-temporal stores
#if defined(PSTL_USE_NONTEMPORAL_STORES) && (__INTEL_COMPILER >= 1600)
#    define __PSTL_USE_NONTEMPORAL_STORES_IF_ALLOWED __PSTL_PRAGMA(vector nontemporal)
#else
#    define __PSTL_USE_NONTEMPORAL_STORES_IF_ALLOWED
#endif

#if _MSC_VER || __INTEL_COMPILER //the preprocessors don't type a message location
#    define __PSTL_PRAGMA_LOCATION __FILE__ ":" __PSTL_STRING(__LINE__) ": [Parallel STL message]: "
#else
#    define __PSTL_PRAGMA_LOCATION " [Parallel STL message]: "
#endif

#define __PSTL_PRAGMA_MESSAGE_IMPL(x) __PSTL_PRAGMA(message(__PSTL_STRING_CONCAT(__PSTL_PRAGMA_LOCATION, x)))

#if __PSTL_USAGE_WARNINGS
#    define __PSTL_PRAGMA_MESSAGE(x) __PSTL_PRAGMA_MESSAGE_IMPL(x)
#    define __PSTL_PRAGMA_MESSAGE_POLICIES(x) __PSTL_PRAGMA_MESSAGE_IMPL(x)
#else
#    define __PSTL_PRAGMA_MESSAGE(x)
#    define __PSTL_PRAGMA_MESSAGE_POLICIES(x)
#endif

// broken macros
#define __PSTL_CPP11_STD_ROTATE_BROKEN ((__GLIBCXX__ && __GLIBCXX__ < 20150716) || (_MSC_VER && _MSC_VER < 1800))

#define __PSTL_ICC_18_OMP_SIMD_BROKEN (__INTEL_COMPILER == 1800)

#endif /* __PSTL_config_H */
