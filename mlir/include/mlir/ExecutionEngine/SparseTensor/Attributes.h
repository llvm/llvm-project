//===- Attributes.h - C++ attributes for SparseTensorRuntime ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines various macros for using C++ attributes whenever
// they're supported by the compiler.  These macros are the same as the
// versions in the LLVMSupport library, but we define our own versions
// in order to avoid introducing that dependency just for the sake of
// these macros.  (If we ever do end up depending on LLVMSupport, then
// we should remove this header and use "llvm/Support/Compiler.h" instead.)
//
// This file is part of the lightweight runtime support library for sparse
// tensor manipulations.  The functionality of the support library is meant
// to simplify benchmarking, testing, and debugging MLIR code operating on
// sparse tensors.  However, the provided functionality is **not** part of
// core MLIR itself.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSOR_ATTRIBUTES_H
#define MLIR_EXECUTIONENGINE_SPARSETENSOR_ATTRIBUTES_H

// A wrapper around `__has_cpp_attribute` for C++11 style attributes,
// which are defined by ISO C++ SD-6.
// <https://en.cppreference.com/w/cpp/experimental/feature_test>
#if defined(__cplusplus) && defined(__has_cpp_attribute)
// NOTE: The __cplusplus requirement should be unnecessary, but guards
// against issues with GCC <https://bugs.llvm.org/show_bug.cgi?id=23435>.
#define MLIR_SPARSETENSOR_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define MLIR_SPARSETENSOR_HAS_CPP_ATTRIBUTE(x) 0
#endif

// A wrapper around `__has_attribute`, which is defined by GCC 5+ and Clang.
// GCC: <https://gcc.gnu.org/gcc-5/changes.html>
// Clang: <https://clang.llvm.org/docs/LanguageExtensions.html>
#ifdef __has_attribute
#define MLIR_SPARSETENSOR_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#define MLIR_SPARSETENSOR_HAS_ATTRIBUTE(x) 0
#endif

// An attribute for non-owning classes (like `PermutationRef`) to enable
// lifetime warnings.
#if MLIR_SPARSETENSOR_HAS_CPP_ATTRIBUTE(gsl::Pointer)
#define MLIR_SPARSETENSOR_GSL_POINTER [[gsl::Pointer]]
#else
#define MLIR_SPARSETENSOR_GSL_POINTER
#endif

// An attribute for functions which are "pure" in the following sense:
// * the result depends only on the function arguments
// * pointer arguments are allowed to be read from, but not written to
// * has no observable effects other than the return value
//
// This allows the compiler to avoid repeated function calls whenever
// it can determine that the arguments are the same and that memory has
// not changed.
//
// This macro is called `LLVM_READONLY` by LLVMSupport.  This definition
// differs slightly from the LLVM version by using `gnu::pure` when
// available, like Abseil's `ABSL_ATTRIBUTE_PURE_FUNCTION`.
#if MLIR_SPARSETENSOR_HAS_CPP_ATTRIBUTE(gnu::pure)
#define MLIR_SPARSETENSOR_PURE [[gnu::pure]]
#elif MLIR_SPARSETENSOR_HAS_ATTRIBUTE(pure) || defined(__GNUC__)
#define MLIR_SPARSETENSOR_PURE __attribute__((pure))
#else
#define MLIR_SPARSETENSOR_PURE
#endif

#endif // MLIR_EXECUTIONENGINE_SPARSETENSOR_ATTRIBUTES_H
