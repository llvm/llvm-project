//===- Wrap.h - C API Utilities ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains common definitions for wrapping opaque C++ pointers into
// C structures for the purpose of C API. This file should not be included from
// C++ code other than C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_WRAP_H
#define MLIR_CAPI_WRAP_H

#include "mlir-c/IR.h"
#include "mlir/Support/LLVM.h"

/* ========================================================================== */
/* Definitions of methods for non-owning structures used in C API.            */
/* ========================================================================== */

#define DEFINE_C_API_PTR_METHODS(name, cpptype)                                \
  static inline name wrap(cpptype *cpp) { return name{cpp}; }                  \
  static inline cpptype *unwrap(name c) {                                      \
    return static_cast<cpptype *>(c.ptr);                                      \
  }

#define DEFINE_C_API_METHODS(name, cpptype)                                    \
  static inline name wrap(cpptype cpp) {                                       \
    return name{cpp.getAsOpaquePointer()};                                     \
  }                                                                            \
  static inline cpptype unwrap(name c) {                                       \
    return cpptype::getFromOpaquePointer(c.ptr);                               \
  }

template <typename CppTy, typename CTy>
static llvm::ArrayRef<CppTy> unwrapList(size_t size, CTy *first,
                                        llvm::SmallVectorImpl<CppTy> &storage) {
  static_assert(
      std::is_same<decltype(unwrap(std::declval<CTy>())), CppTy>::value,
      "incompatible C and C++ types");

  if (size == 0)
    return llvm::None;

  assert(storage.empty() && "expected to populate storage");
  storage.reserve(size);
  for (size_t i = 0; i < size; ++i)
    storage.push_back(unwrap(*(first + i)));
  return storage;
}

#endif // MLIR_CAPI_WRAP_H
