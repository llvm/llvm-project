//===- Support.h - C API Helpers Implementation -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for converting MLIR C++ objects into helper
// C structures for the purpose of C API. This file should not be included from
// C++ code other than C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_SUPPORT_H
#define MLIR_CAPI_SUPPORT_H

#include "mlir-c/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace llvm {
class ThreadPoolInterface;
} // namespace llvm

/// Converts a StringRef into its MLIR C API equivalent.
inline MlirStringRef wrap(llvm::StringRef ref) {
  return mlirStringRefCreate(ref.data(), ref.size());
}

/// Creates a StringRef out of its MLIR C API equivalent.
inline llvm::StringRef unwrap(MlirStringRef ref) {
  return llvm::StringRef(ref.data, ref.length);
}

inline MlirLogicalResult wrap(llvm::LogicalResult res) {
  if (mlir::succeeded(res))
    return mlirLogicalResultSuccess();
  return mlirLogicalResultFailure();
}

inline llvm::LogicalResult unwrap(MlirLogicalResult res) {
  return mlir::success(mlirLogicalResultIsSuccess(res));
}

DEFINE_C_API_PTR_METHODS(MlirLlvmThreadPool, llvm::ThreadPoolInterface)
DEFINE_C_API_METHODS(MlirTypeID, mlir::TypeID)
DEFINE_C_API_PTR_METHODS(MlirTypeIDAllocator, mlir::TypeIDAllocator)

#endif // MLIR_CAPI_SUPPORT_H
