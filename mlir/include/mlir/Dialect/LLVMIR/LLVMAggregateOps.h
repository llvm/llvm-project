//===- LLVMAggregateOps.h - LLVM dialect aggregate ops ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ops of op partition `Aggregate` of the LLVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMAGGREGATEOPS_H_
#define MLIR_DIALECT_LLVMIR_LLVMAGGREGATEOPS_H_

#include "mlir/Dialect/LLVMIR/LLVMDialectDecl.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMAggregateOps.h.inc"

namespace mlir {
namespace LLVM {
/// Convert an array of integer attributes to a vector of integers that can be
/// used as indices in LLVM operations.
template <typename IntT = int64_t>
SmallVector<IntT> convertArrayToIndices(ArrayRef<Attribute> attrs) {
  SmallVector<IntT> indices;
  indices.reserve(attrs.size());
  for (Attribute attr : attrs)
    indices.push_back(cast<IntegerAttr>(attr).getInt());
  return indices;
}

/// Convert an `ArrayAttr` of integer attributes to a vector of integers that
/// can be used as indices in LLVM operations.
template <typename IntT = int64_t>
SmallVector<IntT> convertArrayToIndices(ArrayAttr attrs) {
  return convertArrayToIndices<IntT>(attrs.getValue());
}
} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_LLVMAGGREGATEOPS_H_
