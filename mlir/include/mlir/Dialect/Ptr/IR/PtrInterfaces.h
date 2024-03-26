//===- PtrInterfaces.h - Ptr Interfaces -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op interfaces for the Ptr dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTR_IR_PTRINTERFACES_H
#define MLIR_DIALECT_PTR_IR_PTRINTERFACES_H

#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"

namespace mlir {
namespace ptr {
namespace detail {
/// Verifies the access groups attribute of memory operations that implement the
/// access group interface.
LogicalResult verifyAccessGroupOpInterface(Operation *op);

/// Verifies the alias analysis attributes of memory operations that implement
/// the alias analysis interface.
LogicalResult verifyAliasAnalysisOpInterface(Operation *op);
} // namespace detail
} // namespace ptr
} // namespace mlir

#include "mlir/Dialect/Ptr/IR/PtrInterfaces.h.inc"

#endif // MLIR_DIALECT_PTR_IR_PTRINTERFACES_H
