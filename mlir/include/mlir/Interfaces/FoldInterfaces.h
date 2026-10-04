//===- FoldInterfaces.h - Folding Interfaces --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_INTERFACES_FOLDINTERFACES_H_
#define MLIR_INTERFACES_FOLDINTERFACES_H_

#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpFoldResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Attribute;
class DialectFoldInterface;
class Operation;
class Region;

namespace detail {
/// Fold `op` with the legacy vector `fold` method of `iface`, with the strict
/// legacy contract.
OpFoldResults
foldWithLegacyDialectFoldInterface(const DialectFoldInterface *iface,
                                   Operation *op, ArrayRef<Attribute> operands);
} // namespace detail
} // namespace mlir

#include "mlir/Interfaces/DialectFoldInterface.h.inc"

#endif // MLIR_INTERFACES_FOLDINTERFACES_H_
