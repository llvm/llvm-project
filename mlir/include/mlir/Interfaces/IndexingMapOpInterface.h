//===- IndexingMapOpInterface.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_INDEXING_MAP_OP_INTERFACE_H_
#define MLIR_INTERFACES_INDEXING_MAP_OP_INTERFACE_H_

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace detail {
/// Verify that `op` conforms to the invariants of StructuredOpInterface
LogicalResult verifyIndexingMapOpInterface(Operation *op);
} // namespace detail
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/IndexingMapOpInterface.h.inc"

#endif // MLIR_INTERFACES_INDEXING_MAP_OP_INTERFACE_H_
