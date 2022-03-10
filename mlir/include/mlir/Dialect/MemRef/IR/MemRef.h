//===- MemRef.h - MemRef dialect --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_IR_MEMREF_H_
#define MLIR_DIALECT_MEMREF_IR_MEMREF_H_

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {

class Location;
class OpBuilder;

raw_ostream &operator<<(raw_ostream &os, const Range &range);

/// Return the list of Range (i.e. offset, size, stride). Each Range
/// entry contains either the dynamic value or a ConstantIndexOp constructed
/// with `b` at location `loc`.
SmallVector<Range, 8> getOrCreateRanges(OffsetSizeAndStrideOpInterface op,
                                        OpBuilder &b, Location loc);

namespace memref {

/// This is a common utility used for patterns of the form
/// "someop(memref.cast) -> someop". It folds the source of any memref.cast
/// into the root operation directly.
LogicalResult foldMemRefCast(Operation *op, Value inner = nullptr);

/// Return an unranked/ranked tensor type for the given unranked/ranked memref
/// type.
Type getTensorTypeFromMemRefType(Type type);

/// Finds a single dealloc operation for the given allocated value. If there
/// are > 1 deallocates for `allocValue`, returns None, else returns the single
/// deallocate if it exists or nullptr.
Optional<Operation *> findDealloc(Value allocValue);

} // namespace memref
} // namespace mlir

//===----------------------------------------------------------------------===//
// MemRef Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRefOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// MemRef Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MemRef/IR/MemRefOps.h.inc"

#endif // MLIR_DIALECT_MEMREF_IR_MEMREF_H_
