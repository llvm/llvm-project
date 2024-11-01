//===- VectorOps.h - MLIR Vector Dialect Operations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Vector dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_IR_VECTOROPS_H
#define MLIR_DIALECT_VECTOR_IR_VECTOROPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Vector/Interfaces/MaskableOpInterface.h"
#include "mlir/Dialect/Vector/Interfaces/MaskingOpInterface.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/StringExtras.h"

// Pull in all enum type definitions and utility function declarations.
#include "mlir/Dialect/Vector/IR/VectorOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Vector/IR/VectorOpsAttrDefs.h.inc"

namespace mlir {
class MLIRContext;
class RewritePatternSet;

namespace arith {
enum class AtomicRMWKind : uint64_t;
} // namespace arith

namespace vector {
class ContractionOp;
class TransferReadOp;
class TransferWriteOp;
class VectorDialect;

namespace detail {
struct BitmaskEnumStorage;
} // namespace detail

/// Default callback to build a region with a 'vector.yield' terminator with no
/// arguments.
void buildTerminatedBody(OpBuilder &builder, Location loc);

/// Return whether `srcType` can be broadcast to `dstVectorType` under the
/// semantics of the `vector.broadcast` op.
enum class BroadcastableToResult {
  Success = 0,
  SourceRankHigher = 1,
  DimensionMismatch = 2,
  SourceTypeNotAVector = 3
};
BroadcastableToResult
isBroadcastableTo(Type srcType, VectorType dstVectorType,
                  std::pair<int, int> *mismatchingDims = nullptr);

/// Collect a set of vector-to-vector canonicalization patterns.
void populateVectorToVectorCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    PatternBenefit benefit = 1);

/// Collect a set of patterns that fold arithmetic extension on floating point
/// into vector contract for the backends with native support.
void populateFoldArithExtensionPatterns(RewritePatternSet &patterns);

/// Returns the integer type required for subscripts in the vector dialect.
IntegerType getVectorSubscriptType(Builder &builder);

/// Returns an integer array attribute containing the given values using
/// the integer type required for subscripts in the vector dialect.
ArrayAttr getVectorSubscriptAttr(Builder &b, ArrayRef<int64_t> values);

/// Returns the value obtained by reducing the vector into a scalar using the
/// operation kind associated with a binary AtomicRMWKind op.
Value getVectorReductionOp(arith::AtomicRMWKind op, OpBuilder &builder,
                           Location loc, Value vector);

/// Build the default minor identity map suitable for a vector transfer. This
/// also handles the case memref<... x vector<...>> -> vector<...> in which the
/// rank of the identity map must take the vector element type into account.
AffineMap getTransferMinorIdentityMap(ShapedType shapedType,
                                      VectorType vectorType);

/// Return true if the transfer_write fully writes the data accessed by the
/// transfer_read.
bool checkSameValueRAW(TransferWriteOp defWrite, TransferReadOp read);

/// Return true if the write op fully over-write the priorWrite transfer_write
/// op.
bool checkSameValueWAW(TransferWriteOp write, TransferWriteOp priorWrite);

/// Same behavior as `isDisjointTransferSet` but doesn't require the operations
/// to have the same tensor/memref. This allows comparing operations accessing
/// different tensors.
bool isDisjointTransferIndices(VectorTransferOpInterface transferA,
                               VectorTransferOpInterface transferB);

/// Return true if we can prove that the transfer operations access disjoint
/// memory.
bool isDisjointTransferSet(VectorTransferOpInterface transferA,
                           VectorTransferOpInterface transferB);

/// Return the result value of reducing two scalar/vector values with the
/// corresponding arith operation.
Value makeArithReduction(OpBuilder &b, Location loc, CombiningKind kind,
                         Value v1, Value acc, Value mask = Value());

/// Returns true if `attr` has "parallel" iterator type semantics.
inline bool isParallelIterator(Attribute attr) {
  return cast<IteratorTypeAttr>(attr).getValue() == IteratorType::parallel;
}

/// Returns true if `attr` has "reduction" iterator type semantics.
inline bool isReductionIterator(Attribute attr) {
  return cast<IteratorTypeAttr>(attr).getValue() == IteratorType::reduction;
}

//===----------------------------------------------------------------------===//
// Vector Masking Utilities
//===----------------------------------------------------------------------===//

/// Create the vector.yield-ended region of a vector.mask op with `maskableOp`
/// as masked operation.
void createMaskOpRegion(OpBuilder &builder, Operation *maskableOp);

/// Creates a vector.mask operation around a maskable operation. Returns the
/// vector.mask operation if the mask provided is valid. Otherwise, returns the
/// maskable operation itself.
Operation *maskOperation(OpBuilder &builder, Operation *maskableOp, Value mask,
                         Value passthru = Value());

/// Creates a vector select operation that picks values from `newValue` or
/// `passthru` for each result vector lane based on `mask`. This utility is used
/// to propagate the pass-thru value for masked-out or expeculatively executed
/// lanes. VP intrinsics do not support pass-thru values and every mask-out lane
/// is set to poison. LLVM backends are usually able to match op + select
/// patterns and fold them into a native target instructions.
Value selectPassthru(OpBuilder &builder, Value mask, Value newValue,
                     Value passthru);

} // namespace vector
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Vector/IR/VectorOps.h.inc"
#include "mlir/Dialect/Vector/IR/VectorOpsDialect.h.inc"

#endif // MLIR_DIALECT_VECTOR_IR_VECTOROPS_H
