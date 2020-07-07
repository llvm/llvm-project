//===- Ops.h - Standard MLIR Operations -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines convenience types for working with standard operations
// in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_STANDARDOPS_IR_OPS_H
#define MLIR_DIALECT_STANDARDOPS_IR_OPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorUnrollInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// Pull in all enum type definitions and utility function declarations.
#include "mlir/Dialect/StandardOps/IR/OpsEnums.h.inc"

namespace mlir {
class AffineMap;
class Builder;
class FuncOp;
class OpBuilder;

#define GET_OP_CLASSES
#include "mlir/Dialect/StandardOps/IR/Ops.h.inc"

#include "mlir/Dialect/StandardOps/IR/OpsDialect.h.inc"

/// This is a refinement of the "constant" op for the case where it is
/// returning a float value of FloatType.
///
///   %1 = "std.constant"(){value: 42.0} : bf16
///
class ConstantFloatOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;

  /// Builds a constant float op producing a float of the specified type.
  static void build(OpBuilder &builder, OperationState &result,
                    const APFloat &value, FloatType type);

  APFloat getValue() { return getAttrOfType<FloatAttr>("value").getValue(); }

  static bool classof(Operation *op);
};

/// This is a refinement of the "constant" op for the case where it is
/// returning an integer value of IntegerType.
///
///   %1 = "std.constant"(){value: 42} : i32
///
class ConstantIntOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;
  /// Build a constant int op producing an integer of the specified width.
  static void build(OpBuilder &builder, OperationState &result, int64_t value,
                    unsigned width);

  /// Build a constant int op producing an integer with the specified type,
  /// which must be an integer type.
  static void build(OpBuilder &builder, OperationState &result, int64_t value,
                    Type type);

  int64_t getValue() { return getAttrOfType<IntegerAttr>("value").getInt(); }

  static bool classof(Operation *op);
};

/// This is a refinement of the "constant" op for the case where it is
/// returning an integer value of Index type.
///
///   %1 = "std.constant"(){value: 99} : () -> index
///
class ConstantIndexOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;

  /// Build a constant int op producing an index.
  static void build(OpBuilder &builder, OperationState &result, int64_t value);

  int64_t getValue() { return getAttrOfType<IntegerAttr>("value").getInt(); }

  static bool classof(Operation *op);
};

// DmaStartOp starts a non-blocking DMA operation that transfers data from a
// source memref to a destination memref. The source and destination memref need
// not be of the same dimensionality, but need to have the same elemental type.
// The operands include the source and destination memref's each followed by its
// indices, size of the data transfer in terms of the number of elements (of the
// elemental type of the memref), a tag memref with its indices, and optionally
// at the end, a stride and a number_of_elements_per_stride arguments. The tag
// location is used by a DmaWaitOp to check for completion. The indices of the
// source memref, destination memref, and the tag memref have the same
// restrictions as any load/store. The optional stride arguments should be of
// 'index' type, and specify a stride for the slower memory space (memory space
// with a lower memory space id), transferring chunks of
// number_of_elements_per_stride every stride until %num_elements are
// transferred. Either both or no stride arguments should be specified.
//
// For example, a DmaStartOp operation that transfers 256 elements of a memref
// '%src' in memory space 0 at indices [%i, %j] to memref '%dst' in memory space
// 1 at indices [%k, %l], would be specified as follows:
//
//   %num_elements = constant 256
//   %idx = constant 0 : index
//   %tag = alloc() : memref<1 x i32, (d0) -> (d0), 4>
//   dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%idx] :
//     memref<40 x 128 x f32>, (d0) -> (d0), 0>,
//     memref<2 x 1024 x f32>, (d0) -> (d0), 1>,
//     memref<1 x i32>, (d0) -> (d0), 2>
//
//   If %stride and %num_elt_per_stride are specified, the DMA is expected to
//   transfer %num_elt_per_stride elements every %stride elements apart from
//   memory space 0 until %num_elements are transferred.
//
//   dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%idx], %stride,
//             %num_elt_per_stride :
//
// TODO: add additional operands to allow source and destination striding, and
// multiple stride levels.
// TODO: Consider replacing src/dst memref indices with view memrefs.
class DmaStartOp
    : public Op<DmaStartOp, OpTrait::VariadicOperands, OpTrait::ZeroResult> {
public:
  using Op::Op;

  static void build(OpBuilder &builder, OperationState &result, Value srcMemRef,
                    ValueRange srcIndices, Value destMemRef,
                    ValueRange destIndices, Value numElements, Value tagMemRef,
                    ValueRange tagIndices, Value stride = nullptr,
                    Value elementsPerStride = nullptr);

  // Returns the source MemRefType for this DMA operation.
  Value getSrcMemRef() { return getOperand(0); }
  // Returns the rank (number of indices) of the source MemRefType.
  unsigned getSrcMemRefRank() {
    return getSrcMemRef().getType().cast<MemRefType>().getRank();
  }
  // Returns the source memref indices for this DMA operation.
  operand_range getSrcIndices() {
    return {getOperation()->operand_begin() + 1,
            getOperation()->operand_begin() + 1 + getSrcMemRefRank()};
  }

  // Returns the destination MemRefType for this DMA operations.
  Value getDstMemRef() { return getOperand(1 + getSrcMemRefRank()); }
  // Returns the rank (number of indices) of the destination MemRefType.
  unsigned getDstMemRefRank() {
    return getDstMemRef().getType().cast<MemRefType>().getRank();
  }
  unsigned getSrcMemorySpace() {
    return getSrcMemRef().getType().cast<MemRefType>().getMemorySpace();
  }
  unsigned getDstMemorySpace() {
    return getDstMemRef().getType().cast<MemRefType>().getMemorySpace();
  }

  // Returns the destination memref indices for this DMA operation.
  operand_range getDstIndices() {
    return {getOperation()->operand_begin() + 1 + getSrcMemRefRank() + 1,
            getOperation()->operand_begin() + 1 + getSrcMemRefRank() + 1 +
                getDstMemRefRank()};
  }

  // Returns the number of elements being transferred by this DMA operation.
  Value getNumElements() {
    return getOperand(1 + getSrcMemRefRank() + 1 + getDstMemRefRank());
  }

  // Returns the Tag MemRef for this DMA operation.
  Value getTagMemRef() {
    return getOperand(1 + getSrcMemRefRank() + 1 + getDstMemRefRank() + 1);
  }
  // Returns the rank (number of indices) of the tag MemRefType.
  unsigned getTagMemRefRank() {
    return getTagMemRef().getType().cast<MemRefType>().getRank();
  }

  // Returns the tag memref index for this DMA operation.
  operand_range getTagIndices() {
    unsigned tagIndexStartPos =
        1 + getSrcMemRefRank() + 1 + getDstMemRefRank() + 1 + 1;
    return {getOperation()->operand_begin() + tagIndexStartPos,
            getOperation()->operand_begin() + tagIndexStartPos +
                getTagMemRefRank()};
  }

  /// Returns true if this is a DMA from a faster memory space to a slower one.
  bool isDestMemorySpaceFaster() {
    return (getSrcMemorySpace() < getDstMemorySpace());
  }

  /// Returns true if this is a DMA from a slower memory space to a faster one.
  bool isSrcMemorySpaceFaster() {
    // Assumes that a lower number is for a slower memory space.
    return (getDstMemorySpace() < getSrcMemorySpace());
  }

  /// Given a DMA start operation, returns the operand position of either the
  /// source or destination memref depending on the one that is at the higher
  /// level of the memory hierarchy. Asserts failure if neither is true.
  unsigned getFasterMemPos() {
    assert(isSrcMemorySpaceFaster() || isDestMemorySpaceFaster());
    return isSrcMemorySpaceFaster() ? 0 : getSrcMemRefRank() + 1;
  }

  static StringRef getOperationName() { return "std.dma_start"; }
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();

  LogicalResult fold(ArrayRef<Attribute> cstOperands,
                     SmallVectorImpl<OpFoldResult> &results);

  bool isStrided() {
    return getNumOperands() != 1 + getSrcMemRefRank() + 1 + getDstMemRefRank() +
                                   1 + 1 + getTagMemRefRank();
  }

  Value getStride() {
    if (!isStrided())
      return nullptr;
    return getOperand(getNumOperands() - 1 - 1);
  }

  Value getNumElementsPerStride() {
    if (!isStrided())
      return nullptr;
    return getOperand(getNumOperands() - 1);
  }
};

// DmaWaitOp blocks until the completion of a DMA operation associated with the
// tag element '%tag[%index]'. %tag is a memref, and %index has to be an index
// with the same restrictions as any load/store index. %num_elements is the
// number of elements associated with the DMA operation. For example:
//
//   dma_start %src[%i, %j], %dst[%k, %l], %num_elements, %tag[%index] :
//     memref<2048 x f32>, (d0) -> (d0), 0>,
//     memref<256 x f32>, (d0) -> (d0), 1>
//     memref<1 x i32>, (d0) -> (d0), 2>
//   ...
//   ...
//   dma_wait %tag[%index], %num_elements : memref<1 x i32, (d0) -> (d0), 2>
//
class DmaWaitOp
    : public Op<DmaWaitOp, OpTrait::VariadicOperands, OpTrait::ZeroResult> {
public:
  using Op::Op;

  static void build(OpBuilder &builder, OperationState &result, Value tagMemRef,
                    ValueRange tagIndices, Value numElements);

  static StringRef getOperationName() { return "std.dma_wait"; }

  // Returns the Tag MemRef associated with the DMA operation being waited on.
  Value getTagMemRef() { return getOperand(0); }

  // Returns the tag memref index for this DMA operation.
  operand_range getTagIndices() {
    return {getOperation()->operand_begin() + 1,
            getOperation()->operand_begin() + 1 + getTagMemRefRank()};
  }

  // Returns the rank (number of indices) of the tag memref.
  unsigned getTagMemRefRank() {
    return getTagMemRef().getType().cast<MemRefType>().getRank();
  }

  // Returns the number of elements transferred in the associated DMA operation.
  Value getNumElements() { return getOperand(1 + getTagMemRefRank()); }

  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult fold(ArrayRef<Attribute> cstOperands,
                     SmallVectorImpl<OpFoldResult> &results);
  LogicalResult verify();
};

/// Prints dimension and symbol list.
void printDimAndSymbolList(Operation::operand_iterator begin,
                           Operation::operand_iterator end, unsigned numDims,
                           OpAsmPrinter &p);

/// Parses dimension and symbol list and returns true if parsing failed.
ParseResult parseDimAndSymbolList(OpAsmParser &parser,
                                  SmallVectorImpl<Value> &operands,
                                  unsigned &numDims);

raw_ostream &operator<<(raw_ostream &os, SubViewOp::Range &range);

/// Determines whether MemRefCastOp casts to a more dynamic version of the
/// source memref. This is useful to to fold a memref_cast into a consuming op
/// and implement canonicalization patterns for ops in different dialects that
/// may consume the results of memref_cast operations. Such foldable memref_cast
/// operations are typically inserted as `view` and `subview` ops are
/// canonicalized, to preserve the type compatibility of their uses.
///
/// Returns true when all conditions are met:
/// 1. source and result are ranked memrefs with strided semantics and same
/// element type and rank.
/// 2. each of the source's size, offset or stride has more static information
/// than the corresponding result's size, offset or stride.
///
/// Example 1:
/// ```mlir
///   %1 = memref_cast %0 : memref<8x16xf32> to memref<?x?xf32>
///   %2 = consumer %1 ... : memref<?x?xf32> ...
/// ```
///
/// may fold into:
///
/// ```mlir
///   %2 = consumer %0 ... : memref<8x16xf32> ...
/// ```
///
/// Example 2:
/// ```
///   %1 = memref_cast %0 : memref<?x16xf32, affine_map<(i, j)->(16 * i + j)>>
///          to memref<?x?xf32>
///   consumer %1 : memref<?x?xf32> ...
/// ```
///
/// may fold into:
///
/// ```
///   consumer %0 ... : memref<?x16xf32, affine_map<(i, j)->(16 * i + j)>>
/// ```
bool canFoldIntoConsumerOp(MemRefCastOp castOp);

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known integer
/// comparison predicates.
bool applyCmpPredicate(CmpIPredicate predicate, const APInt &lhs,
                       const APInt &rhs);

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known floating point
/// comparison predicates.
bool applyCmpPredicate(CmpFPredicate predicate, const APFloat &lhs,
                       const APFloat &rhs);
} // end namespace mlir

#endif // MLIR_DIALECT_IR_STANDARDOPS_IR_OPS_H
