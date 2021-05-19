//===- MemRef.h - MemRef dialect --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_IR_MEMREF_H_
#define MLIR_DIALECT_MEMREF_IR_MEMREF_H_

#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {

class Location;
class OpBuilder;

raw_ostream &operator<<(raw_ostream &os, Range &range);

/// Return the list of Range (i.e. offset, size, stride). Each Range
/// entry contains either the dynamic value or a ConstantIndexOp constructed
/// with `b` at location `loc`.
SmallVector<Range, 8> getOrCreateRanges(OffsetSizeAndStrideOpInterface op,
                                        OpBuilder &b, Location loc);

/// Given an operation, retrieves the value of each dynamic dimension through
/// constructing the necessary DimOp operators.
SmallVector<Value, 4> getDynOperands(Location loc, Value val, OpBuilder &b);
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

namespace mlir {
namespace memref {
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
// transferred. Either both or no stride arguments should be specified. If the
// source and destination locations overlap the behavior of this operation is
// not defined.
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
    return {(*this)->operand_begin() + 1,
            (*this)->operand_begin() + 1 + getSrcMemRefRank()};
  }

  // Returns the destination MemRefType for this DMA operations.
  Value getDstMemRef() { return getOperand(1 + getSrcMemRefRank()); }
  // Returns the rank (number of indices) of the destination MemRefType.
  unsigned getDstMemRefRank() {
    return getDstMemRef().getType().cast<MemRefType>().getRank();
  }
  unsigned getSrcMemorySpace() {
    return getSrcMemRef().getType().cast<MemRefType>().getMemorySpaceAsInt();
  }
  unsigned getDstMemorySpace() {
    return getDstMemRef().getType().cast<MemRefType>().getMemorySpaceAsInt();
  }

  // Returns the destination memref indices for this DMA operation.
  operand_range getDstIndices() {
    return {(*this)->operand_begin() + 1 + getSrcMemRefRank() + 1,
            (*this)->operand_begin() + 1 + getSrcMemRefRank() + 1 +
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
    return {(*this)->operand_begin() + tagIndexStartPos,
            (*this)->operand_begin() + tagIndexStartPos + getTagMemRefRank()};
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

  static StringRef getOperationName() { return "memref.dma_start"; }
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

  static StringRef getOperationName() { return "memref.dma_wait"; }

  // Returns the Tag MemRef associated with the DMA operation being waited on.
  Value getTagMemRef() { return getOperand(0); }

  // Returns the tag memref index for this DMA operation.
  operand_range getTagIndices() {
    return {(*this)->operand_begin() + 1,
            (*this)->operand_begin() + 1 + getTagMemRefRank()};
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
} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_IR_MEMREF_H_
