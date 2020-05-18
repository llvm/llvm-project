//===- AffineOps.h - MLIR Affine Operations -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines convenience types for working with Affine operations
// in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_IR_AFFINEOPS_H
#define MLIR_DIALECT_AFFINE_IR_AFFINEOPS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class AffineApplyOp;
class AffineBound;
class AffineDimExpr;
class AffineValueMap;
class AffineTerminatorOp;
class FlatAffineConstraints;
class OpBuilder;

/// A utility function to check if a value is defined at the top level of an
/// op with trait `AffineScope` or is a region argument for such an op. A value
/// of index type defined at the top level is always a valid symbol for all its
/// uses.
bool isTopLevelValue(Value value);

/// AffineDmaStartOp starts a non-blocking DMA operation that transfers data
/// from a source memref to a destination memref. The source and destination
/// memref need not be of the same dimensionality, but need to have the same
/// elemental type. The operands include the source and destination memref's
/// each followed by its indices, size of the data transfer in terms of the
/// number of elements (of the elemental type of the memref), a tag memref with
/// its indices, and optionally at the end, a stride and a
/// number_of_elements_per_stride arguments. The tag location is used by an
/// AffineDmaWaitOp to check for completion. The indices of the source memref,
/// destination memref, and the tag memref have the same restrictions as any
/// affine.load/store. In particular, index for each memref dimension must be an
/// affine expression of loop induction variables and symbols.
/// The optional stride arguments should be of 'index' type, and specify a
/// stride for the slower memory space (memory space with a lower memory space
/// id), transferring chunks of number_of_elements_per_stride every stride until
/// %num_elements are transferred. Either both or no stride arguments should be
/// specified. The value of 'num_elements' must be a multiple of
/// 'number_of_elements_per_stride'.
//
// For example, a DmaStartOp operation that transfers 256 elements of a memref
// '%src' in memory space 0 at indices [%i + 3, %j] to memref '%dst' in memory
// space 1 at indices [%k + 7, %l], would be specified as follows:
//
//   %num_elements = constant 256
//   %idx = constant 0 : index
//   %tag = alloc() : memref<1xi32, 4>
//   affine.dma_start %src[%i + 3, %j], %dst[%k + 7, %l], %tag[%idx],
//     %num_elements :
//       memref<40x128xf32, 0>, memref<2x1024xf32, 1>, memref<1xi32, 2>
//
//   If %stride and %num_elt_per_stride are specified, the DMA is expected to
//   transfer %num_elt_per_stride elements every %stride elements apart from
//   memory space 0 until %num_elements are transferred.
//
//   affine.dma_start %src[%i, %j], %dst[%k, %l], %tag[%idx], %num_elements,
//     %stride, %num_elt_per_stride : ...
//
// TODO(mlir-team): add additional operands to allow source and destination
// striding, and multiple stride levels (possibly using AffineMaps to specify
// multiple levels of striding).
// TODO(andydavis) Consider replacing src/dst memref indices with view memrefs.
class AffineDmaStartOp : public Op<AffineDmaStartOp, OpTrait::VariadicOperands,
                                   OpTrait::ZeroResult> {
public:
  using Op::Op;

  static void build(OpBuilder &builder, OperationState &result, Value srcMemRef,
                    AffineMap srcMap, ValueRange srcIndices, Value destMemRef,
                    AffineMap dstMap, ValueRange destIndices, Value tagMemRef,
                    AffineMap tagMap, ValueRange tagIndices, Value numElements,
                    Value stride = nullptr, Value elementsPerStride = nullptr);

  /// Returns the operand index of the src memref.
  unsigned getSrcMemRefOperandIndex() { return 0; }

  /// Returns the source MemRefType for this DMA operation.
  Value getSrcMemRef() { return getOperand(getSrcMemRefOperandIndex()); }
  MemRefType getSrcMemRefType() {
    return getSrcMemRef().getType().cast<MemRefType>();
  }

  /// Returns the rank (number of indices) of the source MemRefType.
  unsigned getSrcMemRefRank() { return getSrcMemRefType().getRank(); }

  /// Returns the affine map used to access the src memref.
  AffineMap getSrcMap() { return getSrcMapAttr().getValue(); }
  AffineMapAttr getSrcMapAttr() {
    return getAttr(getSrcMapAttrName()).cast<AffineMapAttr>();
  }

  /// Returns the source memref affine map indices for this DMA operation.
  operand_range getSrcIndices() {
    return {operand_begin() + getSrcMemRefOperandIndex() + 1,
            operand_begin() + getSrcMemRefOperandIndex() + 1 +
                getSrcMap().getNumInputs()};
  }

  /// Returns the memory space of the src memref.
  unsigned getSrcMemorySpace() {
    return getSrcMemRef().getType().cast<MemRefType>().getMemorySpace();
  }

  /// Returns the operand index of the dst memref.
  unsigned getDstMemRefOperandIndex() {
    return getSrcMemRefOperandIndex() + 1 + getSrcMap().getNumInputs();
  }

  /// Returns the destination MemRefType for this DMA operations.
  Value getDstMemRef() { return getOperand(getDstMemRefOperandIndex()); }
  MemRefType getDstMemRefType() {
    return getDstMemRef().getType().cast<MemRefType>();
  }

  /// Returns the rank (number of indices) of the destination MemRefType.
  unsigned getDstMemRefRank() {
    return getDstMemRef().getType().cast<MemRefType>().getRank();
  }

  /// Returns the memory space of the src memref.
  unsigned getDstMemorySpace() {
    return getDstMemRef().getType().cast<MemRefType>().getMemorySpace();
  }

  /// Returns the affine map used to access the dst memref.
  AffineMap getDstMap() { return getDstMapAttr().getValue(); }
  AffineMapAttr getDstMapAttr() {
    return getAttr(getDstMapAttrName()).cast<AffineMapAttr>();
  }

  /// Returns the destination memref indices for this DMA operation.
  operand_range getDstIndices() {
    return {operand_begin() + getDstMemRefOperandIndex() + 1,
            operand_begin() + getDstMemRefOperandIndex() + 1 +
                getDstMap().getNumInputs()};
  }

  /// Returns the operand index of the tag memref.
  unsigned getTagMemRefOperandIndex() {
    return getDstMemRefOperandIndex() + 1 + getDstMap().getNumInputs();
  }

  /// Returns the Tag MemRef for this DMA operation.
  Value getTagMemRef() { return getOperand(getTagMemRefOperandIndex()); }
  MemRefType getTagMemRefType() {
    return getTagMemRef().getType().cast<MemRefType>();
  }

  /// Returns the rank (number of indices) of the tag MemRefType.
  unsigned getTagMemRefRank() {
    return getTagMemRef().getType().cast<MemRefType>().getRank();
  }

  /// Returns the affine map used to access the tag memref.
  AffineMap getTagMap() { return getTagMapAttr().getValue(); }
  AffineMapAttr getTagMapAttr() {
    return getAttr(getTagMapAttrName()).cast<AffineMapAttr>();
  }

  /// Returns the tag memref indices for this DMA operation.
  operand_range getTagIndices() {
    return {operand_begin() + getTagMemRefOperandIndex() + 1,
            operand_begin() + getTagMemRefOperandIndex() + 1 +
                getTagMap().getNumInputs()};
  }

  /// Returns the number of elements being transferred by this DMA operation.
  Value getNumElements() {
    return getOperand(getTagMemRefOperandIndex() + 1 +
                      getTagMap().getNumInputs());
  }

  /// Returns the AffineMapAttr associated with 'memref'.
  NamedAttribute getAffineMapAttrForMemRef(Value memref) {
    if (memref == getSrcMemRef())
      return {Identifier::get(getSrcMapAttrName(), getContext()),
              getSrcMapAttr()};
    else if (memref == getDstMemRef())
      return {Identifier::get(getDstMapAttrName(), getContext()),
              getDstMapAttr()};
    assert(memref == getTagMemRef() &&
           "DmaStartOp expected source, destination or tag memref");
    return {Identifier::get(getTagMapAttrName(), getContext()),
            getTagMapAttr()};
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
    return isSrcMemorySpaceFaster() ? 0 : getDstMemRefOperandIndex();
  }

  static StringRef getSrcMapAttrName() { return "src_map"; }
  static StringRef getDstMapAttrName() { return "dst_map"; }
  static StringRef getTagMapAttrName() { return "tag_map"; }

  static StringRef getOperationName() { return "affine.dma_start"; }
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  LogicalResult fold(ArrayRef<Attribute> cstOperands,
                     SmallVectorImpl<OpFoldResult> &results);

  /// Returns true if this DMA operation is strided, returns false otherwise.
  bool isStrided() {
    return getNumOperands() !=
           getTagMemRefOperandIndex() + 1 + getTagMap().getNumInputs() + 1;
  }

  /// Returns the stride value for this DMA operation.
  Value getStride() {
    if (!isStrided())
      return nullptr;
    return getOperand(getNumOperands() - 1 - 1);
  }

  /// Returns the number of elements to transfer per stride for this DMA op.
  Value getNumElementsPerStride() {
    if (!isStrided())
      return nullptr;
    return getOperand(getNumOperands() - 1);
  }
};

/// AffineDmaWaitOp blocks until the completion of a DMA operation associated
/// with the tag element '%tag[%index]'. %tag is a memref, and %index has to be
/// an index with the same restrictions as any load/store index. In particular,
/// index for each memref dimension must be an affine expression of loop
/// induction variables and symbols. %num_elements is the number of elements
/// associated with the DMA operation. For example:
//
//   affine.dma_start %src[%i, %j], %dst[%k, %l], %tag[%index], %num_elements :
//     memref<2048xf32, 0>, memref<256xf32, 1>, memref<1xi32, 2>
//   ...
//   ...
//   affine.dma_wait %tag[%index], %num_elements : memref<1xi32, 2>
//
class AffineDmaWaitOp : public Op<AffineDmaWaitOp, OpTrait::VariadicOperands,
                                  OpTrait::ZeroResult> {
public:
  using Op::Op;

  static void build(OpBuilder &builder, OperationState &result, Value tagMemRef,
                    AffineMap tagMap, ValueRange tagIndices, Value numElements);

  static StringRef getOperationName() { return "affine.dma_wait"; }

  // Returns the Tag MemRef associated with the DMA operation being waited on.
  Value getTagMemRef() { return getOperand(0); }
  MemRefType getTagMemRefType() {
    return getTagMemRef().getType().cast<MemRefType>();
  }

  /// Returns the affine map used to access the tag memref.
  AffineMap getTagMap() { return getTagMapAttr().getValue(); }
  AffineMapAttr getTagMapAttr() {
    return getAttr(getTagMapAttrName()).cast<AffineMapAttr>();
  }

  // Returns the tag memref index for this DMA operation.
  operand_range getTagIndices() {
    return {operand_begin() + 1,
            operand_begin() + 1 + getTagMap().getNumInputs()};
  }

  // Returns the rank (number of indices) of the tag memref.
  unsigned getTagMemRefRank() {
    return getTagMemRef().getType().cast<MemRefType>().getRank();
  }

  /// Returns the AffineMapAttr associated with 'memref'.
  NamedAttribute getAffineMapAttrForMemRef(Value memref) {
    assert(memref == getTagMemRef());
    return {Identifier::get(getTagMapAttrName(), getContext()),
            getTagMapAttr()};
  }

  /// Returns the number of elements transferred in the associated DMA op.
  Value getNumElements() { return getOperand(1 + getTagMap().getNumInputs()); }

  static StringRef getTagMapAttrName() { return "tag_map"; }
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  LogicalResult fold(ArrayRef<Attribute> cstOperands,
                     SmallVectorImpl<OpFoldResult> &results);
};

/// Returns true if the given Value can be used as a dimension id in the region
/// of the closest surrounding op that has the trait `AffineScope`.
bool isValidDim(Value value);

/// Returns true if the given Value can be used as a dimension id in `region`,
/// i.e., for all its uses in `region`.
bool isValidDim(Value value, Region *region);

/// Returns true if the given value can be used as a symbol in the region of the
/// closest surrounding op that has the trait `AffineScope`.
bool isValidSymbol(Value value);

/// Returns true if the given Value can be used as a symbol for `region`, i.e.,
/// for all its uses in `region`.
bool isValidSymbol(Value value, Region *region);

/// Modifies both `map` and `operands` in-place so as to:
/// 1. drop duplicate operands
/// 2. drop unused dims and symbols from map
/// 3. promote valid symbols to symbolic operands in case they appeared as
///    dimensional operands
/// 4. propagate constant operands and drop them
void canonicalizeMapAndOperands(AffineMap *map,
                                SmallVectorImpl<Value> *operands);

/// Canonicalizes an integer set the same way canonicalizeMapAndOperands does
/// for affine maps.
void canonicalizeSetAndOperands(IntegerSet *set,
                                SmallVectorImpl<Value> *operands);

/// Returns a composed AffineApplyOp by composing `map` and `operands` with
/// other AffineApplyOps supplying those operands. The operands of the resulting
/// AffineApplyOp do not change the length of  AffineApplyOp chains.
AffineApplyOp makeComposedAffineApply(OpBuilder &b, Location loc, AffineMap map,
                                      ArrayRef<Value> operands);

/// Given an affine map `map` and its input `operands`, this method composes
/// into `map`, maps of AffineApplyOps whose results are the values in
/// `operands`, iteratively until no more of `operands` are the result of an
/// AffineApplyOp. When this function returns, `map` becomes the composed affine
/// map, and each Value in `operands` is guaranteed to be either a loop IV or a
/// terminal symbol, i.e., a symbol defined at the top level or a block/function
/// argument.
void fullyComposeAffineMapAndOperands(AffineMap *map,
                                      SmallVectorImpl<Value> *operands);

#include "mlir/Dialect/Affine/IR/AffineOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Affine/IR/AffineOps.h.inc"

/// Returns if the provided value is the induction variable of a AffineForOp.
bool isForInductionVar(Value val);

/// Returns the loop parent of an induction variable. If the provided value is
/// not an induction variable, then return nullptr.
AffineForOp getForInductionVarOwner(Value val);

/// Extracts the induction variables from a list of AffineForOps and places them
/// in the output argument `ivs`.
void extractForInductionVars(ArrayRef<AffineForOp> forInsts,
                             SmallVectorImpl<Value> *ivs);

/// AffineBound represents a lower or upper bound in the for operation.
/// This class does not own the underlying operands. Instead, it refers
/// to the operands stored in the AffineForOp. Its life span should not exceed
/// that of the for operation it refers to.
class AffineBound {
public:
  AffineForOp getAffineForOp() { return op; }
  AffineMap getMap() { return map; }

  unsigned getNumOperands() { return opEnd - opStart; }
  Value getOperand(unsigned idx) { return op.getOperand(opStart + idx); }

  using operand_iterator = AffineForOp::operand_iterator;
  using operand_range = AffineForOp::operand_range;

  operand_iterator operand_begin() { return op.operand_begin() + opStart; }
  operand_iterator operand_end() { return op.operand_begin() + opEnd; }
  operand_range getOperands() { return {operand_begin(), operand_end()}; }

private:
  // 'affine.for' operation that contains this bound.
  AffineForOp op;
  // Start and end positions of this affine bound operands in the list of
  // the containing 'affine.for' operation operands.
  unsigned opStart, opEnd;
  // Affine map for this bound.
  AffineMap map;

  AffineBound(AffineForOp op, unsigned opStart, unsigned opEnd, AffineMap map)
      : op(op), opStart(opStart), opEnd(opEnd), map(map) {}

  friend class AffineForOp;
};

/// An `AffineApplyNormalizer` is a helper class that supports renumbering
/// operands of AffineApplyOp. This acts as a reindexing map of Value to
/// positional dims or symbols and allows simplifications such as:
///
/// ```mlir
///    %1 = affine.apply (d0, d1) -> (d0 - d1) (%0, %0)
/// ```
///
/// into:
///
/// ```mlir
///    %1 = affine.apply () -> (0)
/// ```
struct AffineApplyNormalizer {
  AffineApplyNormalizer(AffineMap map, ArrayRef<Value> operands);

  /// Returns the AffineMap resulting from normalization.
  AffineMap getAffineMap() { return affineMap; }

  SmallVector<Value, 8> getOperands() {
    SmallVector<Value, 8> res(reorderedDims);
    res.append(concatenatedSymbols.begin(), concatenatedSymbols.end());
    return res;
  }

  unsigned getNumSymbols() { return concatenatedSymbols.size(); }
  unsigned getNumDims() { return reorderedDims.size(); }

  /// Normalizes 'otherMap' and its operands 'otherOperands' to map to this
  /// normalizer's coordinate space.
  void normalize(AffineMap *otherMap, SmallVectorImpl<Value> *otherOperands);

private:
  /// Helper function to insert `v` into the coordinate system of the current
  /// AffineApplyNormalizer. Returns the AffineDimExpr with the corresponding
  /// renumbered position.
  AffineDimExpr renumberOneDim(Value v);

  /// Given an `other` normalizer, this rewrites `other.affineMap` in the
  /// coordinate system of the current AffineApplyNormalizer.
  /// Returns the rewritten AffineMap and updates the dims and symbols of
  /// `this`.
  AffineMap renumber(const AffineApplyNormalizer &other);

  /// Maps of Value to position in `affineMap`.
  DenseMap<Value, unsigned> dimValueToPosition;

  /// Ordered dims and symbols matching positional dims and symbols in
  /// `affineMap`.
  SmallVector<Value, 8> reorderedDims;
  SmallVector<Value, 8> concatenatedSymbols;

  /// The number of symbols in concatenated symbols that belong to the original
  /// map as opposed to those concatendated during map composition.
  unsigned numProperSymbols;

  AffineMap affineMap;

  /// Used with RAII to control the depth at which AffineApply are composed
  /// recursively. Only accepts depth 1 for now to allow a behavior where a
  /// newly composed AffineApplyOp does not increase the length of the chain of
  /// AffineApplyOps. Full composition is implemented iteratively on top of
  /// this behavior.
  static unsigned &affineApplyDepth() {
    static thread_local unsigned depth = 0;
    return depth;
  }
  static constexpr unsigned kMaxAffineApplyDepth = 1;

  AffineApplyNormalizer() : numProperSymbols(0) { affineApplyDepth()++; }

public:
  ~AffineApplyNormalizer() { affineApplyDepth()--; }
};

} // end namespace mlir

#endif
