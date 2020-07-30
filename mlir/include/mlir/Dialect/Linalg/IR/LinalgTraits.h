//===- LinalgTraits.h - Linalg Traits ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_LINALGTRAITS_H_
#define MLIR_DIALECT_LINALG_LINALGTRAITS_H_

#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace OpTrait {
namespace linalg {

/// This class provides the API for ops that are known to have a specified
/// number of inputs, all passed as operands. Use as a trait as follows:
///
///   class DotOp : public Op<DotOp, OpTrait::NInputs<2>::Impl> {
///
template <unsigned N> class NInputs {
public:
  template <typename ConcreteType>
  class Impl : public OpTrait::TraitBase<ConcreteType, NInputs<N>::Impl> {
  public:
    static unsigned getNumInputs() { return N; }
  };
};

/// This class provides the API for ops that are known to have a specified
/// number of outputs, all passed as operands. Use as a trait as follows:
///
///   class DotOp : public Op<DotOp, OpTrait::NOutputs<2>::Impl> {
///
template <unsigned N> class NOutputs {
public:
  template <typename ConcreteType>
  class Impl : public OpTrait::TraitBase<ConcreteType, NOutputs<N>::Impl> {
  public:
    static unsigned getNumOutputs() { return N; }
  };
};

/// This class provides the API for structured ops that are known to operate on
/// buffers or tensors. This trait must be used in conjunction with an op
/// definition or a trait that provides the methods `getNumInputs` and
/// `getNumOutputs`. Use as a trait as follows:
///
///   class DotOp : public Op<DotOp, OpTrait::StructuredOpTraits> {
///
template <typename ConcreteType>
class StructuredOpTraits
    : public OpTrait::TraitBase<ConcreteType, StructuredOpTraits> {
private:
  /// Return the number of inputs, irrespective of their buffer or tensor type.
  /// For internal use only.
  unsigned nInputs() {
    return cast<ConcreteType>(this->getOperation()).getNumInputs();
  }
  /// Return the number of outputs, irrespective of their buffer or tensor type.
  /// For internal use only.
  unsigned nOutputs() {
    return cast<ConcreteType>(this->getOperation()).getNumOutputs();
  }

public:
  //==========================================================================//
  // Loop types handling.
  //==========================================================================//
  unsigned getNumParallelLoops() {
    return getNumIterators(
        getParallelIteratorTypeName(),
        cast<ConcreteType>(this->getOperation()).iterator_types());
  }
  unsigned getNumReductionLoops() {
    return getNumIterators(
        getReductionIteratorTypeName(),
        cast<ConcreteType>(this->getOperation()).iterator_types());
  }
  unsigned getNumWindowLoops() {
    return getNumIterators(
        getWindowIteratorTypeName(),
        cast<ConcreteType>(this->getOperation()).iterator_types());
  }
  unsigned getNumLoops() {
    return getNumIterators(
        cast<ConcreteType>(this->getOperation()).iterator_types());
  }

  bool hasSingleReductionLoop() {
    auto iterators = cast<ConcreteType>(this->getOperation()).iterator_types();
    return iterators.size() == 1 &&
           getNumIterators(getReductionIteratorTypeName(), iterators);
  }

  //==========================================================================//
  // Input arguments handling.
  //==========================================================================//
  // The `i^th` input argument is always the `i^th` operand regardless of
  // whether we have tensors or buffers.
  //
  /// Return the `i`-th input value.
  Value getInput(unsigned i) {
    assert(i < nInputs());
    return this->getOperation()->getOperand(i);
  }
  /// Return the index of `value` in the list of inputs if found, llvm::None
  /// otherwise.
  Optional<unsigned> getIndexOfInput(Value value) {
    auto it = llvm::find(getInputs(), value);
    if (it != getInputs().end())
      return it - getInputs().begin();
    return llvm::None;
  }
  /// Return the `i`-th input shaped type, irrespective of buffer or tensor
  /// type.
  ShapedType getInputShapedType(unsigned i) {
    return getInput(i).getType().template cast<ShapedType>();
  }
  /// Return the range over inputs.
  Operation::operand_range getInputs() {
    auto range = this->getOperation()->getOperands();
    return {range.begin(), range.begin() + nInputs()};
  }
  /// Query the subset of input operands that are of ranked tensor type.
  SmallVector<RankedTensorType, 4> getInputTensorTypes() {
    SmallVector<RankedTensorType, 4> res;
    for (Type type : getInputs().getTypes())
      if (auto t = type.template dyn_cast<RankedTensorType>())
        res.push_back(t);
    return res;
  }

  //==========================================================================//
  // Output arguments handling.
  //==========================================================================//
  // The `i^th` output argument is an operand (resp. a return value) iff it is
  // a value of buffer type (resp. a return value of tensor type).

  /// Return the `i`-th output, asserts that this is a buffer operand and not
  /// a tensor result.
  Value getOutputBuffer(unsigned i) {
    assert(i + this->getOperation()->getNumResults() < nOutputs() &&
           "overflowing output buffer index");
    return this->getOperation()->getOperand(nInputs() + i);
  }
  /// Return the index of `value` in the list of output buffers if found,
  /// llvm::None otherwise.
  Optional<unsigned> getIndexOfOutputBuffer(Value value) {
    auto it = llvm::find(getOutputBuffers(), value);
    if (it != getOutputBuffers().end())
      return it - getOutputBuffers().begin();
    return llvm::None;
  }
  /// Return the `i`-th output buffer type.
  MemRefType getOutputBufferType(unsigned i) {
    return getOutputBuffer(i).getType().template cast<MemRefType>();
  }
  /// Return the `i`-th output shaped type, irrespective of buffer of tensor
  /// type.
  ShapedType getOutputShapedType(unsigned i) {
    return getShapedType(i + nInputs());
  }
  /// Query the subset of results that are of ranked tensor type.
  SmallVector<RankedTensorType, 4> getOutputTensorTypes() {
    SmallVector<RankedTensorType, 4> res;
    for (Type type : this->getOperation()->getResults().getTypes())
      res.push_back(type.template cast<RankedTensorType>());
    return res;
  }
  /// Return the range over outputs.
  Operation::operand_range getOutputBuffers() {
    auto range = this->getOperation()->getOperands();
    return {range.begin() + nInputs(),
            range.begin() + getNumInputsAndOutputBuffers()};
  }

  //==========================================================================//
  // Input and Output arguments handling.
  //==========================================================================//
  Value getBuffer(unsigned i) {
    assert(i < getNumInputsAndOutputBuffers() && "overflowing buffers index");
    return this->getOperation()->getOperand(i);
  }
  /// Return the number of inputs and outputs, irrespective of their buffer or
  /// tensor type.
  unsigned getNumInputsAndOutputs() { return nInputs() + nOutputs(); }
  /// Return the number of inputs, irrespective of their buffer or tensor type,
  /// and output buffers.
  unsigned getNumInputsAndOutputBuffers() {
    assert(this->getOperation()->getNumResults() <= nOutputs());
    return nInputs() + nOutputs() - this->getOperation()->getNumResults();
  }
  /// Return the range over inputs (irrespective of type) and output buffers.
  Operation::operand_range getInputsAndOutputBuffers() {
    auto range = this->getOperation()->getOperands();
    return {range.begin(), range.begin() + getNumInputsAndOutputBuffers()};
  }
  /// Return the `i`-th shaped type, there are 3 cases:
  ///   1. if `i < nInputs()` then return `getInputShapedType(i)`; otherwise
  ///   2. if `i < getNumInputsAndOutputBuffers()` then return the
  ///      `getOutputBufferType(i - nInputs())`; otherwise
  ///   3. return the `i - getNumInputsAndOutputBuffers()` result type.
  ShapedType getShapedType(unsigned i) {
    if (i < nInputs())
      return getInputShapedType(i);
    if (i < getNumInputsAndOutputBuffers())
      return getOutputBufferType(i - nInputs()).template cast<ShapedType>();
    return getOutputTensorTypes()[i - getNumInputsAndOutputBuffers()]
        .template cast<ShapedType>();
  }
  /// Return the shaped types for all the inputs and outputs
  SmallVector<ShapedType, 4> getInputOutputShapedTypes() {
    SmallVector<Type, 4> inputOutputTypes(
        this->getOperation()->operand_type_begin(),
        this->getOperation()->operand_type_end());
    inputOutputTypes.append(this->getOperation()->result_type_begin(),
                            this->getOperation()->result_type_end());
    return llvm::to_vector<4>(
        llvm::map_range(inputOutputTypes, [](Type type) -> ShapedType {
          return type.cast<ShapedType>();
        }));
  }

  //==========================================================================//
  // Other interface methods.
  //==========================================================================//

  // Get or build the indexing_maps ArrayAttr.
  ArrayAttr iterator_types() {
    // Return the attribute if it is present.
    if (auto attr = this->getOperation()->getAttr("iterator_types"))
      return attr.template cast<ArrayAttr>();

    // If not, form the attribute using the reference iterator types for the
    // ConcreteType.
    auto maybeReferenceIteratorTypes =
        cast<ConcreteType>(this->getOperation()).referenceIterators();

    // If there is no reference, this must be a generic op.
    // TODO: Traits are used to define ops. Split into cpp to avoid cyclic
    // dependency.
    auto name = this->getOperation()->getName().getStringRef();
    if (!maybeReferenceIteratorTypes && name != "generic" &&
        name != "indexed_generic") {
      this->getOperation()->dump();
      llvm_unreachable("Op missing referenceIterators");
    }

    // If we have a reference, build the reference attribute and set it in the
    // op before returning.
    auto *ctx = this->getOperation()->getContext();
    auto attrRange = llvm::map_range(*maybeReferenceIteratorTypes,
                                     [ctx](StringRef str) -> Attribute {
                                       return StringAttr::get(str, ctx);
                                     });
    auto attr = ArrayAttr::get(llvm::to_vector<4>(attrRange), ctx);
    // TODO: Need to memoize this. Can't just store as an attribute atm as it
    // will impact parser, printer and tests.
    // this->getOperation()->setAttr("iterator_types", attr);
    return attr;
  }

  // Get or build the indexing_maps ArrayAttr.
  ArrayAttr indexing_maps() {
    // Return the attribute if it is present.
    if (auto attr = this->getOperation()->getAttr("indexing_maps"))
      return attr.template cast<ArrayAttr>();

    // If not, form the attribute using the reference indexing map for the
    // ConcreteType.
    auto maybeReferenceIndexingMaps =
        cast<ConcreteType>(this->getOperation()).referenceIndexingMaps();

    // If there is no reference, this must be a generic op.
    auto name = this->getOperation()->getName().getStringRef();
    if (!maybeReferenceIndexingMaps && name != "generic" &&
        name != "indexed_generic") {
      this->getOperation()->dump();
      llvm_unreachable("Op missing referenceIndexingMaps");
    }

    // If we have a reference, build the reference attribute and set it in the
    // op before returning.
    auto *ctx = this->getOperation()->getContext();
    auto attrRange =
        llvm::map_range(*maybeReferenceIndexingMaps, [ctx](AffineMap map) {
          // 0-D corner case because there is no such thing as a concrete empty
          // map type.
          if (!map)
            map = AffineMap::get(0, 0, getAffineConstantExpr(0, ctx));
          return AffineMapAttr::get(map);
        });
    SmallVector<Attribute, 4> attrs{attrRange.begin(), attrRange.end()};
    auto attr = ArrayAttr::get(attrs, ctx);
    // TODO: Need to memoize this. Can't just store as an attribute atm as it
    // will impact parser, printer and tests.
    // this->getOperation()->setAttr("indexing_maps", attr);
    return attr;
  }

  SmallVector<AffineMap, 4> getIndexingMaps() {
    return llvm::to_vector<4>(
        llvm::map_range(indexing_maps(), [](Attribute attr) -> AffineMap {
          return attr.cast<AffineMapAttr>().getValue();
        }));
  }

  AffineMap getIndexingMap(unsigned i) {
    assert(i < getNumInputsAndOutputs());
    return indexing_maps()
        .getValue()[i]
        .template cast<AffineMapAttr>()
        .getValue();
  }

  AffineMap getInputIndexingMap(unsigned i) {
    assert(i < nInputs());
    return indexing_maps()
        .getValue()[i]
        .template cast<AffineMapAttr>()
        .getValue();
  }

  AffineMap getOutputIndexingMap(unsigned i) {
    assert(i < nOutputs());
    return indexing_maps()
        .getValue()[i + nInputs()]
        .template cast<AffineMapAttr>()
        .getValue();
  }

  /// Query whether the op has only buffer inputs and no returns.
  bool hasBufferSemantics() {
    return this->getOperation()->getNumResults() == 0 &&
           llvm::all_of(getInputs(),
                        [](Value v) { return v.getType().isa<MemRefType>(); });
  }

  /// Query whether the op has only tensor inputs and outputs.
  bool hasTensorSemantics() {
    auto isTensorType = [](Value v) {
      return v.getType().isa<RankedTensorType>();
    };
    return llvm::all_of(getInputs(), isTensorType) &&
           llvm::all_of(this->getOperation()->getResults(), isTensorType);
  }

  //==========================================================================//
  // Other static interface methods.
  //==========================================================================//
  static LogicalResult verifyTrait(Operation *op) {
    auto nOperands = cast<ConcreteType>(op).getNumInputsAndOutputBuffers();
    if (failed(OpTrait::impl::verifyAtLeastNOperands(op, nOperands)))
      return failure();
    return success();
  }
};

/// This class provides the API for named Linalg StructuredOps.
template <typename ConcreteType>
class NamedStructuredOpTraits
    : public OpTrait::TraitBase<ConcreteType, NamedStructuredOpTraits> {
public:
  static SmallVector<StringRef, 8> referenceIterators(TypeRange inputTypes,
                                                      TypeRange outputTypes);

  static SmallVector<AffineMap, 8> referenceIndexingMaps(TypeRange inputTypes,
                                                         TypeRange outputTypes);
};

} // namespace linalg
} // namespace OpTrait
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_LINALGTRAITS_H_
