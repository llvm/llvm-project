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

/// This class provides the API for ops that are known to not have init tensor
/// operands. Use as a trait as follows:
///
///   class CopyOp : public Op<CopyOp, OpTrait::ZeroInitTensors> {
///
template <typename ConcreteType>
class ZeroInitTensors : public TraitBase<ConcreteType, ZeroInitTensors> {
public:
  static unsigned getNumInitTensors() { return 0; }
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

/// This class provides a verifier for structured ops that are known to operate
/// on buffers or tensors. This trait must be used in conjunction with an op
/// definition or a trait that provides the methods `getNumInputs` and
/// `getNumOutputs`. Use as a trait as follows:
///
///   class DotOp : public Op<DotOp, OpTrait::StructuredOpTraits> {
///
template <typename ConcreteType>
class StructuredOpTraits
    : public OpTrait::TraitBase<ConcreteType, StructuredOpTraits> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    ConcreteType concreteOp = cast<ConcreteType>(op);
    auto nOperands = concreteOp.getNumInputsAndOutputBuffers();
    if (failed(OpTrait::impl::verifyAtLeastNOperands(op, nOperands)))
      return failure();
    if (op->getNumResults() > concreteOp.getNumOutputs())
      return op->emitError("unexpected #results > #outputs");
    return success();
  }
};

/// This class provides a verifier for structured ops that are known to operate
/// on buffers or tensors and that support `ins`, `outs` and `init` arguments.
/// This trait must be used in conjunction with an op definition or a trait that
/// provides the methods `getNumInputs` and `getNumOutputs`.
///
/// Use as a trait as follows:
///
///   class MatmulOp : public Op<MatmulOp, OpTrait::NamedStructuredOpTrait> {
///
template <typename ConcreteType>
class NamedStructuredOpTrait
    : public OpTrait::TraitBase<ConcreteType, NamedStructuredOpTrait> {
public:
  unsigned getNumInputs() {
    return cast<ConcreteType>(this->getOperation()).inputs().size();
  }
  unsigned getNumInitTensors() {
    return cast<ConcreteType>(this->getOperation()).init_tensors().size();
  }
  unsigned getNumOutputs() {
    ConcreteType concreteOp = cast<ConcreteType>(this->getOperation());
    return concreteOp.output_buffers().size() +
           concreteOp.result_tensors().size();
  }
  static LogicalResult verifyTrait(Operation *op) {
    ConcreteType concreteOp = cast<ConcreteType>(op);
    unsigned nInputAndBufferOperands =
        concreteOp.getNumInputsAndOutputBuffers();
    if (failed(
            OpTrait::impl::verifyAtLeastNOperands(op, nInputAndBufferOperands)))
      return failure();

    SmallVector<AffineExpr, 4> redDims;
    concreteOp.getReductionDims(redDims);
    // If no result and no reduction, only check there is no init tensor and we
    // are done.
    if (redDims.empty() || op->getNumResults() == 0) {
      if (!concreteOp.init_tensors().empty())
        return op->emitError("expected empty `init` when op has no "
                             "results or no reduction dims");
      return success();
    }

    // Only a single tensor result supported atm.
    if (op->getNumResults() != 1)
      return op->emitError(
          "expected single tensor result when reduction present");

    if (concreteOp.init_tensors().size() != op->getNumResults())
      return op->emitError(
          "expected #init tensors to match #results when reduction present");

    for (unsigned idx = 0, e = op->getNumResults(); idx < e; ++idx)
      if (concreteOp.init_tensors()[idx].getType() != op->getResultTypes()[idx])
        return op->emitError("expected init tensor #")
               << idx << " of the same type as result #" << idx;

    // Output tensor indexing map may not depend on reduction index.
    // TODO: this is not yet tested. Add a test when linalg.generic switches to
    // this representation.
    for (unsigned idx = 0, e = concreteOp.getNumOutputs(); idx < e; ++idx) {
      AffineMap outputMap = concreteOp.getOutputIndexingMap(idx);
      for (auto expr : outputMap.getResults()) {
        for (auto dim : redDims) {
          unsigned pos = dim.cast<AffineDimExpr>().getPosition();
          if (expr.isFunctionOfDim(pos))
            return op->emitError(
                       "unexpected single tensor output indexing map ")
                   << "is function of reduction dim @" << pos;
        }
      }
    }

    return success();
  }
};

} // namespace linalg
} // namespace OpTrait
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_LINALGTRAITS_H_
