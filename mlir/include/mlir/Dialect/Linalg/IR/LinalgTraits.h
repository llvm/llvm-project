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
    auto nOperands = cast<ConcreteType>(op).getNumInputsAndOutputBuffers();
    if (failed(OpTrait::impl::verifyAtLeastNOperands(op, nOperands)))
      return failure();
    if (op->getNumResults() > concreteOp.getNumOutputs())
      return op->emitError("unexpected #results > #outputs");
    return success();
  }
};

} // namespace linalg
} // namespace OpTrait
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_LINALGTRAITS_H_
