//===- Traits.h - Trait Declaration for MLIR DLTI dialect -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DLTI_TRAITS_H
#define MLIR_DIALECT_DLTI_TRAITS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace mlir {
class DataLayoutSpecAttr;

namespace impl {
LogicalResult verifyHasDefaultDLTIDataLayoutTrait(Operation *op);
DataLayoutSpecInterface getDataLayoutSpec(Operation *op);
TargetSystemSpecInterface getTargetSystemSpec(Operation *op);
TargetAttrInterface getTargetAttr(Operation *op);
void setDataLayoutSpec(Operation *op, DataLayoutSpecInterface spec);
void setTargetSystemSpec(Operation *op, TargetSystemSpecInterface spec);
void setTargetAttr(Operation *op, TargetAttrInterface target);
} // namespace impl

/// Trait to be used by operations willing to use the implementation of the
/// data layout interfaces provided by the Target dialect.
template <typename ConcreteOp>
class HasDefaultDLTIDataLayout
    : public OpTrait::TraitBase<ConcreteOp, HasDefaultDLTIDataLayout> {
public:
  /// Verifies that the operation to which this trait is attached is valid for
  /// the trait, i.e., that it implements the data layout operation interface.
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyHasDefaultDLTIDataLayoutTrait(op);
  }

  /// Returns the data layout specification as provided by the Target dialect
  /// specification attribute.
  DataLayoutSpecInterface getDataLayoutSpec() {
    return impl::getDataLayoutSpec(this->getOperation());
  }

  /// Sets the data layout specification.
  void setDataLayoutSpec(DataLayoutSpecInterface spec) {
    impl::setDataLayoutSpec(this->getOperation(), spec);
  }
  /// Returns the target system description specification as provided by DLTI
  /// dialect
  TargetSystemSpecInterface getTargetSystemSpec() {
    return impl::getTargetSystemSpec(this->getOperation());
  }
  /// Sets the target system description specification.
  void setTargetSystemSpec(TargetSystemSpecInterface spec) {
    impl::setTargetSystemSpec(this->getOperation(), spec);
  }
  /// Returns the target information as provided by DLTI dialect.
  TargetAttrInterface getTargetAttr() {
    return impl::getTargetAttr(this->getOperation());
  }
  /// Sets the target information.
  void setTargetAttr(TargetAttrInterface target) {
    impl::setTargetAttr(this->getOperation(), target);
  }
};
} // namespace mlir

#endif // MLIR_DIALECT_DLTI_TRAITS_H
