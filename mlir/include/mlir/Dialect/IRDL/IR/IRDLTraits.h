//===- IRDLTraits.h - IRDL traits definition ---------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the traits used by the IR Definition Language dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IR_IRDLTRAITS_H_
#define MLIR_DIALECT_IRDL_IR_IRDLTRAITS_H_

#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/Casting.h"

namespace mlir {
namespace OpTrait {

/// Characterize operations that have at most a single operation of certain
/// types in their region.
/// This check is only done on the children that are immediate children of the
/// operation, and does not recurse into the children's regions.
/// This trait expects the Op to satisfy the `OneRegion` trait.
template <typename... ChildOps>
class AtMostOneChildOf {
public:
  template <typename ConcreteType>
  class Impl
      : public TraitBase<ConcreteType, AtMostOneChildOf<ChildOps...>::Impl> {
  public:
    static LogicalResult verifyTrait(Operation *op) {
      static_assert(
          ConcreteType::template hasTrait<::mlir::OpTrait::OneRegion>(),
          "expected operation to have a single region");
      static_assert(sizeof...(ChildOps) > 0,
                    "expected at least one child operation type");

      // Contains `true` if the corresponding child op has been seen.
      bool satisfiedOps[sizeof...(ChildOps)] = {};

      for (Operation &child : cast<ConcreteType>(op).getOps()) {
        int childOpIndex = 0;
        if (((isa<ChildOps>(child) ? false : (++childOpIndex, true)) && ...))
          continue;

        // Check that the operation has not been seen before.
        if (satisfiedOps[childOpIndex])
          return op->emitError()
                 << "failed to verify AtMostOneChildOf trait: the operation "
                    "contains at least two operations of type "
                 << child.getName();

        // Mark the operation as seen.
        satisfiedOps[childOpIndex] = true;
      }
      return success();
    }

    /// Get the unique operation of a specific op that is in the operation
    /// region.
    template <typename OpT>
    std::enable_if_t<std::disjunction<std::is_same<OpT, ChildOps>...>::value,
                     std::optional<OpT>>
    getOp() {
      auto ops =
          cast<ConcreteType>(this->getOperation()).template getOps<OpT>();
      if (ops.empty())
        return {};
      return {*ops.begin()};
    }
  };
};
} // namespace OpTrait
} // namespace mlir

#endif // MLIR_DIALECT_IRDL_IR_IRDLTRAITS_H_
