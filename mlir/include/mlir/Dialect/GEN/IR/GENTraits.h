//===--- GENTraits.h - GEN Dialect Traits -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GEN_IR_GENTRAITS_H
#define MLIR_DIALECT_GEN_IR_GENTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {
namespace GEN {
namespace detail {
LogicalResult verifyGEN3DNDRange(Operation *op);
} // namespace detail

template <typename ConcreteType>
class GEN3DNDRange : public TraitBase<ConcreteType, GEN3DNDRange> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return detail::verifyGEN3DNDRange(op);
  }
};
} // namespace GEN
} // namespace OpTrait
} // namespace mlir

#endif // MLIR_DIALECT_GEN_IR_GENTRAITS_H
