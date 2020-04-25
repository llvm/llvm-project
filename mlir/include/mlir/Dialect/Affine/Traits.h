//===- Traits.h - Traits for the affine dialect -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares traits that the affine dialect relies upon for analysis
// and transformation purposes, and that are also potentially used by other
// dialect entities not depending on the affine dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_TRAITS
#define MLIR_DIALECT_AFFINE_TRAITS

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {

/// A trait of region holding operations that defines a new scope for polyhedral
/// optimization purposes. Any SSA values of 'index' type that either dominate
/// such an operation or are used at the top-level of such an operation
/// automatically become valid symbols for the polyhedral scope defined by that
/// operation. For more details, see `Traits.md#PolyhedralScope`.
template <typename ConcreteType>
class PolyhedralScope : public TraitBase<ConcreteType, PolyhedralScope> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    static_assert(!ConcreteType::template hasTrait<ZeroRegion>(),
                  "expected operation to have one or more regions");
    return success();
  }
};

} // end namespace OpTrait
} // end namespace mlir

#endif // MLIR_DIALECT_TRAITS
