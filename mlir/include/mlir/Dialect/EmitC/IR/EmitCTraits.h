//===- EmitCTraits.h - EmitC trait definitions ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares C++ classes for some of the traits used in the EmitC
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_IR_EMITCTRAITS_H
#define MLIR_DIALECT_EMITC_IR_EMITCTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {
namespace emitc {

template <typename ConcreteType>
class CExpression : public TraitBase<ConcreteType, CExpression> {};

} // namespace emitc
} // namespace OpTrait
} // namespace mlir

#endif // MLIR_DIALECT_EMITC_IR_EMITCTRAITS_H
