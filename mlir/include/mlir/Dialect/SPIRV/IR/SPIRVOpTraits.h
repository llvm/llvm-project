//===- SPIRVOps.h - MLIR SPIR-V operation traits ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares C++ classes for some of operation traits in the SPIR-V
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_SPIRVOPTRAITS_H_
#define MLIR_DIALECT_SPIRV_IR_SPIRVOPTRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {
namespace spirv {

template <typename ConcreteType>
class UnsignedOp : public TraitBase<ConcreteType, UnsignedOp> {};

/// A trait to mark ops that can be enclosed/wrapped in a
/// `SpecConstantOperation` op.
template <typename ConcreteType>
class UsableInSpecConstantOp
    : public TraitBase<ConcreteType, UsableInSpecConstantOp> {};

} // namespace spirv
} // namespace OpTrait
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_IR_SPIRVOPTRAITS_H_
