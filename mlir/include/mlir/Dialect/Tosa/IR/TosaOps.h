//===-- TosaOps.h - TOSA dialect operation definitions *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the TOSA Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TOSA_IR_TOSA_OPS_H
#define MLIR_DIALECT_TOSA_IR_TOSA_OPS_H

#include <initializer_list>
#include <unordered_map>

#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Tosa/IR/TosaTraits.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Dialect/Tosa/IR/TosaStructs.h.inc"

namespace mlir {
namespace tosa {

class TosaDialect : public Dialect {

public:
  explicit TosaDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "tosa"; }

  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;
};

#include "mlir/Dialect/Tosa/IR/TosaInterfaces.h.inc"

} // end namespace tosa
} // end namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaOps.h.inc"

#endif // TOSA_OPS_H
