//===- MathToEmitC.h - Math to EmitC Pass -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MATHTOEMITC_MATHTOEMITC_H
#define MLIR_CONVERSION_MATHTOEMITC_MATHTOEMITC_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

#define GEN_PASS_DECL_CONVERTMATHTOEMITC
#include "mlir/Conversion/Passes.h.inc"

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertMathToEmitCPass();

} // namespace mlir

#endif // MLIR_CONVERSION_MATHTOEMITC_MATHTOEMITC_H
