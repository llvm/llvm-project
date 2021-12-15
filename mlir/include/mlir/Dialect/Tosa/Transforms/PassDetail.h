//===- PassDetail.h - TOSA Pass class details -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TOSA_TRANSFORMS_PASSDETAIL_H
#define MLIR_DIALECT_TOSA_TRANSFORMS_PASSDETAIL_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_TOSA_TRANSFORMS_PASSDETAIL_H
