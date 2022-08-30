//===- PassDetail.h - Loop Pass class details -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_SCF_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_SCF_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

class AffineDialect;

namespace arith {
class ArithmeticDialect;
} // namespace arith

namespace bufferization {
class BufferizationDialect;
} // namespace bufferization

namespace memref {
class MemRefDialect;
} // namespace memref

namespace tensor {
class TensorDialect;
} // namespace tensor

#define GEN_PASS_CLASSES
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"

} // namespace mlir

#endif // DIALECT_SCF_TRANSFORMS_PASSDETAIL_H_
