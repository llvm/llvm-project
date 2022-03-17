//===- PassDetail.h - Affine Pass class details -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_AFFINE_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_AFFINE_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arith {
class ArithmeticDialect;
} // namespace arith

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace memref {
class MemRefDialect;
} // namespace memref

namespace vector {
class VectorDialect;
} // namespace vector

#define GEN_PASS_CLASSES
#include "mlir/Dialect/Affine/Passes.h.inc"

} // namespace mlir

#endif // DIALECT_AFFINE_TRANSFORMS_PASSDETAIL_H_
