//===- PassDetail.h - MemRef Pass class details -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_MEMREF_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_MEMREF_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

class AffineDialect;

// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arith {
class ArithmeticDialect;
} // namespace arith

namespace func {
class FuncDialect;
} // namespace func

namespace memref {
class MemRefDialect;
} // namespace memref

namespace tensor {
class TensorDialect;
} // namespace tensor

namespace vector {
class VectorDialect;
} // namespace vector

#define GEN_PASS_CLASSES
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"

} // namespace mlir

#endif // DIALECT_MEMREF_TRANSFORMS_PASSDETAIL_H_
