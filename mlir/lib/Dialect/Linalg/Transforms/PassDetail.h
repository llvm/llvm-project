//===- PassDetail.h - Linalg Pass class details -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_LINALG_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_LINALG_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace arith {
class ArithmeticDialect;
} // namespace arith

namespace bufferization {
class BufferizationDialect;
} // namespace bufferization

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace scf {
class SCFDialect;
} // namespace scf

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
#include "mlir/Dialect/Linalg/Passes.h.inc"

} // namespace mlir

#endif // DIALECT_LINALG_TRANSFORMS_PASSDETAIL_H_
