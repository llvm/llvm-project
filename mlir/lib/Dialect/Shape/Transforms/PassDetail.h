//===- PassDetail.h - Shape Pass class details ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_SHAPE_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_SHAPE_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace bufferization {
class BufferizationDialect;
} // namespace bufferization

namespace memref {
class MemRefDialect;
} // namespace memref

namespace shape {
class ShapeDialect;
} // namespace shape

#define GEN_PASS_CLASSES
#include "mlir/Dialect/Shape/Transforms/Passes.h.inc"

} // namespace mlir

#endif // DIALECT_SHAPE_TRANSFORMS_PASSDETAIL_H_
