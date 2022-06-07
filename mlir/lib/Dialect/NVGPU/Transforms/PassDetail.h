//===- PassDetail.h - NVGPU Pass class details -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef DIALECT_NVGPU_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_NVGPU_TRANSFORMS_PASSDETAIL_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace arith {
class ArithmeticDialect;
} // namespace arith

namespace memref {
class MemRefDialect;
} // namespace memref

namespace vector {
class VectorDialect;
} // namespace vector

#define GEN_PASS_CLASSES
#include "mlir/Dialect/NVGPU/Passes.h.inc"

} // namespace mlir

#endif // DIALECT_NVGPU_TRANSFORMS_PASSDETAIL_H_
