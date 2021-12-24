//===- PassDetail.h - GPU Pass class details --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_TENSOR_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_TENSOR_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

namespace bufferization {
class BufferizationDialect;
} // namespace bufferization

namespace memref {
class MemRefDialect;
} // namespace memref

namespace scf {
class SCFDialect;
} // namespace scf

#define GEN_PASS_CLASSES
#include "mlir/Dialect/Tensor/Transforms/Passes.h.inc"

} // namespace mlir

#endif // DIALECT_TENSOR_TRANSFORMS_PASSDETAIL_H_
