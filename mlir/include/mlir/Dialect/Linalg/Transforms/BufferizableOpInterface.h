//===- BufferizableOpInterface.h - Comprehensive Bufferize ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_BUFFERIZABLEOPINTERFACE_H_
#define MLIR_DIALECT_LINALG_TRANSFORMS_BUFFERIZABLEOPINTERFACE_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class BlockAndValueMapping;

namespace linalg {
class AllocationCallbacks;
class BufferizationAliasInfo;

/// Specify fine-grain relationship between buffers to enable more analysis.
enum class BufferRelation {
  None,
  // TODO: ResultContainsOperand,
  // TODO: OperandContainsResult,
  Equivalent
};
} // namespace linalg
} // namespace mlir

#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterface.h.inc"

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_BUFFERIZABLEOPINTERFACE_H_
