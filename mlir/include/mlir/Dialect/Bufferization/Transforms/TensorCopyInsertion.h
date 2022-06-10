//===- TensorCopyInsertion.h - Resolve Bufferization Conflicts w/ Copies --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_TENSORCOPYINSERTION_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_TENSORCOPYINSERTION_H

#include "mlir/IR/Operation.h"

namespace mlir {
namespace bufferization {
class AnalysisState;
struct OneShotBufferizationOptions;

LogicalResult insertTensorCopies(Operation *op,
                                 const OneShotBufferizationOptions &options);

LogicalResult insertTensorCopies(Operation *op, const AnalysisState &state);
} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_TENSORCOPYINSERTION_H
