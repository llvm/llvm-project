//===- BufferizableOpInterface.h - Comprehensive Bufferize ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_BUFFERIZABLEOPINTERFACE_H_
#define MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_BUFFERIZABLEOPINTERFACE_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class BlockAndValueMapping;

namespace linalg {
namespace comprehensive_bufferize {

struct AllocationCallbacks;
class BufferizationAliasInfo;

/// Specify fine-grain relationship between buffers to enable more analysis.
enum class BufferRelation {
  None,
  // TODO: ResultContainsOperand,
  // TODO: OperandContainsResult,
  Equivalent
};

/// Determine which OpOperand* will alias with `result` if the op is bufferized
/// in place. Return an empty vector if the op is not bufferizable.
SmallVector<OpOperand *> getAliasingOpOperand(OpResult result);

/// Determine which OpResult will alias with `opOperand` if the op is bufferized
/// in place. Return an empty OpResult if the op is not bufferizable.
OpResult getAliasingOpResult(OpOperand &opOperand);

/// Return true if `opOperand` bufferizes to a memory read. Return `true` if the
/// op is not bufferizable.
bool bufferizesToMemoryRead(OpOperand &opOperand);

/// Return true if `opOperand` bufferizes to a memory write. Return
/// `true` if the op is not bufferizable.
bool bufferizesToMemoryWrite(OpOperand &opOperand);

/// Return true if `opOperand` does neither read nor write but bufferizes to an
/// alias. Return false if the op is not bufferizable.
bool bufferizesToAliasOnly(OpOperand &opOperand);

/// Return true if the given value is read by an op that bufferizes to a memory
/// read. Also takes into account ops that create an alias but do not read by
/// themselves (e.g., ExtractSliceOp).
bool isValueRead(Value value);

/// Return the relationship between the operand and the its corresponding
/// OpResult that it may alias with. Return None if the op is not bufferizable.
BufferRelation bufferRelation(OpOperand &opOperand);

} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h.inc"

#endif // MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_BUFFERIZABLEOPINTERFACE_H_
