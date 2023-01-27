//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These BufferizableOpInterface implementations provide analysis-related
// interface methods only. They are getting bufferized by the
// SparseTensorConversion pass.

#include "mlir/Dialect/SparseTensor/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir::bufferization;
using namespace mlir::sparse_tensor;

namespace mlir {
namespace sparse_tensor {
namespace {

struct ConcatenateOpInterface
    : public BufferizableOpInterface::ExternalModel<
          ConcatenateOpInterface, sparse_tensor::ConcatenateOp> {
  bool bufferizesToAllocation(Operation *op, OpResult opResult) const {
    return true;
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    return true;
  }
};

struct ConvertOpInterface
    : public BufferizableOpInterface::ExternalModel<ConvertOpInterface,
                                                    sparse_tensor::ConvertOp> {
  bool bufferizesToAllocation(Operation *op, OpResult opResult) const {
    // ConvertOps may allocate. (Unless they convert between two identical
    // types, then they fold away.)
    return true;
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    return true;
  }
};

struct LoadOpInterface
    : public BufferizableOpInterface::ExternalModel<LoadOpInterface,
                                                    sparse_tensor::LoadOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {op->getOpResult(0)};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }
};

struct NewOpInterface
    : public BufferizableOpInterface::ExternalModel<NewOpInterface,
                                                    sparse_tensor::NewOp> {
  bool isMemoryWrite(Operation *op, OpResult opResult,
                     const AnalysisState &state) const {
    // NewOps allocate but do not write.
    return false;
  }

  bool bufferizesToAllocation(Operation *op, OpResult opResult) const {
    return true;
  }
};

struct InsertOpInterface
    : public BufferizableOpInterface::ExternalModel<InsertOpInterface,
                                                    sparse_tensor::InsertOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // InsertOp writes to memory.
    return true;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    // InsertOp returns an alias of its operand.
    assert(op->getNumResults() == 1);
    return op->getResults();
  }

  BufferRelation bufferRelation(Operation *oo, OpResult opResult,
                                const AnalysisState &state) const {
    // InsertOp returns the same object (realloc should not invalidate
    // aliases).
    return BufferRelation::Equivalent;
  }
};

struct NumberOfEntriesOpInterface
    : public BufferizableOpInterface::ExternalModel<
          NumberOfEntriesOpInterface, sparse_tensor::NumberOfEntriesOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }
};

struct ToIndicesBufferOpInterface
    : public BufferizableOpInterface::ExternalModel<
          ToIndicesBufferOpInterface, sparse_tensor::ToIndicesBufferOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Potential writes into memory through the result of sparse_tensor.indices
    // are not considered.
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }
};

struct ToIndicesOpInterface
    : public BufferizableOpInterface::ExternalModel<
          ToIndicesOpInterface, sparse_tensor::ToIndicesOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Potential writes into memory through the result of sparse_tensor.indices
    // are not considered.
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }
};

struct ToPointersOpInterface
    : public BufferizableOpInterface::ExternalModel<
          ToPointersOpInterface, sparse_tensor::ToPointersOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Potential writes into memory through the result of sparse_tensor.pointers
    // are not considered.
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }
};

struct ToValuesOpInterface
    : public BufferizableOpInterface::ExternalModel<ToValuesOpInterface,
                                                    sparse_tensor::ToValuesOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Potential writes into memory through the result of sparse_tensor.values
    // are not considered.
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }
};

} // namespace
} // namespace sparse_tensor
} // namespace mlir

void mlir::sparse_tensor::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            sparse_tensor::SparseTensorDialect *dialect) {
    sparse_tensor::ConcatenateOp::attachInterface<ConcatenateOpInterface>(*ctx);
    sparse_tensor::ConvertOp::attachInterface<ConvertOpInterface>(*ctx);
    sparse_tensor::LoadOp::attachInterface<LoadOpInterface>(*ctx);
    sparse_tensor::NewOp::attachInterface<NewOpInterface>(*ctx);
    sparse_tensor::InsertOp::attachInterface<InsertOpInterface>(*ctx);
    sparse_tensor::NumberOfEntriesOp::attachInterface<
        NumberOfEntriesOpInterface>(*ctx);
    sparse_tensor::ToIndicesBufferOp::attachInterface<
        ToIndicesBufferOpInterface>(*ctx);
    sparse_tensor::ToIndicesOp::attachInterface<ToIndicesOpInterface>(*ctx);
    sparse_tensor::ToPointersOp::attachInterface<ToPointersOpInterface>(*ctx);
    sparse_tensor::ToValuesOp::attachInterface<ToValuesOpInterface>(*ctx);
  });
}
