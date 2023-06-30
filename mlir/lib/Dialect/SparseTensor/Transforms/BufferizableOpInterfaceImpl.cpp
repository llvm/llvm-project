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

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
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

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
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

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {{op->getOpResult(0), BufferRelation::Equivalent}};
  }
};

struct NewOpInterface
    : public BufferizableOpInterface::ExternalModel<NewOpInterface,
                                                    sparse_tensor::NewOp> {
  bool resultBufferizesToMemoryWrite(Operation *op, OpResult opResult,
                                     const AnalysisState &state) const {
    // NewOps allocate but do not write.
    return false;
  }

  bool bufferizesToAllocation(Operation *op, OpResult opResult) const {
    return true;
  }
};

struct PackOpInterface
    : public BufferizableOpInterface::ExternalModel<PackOpInterface,
                                                    sparse_tensor::PackOp> {
  bool bufferizesToAllocation(Operation *op, OpResult opResult) const {
    // PackOp reuses all the buffers instead of allocating new ones
    return false;
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    assert(op->getNumResults() == 1);
    // PackOp reuses the input tensors as values/coordinates instead of
    // creating new ones when packing into a COO format.
    return {{op->getOpResult(0), BufferRelation::Equivalent}};
  }

  BufferRelation bufferRelation(Operation *oo, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Unknown;
  }
};

struct UnpackOpInterface
    : public BufferizableOpInterface::ExternalModel<UnpackOpInterface,
                                                    sparse_tensor::UnpackOp> {
  bool bufferizesToAllocation(Operation *op, OpResult opResult) const {
    // The output buffer is pre-allocated by the user.
    return false;
  }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // The first operand is the sparse tensor that we are unpacking.
    return opOperand.getOperandNumber() == 0;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // We write into the output operand.
    assert(2 * (op->getNumOperands() - 1) == op->getNumResults());
    return opOperand.getOperandNumber() > 0;
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    assert(2 * (op->getNumOperands() - 1) == op->getNumResults());

    if (opOperand.getOperandNumber() == 0)
      return {};
    // We write directly into the output tensors and returns them.
    return {{op->getResult(opOperand.getOperandNumber() - 1),
             BufferRelation::Equivalent}};
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

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    // InsertOp returns an alias of its operand.
    assert(op->getNumResults() == 1);
    return {{op->getOpResult(0), BufferRelation::Equivalent}};
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

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }
};

struct ToCoordinatesBufferOpInterface
    : public BufferizableOpInterface::ExternalModel<
          ToCoordinatesBufferOpInterface,
          sparse_tensor::ToCoordinatesBufferOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Potential writes into memory through the result of
    // `sparse_tensor.coordinates` are not considered.
    return false;
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }
};

struct ToCoordinatesOpInterface
    : public BufferizableOpInterface::ExternalModel<
          ToCoordinatesOpInterface, sparse_tensor::ToCoordinatesOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Potential writes into memory through the result of
    // `sparse_tensor.coordinates` are not considered.
    return false;
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }
};

struct ToPositionsOpInterface
    : public BufferizableOpInterface::ExternalModel<
          ToPositionsOpInterface, sparse_tensor::ToPositionsOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Potential writes into memory through the result of
    // `sparse_tensor.positions` are not considered.
    return false;
  }

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
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

  AliasingOpResultList getAliasingOpResults(Operation *op, OpOperand &opOperand,
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
    sparse_tensor::PackOp::attachInterface<PackOpInterface>(*ctx);
    sparse_tensor::UnpackOp::attachInterface<UnpackOpInterface>(*ctx);
    sparse_tensor::ToCoordinatesBufferOp::attachInterface<
        ToCoordinatesBufferOpInterface>(*ctx);
    sparse_tensor::ToCoordinatesOp::attachInterface<ToCoordinatesOpInterface>(
        *ctx);
    sparse_tensor::ToPositionsOp::attachInterface<ToPositionsOpInterface>(*ctx);
    sparse_tensor::ToValuesOp::attachInterface<ToValuesOpInterface>(*ctx);
  });
}
