//===- MemoryAccessOpInterfacesImpl.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/NVGPU/Transforms/MemoryAccessOpInterfacesImpl.h"

#include "mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::memref;
using namespace mlir::nvgpu;

namespace {
struct LdMatrixOpInterface final
    : IndexedAccessOpInterface::ExternalModel<LdMatrixOpInterface, LdMatrixOp> {
  TypedValue<MemRefType> getAccessedMemref(Operation *op) const {
    return cast<LdMatrixOp>(op).getSrcMemref();
  }

  Operation::operand_range getIndices(Operation *op) const {
    return cast<LdMatrixOp>(op).getIndices();
  }

  SmallVector<int64_t> getAccessedShape(Operation *op) const {
    VectorType vecTy = cast<LdMatrixOp>(op).getRes().getType();
    return llvm::to_vector(vecTy.getShape());
  }

  std::optional<SmallVector<Value>>
  updateMemrefAndIndices(Operation *op, RewriterBase &rewriter, Value newMemref,
                         ValueRange newIndices) const {
    auto ldMatrixOp = cast<LdMatrixOp>(op);
    rewriter.modifyOpInPlace(ldMatrixOp, [&]() {
      ldMatrixOp.getSrcMemrefMutable().assign(newMemref);
      ldMatrixOp.getIndicesMutable().assign(newIndices);
    });
    return std::nullopt;
  }

  bool hasInboundsIndices(Operation *) const { return true; }
};

struct DeviceAsyncCopyOpInterface final
    : IndexedMemCopyOpInterface::ExternalModel<DeviceAsyncCopyOpInterface,
                                               DeviceAsyncCopyOp> {
  TypedValue<MemRefType> getSrc(Operation *op) const {
    return cast<DeviceAsyncCopyOp>(op).getSrc();
  }

  Operation::operand_range getSrcIndices(Operation *op) const {
    return cast<DeviceAsyncCopyOp>(op).getSrcIndices();
  }

  TypedValue<MemRefType> getDst(Operation *op) const {
    return cast<DeviceAsyncCopyOp>(op).getDst();
  }

  Operation::operand_range getDstIndices(Operation *op) const {
    return cast<DeviceAsyncCopyOp>(op).getDstIndices();
  }

  void setMemrefsAndIndices(Operation *op, RewriterBase &rewriter, Value newSrc,
                            ValueRange newSrcIndices, Value newDst,
                            ValueRange newDstIndices) const {
    auto copyOp = cast<DeviceAsyncCopyOp>(op);
    rewriter.modifyOpInPlace(copyOp, [&]() {
      copyOp.getSrcMutable().assign(newSrc);
      copyOp.getSrcIndicesMutable().assign(newSrcIndices);
      copyOp.getDstMutable().assign(newDst);
      copyOp.getDstIndicesMutable().assign(newDstIndices);
    });
  }
};
} // namespace

void mlir::nvgpu::registerMemoryAccessOpInterfacesExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, nvgpu::NVGPUDialect *dialect) {
    LdMatrixOp::attachInterface<LdMatrixOpInterface>(*ctx);
    DeviceAsyncCopyOp::attachInterface<DeviceAsyncCopyOpInterface>(*ctx);
  });
}
