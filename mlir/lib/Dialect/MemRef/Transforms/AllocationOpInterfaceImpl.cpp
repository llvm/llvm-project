//===- AllocationOpInterfaceImpl.cpp - Impl. of AllocationOpInterface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"

#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace {
struct DefaultAllocationInterface
    : public bufferization::AllocationOpInterface::ExternalModel<
          DefaultAllocationInterface, memref::AllocOp> {
  static std::optional<Operation *> buildDealloc(OpBuilder &builder,
                                                 Value alloc) {
    return builder.create<memref::DeallocOp>(alloc.getLoc(), alloc)
        .getOperation();
  }
  static std::optional<Value> buildClone(OpBuilder &builder, Value alloc) {
    return builder.create<bufferization::CloneOp>(alloc.getLoc(), alloc)
        .getResult();
  }
  static ::mlir::HoistingKind getHoistingKind() {
    return HoistingKind::Loop | HoistingKind::Block;
  }
  static ::std::optional<::mlir::Operation *>
  buildPromotedAlloc(OpBuilder &builder, Value alloc) {
    Operation *definingOp = alloc.getDefiningOp();
    return builder.create<memref::AllocaOp>(
        definingOp->getLoc(), cast<MemRefType>(definingOp->getResultTypes()[0]),
        definingOp->getOperands(), definingOp->getAttrs());
  }
};

struct DefaultAutomaticAllocationHoistingInterface
    : public bufferization::AllocationOpInterface::ExternalModel<
          DefaultAutomaticAllocationHoistingInterface, memref::AllocaOp> {
  static ::mlir::HoistingKind getHoistingKind() { return HoistingKind::Loop; }
};

struct DefaultReallocationInterface
    : public bufferization::AllocationOpInterface::ExternalModel<
          DefaultAllocationInterface, memref::ReallocOp> {
  static std::optional<Operation *> buildDealloc(OpBuilder &builder,
                                                 Value realloc) {
    return builder.create<memref::DeallocOp>(realloc.getLoc(), realloc)
        .getOperation();
  }
};
} // namespace

void mlir::memref::registerAllocationOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    memref::AllocOp::attachInterface<DefaultAllocationInterface>(*ctx);
    memref::AllocaOp::attachInterface<
        DefaultAutomaticAllocationHoistingInterface>(*ctx);
    memref::ReallocOp::attachInterface<DefaultReallocationInterface>(*ctx);
  });
}
