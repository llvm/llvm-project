//===- BufferDeallocationOpInterfaceImpl.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {
/// The `scf.forall.in_parallel` terminator is special in a few ways:
/// * It does not implement the BranchOpInterface or
///   RegionBranchTerminatorOpInterface, but the ParallelCombiningOpInterface
///   which is not supported by BufferDeallocation.
/// * It has a graph-like region which only allows one specific tensor op
/// * After bufferization the nested region is always empty
/// For these reasons we provide custom deallocation logic via this external
/// model.
///
/// Example:
/// ```mlir
/// scf.forall (%arg1) in (%arg0) {
///   %alloc = memref.alloc() : memref<2xf32>
///   ...
///   <implicit in_parallel terminator here>
/// }
/// ```
/// gets transformed to
/// ```mlir
/// scf.forall (%arg1) in (%arg0) {
///   %alloc = memref.alloc() : memref<2xf32>
///   ...
///   bufferization.dealloc (%alloc : memref<2xf32>) if (%true)
///   <implicit in_parallel terminator here>
/// }
/// ```
struct InParallelOpInterface
    : public BufferDeallocationOpInterface::ExternalModel<InParallelOpInterface,
                                                          scf::InParallelOp> {
  FailureOr<Operation *> process(Operation *op, DeallocationState &state,
                                 const DeallocationOptions &options) const {
    auto inParallelOp = cast<scf::InParallelOp>(op);
    if (!inParallelOp.getBody()->empty())
      return op->emitError("only supported when nested region is empty");

    SmallVector<Value> updatedOperandOwnership;
    return deallocation_impl::insertDeallocOpForReturnLike(
        state, op, {}, updatedOperandOwnership);
  }
};

struct ReduceReturnOpInterface
    : public BufferDeallocationOpInterface::ExternalModel<
          ReduceReturnOpInterface, scf::ReduceReturnOp> {
  FailureOr<Operation *> process(Operation *op, DeallocationState &state,
                                 const DeallocationOptions &options) const {
    auto reduceReturnOp = cast<scf::ReduceReturnOp>(op);
    if (isa<BaseMemRefType>(reduceReturnOp.getOperand().getType()))
      return op->emitError("only supported when operand is not a MemRef");

    SmallVector<Value> updatedOperandOwnership;
    return deallocation_impl::insertDeallocOpForReturnLike(
        state, op, {}, updatedOperandOwnership);
  }
};

} // namespace

void mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, SCFDialect *dialect) {
    InParallelOp::attachInterface<InParallelOpInterface>(*ctx);
    ReduceReturnOp::attachInterface<ReduceReturnOpInterface>(*ctx);
  });
}
