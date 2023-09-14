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
    OpBuilder builder(op);
    if (!inParallelOp.getBody()->empty())
      return op->emitError("only supported when nested region is empty");

    // Collect the values to deallocate and retain and use them to create the
    // dealloc operation.
    Block *block = op->getBlock();
    SmallVector<Value> memrefs, conditions, toRetain;
    if (failed(state.getMemrefsAndConditionsToDeallocate(
            builder, op->getLoc(), block, memrefs, conditions)))
      return failure();

    state.getMemrefsToRetain(block, /*toBlock=*/nullptr, {}, toRetain);
    if (memrefs.empty() && toRetain.empty())
      return op;

    auto deallocOp = builder.create<bufferization::DeallocOp>(
        op->getLoc(), memrefs, conditions, toRetain);

    // We want to replace the current ownership of the retained values with the
    // result values of the dealloc operation as they are always unique.
    state.resetOwnerships(deallocOp.getRetained(), block);
    for (auto [retained, ownership] :
         llvm::zip(deallocOp.getRetained(), deallocOp.getUpdatedConditions()))
      state.updateOwnership(retained, ownership, block);

    return op;
  }
};

} // namespace

void mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, SCFDialect *dialect) {
    InParallelOp::attachInterface<InParallelOpInterface>(*ctx);
  });
}
