//===- StackToShared.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements transforms to swap stack allocations on the target
// device with device shared memory where applicable.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace omp {
#define GEN_PASS_DEF_STACKTOSHAREDPASS
#include "mlir/Dialect/OpenMP/Transforms/Passes.h.inc"
} // namespace omp
} // namespace mlir

using namespace mlir;

/// Tell whether to replace an operation representing a stack allocation with a
/// device shared memory allocation/deallocation pair based on the location of
/// the allocation and its uses.
static bool shouldReplaceAllocaWithDeviceSharedMem(Operation &op) {
  return omp::opInSharedDeviceContext(op) &&
         llvm::any_of(op.getResults(), [&](Value result) {
           return omp::allocaUsesRequireSharedMem(result);
         });
}

/// Based on the location of the definition of the given value representing the
/// result of a device shared memory allocation, find the corresponding points
/// where its deallocation should be placed and introduce `omp.free_shared_mem`
/// ops at those points.
static void insertDeviceSharedMemDeallocation(OpBuilder &builder,
                                              Value allocVal) {
  Block *allocaBlock = allocVal.getParentBlock();
  DominanceInfo domInfo;
  for (Block &block : allocVal.getParentRegion()->getBlocks()) {
    Operation *terminator = block.getTerminator();
    if (!terminator->hasSuccessors() &&
        domInfo.dominates(allocaBlock, &block)) {
      builder.setInsertionPoint(terminator);
      omp::FreeSharedMemOp::create(builder, allocVal.getLoc(), allocVal);
    }
  }
}

namespace {
class StackToSharedPass
    : public omp::impl::StackToSharedPassBase<StackToSharedPass> {
public:
  StackToSharedPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    LLVM::LLVMFuncOp funcOp = getOperation();
    auto offloadIface = funcOp->getParentOfType<omp::OffloadModuleInterface>();
    if (!offloadIface || !offloadIface.getIsTargetDevice())
      return;

    llvm::SmallVector<Operation *> toBeDeleted;
    funcOp->walk([&](LLVM::AllocaOp allocaOp) {
      if (!shouldReplaceAllocaWithDeviceSharedMem(*allocaOp))
        return;
      // Replace llvm.alloca with omp.alloc_shared_mem.
      Type resultType = allocaOp.getResult().getType();

      // TODO: The handling of non-default address spaces might need to be
      // improved. This currently only handles the case where an alloca to
      // non-default address space must only be used by a single addrspacecast
      // to default address space.
      bool nonDefaultAddrSpace = false;
      if (auto llvmPtrType = dyn_cast<LLVM::LLVMPointerType>(resultType))
        nonDefaultAddrSpace = llvmPtrType.getAddressSpace() != 0;

      builder.setInsertionPoint(allocaOp);
      auto sharedAllocOp = omp::AllocSharedMemOp::create(
          builder, allocaOp->getLoc(), LLVM::LLVMPointerType::get(context),
          allocaOp.getElemTypeAttr(), allocaOp.getArraySize(),
          allocaOp.getAlignmentAttr());
      if (nonDefaultAddrSpace) {
        assert(allocaOp->hasOneUse() && "alloca must have only one use");
        auto asCastOp =
            cast<LLVM::AddrSpaceCastOp>(*allocaOp->getUsers().begin());
        asCastOp.replaceAllUsesWith(sharedAllocOp.getOperation());
        // Delete later because we can't delete the cast op before the top-level
        // iteration visits it. Also, the alloca can't be deleted before because
        // it's used by it.
        toBeDeleted.push_back(asCastOp);
        toBeDeleted.push_back(allocaOp);
      } else {
        allocaOp.replaceAllUsesWith(sharedAllocOp.getOperation());
        allocaOp.erase();
      }

      // Create a new omp.free_shared_mem for the allocated buffer prior to
      // exiting the region.
      insertDeviceSharedMemDeallocation(builder, sharedAllocOp.getResult());
    });
    for (Operation *op : toBeDeleted)
      op->erase();
  }
};
} // namespace
