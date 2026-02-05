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
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace omp {
#define GEN_PASS_DEF_STACKTOSHAREDPASS
#include "mlir/Dialect/OpenMP/Transforms/Passes.h.inc"
} // namespace omp
} // namespace mlir

using namespace mlir;

/// When a use takes place inside an omp.parallel region and it's not as a
/// private clause argument, or when it is a reduction argument passed to
/// omp.parallel or a function call argument, then the defining allocation is
/// eligible for replacement with shared memory.
static bool allocaUseRequiresDeviceSharedMem(const OpOperand &use) {
  Operation *owner = use.getOwner();
  if (auto parallelOp = dyn_cast<omp::ParallelOp>(owner)) {
    if (llvm::is_contained(parallelOp.getReductionVars(), use.get()))
      return true;
  } else if (auto callOp = dyn_cast<CallOpInterface>(owner)) {
    if (llvm::is_contained(callOp.getArgOperands(), use.get()))
      return true;
  }

  // If it is used directly inside of a parallel region, it has to be replaced
  // unless the use is a private clause.
  if (owner->getParentOfType<omp::ParallelOp>()) {
    if (auto argIface = dyn_cast<omp::BlockArgOpenMPOpInterface>(owner)) {
      if (auto privateSyms =
              cast_or_null<ArrayAttr>(owner->getAttr("private_syms"))) {
        for (auto [var, sym] :
             llvm::zip_equal(argIface.getPrivateVars(), privateSyms)) {
          if (var != use.get())
            continue;

          auto moduleOp = owner->getParentOfType<ModuleOp>();
          auto privateOp = cast<omp::PrivateClauseOp>(
              moduleOp.lookupSymbol(cast<SymbolRefAttr>(sym)));
          return privateOp.getDataSharingType() !=
                 omp::DataSharingClauseType::Private;
        }
      }
    }
    return true;
  }
  return false;
}

static bool shouldReplaceAllocaWithUses(const Operation::use_range &uses) {
  // Check direct uses and also follow hlfir.declare/fir.convert uses.
  for (const OpOperand &use : uses) {
    Operation *owner = use.getOwner();
    if (llvm::isa<LLVM::AddrSpaceCastOp, LLVM::GEPOp>(owner)) {
      if (shouldReplaceAllocaWithUses(owner->getUses()))
        return true;
    } else if (allocaUseRequiresDeviceSharedMem(use)) {
      return true;
    }
  }

  return false;
}

// TODO: Refactor the logic in `shouldReplaceAllocaWithDeviceSharedMem`,
// `shouldReplaceAllocaWithUses` and `allocaUseRequiresDeviceSharedMem` to
// be reusable by the MLIR to LLVM IR translation stage, as something very
// similar is also implemented there to choose between allocas and device
// shared memory allocations when processing OpenMP reductions, mapping and
// privatization.
bool shouldReplaceAllocaWithDeviceSharedMem(Operation &op) {
  auto offloadIface = op.getParentOfType<omp::OffloadModuleInterface>();
  if (!offloadIface || !offloadIface.getIsTargetDevice())
    return false;

  auto targetOp = op.getParentOfType<omp::TargetOp>();

  // It must be inside of a generic omp.target or in a target device function,
  // and not inside of omp.parallel.
  if (auto parallelOp = op.getParentOfType<omp::ParallelOp>()) {
    if (!targetOp || targetOp->isProperAncestor(parallelOp))
      return false;
  }

  if (targetOp) {
    if (targetOp.getKernelExecFlags(targetOp.getInnermostCapturedOmpOp()) !=
        omp::TargetExecMode::generic)
      return false;
  } else {
    auto declTargetIface = op.getParentOfType<omp::DeclareTargetInterface>();
    if (!declTargetIface || !declTargetIface.isDeclareTarget() ||
        declTargetIface.getDeclareTargetDeviceType() ==
            omp::DeclareTargetDeviceType::host)
      return false;
  }

  return shouldReplaceAllocaWithUses(op.getUses());
}

void insertDeviceSharedMemDeallocation(OpBuilder &builder, Value allocVal) {
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
