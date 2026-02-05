//===- StackToShared.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements various OpenMP dialect utilities.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/Utils/Utils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

using namespace mlir;

static bool allocaUseRequiresSharedMem(const OpOperand &use) {
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

bool mlir::omp::allocaUsesRequireSharedMem(Value alloc) {
  for (const OpOperand &use : alloc.getUses()) {
    Operation *owner = use.getOwner();
    if (isa<LLVM::AddrSpaceCastOp, LLVM::GEPOp>(owner)) {
      if (llvm::any_of(owner->getResults(), [&](Value result) {
            return allocaUsesRequireSharedMem(result);
          }))
        return true;
    } else if (allocaUseRequiresSharedMem(use)) {
      return true;
    }
  }
  return false;
}

bool mlir::omp::opInSharedDeviceContext(Operation &op) {
  if (isa<omp::ParallelOp>(op))
    return false;

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

  // The omp.target operation itself is considered in a shared device context in
  // order to properly process its own allocation-defining entry block
  // arguments.
  if (!targetOp)
    targetOp = dyn_cast<omp::TargetOp>(op);

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
  return true;
}
