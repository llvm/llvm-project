//===-- AMDGPUMemoryUtils.cpp - -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMemoryUtils.h"
#include "AMDGPU.h"
#include "AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/ReplaceConstant.h"

#define DEBUG_TYPE "amdgpu-memory-utils"

using namespace llvm;

namespace llvm {

namespace AMDGPU {

Align getAlign(DataLayout const &DL, const GlobalVariable *GV) {
  return DL.getValueOrABITypeAlignment(GV->getPointerAlignment(DL),
                                       GV->getValueType());
}

bool isDynamicLDS(const GlobalVariable &GV) {
  // external zero size addrspace(3) without initializer is dynlds.
  const Module *M = GV.getParent();
  const DataLayout &DL = M->getDataLayout();
  if (GV.getType()->getPointerAddressSpace() != AMDGPUAS::LOCAL_ADDRESS)
    return false;
  return DL.getTypeAllocSize(GV.getValueType()) == 0;
}

bool isLDSVariableToLower(const GlobalVariable &GV) {
  if (GV.getType()->getPointerAddressSpace() != AMDGPUAS::LOCAL_ADDRESS) {
    return false;
  }
  if (isDynamicLDS(GV)) {
    return true;
  }
  if (GV.isConstant()) {
    // A constant undef variable can't be written to, and any load is
    // undef, so it should be eliminated by the optimizer. It could be
    // dropped by the back end if not. This pass skips over it.
    return false;
  }
  if (GV.hasInitializer() && !isa<UndefValue>(GV.getInitializer())) {
    // Initializers are unimplemented for LDS address space.
    // Leave such variables in place for consistent error reporting.
    return false;
  }
  return true;
}

bool isReallyAClobber(const Value *Ptr, MemoryDef *Def, AAResults *AA) {
  Instruction *DefInst = Def->getMemoryInst();

  if (isa<FenceInst>(DefInst))
    return false;

  if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(DefInst)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::amdgcn_s_barrier:
    case Intrinsic::amdgcn_s_barrier_signal:
    case Intrinsic::amdgcn_s_barrier_signal_var:
    case Intrinsic::amdgcn_s_barrier_signal_isfirst:
    case Intrinsic::amdgcn_s_barrier_signal_isfirst_var:
    case Intrinsic::amdgcn_s_barrier_init:
    case Intrinsic::amdgcn_s_barrier_join:
    case Intrinsic::amdgcn_s_barrier_wait:
    case Intrinsic::amdgcn_s_barrier_leave:
    case Intrinsic::amdgcn_s_get_barrier_state:
    case Intrinsic::amdgcn_s_wakeup_barrier:
    case Intrinsic::amdgcn_wave_barrier:
    case Intrinsic::amdgcn_sched_barrier:
    case Intrinsic::amdgcn_sched_group_barrier:
      return false;
    default:
      break;
    }
  }

  // Ignore atomics not aliasing with the original load, any atomic is a
  // universal MemoryDef from MSSA's point of view too, just like a fence.
  const auto checkNoAlias = [AA, Ptr](auto I) -> bool {
    return I && AA->isNoAlias(I->getPointerOperand(), Ptr);
  };

  if (checkNoAlias(dyn_cast<AtomicCmpXchgInst>(DefInst)) ||
      checkNoAlias(dyn_cast<AtomicRMWInst>(DefInst)))
    return false;

  return true;
}

bool isClobberedInFunction(const LoadInst *Load, MemorySSA *MSSA,
                           AAResults *AA) {
  MemorySSAWalker *Walker = MSSA->getWalker();
  SmallVector<MemoryAccess *> WorkList{Walker->getClobberingMemoryAccess(Load)};
  SmallSet<MemoryAccess *, 8> Visited;
  MemoryLocation Loc(MemoryLocation::get(Load));

  LLVM_DEBUG(dbgs() << "Checking clobbering of: " << *Load << '\n');

  // Start with a nearest dominating clobbering access, it will be either
  // live on entry (nothing to do, load is not clobbered), MemoryDef, or
  // MemoryPhi if several MemoryDefs can define this memory state. In that
  // case add all Defs to WorkList and continue going up and checking all
  // the definitions of this memory location until the root. When all the
  // defs are exhausted and came to the entry state we have no clobber.
  // Along the scan ignore barriers and fences which are considered clobbers
  // by the MemorySSA, but not really writing anything into the memory.
  while (!WorkList.empty()) {
    MemoryAccess *MA = WorkList.pop_back_val();
    if (!Visited.insert(MA).second)
      continue;

    if (MSSA->isLiveOnEntryDef(MA))
      continue;

    if (MemoryDef *Def = dyn_cast<MemoryDef>(MA)) {
      LLVM_DEBUG(dbgs() << "  Def: " << *Def->getMemoryInst() << '\n');

      if (isReallyAClobber(Load->getPointerOperand(), Def, AA)) {
        LLVM_DEBUG(dbgs() << "      -> load is clobbered\n");
        return true;
      }

      WorkList.push_back(
          Walker->getClobberingMemoryAccess(Def->getDefiningAccess(), Loc));
      continue;
    }

    const MemoryPhi *Phi = cast<MemoryPhi>(MA);
    for (const auto &Use : Phi->incoming_values())
      WorkList.push_back(cast<MemoryAccess>(&Use));
  }

  LLVM_DEBUG(dbgs() << "      -> no clobber\n");
  return false;
}

} // end namespace AMDGPU

} // end namespace llvm
