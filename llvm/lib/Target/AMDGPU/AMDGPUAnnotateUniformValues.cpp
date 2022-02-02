//===-- AMDGPUAnnotateUniformValues.cpp - ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass adds amdgpu.uniform metadata to IR values so this information
/// can be used during instruction selection.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "amdgpu-annotate-uniform"

using namespace llvm;

namespace {

class AMDGPUAnnotateUniformValues : public FunctionPass,
                       public InstVisitor<AMDGPUAnnotateUniformValues> {
  LegacyDivergenceAnalysis *DA;
  MemorySSA *MSSA;
  AliasAnalysis *AA;
  DenseMap<Value*, GetElementPtrInst*> noClobberClones;
  bool isEntryFunc;

public:
  static char ID;
  AMDGPUAnnotateUniformValues() :
    FunctionPass(ID) { }
  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;
  StringRef getPassName() const override {
    return "AMDGPU Annotate Uniform Values";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LegacyDivergenceAnalysis>();
    AU.addRequired<MemorySSAWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.setPreservesAll();
 }

  void visitBranchInst(BranchInst &I);
  void visitLoadInst(LoadInst &I);
  bool isClobberedInFunction(LoadInst * Load);
};

} // End anonymous namespace

INITIALIZE_PASS_BEGIN(AMDGPUAnnotateUniformValues, DEBUG_TYPE,
                      "Add AMDGPU uniform metadata", false, false)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(MemorySSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(AMDGPUAnnotateUniformValues, DEBUG_TYPE,
                    "Add AMDGPU uniform metadata", false, false)

char AMDGPUAnnotateUniformValues::ID = 0;

static void setUniformMetadata(Instruction *I) {
  I->setMetadata("amdgpu.uniform", MDNode::get(I->getContext(), {}));
}
static void setNoClobberMetadata(Instruction *I) {
  I->setMetadata("amdgpu.noclobber", MDNode::get(I->getContext(), {}));
}

bool AMDGPUAnnotateUniformValues::isClobberedInFunction(LoadInst *Load) {
  MemorySSAWalker *Walker = MSSA->getWalker();
  SmallVector<MemoryAccess *> WorkList{Walker->getClobberingMemoryAccess(Load)};
  SmallSet<MemoryAccess *, 8> Visited;
  MemoryLocation Loc(MemoryLocation::get(Load));

  const auto isReallyAClobber = [this, Load](MemoryDef *Def) -> bool {
    Instruction *DefInst = Def->getMemoryInst();
    LLVM_DEBUG(dbgs() << "  Def: " << *DefInst << '\n');

    if (isa<FenceInst>(DefInst))
      return false;

    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(DefInst)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::amdgcn_s_barrier:
      case Intrinsic::amdgcn_wave_barrier:
        return false;
      default:
        break;
      }
    }

    // Ignore atomics not aliasing with the original load, any atomic is a
    // universal MemoryDef from MSSA's point of view too, just like a fence.
    const auto checkNoAlias = [this, Load](auto I) -> bool {
      return I && AA->isNoAlias(I->getPointerOperand(),
                                Load->getPointerOperand());
    };

    if (checkNoAlias(dyn_cast<AtomicCmpXchgInst>(DefInst)) ||
        checkNoAlias(dyn_cast<AtomicRMWInst>(DefInst)))
      return false;

    return true;
  };

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
      if (isReallyAClobber(Def)) {
        LLVM_DEBUG(dbgs() << "      -> load is clobbered\n");
        return true;
      }

      WorkList.push_back(
          Walker->getClobberingMemoryAccess(Def->getDefiningAccess(), Loc));
      continue;
    }

    const MemoryPhi *Phi = cast<MemoryPhi>(MA);
    for (auto &Use : Phi->incoming_values())
      WorkList.push_back(cast<MemoryAccess>(&Use));
  }

  LLVM_DEBUG(dbgs() << "      -> no clobber\n");
  return false;
}

void AMDGPUAnnotateUniformValues::visitBranchInst(BranchInst &I) {
  if (DA->isUniform(&I))
    setUniformMetadata(&I);
}

void AMDGPUAnnotateUniformValues::visitLoadInst(LoadInst &I) {
  Value *Ptr = I.getPointerOperand();
  if (!DA->isUniform(Ptr))
    return;
  // We're tracking up to the Function boundaries, and cannot go beyond because
  // of FunctionPass restrictions. We can ensure that is memory not clobbered
  // for memory operations that are live in to entry points only.
  Instruction *PtrI = dyn_cast<Instruction>(Ptr);

  if (!isEntryFunc) {
    if (PtrI)
      setUniformMetadata(PtrI);
    return;
  }

  bool NotClobbered = false;
  bool GlobalLoad = I.getPointerAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS;
  if (PtrI)
    NotClobbered = GlobalLoad && !isClobberedInFunction(&I);
  else if (isa<Argument>(Ptr) || isa<GlobalValue>(Ptr)) {
    if (GlobalLoad && !isClobberedInFunction(&I)) {
      NotClobbered = true;
      // Lookup for the existing GEP
      if (noClobberClones.count(Ptr)) {
        PtrI = noClobberClones[Ptr];
      } else {
        // Create GEP of the Value
        Function *F = I.getParent()->getParent();
        Value *Idx = Constant::getIntegerValue(
          Type::getInt32Ty(Ptr->getContext()), APInt(64, 0));
        // Insert GEP at the entry to make it dominate all uses
        PtrI = GetElementPtrInst::Create(I.getType(), Ptr,
                                         ArrayRef<Value *>(Idx), Twine(""),
                                         F->getEntryBlock().getFirstNonPHI());
      }
      I.replaceUsesOfWith(Ptr, PtrI);
    }
  }

  if (PtrI) {
    setUniformMetadata(PtrI);
    if (NotClobbered)
      setNoClobberMetadata(PtrI);
  }
}

bool AMDGPUAnnotateUniformValues::doInitialization(Module &M) {
  return false;
}

bool AMDGPUAnnotateUniformValues::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  DA = &getAnalysis<LegacyDivergenceAnalysis>();
  MSSA = &getAnalysis<MemorySSAWrapperPass>().getMSSA();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  isEntryFunc = AMDGPU::isEntryFunctionCC(F.getCallingConv());

  visit(F);
  noClobberClones.clear();
  return true;
}

FunctionPass *
llvm::createAMDGPUAnnotateUniformValues() {
  return new AMDGPUAnnotateUniformValues();
}
