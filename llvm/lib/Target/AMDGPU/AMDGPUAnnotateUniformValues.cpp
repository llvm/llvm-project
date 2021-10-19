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
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "amdgpu-annotate-uniform"

using namespace llvm;

namespace {

class AMDGPUAnnotateUniformValues : public FunctionPass,
                       public InstVisitor<AMDGPUAnnotateUniformValues> {
  LegacyDivergenceAnalysis *DA;
  MemorySSA *MSSA;
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
INITIALIZE_PASS_END(AMDGPUAnnotateUniformValues, DEBUG_TYPE,
                    "Add AMDGPU uniform metadata", false, false)

char AMDGPUAnnotateUniformValues::ID = 0;

static void setUniformMetadata(Instruction *I) {
  I->setMetadata("amdgpu.uniform", MDNode::get(I->getContext(), {}));
}
static void setNoClobberMetadata(Instruction *I) {
  I->setMetadata("amdgpu.noclobber", MDNode::get(I->getContext(), {}));
}

bool AMDGPUAnnotateUniformValues::isClobberedInFunction(LoadInst * Load) {
  const MemoryAccess *MA = MSSA->getWalker()->getClobberingMemoryAccess(Load);
  return !MSSA->isLiveOnEntryDef(MA);
}

void AMDGPUAnnotateUniformValues::visitBranchInst(BranchInst &I) {
  if (DA->isUniform(&I))
    setUniformMetadata(&I);
}

void AMDGPUAnnotateUniformValues::visitLoadInst(LoadInst &I) {
  Value *Ptr = I.getPointerOperand();
  if (!DA->isUniform(Ptr))
    return;
  auto isGlobalLoad = [&](LoadInst &Load)->bool {
    return Load.getPointerAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS;
  };
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
  bool GlobalLoad = isGlobalLoad(I);
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
  isEntryFunc = AMDGPU::isEntryFunctionCC(F.getCallingConv());

  visit(F);
  noClobberClones.clear();
  return true;
}

FunctionPass *
llvm::createAMDGPUAnnotateUniformValues() {
  return new AMDGPUAnnotateUniformValues();
}
