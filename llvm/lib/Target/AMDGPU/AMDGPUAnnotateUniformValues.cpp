//===-- AMDGPUAnnotateUniformValues.cpp - ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass adds amdgpu.uniform metadata to IR values so this information
/// can be used during instruction selection.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUIntrinsicInfo.h"
#include "llvm/Analysis/DivergenceAnalysis.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "amdgpu-annotate-uniform"

using namespace llvm;

namespace {

class AMDGPUAnnotateUniformValues : public FunctionPass,
                       public InstVisitor<AMDGPUAnnotateUniformValues> {
  DivergenceAnalysis *DA;

public:
  static char ID;
  AMDGPUAnnotateUniformValues() :
    FunctionPass(ID) { }
  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;
  const char *getPassName() const override { return "AMDGPU Annotate Uniform Values"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DivergenceAnalysis>();
    AU.setPreservesAll();
 }

  void visitBranchInst(BranchInst &I);
  void visitLoadInst(LoadInst &I);

};

} // End anonymous namespace

INITIALIZE_PASS_BEGIN(AMDGPUAnnotateUniformValues, DEBUG_TYPE,
                      "Add AMDGPU uniform metadata", false, false)
INITIALIZE_PASS_DEPENDENCY(DivergenceAnalysis)
INITIALIZE_PASS_END(AMDGPUAnnotateUniformValues, DEBUG_TYPE,
                    "Add AMDGPU uniform metadata", false, false)

char AMDGPUAnnotateUniformValues::ID = 0;

static void setUniformMetadata(Instruction *I) {
  I->setMetadata("amdgpu.uniform", MDNode::get(I->getContext(), {}));
}

void AMDGPUAnnotateUniformValues::visitBranchInst(BranchInst &I) {
  if (I.isUnconditional())
    return;

  Value *Cond = I.getCondition();
  if (!DA->isUniform(Cond))
    return;

  setUniformMetadata(I.getParent()->getTerminator());
}

void AMDGPUAnnotateUniformValues::visitLoadInst(LoadInst &I) {
  Value *Ptr = I.getPointerOperand();
  if (!DA->isUniform(Ptr))
    return;

  if (Instruction *PtrI = dyn_cast<Instruction>(Ptr))
    setUniformMetadata(PtrI);

}

bool AMDGPUAnnotateUniformValues::doInitialization(Module &M) {
  return false;
}

bool AMDGPUAnnotateUniformValues::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  DA = &getAnalysis<DivergenceAnalysis>();
  visit(F);

  return true;
}

FunctionPass *
llvm::createAMDGPUAnnotateUniformValues() {
  return new AMDGPUAnnotateUniformValues();
}
