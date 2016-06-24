//===-- AMDGPUCodeGenPrepare.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass does misc. AMDGPU optimizations on IR before instruction
/// selection.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"

#include "llvm/Analysis/DivergenceAnalysis.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "amdgpu-codegenprepare"

using namespace llvm;

namespace {

class AMDGPUCodeGenPrepare : public FunctionPass,
                             public InstVisitor<AMDGPUCodeGenPrepare> {
  DivergenceAnalysis *DA;
  const TargetMachine *TM;

public:
  static char ID;
  AMDGPUCodeGenPrepare(const TargetMachine *TM = nullptr) :
    FunctionPass(ID),
    TM(TM) { }

  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;

  const char *getPassName() const override {
    return "AMDGPU IR optimizations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DivergenceAnalysis>();
    AU.setPreservesAll();
 }
};

} // End anonymous namespace

bool AMDGPUCodeGenPrepare::doInitialization(Module &M) {
  return false;
}

bool AMDGPUCodeGenPrepare::runOnFunction(Function &F) {
  if (!TM || skipFunction(F))
    return false;

  DA = &getAnalysis<DivergenceAnalysis>();
  visit(F);

  return true;
}

INITIALIZE_TM_PASS_BEGIN(AMDGPUCodeGenPrepare, DEBUG_TYPE,
                      "AMDGPU IR optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(DivergenceAnalysis)
INITIALIZE_TM_PASS_END(AMDGPUCodeGenPrepare, DEBUG_TYPE,
                       "AMDGPU IR optimizations", false, false)

char AMDGPUCodeGenPrepare::ID = 0;

FunctionPass *llvm::createAMDGPUCodeGenPreparePass(const TargetMachine *TM) {
  return new AMDGPUCodeGenPrepare(TM);
}
