//===-- CallBrPrepare - Prepare callbr for code generation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers callbrs in LLVM IR in order to to assist SelectionDAG's
// codegen.
//
// In particular, this pass assists in inserting register copies for the output
// values of a callbr along the edges leading to the indirect target blocks.
// Though the output SSA value is defined by the callbr instruction itself in
// the IR representation, the value cannot be copied to the appropriate virtual
// registers prior to jumping to an indirect label, since the jump occurs
// within the user-provided assembly blob.
//
// Instead, those copies must occur separately at the beginning of each
// indirect target. That requires that we create a separate SSA definition in
// each of them (via llvm.callbr.landingpad), and may require splitting
// critical edges so we have a location to place the intrinsic. Finally, we
// remap users of the original callbr output SSA value to instead point to the
// appropriate llvm.callbr.landingpad value.
//
// Ideally, this could be done inside SelectionDAG, or in the
// MachineInstruction representation, without the use of an IR-level intrinsic.
// But, within the current framework, it’s simpler to implement as an IR pass.
// (If support for callbr in GlobalISel is implemented, it’s worth considering
// whether this is still required.)
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "callbrprepare"

namespace {

class CallBrPrepare : public FunctionPass {
public:
  CallBrPrepare() : FunctionPass(ID) {}
  static char ID;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnFunction(Function &Fn) override;
};

} // end anonymous namespace

char CallBrPrepare::ID = 0;
INITIALIZE_PASS(CallBrPrepare, DEBUG_TYPE, "Prepare callbr", false, false)

FunctionPass *llvm::createCallBrPass() { return new CallBrPrepare(); }

void CallBrPrepare::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

bool CallBrPrepare::runOnFunction(Function &Fn) {
  for (BasicBlock &BB : Fn) {
    auto *CBR = dyn_cast<CallBrInst>(BB.getTerminator());
    if (!CBR)
      continue;
    // TODO: something interesting.
    // https://discourse.llvm.org/t/rfc-syncing-asm-goto-with-outputs-with-gcc/65453/8
  }
  return false;
}
