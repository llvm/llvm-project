//===--------- BPFCheckUnreachableIR.cpp - Issue Unreachable Error --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Check 'unreachable' IRs and issue proper errors.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "bpf-check-unreachable-ir"

using namespace llvm;

static cl::opt<bool>
    DisableCheckUnreachableIR("bpf-disable-check-unreachable-ir", cl::Hidden,
                              cl::desc("BPF: Disable Checking Unreachable IR"),
                              cl::init(false));

namespace {

class BPFCheckUnreachableIR final : public ModulePass {
  bool runOnModule(Module &F) override;

public:
  static char ID;
  BPFCheckUnreachableIR() : ModulePass(ID) {}

private:
  void BPFCheckUnreachableIRImpl(Function &F);
  void BPFCheckInst(Function &F, BasicBlock &BB, Instruction &I);
  void HandleUnreachableInsn(Function &F, BasicBlock &BB, Instruction &I);
};
} // End anonymous namespace

char BPFCheckUnreachableIR::ID = 0;
INITIALIZE_PASS(BPFCheckUnreachableIR, DEBUG_TYPE, "BPF Check Unreachable IRs",
                false, false)

ModulePass *llvm::createBPFCheckUnreachableIR() {
  return new BPFCheckUnreachableIR();
}

void BPFCheckUnreachableIR::HandleUnreachableInsn(Function &F, BasicBlock &BB,
                                                  Instruction &I) {
  // LLVM may create a switch statement with default to a 'unreachable' basic
  // block. Do not warn for such cases.
  unsigned NumNoSwitches = 0, NumSwitches = 0;
  for (BasicBlock *Pred : predecessors(&BB)) {
    const Instruction *Term = Pred->getTerminator();
    if (Term && Term->getOpcode() == Instruction::Switch) {
      NumSwitches++;
      continue;
    }
    NumNoSwitches++;
  }
  if (NumSwitches > 0 && NumNoSwitches == 0)
    return;

  // If the previous insn is no return, do not warn for such cases.
  // One example is __bpf_unreachable from libbpf bpf_headers.h.
  Instruction *PrevI = I.getPrevNonDebugInstruction();
  if (PrevI) {
    auto *CI = dyn_cast<CallInst>(PrevI);
    if (CI && CI->doesNotReturn())
      return;
  }

  // Typically the 'unreachable' insn is the last insn in the function.
  // Find the closest line number to this insn and report such info to users.
  uint32_t LineNum = 0;
  for (Instruction &Insn : llvm::reverse(BB)) {
    const DebugLoc &DL = Insn.getDebugLoc();
    if (!DL || DL.getLine() == 0)
      continue;
    LineNum = DL.getLine();
    break;
  }

  std::string LineInfo;
  if (LineNum)
    LineInfo =
        " from line " + std::to_string(LineNum) + " to the end of function";

  F.getContext().diagnose(DiagnosticInfoGeneric(
      Twine("in function ")
          .concat("\"" + F.getName() + "\"")
          .concat(LineInfo)
          .concat(" that code was deleted as unreachable.\n")
          .concat("       due to uninitialized variable? try -Wuninitialized?"),
      DS_Error));
}

void BPFCheckUnreachableIR::BPFCheckInst(Function &F, BasicBlock &BB,
                                         Instruction &I) {
  if (I.getOpcode() == Instruction::Unreachable)
    HandleUnreachableInsn(F, BB, I);
}

void BPFCheckUnreachableIR::BPFCheckUnreachableIRImpl(Function &F) {
  // A 'unreachable' will be added to the end of naked function.
  // Let ignore these naked functions.
  if (F.hasFnAttribute(Attribute::Naked))
    return;

  for (auto &BB : F) {
    for (auto &I : BB)
      BPFCheckInst(F, BB, I);
  }
}

bool BPFCheckUnreachableIR::runOnModule(Module &M) {
  if (DisableCheckUnreachableIR)
    return false;
  for (Function &F : M)
    BPFCheckUnreachableIRImpl(F);
  return false;
}
