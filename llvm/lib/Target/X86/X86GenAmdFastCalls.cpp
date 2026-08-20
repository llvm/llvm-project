//===-- X86GenAmdFastCalls.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation converts standard math function calls into their
// corresponding AMD AOCL fast entry points for X86 targets, e.g.:
//     tan ---> amd_fasttan
// Such lowering is only legal under fast-math semantics and when the AMD
// fast math library has been selected (-fast-library=AMDLIBM /
// -ffastlib=AMDLIBM).
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "x86-gen-aocl-fast"

using namespace llvm;

namespace {

class X86GenAmdFastCalls : public MachineFunctionPass {
public:
  static char ID;

  X86GenAmdFastCalls() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &F) override;

  StringRef getPassName() const override {
    return "X86 Generate AOCL Fast Entries";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<MachineOptimizationRemarkEmitterPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  TargetLibraryInfo *TLI = nullptr;
  MachineOptimizationRemarkEmitter *ORE = nullptr;
  bool isCandidateSafeToLower(MachineInstr *MI) const;
  bool createAmdFastCall(MachineInstr *MI) const;
};

} // namespace

// Rewriting a math call to its AOCL fast-call variant is only legal under
// fast-math semantics. By the time this late machine pass runs, the
// per-operation fast-math flags carried by the original call have already been
// lowered away, so we rely on the function-level fast-math attribute that the
// frontend sets under -ffast-math. Together with the explicit
// -fast-library=AMDLIBM opt-in (checked in runOnMachineFunction) this gates
// the transformation.
bool X86GenAmdFastCalls::isCandidateSafeToLower(MachineInstr *MI) const {
  const Function &F = MI->getMF()->getFunction();
  return F.getFnAttribute("no-signed-zeros-fp-math").getValueAsBool();
}

/// Lowers math functions to AOCL fast entry points.
///     e.g.: tan         --> amd_fasttan
/// The callsite symbol is updated during lowering.
bool X86GenAmdFastCalls::createAmdFastCall(MachineInstr *MI) const {
  StringRef CallSiteName = "";
  StringRef LibFastFnName = "";
  if (MI->getOperand(0).isSymbol()) {
    CallSiteName = MI->getOperand(0).getSymbolName();
  } else if (MI->getOperand(0).isGlobal()) {
    CallSiteName = MI->getOperand(0).getGlobal()->getName();
  } else {
    return false;
  }

  LLVM_DEBUG(dbgs() << "Candidate Func = " << CallSiteName << "\n";);
  if (CallSiteName.empty()) {
    return false;
  }
  LibFastFnName = TLI->getFastFunctionFromMathLib(CallSiteName);
  if (LibFastFnName.empty()) {
    LLVM_DEBUG(dbgs() << "Fast call not supported\n";);
    return false;
  }
  LLVM_DEBUG(dbgs() << "Candidate Func has fast Call variant available = "
                    << LibFastFnName << "\n";);
  MI->getOperand(0).ChangeToES(LibFastFnName.data(),
                               MI->getOperand(0).getTargetFlags());

  LLVM_DEBUG(dbgs() << "Successfully replaced with fastcall= "
                    << LibFastFnName << "\n";);

  ORE->emit([&]() {
    return MachineOptimizationRemark(DEBUG_TYPE, "Passed", MI->getDebugLoc(),
                                     MI->getParent())
           << "Successfully replaced with fastcall= " << LibFastFnName
           << "\n";
  });
  return true;
}

bool X86GenAmdFastCalls::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  if (skipFunction(MF.getFunction()))
    return Changed;
  if (MF.getFunction().isDeclaration())
    return Changed;
  SmallVector<MachineInstr *, 4> Callsites;
  for (auto &BB : MF) {
    for (auto &I : BB) {
      if (I.isCall()) {
        Callsites.push_back(&I);
      }
    }
  }

  if (Callsites.empty()) {
    return Changed;
  }

  TLI = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(MF.getFunction());
  ORE = &getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE();
  if (!TLI)
    return Changed;

  if (TLI->getFastMathLib() !=
      TargetLibraryInfoImpl::FastLibrary::FAST_AMDLIBM) {
    LLVM_DEBUG(dbgs() << "-fast-library=AMDLIBM not used so bailing out.\n";);
    return Changed;
  }

  for (auto *CI : Callsites) {
    if (isCandidateSafeToLower(CI)) {
      LLVM_DEBUG(dbgs() << "Call Inst has fastMath flags\n";);
      Changed |= createAmdFastCall(CI);
    } else
      LLVM_DEBUG(dbgs() << "Call Inst does not have fastMath flags\n";);
  }
  return Changed;
}

char X86GenAmdFastCalls::ID = 0;

char &llvm::X86GenAmdFastCallsID = X86GenAmdFastCalls::ID;

INITIALIZE_PASS_BEGIN(X86GenAmdFastCalls, DEBUG_TYPE,
                      "Generate AMD Fast calls", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineOptimizationRemarkEmitterPass)
INITIALIZE_PASS_END(X86GenAmdFastCalls, DEBUG_TYPE,
                    "Generate AMD Fast calls", false, false)

FunctionPass *llvm::createX86GenAmdFastCallsPass() {
  return new X86GenAmdFastCalls();
}
