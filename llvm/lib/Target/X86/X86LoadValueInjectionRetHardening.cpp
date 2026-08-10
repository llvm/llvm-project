//===-- X86LoadValueInjectionRetHardening.cpp - LVI RET hardening for x86 --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Description: Replaces every `ret` instruction with the sequence:
/// ```
/// pop <scratch-reg>
/// lfence
/// jmp *<scratch-reg>
/// ```
/// where `<scratch-reg>` is some available scratch register, according to the
/// calling convention of the function being mitigated.
///
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define PASS_KEY "x86-lvi-ret"
#define DEBUG_TYPE PASS_KEY

STATISTIC(NumFences, "Number of LFENCEs inserted for LVI mitigation");
STATISTIC(NumFunctionsConsidered, "Number of functions analyzed");
STATISTIC(NumFunctionsMitigated, "Number of functions for which mitigations "
                                 "were deployed");

namespace {

constexpr StringRef X86LVIRetPassName =
    "X86 Load Value Injection (LVI) Ret-Hardening";

class X86LoadValueInjectionRetHardeningLegacy : public MachineFunctionPass {
public:
  X86LoadValueInjectionRetHardeningLegacy() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return X86LVIRetPassName; }
  bool runOnMachineFunction(MachineFunction &MF) override;

  static char ID;
};

} // end anonymous namespace

char X86LoadValueInjectionRetHardeningLegacy::ID = 0;

static bool runX86LoadValueInjectionRetHardening(MachineFunction &MF) {
  const X86Subtarget *Subtarget = &MF.getSubtarget<X86Subtarget>();
  if (!Subtarget->useLVIControlFlowIntegrity() || !Subtarget->is64Bit())
    return false; // FIXME: support 32-bit

  LLVM_DEBUG(dbgs() << "***** " << X86LVIRetPassName << " : " << MF.getName()
                    << " *****\n");
  ++NumFunctionsConsidered;
  const X86RegisterInfo *TRI = Subtarget->getRegisterInfo();
  const X86InstrInfo *TII = Subtarget->getInstrInfo();

  bool Modified = false;
  for (auto &MBB : MF) {
    for (auto MBBI = MBB.begin(); MBBI != MBB.end(); ++MBBI) {
      if (MBBI->getOpcode() != X86::RET64)
        continue;

      unsigned ClobberReg = TRI->findDeadCallerSavedReg(MBB, MBBI);
      if (ClobberReg != X86::NoRegister) {
        BuildMI(MBB, MBBI, DebugLoc(), TII->get(X86::POP64r))
            .addReg(ClobberReg, RegState::Define)
            .setMIFlag(MachineInstr::FrameDestroy);
        BuildMI(MBB, MBBI, DebugLoc(), TII->get(X86::LFENCE));
        BuildMI(MBB, MBBI, DebugLoc(), TII->get(X86::JMP64r))
            .addReg(ClobberReg);
        MBB.erase(MBBI);
      } else {
        // In case there is no available scratch register, we can still read
        // from RSP to assert that RSP points to a valid page. The write to RSP
        // is also helpful because it verifies that the stack's write
        // permissions are intact.
        MachineInstr *Fence =
            BuildMI(MBB, MBBI, DebugLoc(), TII->get(X86::LFENCE));
        addRegOffset(BuildMI(MBB, Fence, DebugLoc(), TII->get(X86::SHL64mi)),
                     X86::RSP, false, 0)
            .addImm(0)
            ->addRegisterDead(X86::EFLAGS, TRI);
      }

      ++NumFences;
      Modified = true;
      break;
    }
  }

  if (Modified)
    ++NumFunctionsMitigated;
  return Modified;
}

bool X86LoadValueInjectionRetHardeningLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  // Don't skip functions with the "optnone" attr but participate in opt-bisect.
  // Note: NewPM implements this behavior by default.
  const Function &F = MF.getFunction();
  if (!F.hasOptNone() && skipFunction(F))
    return false;

  return runX86LoadValueInjectionRetHardening(MF);
}

PreservedAnalyses X86LoadValueInjectionRetHardeningPass::run(
    MachineFunction &MF, MachineFunctionAnalysisManager &MFAM) {
  return runX86LoadValueInjectionRetHardening(MF)
             ? getMachineFunctionPassPreservedAnalyses()
                   .preserveSet<CFGAnalyses>()
             : PreservedAnalyses::all();
}

INITIALIZE_PASS(X86LoadValueInjectionRetHardeningLegacy, PASS_KEY,
                "X86 LVI ret hardener", false, false)

FunctionPass *llvm::createX86LoadValueInjectionRetHardeningLegacyPass() {
  return new X86LoadValueInjectionRetHardeningLegacy();
}
