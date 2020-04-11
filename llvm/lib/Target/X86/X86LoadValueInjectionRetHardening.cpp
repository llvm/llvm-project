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
#include <bitset>

using namespace llvm;

#define PASS_KEY "x86-lvi-ret"
#define DEBUG_TYPE PASS_KEY

STATISTIC(NumFences, "Number of LFENCEs inserted for LVI mitigation");
STATISTIC(NumFunctionsConsidered, "Number of functions analyzed");
STATISTIC(NumFunctionsMitigated, "Number of functions for which mitigations "
                                 "were deployed");

namespace {

class X86LoadValueInjectionRetHardeningPass : public MachineFunctionPass {
public:
  X86LoadValueInjectionRetHardeningPass() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override {
    return "X86 Load Value Injection (LVI) Ret-Hardening";
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

  static char ID;
};

} // end anonymous namespace

char X86LoadValueInjectionRetHardeningPass::ID = 0;

bool X86LoadValueInjectionRetHardeningPass::runOnMachineFunction(
    MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "***** " << getPassName() << " : " << MF.getName()
                    << " *****\n");
  const X86Subtarget *Subtarget = &MF.getSubtarget<X86Subtarget>();
  if (!Subtarget->useLVIControlFlowIntegrity() || !Subtarget->is64Bit())
    return false; // FIXME: support 32-bit

  // Don't skip functions with the "optnone" attr but participate in opt-bisect.
  const Function &F = MF.getFunction();
  if (!F.hasOptNone() && skipFunction(F))
    return false;

  ++NumFunctionsConsidered;
  const X86RegisterInfo *TRI = Subtarget->getRegisterInfo();
  const X86InstrInfo *TII = Subtarget->getInstrInfo();
  unsigned ClobberReg = X86::NoRegister;
  std::bitset<X86::NUM_TARGET_REGS> UnclobberableGR64s;
  UnclobberableGR64s.set(X86::RSP); // can't clobber stack pointer
  UnclobberableGR64s.set(X86::RIP); // can't clobber instruction pointer
  UnclobberableGR64s.set(X86::RAX); // used for function return
  UnclobberableGR64s.set(X86::RDX); // used for function return

  // We can clobber any register allowed by the function's calling convention.
  for (const MCPhysReg *PR = TRI->getCalleeSavedRegs(&MF); auto Reg = *PR; ++PR)
    UnclobberableGR64s.set(Reg);
  for (auto &Reg : X86::GR64RegClass) {
    if (!UnclobberableGR64s.test(Reg)) {
      ClobberReg = Reg;
      break;
    }
  }

  if (ClobberReg != X86::NoRegister) {
    LLVM_DEBUG(dbgs() << "Selected register "
                      << Subtarget->getRegisterInfo()->getRegAsmName(ClobberReg)
                      << " to clobber\n");
  } else {
    LLVM_DEBUG(dbgs() << "Could not find a register to clobber\n");
  }

  bool Modified = false;
  for (auto &MBB : MF) {
    if (MBB.empty())
      continue;

    MachineInstr &MI = MBB.back();
    if (MI.getOpcode() != X86::RETQ)
      continue;

    if (ClobberReg != X86::NoRegister) {
      MBB.erase_instr(&MI);
      BuildMI(MBB, MBB.end(), DebugLoc(), TII->get(X86::POP64r))
          .addReg(ClobberReg, RegState::Define)
          .setMIFlag(MachineInstr::FrameDestroy);
      BuildMI(MBB, MBB.end(), DebugLoc(), TII->get(X86::LFENCE));
      BuildMI(MBB, MBB.end(), DebugLoc(), TII->get(X86::JMP64r))
          .addReg(ClobberReg);
    } else {
      // In case there is no available scratch register, we can still read from
      // RSP to assert that RSP points to a valid page. The write to RSP is
      // also helpful because it verifies that the stack's write permissions
      // are intact.
      MachineInstr *Fence = BuildMI(MBB, MI, DebugLoc(), TII->get(X86::LFENCE));
      addRegOffset(BuildMI(MBB, Fence, DebugLoc(), TII->get(X86::SHL64mi)),
                   X86::RSP, false, 0)
          .addImm(0)
          ->addRegisterDead(X86::EFLAGS, TRI);
    }

    ++NumFences;
    Modified = true;
  }

  if (Modified)
    ++NumFunctionsMitigated;
  return Modified;
}

INITIALIZE_PASS(X86LoadValueInjectionRetHardeningPass, PASS_KEY,
                "X86 LVI ret hardener", false, false)

FunctionPass *llvm::createX86LoadValueInjectionRetHardeningPass() {
  return new X86LoadValueInjectionRetHardeningPass();
}
