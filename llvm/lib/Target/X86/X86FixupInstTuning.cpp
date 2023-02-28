//===-- X86FixupInstTunings.cpp - replace instructions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file does a tuning pass replacing slower machine instructions
// with faster ones. We do this here, as opposed to during normal ISel, as
// attempting to get the "right" instruction can break patterns. This pass
// is not meant search for special cases where an instruction can be transformed
// to another, it is only meant to do transformations where the old instruction
// is always replacable with the new instructions. For example:
//
//      `vpermq ymm` -> `vshufd ymm`
//          -- BAD, not always valid (lane cross/non-repeated mask)
//
//      `vpermilps ymm` -> `vshufd ymm`
//          -- GOOD, always replaceable
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "x86-fixup-inst-tuning"

STATISTIC(NumInstChanges, "Number of instructions changes");

namespace {
class X86FixupInstTuningPass : public MachineFunctionPass {
public:
  static char ID;

  X86FixupInstTuningPass() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "X86 Fixup Inst Tuning"; }

  bool runOnMachineFunction(MachineFunction &MF) override;
  bool processInstruction(MachineFunction &MF, MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator &I);

  // This pass runs after regalloc and doesn't support VReg operands.
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

private:
  const X86InstrInfo *TII = nullptr;
  const X86Subtarget *ST = nullptr;
};
} // end anonymous namespace

char X86FixupInstTuningPass::ID = 0;

INITIALIZE_PASS(X86FixupInstTuningPass, DEBUG_TYPE, DEBUG_TYPE, false, false)

FunctionPass *llvm::createX86FixupInstTuning() {
  return new X86FixupInstTuningPass();
}

bool X86FixupInstTuningPass::processInstruction(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator &I) {
  MachineInstr &MI = *I;
  unsigned Opc = MI.getOpcode();
  unsigned NumOperands = MI.getDesc().getNumOperands();

  // `vpermilps r, i` -> `vshufps r, r, i`
  // `vshufps` is always as fast or faster than `vpermilps` and takes 1 less
  // byte of code size.
  auto ProcessVPERMILPSri = [&](unsigned NewOpc) -> bool {
    unsigned MaskImm = MI.getOperand(NumOperands - 1).getImm();
    MI.removeOperand(NumOperands - 1);
    MI.addOperand(MI.getOperand(1));
    MI.setDesc(TII->get(NewOpc));
    MI.addOperand(MachineOperand::CreateImm(MaskImm));
    return true;
  };

  // `vpermilps m, i` -> `vpshufd m, i` iff no domain delay penalty on shuffles.
  // `vpshufd` is always as fast or faster than `vpermilps` and takes 1 less
  // byte of code size.
  auto ProcessVPERMILPSmi = [&](unsigned NewOpc) -> bool {
    // TODO: Might be work adding bypass delay if -Os/-Oz is enabled as
    // `vpshufd` saves a byte of code size.
    if (!ST->hasNoDomainDelayShuffle())
      return false;
    MI.setDesc(TII->get(NewOpc));
    return true;
  };

  // TODO: Add masked predicate execution variants.
  switch (Opc) {
  case X86::VPERMILPSri:
    return ProcessVPERMILPSri(X86::VSHUFPSrri);
  case X86::VPERMILPSYri:
    return ProcessVPERMILPSri(X86::VSHUFPSYrri);
  case X86::VPERMILPSZ128ri:
    return ProcessVPERMILPSri(X86::VSHUFPSZ128rri);
  case X86::VPERMILPSZ256ri:
    return ProcessVPERMILPSri(X86::VSHUFPSZ256rri);
  case X86::VPERMILPSZri:
    return ProcessVPERMILPSri(X86::VSHUFPSZrri);
  case X86::VPERMILPSmi:
    return ProcessVPERMILPSmi(X86::VPSHUFDmi);
  case X86::VPERMILPSYmi:
    // TODO: See if there is a more generic way we can test if the replacement
    // instruction is supported.
    return ST->hasAVX2() ? ProcessVPERMILPSmi(X86::VPSHUFDYmi) : false;
  case X86::VPERMILPSZ128mi:
    return ProcessVPERMILPSmi(X86::VPSHUFDZ128mi);
  case X86::VPERMILPSZ256mi:
    return ProcessVPERMILPSmi(X86::VPSHUFDZ256mi);
  case X86::VPERMILPSZmi:
    return ProcessVPERMILPSmi(X86::VPSHUFDZmi);
  default:
    return false;
  }
}

bool X86FixupInstTuningPass::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "Start X86FixupInstTuning\n";);
  bool Changed = false;
  ST = &MF.getSubtarget<X86Subtarget>();
  TII = ST->getInstrInfo();
  for (MachineBasicBlock &MBB : MF) {
    for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I) {
      if (processInstruction(MF, MBB, I)) {
        ++NumInstChanges;
        Changed = true;
      }
    }
  }
  LLVM_DEBUG(dbgs() << "End X86FixupInstTuning\n";);
  return Changed;
}
