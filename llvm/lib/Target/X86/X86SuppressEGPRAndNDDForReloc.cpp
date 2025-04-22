//==- X86SuppressEGPRAndNDDForReloc.cpp - Suppress EGPR/NDD for relocations -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This pass is added to suppress EGPR and NDD for relocations. It's used
/// together with disabling emitting APX relocation types for backward
/// compatibility with old version of linker (like before LD 2.43). It can avoid
/// the instructions updated incorrectly by old version of linker if the
/// instructions are with APX EGPR/NDD features + the relocations other than APX
/// ones (like GOTTPOFF).
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "X86.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "x86-suppress-egpr-and-ndd-for-relocation"

static cl::opt<bool> X86SuppressEGPRAndNDDForReloc(
    DEBUG_TYPE,
    cl::desc("Suppress EGPR and NDD for instructions with relocations on "
             "x86-64 ELF"),
    cl::init(true));

namespace {
class X86SuppressEGPRAndNDDForRelocPass : public MachineFunctionPass {
public:
  X86SuppressEGPRAndNDDForRelocPass() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "X86 Suppress EGPR and NDD for relocation";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  static char ID;
};
} // namespace

char X86SuppressEGPRAndNDDForRelocPass::ID = 0;

FunctionPass *llvm::createX86SuppressEGPRAndNDDForRelocPass() {
  return new X86SuppressEGPRAndNDDForRelocPass();
}

static void suppressEGPRRegClass(MachineFunction &MF, MachineInstr &MI) {
  MachineRegisterInfo *MRI = &MF.getRegInfo();
  auto Reg = MI.getOperand(0).getReg();
  if (!Reg.isVirtual()) {
    assert(!X86II::isApxExtendedReg(Reg) && "APX EGPR is used unexpectedly.");
    return;
  }

  auto *RC = MRI->getRegClass(Reg);
  auto *NewRC = X86II::constrainRegClassToNonRex2(RC);
  MRI->setRegClass(Reg, NewRC);
}

bool X86SuppressEGPRAndNDDForRelocPass::runOnMachineFunction(
    MachineFunction &MF) {
  if (MF.getTarget().Options.MCOptions.X86APXRelaxRelocations ||
      !X86SuppressEGPRAndNDDForReloc)
    return false;
  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  if (!ST.hasEGPR() && !ST.hasNDD() && !ST.hasNF())
    return false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      unsigned Opcode = MI.getOpcode();
      switch (Opcode) {
        // For GOTPC32_TLSDESC, it's emitted with physical register (EAX/RAX) in
        // X86AsmPrinter::LowerTlsAddr, and there is no corresponding target
        // flag for it, so we don't need to handle LEA64r with TLSDESC and EGPR
        // in this pass (before emitting assembly).

      case X86::ADC32rm:
      case X86::ADD32rm:
      case X86::AND32rm:
      case X86::CMP32rm:
      case X86::MOV32rm:
      case X86::OR32rm:
      case X86::SBB32rm:
      case X86::SUB32rm:
      case X86::TEST32mr:
      case X86::XOR32rm:
      case X86::CALL64m:
      case X86::JMP64m:
      case X86::TAILJMPm64:
      case X86::TEST64mr:
      case X86::ADC64rm:
      case X86::ADD64rm:
      case X86::AND64rm:
      case X86::CMP64rm:
      case X86::OR64rm:
      case X86::SBB64rm:
      case X86::SUB64rm:
      case X86::XOR64rm: {
        for (auto &MO : MI.operands()) {
          if (MO.getTargetFlags() == X86II::MO_GOTTPOFF ||
              MO.getTargetFlags() == X86II::MO_GOTPCREL)
            suppressEGPRRegClass(MF, MI);
        }
        break;
      }
      case X86::MOV64rm: {
        if (MI.getOperand(4).getTargetFlags() == X86II::MO_GOTTPOFF)
          suppressEGPRRegClass(MF, MI);
        break;
      }
      case X86::ADD64rm_NF:
      case X86::ADD64rm_ND:
      case X86::ADD64mr_ND:
      case X86::ADD64mr_NF_ND:
      case X86::ADD64rm_NF_ND: {
        // TODO: implement this if there is a case of NDD/NF instructions with
        // GOTTPOFF relocation (update the instructions to ADD64rm/ADD64mr and
        // suppress EGPR)
        for (auto &MO : MI.operands())
          assert((MO.getTargetFlags() != X86II::MO_GOTTPOFF) &&
                 "Suppressing NDD/NF instructions with relocation is "
                 "unimplemented!");
        break;
      }
      }
    }
  }

  return true;
}