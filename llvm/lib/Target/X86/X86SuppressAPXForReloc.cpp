//===- X86SuppressAPXForReloc.cpp - Suppress APX features for relocations -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This pass is added to suppress APX features for relocations. It's used
/// together with disabling emitting APX relocation types for backward
/// compatibility with old version of linker (like before LD 2.43). It can avoid
/// the instructions updated incorrectly by old version of linker if the
/// instructions are with APX EGPR/NDD/NF features + the relocations other than
/// APX ones (like GOTTPOFF).
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "MCTargetDesc/X86MCTargetDesc.h"
#include "X86.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "x86-suppress-apx-for-relocation"

static cl::opt<bool> X86EnableAPXForRelocation(
    "x86-enable-apx-for-relocation",
    cl::desc("Enable APX features (EGPR, NDD and NF) for instructions with "
             "relocations on x86-64 ELF"),
    cl::init(false));

namespace {
class X86SuppressAPXForRelocationPass : public MachineFunctionPass {
public:
  X86SuppressAPXForRelocationPass() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "X86 Suppress APX features for relocation";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  static char ID;
};
} // namespace

char X86SuppressAPXForRelocationPass::ID = 0;

INITIALIZE_PASS_BEGIN(X86SuppressAPXForRelocationPass, DEBUG_TYPE,
                      "X86 Suppress APX features for relocation", false, false)
INITIALIZE_PASS_END(X86SuppressAPXForRelocationPass, DEBUG_TYPE,
                    "X86 Suppress APX features for relocation", false, false)

FunctionPass *llvm::createX86SuppressAPXForRelocationPass() {
  return new X86SuppressAPXForRelocationPass();
}

static void suppressEGPRRegClass(MachineFunction &MF, MachineInstr &MI,
                                 unsigned int OpNum) {
  MachineRegisterInfo *MRI = &MF.getRegInfo();
  auto Reg = MI.getOperand(OpNum).getReg();
  if (!Reg.isVirtual()) {
    assert(!X86II::isApxExtendedReg(Reg) && "APX EGPR is used unexpectedly.");
    return;
  }

  auto *RC = MRI->getRegClass(Reg);
  auto *NewRC = X86II::constrainRegClassToNonRex2(RC);
  MRI->setRegClass(Reg, NewRC);
}

bool X86SuppressAPXForRelocationPass::runOnMachineFunction(
    MachineFunction &MF) {
  if (MF.getTarget().Options.MCOptions.X86APXRelaxRelocations ||
      X86EnableAPXForRelocation)
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
              MO.getTargetFlags() == X86II::MO_GOTPCREL) {
            suppressEGPRRegClass(MF, MI, 0);
            break;
          }
        }
        break;
      }
      case X86::MOV64rm: {
        for (auto &MO : MI.operands()) {
          if (MO.getTargetFlags() == X86II::MO_GOTTPOFF) {
            suppressEGPRRegClass(MF, MI, 0);
            break;
          }
        }
        break;
      }
      case X86::ADD64rm_NF:
      case X86::ADD64rm_ND:
      case X86::ADD64rm_NF_ND: {
        for (auto &MO : MI.operands()) {
          if (MO.getTargetFlags() == X86II::MO_GOTTPOFF) {
            suppressEGPRRegClass(MF, MI, 0);
            const MCInstrDesc &NewDesc = ST.getInstrInfo()->get(X86::ADD64rm);
            MI.setDesc(NewDesc);
            if (Opcode == X86::ADD64rm_ND || Opcode == X86::ADD64rm_NF_ND) {
              MI.tieOperands(0, 1);
              MI.getOperand(1).setIsKill(false);
              suppressEGPRRegClass(MF, MI, 1);
            }
            break;
          }
        }
        break;
      }
      case X86::ADD64mr_ND:
      case X86::ADD64mr_NF_ND: {
        for ([[maybe_unused]] auto &MO : MI.operands()) {
          assert((MO.getTargetFlags() != X86II::MO_GOTTPOFF) &&
                 "Suppressing this instruction with relocation is "
                 "unimplemented!");
          break;
        }
        break;
      }
      }
    }
  }

  return true;
}