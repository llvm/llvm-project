//===-- EZHBitSliceInjection.cpp - EZH BitSlice Interrupt Workaround ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements a workaround for EZH core's lack of interrupts.
// It injects a conditional 'e_gotol_bs bitslice_handler' instruction before
// every direct branch or direct call instruction if the bitslice-interrupts
// subtarget feature is enabled.
//
//===----------------------------------------------------------------------===//

#include "EZH.h"
#include "EZHInstrInfo.h"
#include "EZHSubtarget.h"
#include "MCTargetDesc/EZHMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "ezh-bitslice-injection"

namespace {
class EZHBitSliceInjection : public MachineFunctionPass {
public:
  static char ID;
  EZHBitSliceInjection() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    const EZHSubtarget &ST = MF.getSubtarget<EZHSubtarget>();
    if (!ST.hasBitSliceInterrupts())
      return false;

    // Do not recursively inject into the bitslice handler itself!
    if (MF.getName() == "bitslice_handler")
      return false;

    const TargetInstrInfo *TII = ST.getInstrInfo();
    bool Changed = false;

    for (auto &MBB : MF) {
      for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
        MachineInstr &MI = *I;
        ++I; // Increment iterator early as we might insert before MI

        unsigned Opc = MI.getOpcode();
        bool IsBranchOrCall = false;

        switch (Opc) {
        case EZH::GOTO:
        case EZH::GOTO_ze:
        case EZH::GOTO_nz:
        case EZH::GOTO_po:
        case EZH::GOTO_ne:
        case EZH::GOTO_az:
        case EZH::GOTO_zb:
        case EZH::GOTO_ca:
        case EZH::GOTO_nc:
        case EZH::GOTO_cz:
        case EZH::GOTO_spo:
        case EZH::GOTO_sne:
        case EZH::GOTO_nbs:
        case EZH::GOTO_nex:
        case EZH::GOTO_bs:
        case EZH::GOTO_ex:
        case EZH::GOTO_REG:
        case EZH::GOTO_REG_ze:
        case EZH::GOTO_REG_nz:
        case EZH::GOTO_REG_po:
        case EZH::GOTO_REG_ne:
        case EZH::GOTO_REG_az:
        case EZH::GOTO_REG_zb:
        case EZH::GOTO_REG_ca:
        case EZH::GOTO_REG_nc:
        case EZH::GOTO_REG_cz:
        case EZH::GOTO_REG_spo:
        case EZH::GOTO_REG_sne:
        case EZH::GOTO_REG_nbs:
        case EZH::GOTO_REG_nex:
        case EZH::GOTO_REG_bs:
        case EZH::GOTO_REG_ex:
        case EZH::GOTO_REGL:
        case EZH::GOTO_REGL_ze:
        case EZH::GOTO_REGL_nz:
        case EZH::GOTO_REGL_po:
        case EZH::GOTO_REGL_ne:
        case EZH::GOTO_REGL_az:
        case EZH::GOTO_REGL_zb:
        case EZH::GOTO_REGL_ca:
        case EZH::GOTO_REGL_nc:
        case EZH::GOTO_REGL_cz:
        case EZH::GOTO_REGL_spo:
        case EZH::GOTO_REGL_sne:
        case EZH::GOTO_REGL_nbs:
        case EZH::GOTO_REGL_nex:
        case EZH::GOTO_REGL_bs:
        case EZH::GOTO_REGL_ex:
        case EZH::CALL:
        case EZH::CALLExt:
        case EZH::CALL_INDIRECT:
        case EZH::PseudoBR_JT:
          IsBranchOrCall = true;
          break;
        default:
          break;
        }

        if (IsBranchOrCall) {
          BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(EZH::GOTOL_bs))
              .addExternalSymbol("bitslice_handler");
          Changed = true;
        }
      }
    }

    return Changed;
  }
};

char EZHBitSliceInjection::ID = 0;
} // namespace

namespace llvm {
FunctionPass *createEZHBitSliceInjectionPass() {
  return new EZHBitSliceInjection();
}
} // namespace llvm
