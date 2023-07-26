//===- AMDGPUInsertSingleUseVDST.cpp - Insert s_singleuse_vdst instructions ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Insert s_singleuse_vdst instructions on GFX11.5+ to mark regions of VALU
/// instructions that produce single-use VGPR values. If the value is forwarded
/// to the consumer instruction prior to VGPR writeback, the hardware can
/// then skip (kill) the VGPR write.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-insert-single-use-vdst"

namespace {
class AMDGPUInsertSingleUseVDST : public MachineFunctionPass {
private:
  const SIInstrInfo *SII;

public:
  static char ID;

  AMDGPUInsertSingleUseVDST() : MachineFunctionPass(ID) {}

  void emitSingleUseVDST(MachineInstr &MI) const {
    BuildMI(*MI.getParent(), MI, DebugLoc(), SII->get(AMDGPU::S_SINGLEUSE_VDST))
        .addImm(0x1);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    const auto &ST = MF.getSubtarget<GCNSubtarget>();
    if (!ST.hasVGPRSingleUseHintInsts())
      return false;

    SII = ST.getInstrInfo();
    const auto *TRI = MF.getSubtarget().getRegisterInfo();
    bool InstructionEmitted = false;

    for (MachineBasicBlock &MBB : MF) {
      DenseMap<MCPhysReg, unsigned> RegisterUseCount; // TODO: MCRegUnits

      // Handle boundaries at the end of basic block separately to avoid
      // false positives. If they are live at the end of a basic block then
      // assume it has more uses later on.
      for (const auto &Liveouts : MBB.liveouts())
        RegisterUseCount[Liveouts.PhysReg] = 2;

      for (MachineInstr &MI : reverse(MBB.instrs())) {
        // All registers in all operands need to be single use for an
        // instruction to be marked as a single use producer.
        bool AllProducerOperandsAreSingleUse = true;

        for (const auto &Operand : MI.operands()) {
          if (!Operand.isReg())
            continue;
          const auto Reg = Operand.getReg();

          // Count the number of times each register is read.
          if (Operand.readsReg())
            RegisterUseCount[Reg]++;

          // Do not attempt to optimise across exec mask changes.
          if (MI.modifiesRegister(AMDGPU::EXEC, TRI)) {
            for (auto &UsedReg : RegisterUseCount)
              UsedReg.second = 2;
          }

          // If we are at the point where the register first became live,
          // check if the operands are single use.
          if (!MI.modifiesRegister(Reg, TRI))
            continue;
          if (RegisterUseCount[Reg] > 1)
            AllProducerOperandsAreSingleUse = false;
          // Reset uses count when a register is no longer live.
          RegisterUseCount.erase(Reg);
        }
        if (AllProducerOperandsAreSingleUse && SIInstrInfo::isVALU(MI)) {
          // TODO: Replace with candidate logging for instruction grouping
          // later.
          emitSingleUseVDST(MI);
          InstructionEmitted = true;
        }
      }
    }
    return InstructionEmitted;
  }
};
} // namespace

char AMDGPUInsertSingleUseVDST::ID = 0;

char &llvm::AMDGPUInsertSingleUseVDSTID = AMDGPUInsertSingleUseVDST::ID;

INITIALIZE_PASS(AMDGPUInsertSingleUseVDST, DEBUG_TYPE,
                "AMDGPU Insert SingleUseVDST", false, false)
