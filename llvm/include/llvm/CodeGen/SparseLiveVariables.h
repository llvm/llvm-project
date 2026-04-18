//===-- SparseLiveVariables.h - RISC-V Live Variable Analysis ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SPARSELIVEVARIABLES_H
#define LLVM_CODEGEN_SPARSELIVEVARIABLES_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

namespace llvm {

class SparseLiveVariables : public MachineFunctionPass {
public:
  static char ID;

  SparseLiveVariables() : MachineFunctionPass(ID) {}

  struct BlockInfo {
    SparseBitVector<> LiveIn;
    SparseBitVector<> LiveOut;
  };

  DenseMap<const MachineBasicBlock *, BlockInfo> BlockLiveness;

  class LivenessTracker {
    SparseBitVector<> LiveRegs;
    const MachineRegisterInfo *MRI;

  public:
    bool isTrackableRegister(Register Reg) const {
      if (Reg.isVirtual()) return true;
      if (Reg.isPhysical()) {
        if (MRI->isReserved(Reg)) return false;
        return true;
      }
      return false;
    }

    LivenessTracker(const SparseBitVector<> &LiveOut,
                    const MachineRegisterInfo *MRI)
: LiveRegs(LiveOut), MRI(MRI) {}

    void stepBackward(const MachineInstr &MI) {
      if (MI.isDebugInstr() || MI.isMetaInstruction())
        return;

      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.isDef()) {
          Register Reg = MO.getReg();
          if (Reg.isValid() && isTrackableRegister(Reg))
            LiveRegs.reset(Reg.id());
        }
      }

      if (!MI.isPHI()) {
        for (const MachineOperand &MO : MI.operands()) {
          if (MO.isReg() && MO.isUse()) {
            Register Reg = MO.getReg();
            if (Reg.isValid() && isTrackableRegister(Reg))
              LiveRegs.set(Reg.id());
          }
        }
      }
    }

    bool isLive(Register Reg) const {
      if (!Reg.isValid()) return false;
      return LiveRegs.test(Reg.id());
    }

    const SparseBitVector<> &getLiveSet() const { return LiveRegs; }
  };


  const SparseBitVector<> &getLiveInSet(const MachineBasicBlock *MBB) const {
    auto It = BlockLiveness.find(MBB);
    assert(It != BlockLiveness.end() && "Block not analyzed");
    return It->second.LiveIn;
  }

  const SparseBitVector<> &getLiveOutSet(const MachineBasicBlock *MBB) const {
    auto It = BlockLiveness.find(MBB);
    assert(It != BlockLiveness.end() && "Block not analyzed");
    return It->second.LiveOut;
  }


  void verifyLiveness(const MachineFunction &MF) const;

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "RISC-V Live Variable Analysis"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  const MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;
};

} // end namespace llvm

#endif // LLVM_CODEGEN_SPARSELIVEVARIABLES_H
