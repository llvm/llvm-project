//=== PISAOptimizeSubregAccess.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Given two consecutive COPY operations utilizing same destination and source
// registers (and differing in subregister index), attempt to combine them into
// a single COPY operation, e.g.
//
// %v.sub16_2:regv4_16b = COPY %4.sub16_0:regv2_16b
// %v.sub16_3:regv4_16b = COPY %4.sub16_1:regv2_16b
// => undef %v.sub16_zw:regv4_16b = COPY %4:regv2_16b
//
//===----------------------------------------------------------------------===//

#include "PISA.h"
#include "PISAMCInstLower.h"
#include "PISASubtarget.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "pisa-optimize-subreg-access"
#define DEBUG_NAME "PISA optimize subreg accesses"

using namespace llvm;

namespace {

class PISAOptimizeSubregAccess : public MachineFunctionPass {
public:
  static char ID;

  PISAOptimizeSubregAccess();

  StringRef getPassName() const override { return DEBUG_NAME; }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  int getCombineSubreg(unsigned, unsigned, unsigned, unsigned);
};
} // end anonymous namespace

char PISAOptimizeSubregAccess::ID = 0;
INITIALIZE_PASS(PISAOptimizeSubregAccess, DEBUG_TYPE, DEBUG_NAME, false, false)

void PISAOptimizeSubregAccess::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

PISAOptimizeSubregAccess::PISAOptimizeSubregAccess() : MachineFunctionPass(ID) {
  initializePISAOptimizeSubregAccessPass(*PassRegistry::getPassRegistry());
}

int PISAOptimizeSubregAccess::getCombineSubreg(unsigned RegSize,
                                               unsigned SubRegSize,
                                               unsigned Idx0, unsigned Idx1) {
  int NewIdx = -1; // invalid
  switch (SubRegSize) {
  case 8: {
    if ((Idx0 == PISA::sub8_0) && (Idx1 == PISA::sub8_1))
      NewIdx = PISA::sub8_xy;
    if ((Idx0 == PISA::sub8_2) && (Idx1 == PISA::sub8_3))
      NewIdx = PISA::sub8_zw;
  } break;
  case 16: {
    if ((Idx0 == PISA::sub16_0) && (Idx1 == PISA::sub16_1))
      NewIdx = PISA::sub16_xy;
    if ((Idx0 == PISA::sub16_2) && (Idx1 == PISA::sub16_3))
      NewIdx = PISA::sub16_zw;
  } break;
  case 32: {
    if ((Idx0 == PISA::sub32_0) && (Idx1 == PISA::sub32_1))
      NewIdx = PISA::sub32_xy;
    if ((Idx0 == PISA::sub32_2) && (Idx1 == PISA::sub32_3))
      NewIdx = PISA::sub32_zw;
  } break;
  case 64: {
    if ((Idx0 == PISA::sub64_0) && (Idx1 == PISA::sub64_1))
      NewIdx = PISA::sub64_xy;
    if ((Idx0 == PISA::sub64_2) && (Idx1 == PISA::sub64_3))
      NewIdx = PISA::sub64_zw;
  } break;
  default:
    break;
  }
  if ((NewIdx > 0) && (SubRegSize * 2 == RegSize))
    NewIdx = 0; // use full reg
  return NewIdx;
}

bool PISAOptimizeSubregAccess::runOnMachineFunction(MachineFunction &MF) {
  const PISASubtarget &ST = MF.getSubtarget<PISASubtarget>();
  const PISAInstrInfo *TII = ST.getInstrInfo();
  const PISARegisterInfo *TRI = ST.getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  bool Changed = false;
  SmallVector<MachineInstr *> DeleteMIs;
  MachineInstr *LastMI = nullptr;
  for (MachineBasicBlock &MBB : MF) {
    LastMI = nullptr;
    for (MachineInstr &MI : MBB) {
      if (!LastMI || !MI.isCopy() || !LastMI->isCopy()) {
        LastMI = &MI;
        continue;
      }
      MachineOperand &Dst = MI.getOperand(0);
      Register DstReg = Dst.getReg();
      unsigned DstSubreg = Dst.getSubReg();
      if (!(DstReg.isVirtual() && DstSubreg)) {
        LastMI = &MI;
        continue;
      }
      const MachineOperand &LDst = LastMI->getOperand(0);
      Register LDstReg = LDst.getReg();
      unsigned LDstSubreg = LDst.getSubReg();
      if (!(LDstReg.isVirtual() && LDstSubreg && (LDstReg == DstReg))) {
        LastMI = &MI;
        continue;
      }
      const TargetRegisterClass *DstRC =
          TRI->getSubRegisterClass(MRI.getRegClass(DstReg), DstSubreg);
      const TargetRegisterClass *LDstRC =
          TRI->getSubRegisterClass(MRI.getRegClass(LDstReg), LDstSubreg);
      if (!DstRC || !LDstRC) {
        LastMI = &MI;
        continue;
      }
      TypeSize DstRegSize = TRI->getRegSizeInBits(*MRI.getRegClass(DstReg));
      unsigned DstSubRegSize = TRI->getSubRegIdxSize(DstSubreg);
      unsigned LDstSubRegSize = TRI->getSubRegIdxSize(LDstSubreg);
      if ((DstSubRegSize + LDstSubRegSize) > 128) { // exceed max 'mov' size
        LastMI = &MI;
        continue;
      }
      int NewDstIdx =
          getCombineSubreg(DstRegSize, DstSubRegSize, LDstSubreg, DstSubreg);
      if (NewDstIdx >= 0) {
        MachineOperand &Src = MI.getOperand(1);
        Register SrcReg = Src.getReg();
        unsigned SrcSubreg = Src.getSubReg();
        if (!(SrcReg.isVirtual() && SrcSubreg)) {
          LastMI = &MI;
          continue;
        }
        const MachineOperand &LSrc = LastMI->getOperand(1);
        Register LSrcReg = LSrc.getReg();
        unsigned LSrcSubreg = LSrc.getSubReg();
        if (!(LSrcReg.isVirtual() && LSrcSubreg && (LSrcReg == SrcReg))) {
          LastMI = &MI;
          continue;
        }
        const TargetRegisterClass *SrcRC =
            TRI->getSubRegisterClass(MRI.getRegClass(SrcReg), SrcSubreg);
        if (!SrcRC) {
          LastMI = &MI;
          continue;
        }
        TypeSize SrcRegSize = TRI->getRegSizeInBits(*MRI.getRegClass(SrcReg));
        unsigned SrcSubRegSize = TRI->getSubRegIdxSize(SrcSubreg);
        int NewSrcIdx =
            getCombineSubreg(SrcRegSize, SrcSubRegSize, LSrcSubreg, SrcSubreg);
        if (NewSrcIdx >= 0) {
          DebugLoc DL = MI.getDebugLoc();
          MachineInstrBuilder NewMI =
              BuildMI(*MI.getParent(), MI, DL, TII->get(TargetOpcode::COPY));
          RegState DstUndef =
              NewDstIdx == 0 ? RegState::NoFlags : RegState::Undef;
          RegState SrcUndef = (Src.isUndef() && LSrc.isUndef())
                                  ? RegState::Undef
                                  : RegState::NoFlags;
          NewMI.addDef(DstReg, DstUndef, NewDstIdx);
          NewMI.addReg(SrcReg, SrcUndef, NewSrcIdx);
          DeleteMIs.push_back(LastMI);
          DeleteMIs.push_back(&MI);
          Changed = true;
          LastMI = nullptr;
          continue;
        }
      }
      LastMI = &MI;
    }
  }
  for (MachineInstr *MI : DeleteMIs)
    MI->eraseFromParent();
  return Changed;
}

namespace llvm {
FunctionPass *createPISAOptimizeSubregAccess() {
  return new PISAOptimizeSubregAccess();
}
} // end namespace llvm
