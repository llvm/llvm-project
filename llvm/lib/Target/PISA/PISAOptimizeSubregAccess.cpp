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
  auto &ST = MF.getSubtarget<PISASubtarget>();
  auto *TII = ST.getInstrInfo();
  auto *TRI = ST.getRegisterInfo();
  auto &MRI = MF.getRegInfo();

  bool Changed = false;
  SmallVector<MachineInstr *> DeleteMIs;
  MachineInstr *LastMI = nullptr;
  for (auto &MBB : MF) {
    LastMI = nullptr;
    for (auto &MI : MBB) {
      if (!LastMI || !MI.isCopy() || !LastMI->isCopy()) {
        LastMI = &MI;
        continue;
      }
      auto &Dst = MI.getOperand(0);
      auto DstReg = Dst.getReg();
      auto DstSubreg = Dst.getSubReg();
      if (!(DstReg.isVirtual() && DstSubreg)) {
        LastMI = &MI;
        continue;
      }
      auto LDst = LastMI->getOperand(0);
      auto LDstReg = LDst.getReg();
      auto LDstSubreg = LDst.getSubReg();
      if (!(LDstReg.isVirtual() && LDstSubreg && (LDstReg == DstReg))) {
        LastMI = &MI;
        continue;
      }
      auto *DstRC =
          TRI->getSubRegisterClass(MRI.getRegClass(DstReg), DstSubreg);
      auto *LDstRC =
          TRI->getSubRegisterClass(MRI.getRegClass(LDstReg), LDstSubreg);
      if (!DstRC || !LDstRC) {
        LastMI = &MI;
        continue;
      }
      auto DstRegSize = TRI->getRegSizeInBits(*MRI.getRegClass(DstReg));
      auto DstSubRegSize = TRI->getSubRegIdxSize(DstSubreg);
      auto LDstSubRegSize = TRI->getSubRegIdxSize(LDstSubreg);
      if ((DstSubRegSize + LDstSubRegSize) > 128) { // exceed max 'mov' size
        LastMI = &MI;
        continue;
      }
      auto NewDstIdx =
          getCombineSubreg(DstRegSize, DstSubRegSize, LDstSubreg, DstSubreg);
      if (NewDstIdx >= 0) {
        auto &Src = MI.getOperand(1);
        auto SrcReg = Src.getReg();
        auto SrcSubreg = Src.getSubReg();
        if (!(SrcReg.isVirtual() && SrcSubreg)) {
          LastMI = &MI;
          continue;
        }
        auto LSrc = LastMI->getOperand(1);
        auto LSrcReg = LSrc.getReg();
        auto LSrcSubreg = LSrc.getSubReg();
        if (!(LSrcReg.isVirtual() && LSrcSubreg && (LSrcReg == SrcReg))) {
          LastMI = &MI;
          continue;
        }
        auto *SrcRC =
            TRI->getSubRegisterClass(MRI.getRegClass(SrcReg), SrcSubreg);
        if (!SrcRC) {
          LastMI = &MI;
          continue;
        }
        auto SrcRegSize = TRI->getRegSizeInBits(*MRI.getRegClass(SrcReg));
        auto SrcSubRegSize = TRI->getSubRegIdxSize(SrcSubreg);
        auto NewSrcIdx =
            getCombineSubreg(SrcRegSize, SrcSubRegSize, LSrcSubreg, SrcSubreg);
        if (NewSrcIdx >= 0) {
          DebugLoc DL = MI.getDebugLoc();
          auto NewMI =
              BuildMI(*MI.getParent(), MI, DL, TII->get(TargetOpcode::COPY));
          auto DstUndef = NewDstIdx == 0 ? RegState::NoFlags : RegState::Undef;
          auto SrcUndef = (Src.isUndef() && LSrc.isUndef()) ? RegState::Undef
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
  for (auto *MI : DeleteMIs)
    MI->eraseFromParent();
  return Changed;
}

namespace llvm {
FunctionPass *createPISAOptimizeSubregAccess() {
  return new PISAOptimizeSubregAccess();
}
} // end namespace llvm
