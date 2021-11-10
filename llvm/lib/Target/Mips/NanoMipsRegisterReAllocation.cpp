//===- NanoMipsRegisterHinting.cpp - nanoMIPS reg. hinting pass -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a pass that performs register hinting for better
/// register allocation.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsSubtarget.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include <algorithm>
#include <cmath>

using namespace llvm;

#define NM_MOVE_OPT_NAME "nanoMIPS move optimization pass"

namespace {
struct NanoMipsRegisterReAlloc : public MachineFunctionPass {
  using MBBIter = MachineBasicBlock::iterator;
  std::array<unsigned, 8> GPR3{Mips::A0_NM, Mips::A1_NM, Mips::A2_NM,
                               Mips::A3_NM, Mips::S0_NM, Mips::S1_NM,
                               Mips::S2_NM, Mips::S3_NM};
  std::array<unsigned, 16> GPR4{
      Mips::A0_NM, Mips::A1_NM, Mips::A2_NM, Mips::A3_NM,
      Mips::A4_NM, Mips::A5_NM, Mips::A6_NM, Mips::A7_NM,
      Mips::S0_NM, Mips::S1_NM, Mips::S2_NM, Mips::S3_NM,
      Mips::S4_NM, Mips::S5_NM, Mips::S6_NM, Mips::S7_NM};
  static char ID;
  const MipsSubtarget *STI;
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;
  VirtRegMap *VRM;
  LiveRegMatrix *LRM;

  NanoMipsRegisterReAlloc() : MachineFunctionPass(ID) {
    initializeNanoMipsRegisterReAllocPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<VirtRegMap>();
    AU.addRequired<LiveIntervals>();
    AU.addRequired<LiveRegMatrix>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return NM_MOVE_OPT_NAME; }

  bool runOnMachineFunction(MachineFunction &) override;
  bool hintRegister(MachineBasicBlock &);
  bool tryGPR4ReAlloc(Register, Register, Register);
  bool isGPR3(unsigned);
  bool isGPR4(unsigned);
  bool hasNoInterference(LiveInterval &, unsigned);
  bool isValidForReplacement(Register);
};
} // namespace

char NanoMipsRegisterReAlloc::ID = 0;

INITIALIZE_PASS_BEGIN(NanoMipsRegisterReAlloc, "nmregrealloc", "nanoMIPS Register Re-allocation", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrix)
INITIALIZE_PASS_END(NanoMipsRegisterReAlloc, "nmregrealloc", "nanoMIPS Register Re-allocation", false,
                    false)

bool NanoMipsRegisterReAlloc::runOnMachineFunction(MachineFunction &Fn) {
  STI = &static_cast<const MipsSubtarget &>(Fn.getSubtarget());
  TII = STI->getInstrInfo();
  MRI = &Fn.getRegInfo();
  LIS = &getAnalysis<LiveIntervals>();
  VRM = &getAnalysis<VirtRegMap>();
  LRM = &getAnalysis<LiveRegMatrix>();
  bool Modified = false;
  for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
       ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    Modified |= hintRegister(MBB);
  }

  return Modified;
}

bool NanoMipsRegisterReAlloc::isGPR3(unsigned Reg) {
  return std::find(GPR3.begin(), GPR3.end(), Reg) != GPR3.end();
}

bool NanoMipsRegisterReAlloc::isGPR4(unsigned Reg) {
  return std::find(GPR4.begin(), GPR4.end(), Reg) != GPR4.end();
}

bool NanoMipsRegisterReAlloc::hasNoInterference(LiveInterval &VReg, unsigned PReg) {
  return LRM->checkInterference(VReg, PReg) == LiveRegMatrix::IK_Free;
}

bool NanoMipsRegisterReAlloc::isValidForReplacement(Register Reg) {
  if (!Reg || !Reg.isVirtual())
    return false;

  if (!LIS->hasInterval(Reg))
    return false;

  // InlineSpiller does not call LRM::assign() after an LI split leaving
  // it in an inconsistent state, so we cannot call LRM::unassign().
  // See llvm bug #48911.
  // Skip reassign if a register has originated from such split.
  // FIXME: Remove the workaround when bug #48911 is fixed.
  if (VRM->getPreSplitReg(Reg))
    return false;

  Register PhysReg = VRM->getPhys(Reg);

  if (!PhysReg)
    return false;

  const MachineInstr *Def = MRI->getUniqueVRegDef(Reg);
  if (Def && Def->isCopy() && Def->getOperand(1).getReg() == PhysReg)
    return false;

  for (auto U : MRI->use_nodbg_operands(Reg)) {
    if (U.isImplicit())
      return false;
    const MachineInstr *UseInst = U.getParent();
    if (UseInst->isCopy() && UseInst->getOperand(0).getReg() == PhysReg)
      return false;
  }

  return true;
}

bool NanoMipsRegisterReAlloc::tryGPR4ReAlloc(Register Dst, Register Src1, Register Src2) {
  auto &DstInt = LIS->getInterval(Dst);
  auto &Src1Int = LIS->getInterval(Src1);
  if (DstInt.overlaps(Src1Int))
    return false;

  // If SRC2 is not GPR4 and is not valid for replacement, exit immediately.
  Register Src2Phys = VRM->getPhys(Src2);
  if (!isGPR4(Src2Phys) && !isValidForReplacement(Src2))
    return false;

  Register DstPhys = VRM->getPhys(Dst);
  Register Src1Phys = VRM->getPhys(Src1);

  // In phase 1, goal is to allocate the same GPR4 registers for SRC1 and DST.
  bool Phase1Successful = (DstPhys == Src1Phys && isGPR4(DstPhys));
  if (!Phase1Successful) {
    bool ReAllocDst = isValidForReplacement(Dst);
    bool ReAllocSrc1 = isValidForReplacement(Src1);
    if (isGPR4(Src1Phys) && hasNoInterference(DstInt, Src1Phys) && ReAllocDst) {
      LRM->unassign(DstInt);
      LRM->assign(DstInt, Src1Phys);
      Phase1Successful = true;
    } else if (isGPR4(DstPhys) && hasNoInterference(Src1Int, DstPhys) &&
               ReAllocSrc1) {
      LRM->unassign(Src1Int);
      LRM->assign(Src1Int, DstPhys);
      Phase1Successful = true;
    } else if (ReAllocDst && ReAllocSrc1) {
      for (auto GPR4Reg : GPR4)
        if (hasNoInterference(DstInt, GPR4Reg) &&
            hasNoInterference(Src1Int, GPR4Reg)) {
          LRM->unassign(DstInt);
          LRM->assign(DstInt, GPR4Reg);
          LRM->unassign(Src1Int);
          LRM->assign(Src1Int, GPR4Reg);
          Phase1Successful = true;
          break;
        }
    }
  }

  // If realloc of DST/SRC1 pair failed, no need to check SRC2.
  if (!Phase1Successful)
    return false;

  // If realloc was successful and SRC2 is GPR4, then it's already ADDU[4x4].
  if (isGPR4(Src2Phys))
    return true;

  // SRC2 is not GPR4, try to allocate GPR4 for it.
  auto &Src2Int = LIS->getInterval(Src2);
  for (auto GPR4Reg : GPR4)
    if (hasNoInterference(Src2Int, GPR4Reg)) {
      LRM->unassign(Src2Int);
      LRM->assign(Src2Int, GPR4Reg);
      return true;
    }

  return false;
}

bool NanoMipsRegisterReAlloc::hintRegister(MachineBasicBlock &MBB) {
  // If one out of 3 registers is not in the RegSet, return it.
  auto GetOneMissingGPR3 = [this](Register Reg1, Register Reg2,
                                      Register Reg3) {
    bool IsReg1Valid = isGPR3(VRM->getPhys(Reg1));
    bool IsReg2Valid = isGPR3(VRM->getPhys(Reg2));
    bool IsReg3Valid = isGPR3(VRM->getPhys(Reg3));
    if (IsReg1Valid && IsReg2Valid && !IsReg3Valid)
      return Reg3;
    if (IsReg1Valid && !IsReg2Valid && IsReg3Valid)
      return Reg2;
    if (!IsReg1Valid && IsReg2Valid && IsReg3Valid)
      return Reg1;
    return Register();
  };
  bool Modified = false;
  for (auto &MI : MBB) {
    switch (MI.getOpcode()) {
    case Mips::ADDu_NM: {
      Register Dst = MI.getOperand(0).getReg();
      Register Src1 = MI.getOperand(1).getReg();
      Register Src2 = MI.getOperand(2).getReg();

      if (!Dst.isVirtual() || !Src1.isVirtual() || !Src2.isVirtual())
        continue;

      Register DstPhys = VRM->getPhys(Dst);
      Register Src1Phys = VRM->getPhys(Src1);
      Register Src2Phys = VRM->getPhys(Src2);

      // Skip ADDU[16]
      if (isGPR3(Src1Phys) && isGPR3(Src2Phys) && isGPR3(DstPhys))
        continue;
      // Skip ADDU[4x4]
      if (isGPR4(Src1Phys) && isGPR4(Src2Phys) && isGPR4(DstPhys) &&
          (DstPhys == Src1Phys || DstPhys == Src2Phys))
        continue;

      if (tryGPR4ReAlloc(Dst, Src1, Src2)) {
        Modified = true;
        continue;
      }
      if (tryGPR4ReAlloc(Dst, Src2, Src1)) {
        Modified = true;
        continue;
      }

      // Check if 2 registers are GPR3 and one is not. If this is the case,
      // try to allocate 3rd GPR3 register.
      Register RegToReplace = GetOneMissingGPR3(Dst, Src1, Src2);
      if (isValidForReplacement(RegToReplace)) {
        auto &Interval = LIS->getInterval(RegToReplace);
        for (auto GPR3Reg : GPR3)
          if (hasNoInterference(Interval, GPR3Reg)) {
            LRM->unassign(Interval);
            LRM->assign(Interval, GPR3Reg);
            Modified = true;
            break;
          }
      }
    }
    }
  }
  return Modified;
}

namespace llvm {
FunctionPass *createNanoMipsRegisterReAllocationPass() { return new NanoMipsRegisterReAlloc(); }
} // namespace llvm
