//===- SIGlobalLoadSAddrToVAddr.cpp - SADDR global loads to VADDR ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Convert global SADDR memory ops to VADDR when the SGPR address is
/// overwritten soon after, avoiding s_wait_xcnt hazards from replay of loads
/// that share a scratch SGPR pair. Runs after SGPR RA and SILowerSGPRSpills.
//
//===----------------------------------------------------------------------===//

#include "SIGlobalLoadSAddrToVAddr.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "si-global-load-saddr-to-vaddr"

static cl::opt<unsigned> SAddrToVAddrWindow(
    "amdgpu-global-load-saddr-to-vaddr-window",
    cl::desc("Instruction window to scan for saddr redefinition"), cl::init(4),
    cl::Hidden);

static cl::opt<unsigned> SAddrToVAddrKillGroupSize(
    "amdgpu-global-load-saddr-to-vaddr-kill-group-size",
    cl::desc("Max converted loads per KILL group"), cl::init(16), cl::Hidden);

namespace {

class SIGlobalLoadSAddrToVAddr {
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  bool convertToVAddr(MachineInstr &MI, MachineBasicBlock &MBB,
                      Register &NewAddrReg);
  bool isSAddrRedefinedInWindow(MachineInstr &MI, Register SAddr);
  bool processMBB(MachineBasicBlock &MBB);

public:
  bool run(MachineFunction &MF);
};

class SIGlobalLoadSAddrToVAddrLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIGlobalLoadSAddrToVAddrLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!MF.getSubtarget<GCNSubtarget>().hasWaitXcnt())
      return false;
    SIGlobalLoadSAddrToVAddr Impl;
    return Impl.run(MF);
  }

  StringRef getPassName() const override {
    return "SI Global Load SAddr to VAddr";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char SIGlobalLoadSAddrToVAddrLegacy::ID = 0;

INITIALIZE_PASS(SIGlobalLoadSAddrToVAddrLegacy, DEBUG_TYPE,
                "SI Global Load SAddr to VAddr", false, false)

char &llvm::SIGlobalLoadSAddrToVAddrLegacyID =
    SIGlobalLoadSAddrToVAddrLegacy::ID;

PreservedAnalyses
SIGlobalLoadSAddrToVAddrPass::run(MachineFunction &MF,
                                  MachineFunctionAnalysisManager &MFAM) {
  if (!MF.getSubtarget<GCNSubtarget>().hasWaitXcnt())
    return PreservedAnalyses::all();
  SIGlobalLoadSAddrToVAddr Impl;
  if (!Impl.run(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

bool SIGlobalLoadSAddrToVAddr::isSAddrRedefinedInWindow(MachineInstr &MI,
                                                        Register SAddr) {
  unsigned NumScanned = 0;
  auto I = std::next(MI.getIterator());
  auto E = MI.getParent()->end();
  for (; I != E; ++I) {
    if (I->isMetaInstruction())
      continue;

    ++NumScanned;

    for (const MachineOperand &MO : I->operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical() &&
          TRI->regsOverlap(MO.getReg(), SAddr))
        return true;
    }

    if (NumScanned >= SAddrToVAddrWindow)
      return false;
  }
  return false;
}

static bool isGlobalWithPhysSAddr(const MachineInstr &MI,
                                  const SIInstrInfo &TII,
                                  const SIRegisterInfo &TRI,
                                  const MachineRegisterInfo &MRI,
                                  int &SAddrIdx) {
  unsigned Opc = MI.getOpcode();
  SAddrIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::saddr);
  if (SAddrIdx < 0 || !TII.isFLATGlobal(MI))
    return false;
  const MachineOperand &SAddr = MI.getOperand(SAddrIdx);
  return SAddr.isReg() && SAddr.getReg().isPhysical() &&
         TRI.isSGPRReg(MRI, SAddr.getReg());
}

bool SIGlobalLoadSAddrToVAddr::convertToVAddr(MachineInstr &MI,
                                              MachineBasicBlock &MBB,
                                              Register &NewAddrReg) {
  unsigned Opc = MI.getOpcode();
  int NewOpc = AMDGPU::getGlobalVaddrOp(Opc);
  if (NewOpc < 0)
    return false;

  int NewVAddrIdx = AMDGPU::getNamedOperandIdx(NewOpc, AMDGPU::OpName::vaddr);
  if (NewVAddrIdx < 0)
    return false;

  int OldSAddrIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::saddr);
  int OldVAddrIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vaddr);
  assert(OldSAddrIdx >= 0 && OldVAddrIdx >= 0);

  MachineOperand &SAddr = MI.getOperand(OldSAddrIdx);
  MachineOperand &VAddr = MI.getOperand(OldVAddrIdx);
  Register SAddrReg = SAddr.getReg();
  Register OldVAddrReg = VAddr.getReg();
  const DebugLoc &DL = MI.getDebugLoc();

  bool VAddrIsZero = false;
  if (OldVAddrReg.isVirtual()) {
    MachineInstr *VAddrDef = MRI->getVRegDef(OldVAddrReg);
    if (VAddrDef &&
        (VAddrDef->isImplicitDef() ||
         (VAddrDef->isMoveImmediate() && VAddrDef->getOperand(1).isImm() &&
          VAddrDef->getOperand(1).getImm() == 0)))
      VAddrIsZero = true;
  }

  Register NewVAddr = MRI->createVirtualRegister(TRI->getVGPR64Class());

  if (VAddrIsZero) {
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), NewVAddr).addReg(SAddrReg);
  } else {
    assert(OldVAddrReg.isVirtual() &&
           TRI->getRegSizeInBits(*MRI->getRegClass(OldVAddrReg)) == 32 &&
           "Non-zero vaddr path expects a 32-bit VGPR operand");

    MCRegister SAddrLoPhys = TRI->getSubReg(SAddrReg, AMDGPU::sub0);
    MCRegister SAddrHiPhys = TRI->getSubReg(SAddrReg, AMDGPU::sub1);

    Register VAddrHiExt = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_ASHRREV_I32_e64), VAddrHiExt)
        .addImm(31)
        .addReg(OldVAddrReg);

    Register SAddrLo = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    Register SAddrHi = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), SAddrLo).addReg(SAddrLoPhys);
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), SAddrHi).addReg(SAddrHiPhys);

    MCRegister VCC = TRI->getVCC();

    auto AddLo =
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_ADD_CO_U32_e64), NewVAddr)
            .addDef(VCC)
            .addReg(SAddrLo)
            .addReg(OldVAddrReg)
            .addImm(0);
    AddLo->getOperand(0).setSubReg(AMDGPU::sub0);
    AddLo->getOperand(0).setIsUndef(true);

    auto AddHi =
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_ADDC_U32_e64), NewVAddr)
            .addDef(VCC, RegState::Dead)
            .addReg(SAddrHi)
            .addReg(VAddrHiExt)
            .addReg(VCC, RegState::Kill)
            .addImm(0);
    AddHi->getOperand(0).setSubReg(AMDGPU::sub1);
  }

  MI.setDesc(TII->get(NewOpc));
  VAddr.setReg(NewVAddr);
  MI.removeOperand(OldSAddrIdx);

  LLVM_DEBUG(dbgs() << "  Converted to VADDR: " << MI);
  NewAddrReg = NewVAddr;
  return true;
}

bool SIGlobalLoadSAddrToVAddr::processMBB(MachineBasicBlock &MBB) {
  SmallVector<Register, 8> GroupRegs;
  MachineInstr *GroupLast = nullptr;
  bool Changed = false;

  // KILL extends address VGPR liveness to prevent RA reuse within a group.
  auto FlushGroup = [&]() {
    if (GroupLast && GroupRegs.size() > 1) {
      auto InsertPt = std::next(GroupLast->getIterator());
      auto KillMI = BuildMI(MBB, InsertPt, GroupLast->getDebugLoc(),
                            TII->get(TargetOpcode::KILL));
      for (Register R : GroupRegs)
        KillMI.addReg(R, RegState::Kill);
    }
    GroupRegs.clear();
    GroupLast = nullptr;
  };

  for (MachineInstr &MI : MBB) {
    int SAddrIdx;
    if (!isGlobalWithPhysSAddr(MI, *TII, *TRI, *MRI, SAddrIdx))
      continue;
    if (!isSAddrRedefinedInWindow(MI, MI.getOperand(SAddrIdx).getReg()))
      continue;

    LLVM_DEBUG(dbgs() << "  SAddr redef in window: " << MI);
    Register NewAddr;
    if (convertToVAddr(MI, MBB, NewAddr)) {
      GroupRegs.push_back(NewAddr);
      GroupLast = &MI;
      Changed = true;

      if (SAddrToVAddrKillGroupSize > 0 &&
          GroupRegs.size() >= SAddrToVAddrKillGroupSize)
        FlushGroup();
    }
  }

  FlushGroup();
  return Changed;
}

bool SIGlobalLoadSAddrToVAddr::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= processMBB(MBB);

  return Changed;
}
