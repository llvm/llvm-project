//===- SIFixXcntStallSAddrReuse.cpp - Fix xcnt stalls from SADDR reuse ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Convert FLAT global SADDR memory ops to VADDR when the physical SGPR
/// address pair is redefined later in the block. This avoids s_wait_xcnt
/// stalls from XNACK replay of loads that share a reused SGPR pair.
///
/// Runs after SGPR RA and before VGPR RA.
//===----------------------------------------------------------------------===//

#include "SIFixXcntStallSAddrReuse.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "si-fix-xcnt-stall-saddr-reuse"

STATISTIC(NumConverted, "Number of SADDR loads converted to VADDR");

static cl::opt<unsigned> SAddrRedefWindow(
    "amdgpu-saddr-redef-window",
    cl::desc("Maximum instructions to scan for saddr redefines "
             "(0 = scan to end of block)"),
    cl::init(0), cl::Hidden);

static cl::opt<unsigned> AddrLivenessGroupSize(
    "amdgpu-saddr-vaddr-liveness-group",
    cl::desc("Maximum converted VADDRs kept simultaneously live, kill is "
             "issued for each group"),
    cl::init(16), cl::Hidden);

namespace {

class SIFixXcntStallSAddrReuse {
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  bool sAddrRedefinedAfter(MachineInstr &MI, Register SAddr);
  bool convertToVAddr(MachineInstr &MI, MachineBasicBlock &MBB,
                      Register &NewAddrReg);
  bool processMBB(MachineBasicBlock &MBB);

public:
  bool run(MachineFunction &MF);
};

class SIFixXcntStallSAddrReuseLegacy : public MachineFunctionPass {
public:
  static char ID;
  SIFixXcntStallSAddrReuseLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!MF.getSubtarget<GCNSubtarget>().hasWaitXcnt())
      return false;
    return SIFixXcntStallSAddrReuse().run(MF);
  }

  StringRef getPassName() const override {
    return "SI Fix Xcnt Stall SAddr Reuse";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char SIFixXcntStallSAddrReuseLegacy::ID = 0;

INITIALIZE_PASS(SIFixXcntStallSAddrReuseLegacy, DEBUG_TYPE,
                "SI Fix Xcnt Stall SAddr Reuse", false, false)

char &llvm::SIFixXcntStallSAddrReuseLegacyID =
    SIFixXcntStallSAddrReuseLegacy::ID;

PreservedAnalyses
SIFixXcntStallSAddrReusePass::run(MachineFunction &MF,
                                  MachineFunctionAnalysisManager &MFAM) {
  if (!MF.getSubtarget<GCNSubtarget>().hasWaitXcnt())
    return PreservedAnalyses::all();
  if (!SIFixXcntStallSAddrReuse().run(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

/// Return true if \p MI is a FLAT global op with a physical SGPR saddr.
static bool isEligible(const MachineInstr &MI, const SIInstrInfo &TII,
                       const SIRegisterInfo &TRI,
                       const MachineRegisterInfo &MRI, int &SAddrIdx) {
  SAddrIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::saddr);
  if (SAddrIdx < 0 || !TII.isFLATGlobal(MI))
    return false;
  const MachineOperand &SAddr = MI.getOperand(SAddrIdx);
  return SAddr.isReg() && SAddr.getReg().isPhysical() &&
         TRI.isSGPRReg(MRI, SAddr.getReg());
}

bool SIFixXcntStallSAddrReuse::sAddrRedefinedAfter(MachineInstr &MI,
                                                   Register SAddr) {
  unsigned NumScanned = 0;
  MachineBasicBlock::iterator I = std::next(MI.getIterator());
  MachineBasicBlock::iterator E = MI.getParent()->end();
  for (; I != E; ++I) {
    for (const MachineOperand &MO : I->operands())
      if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical() &&
          TRI->regsOverlap(MO.getReg(), SAddr))
        return true;

    if (!I->isMetaInstruction() && SAddrRedefWindow > 0 &&
        ++NumScanned >= SAddrRedefWindow)
      return false;
  }
  return false;
}

bool SIFixXcntStallSAddrReuse::convertToVAddr(MachineInstr &MI,
                                              MachineBasicBlock &MBB,
                                              Register &NewAddrReg) {
  unsigned Opc = MI.getOpcode();
  int NewOpc = AMDGPU::getGlobalVaddrOp(Opc);
  if (NewOpc < 0)
    return false;
  if (AMDGPU::getNamedOperandIdx(NewOpc, AMDGPU::OpName::vaddr) < 0)
    return false;

  // VADDR form does not support scale_offset.
  int CPolIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::cpol);
  if (CPolIdx >= 0 && (MI.getOperand(CPolIdx).getImm() & AMDGPU::CPol::SCAL))
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
    if (auto *Def = MRI->getVRegDef(OldVAddrReg))
      VAddrIsZero = Def->isImplicitDef() ||
                    (Def->isMoveImmediate() && Def->getOperand(1).isImm() &&
                     Def->getOperand(1).getImm() == 0);
  }

  Register NewVAddr = MRI->createVirtualRegister(TRI->getVGPR64Class());

  if (VAddrIsZero) {
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), NewVAddr).addReg(SAddrReg);
  } else {
    MCRegister Lo = TRI->getSubReg(SAddrReg, AMDGPU::sub0);
    MCRegister Hi = TRI->getSubReg(SAddrReg, AMDGPU::sub1);

    Register HiExt = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_ASHRREV_I32_e64), HiExt)
        .addImm(31)
        .addReg(OldVAddrReg);

    Register LoV = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    Register HiV = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), LoV).addReg(Lo);
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), HiV).addReg(Hi);

    MCRegister VCC = TRI->getVCC();
    auto AddLo =
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_ADD_CO_U32_e64), NewVAddr)
            .addDef(VCC)
            .addReg(LoV)
            .addReg(OldVAddrReg)
            .addImm(0);
    AddLo->getOperand(0).setSubReg(AMDGPU::sub0);
    AddLo->getOperand(0).setIsUndef(true);

    auto AddHi =
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_ADDC_U32_e64), NewVAddr)
            .addDef(VCC, RegState::Dead)
            .addReg(HiV)
            .addReg(HiExt)
            .addReg(VCC, RegState::Kill)
            .addImm(0);
    AddHi->getOperand(0).setSubReg(AMDGPU::sub1);
  }

  MI.setDesc(TII->get(NewOpc));
  VAddr.setReg(NewVAddr);
  MI.removeOperand(OldSAddrIdx);

  LLVM_DEBUG(dbgs() << "  Converted: " << MI);
  NewAddrReg = NewVAddr;
  ++NumConverted;
  return true;
}

bool SIFixXcntStallSAddrReuse::processMBB(MachineBasicBlock &MBB) {
  SmallVector<Register, 8> GroupRegs;
  MachineInstr *GroupLast = nullptr;
  bool Changed = false;

  auto FlushGroup = [&]() {
    if (GroupLast && GroupRegs.size() > 1) {
      auto KillMI =
          BuildMI(MBB, std::next(GroupLast->getIterator()),
                  GroupLast->getDebugLoc(), TII->get(TargetOpcode::KILL));
      for (Register R : GroupRegs)
        KillMI.addReg(R, RegState::Kill);
    }
    GroupRegs.clear();
    GroupLast = nullptr;
  };

  for (MachineInstr &MI : MBB) {
    int SAddrIdx;
    if (!isEligible(MI, *TII, *TRI, *MRI, SAddrIdx))
      continue;
    if (!sAddrRedefinedAfter(MI, MI.getOperand(SAddrIdx).getReg()))
      continue;

    Register NewAddr;
    if (convertToVAddr(MI, MBB, NewAddr)) {
      GroupRegs.push_back(NewAddr);
      GroupLast = &MI;
      Changed = true;

      if (AddrLivenessGroupSize > 0 &&
          GroupRegs.size() >= AddrLivenessGroupSize)
        FlushGroup();
    }
  }

  FlushGroup();
  return Changed;
}

bool SIFixXcntStallSAddrReuse::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= processMBB(MBB);
  return Changed;
}
