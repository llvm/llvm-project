//===-- SIGlobalLoadSAddrToVAddr.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert SADDR global loads/stores to VADDR form when the SGPR address pair
// is about to be overwritten. This eliminates s_wait_xcnt hazards that
// serialize loads sharing a single SGPR scratch address register.
//
//===----------------------------------------------------------------------===//

#include "SIGlobalLoadSAddrToVAddr.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "si-global-load-saddr-to-vaddr"

static cl::opt<bool> EnableSAddrToVAddr(
    "amdgpu-global-load-saddr-to-vaddr",
    cl::desc(
        "Convert SADDR global loads/stores to VADDR when saddr is overwritten"),
    cl::Hidden);

static cl::opt<unsigned> SAddrToVAddrWindow(
    "amdgpu-global-load-saddr-to-vaddr-window",
    cl::desc("Instruction window to scan for saddr redefinition"), cl::init(4),
    cl::Hidden);

static cl::opt<unsigned> SAddrToVAddrKillGroupSize(
    "amdgpu-global-load-saddr-to-vaddr-kill-group-size",
    cl::desc("Max converted loads per KILL group (0 = single KILL per block)"),
    cl::init(16), cl::Hidden);

static bool isEnabled(const GCNSubtarget &ST) {
  if (EnableSAddrToVAddr.getNumOccurrences())
    return EnableSAddrToVAddr;
  return ST.hasWaitXcnt();
}

namespace {

class SIGlobalLoadSAddrToVAddr {
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  unsigned MaxAddrVGPRsPerBB = 0;

  bool convertToVAddr(MachineInstr &MI, MachineBasicBlock &MBB,
                      Register &NewAddrReg);
  bool sAddrRedefinedInWindow(MachineInstr &MI, Register SAddr);

public:
  bool run(MachineFunction &MF);
};

class SIGlobalLoadSAddrToVAddrLegacy : public MachineFunctionPass {
public:
  static char ID;

  SIGlobalLoadSAddrToVAddrLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
    if (!isEnabled(ST))
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
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!isEnabled(ST))
    return PreservedAnalyses::all();
  SIGlobalLoadSAddrToVAddr Impl;
  if (!Impl.run(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

bool SIGlobalLoadSAddrToVAddr::sAddrRedefinedInWindow(MachineInstr &MI,
                                                      Register SAddr) {
  unsigned Count = 0;
  MachineBasicBlock::iterator Next = std::next(MI.getIterator());
  MachineBasicBlock::iterator End = MI.getParent()->end();
  for (; Next != End && Count < SAddrToVAddrWindow; ++Next) {
    if (Next->isMetaInstruction())
      continue;
    ++Count;
    for (const MachineOperand &MO : Next->operands()) {
      if (!MO.isReg() || !MO.isDef())
        continue;
      Register Reg = MO.getReg();
      if (Reg.isPhysical() && TRI->regsOverlap(Reg, SAddr))
        return true;
    }
  }
  return false;
}

bool SIGlobalLoadSAddrToVAddr::convertToVAddr(MachineInstr &MI,
                                              MachineBasicBlock &MBB,
                                              Register &NewAddrReg) {
  unsigned Opc = MI.getOpcode();
  int NewOpc = AMDGPU::getGlobalVaddrOp(Opc);
  if (NewOpc < 0)
    return false;

  int NewVAddrIdx =
      AMDGPU::getNamedOperandIdx(NewOpc, AMDGPU::OpName::vaddr);
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
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), NewVAddr)
        .addReg(SAddrReg);
  } else {
    MCRegister SAddrLoPhys = TRI->getSubReg(SAddrReg, AMDGPU::sub0);
    MCRegister SAddrHiPhys = TRI->getSubReg(SAddrReg, AMDGPU::sub1);

    Register VAddrHiExt =
        MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_ASHRREV_I32_e64), VAddrHiExt)
        .addImm(31)
        .addReg(OldVAddrReg);

    Register SAddrLo = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    Register SAddrHi = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), SAddrLo)
        .addReg(SAddrLoPhys);
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), SAddrHi)
        .addReg(SAddrHiPhys);

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

bool SIGlobalLoadSAddrToVAddr::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();

  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  unsigned Occupancy = MFI->getOccupancy();
  unsigned MaxVGPRs = ST.getMaxNumVGPRs(MF);

  // Reserve at most a quarter of the VGPR budget for converted address
  // registers. Each conversion creates one vreg_64 kept live to the
  // group's KILL, so the cost is 2 VGPRs per conversion.
  MaxAddrVGPRsPerBB = MaxVGPRs / 4;

  LLVM_DEBUG(dbgs() << "SAddrToVAddr: occupancy=" << Occupancy
                    << " maxVGPRs=" << MaxVGPRs
                    << " addrBudget=" << MaxAddrVGPRsPerBB << " VGPRs\n");

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    SmallVector<Register, 8> GroupRegs;
    MachineInstr *GroupLast = nullptr;
    unsigned TotalAddrVGPRs = 0;

    auto FlushGroup = [&]() {
      if (GroupLast && GroupRegs.size() > 1) {
        auto InsertPt = std::next(GroupLast->getIterator());
        auto KillMI = BuildMI(MBB, InsertPt, GroupLast->getDebugLoc(),
                              TII->get(TargetOpcode::KILL));
        for (Register R : GroupRegs)
          KillMI.addReg(R);
      }
      GroupRegs.clear();
      GroupLast = nullptr;
    };

    for (auto MII = MBB.begin(), MIE = MBB.end(); MII != MIE; ++MII) {
      MachineInstr &MI = *MII;
      unsigned Opc = MI.getOpcode();

      int SAddrIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::saddr);
      if (SAddrIdx < 0)
        continue;

      if (!TII->isFLATGlobal(MI))
        continue;

      MachineOperand &SAddr = MI.getOperand(SAddrIdx);
      if (!SAddr.isReg() || !SAddr.getReg().isPhysical())
        continue;

      if (!TRI->isSGPRReg(*MRI, SAddr.getReg()))
        continue;

      if (!sAddrRedefinedInWindow(MI, SAddr.getReg()))
        continue;

      // Stop converting in this BB if we would exceed the VGPR budget.
      if (TotalAddrVGPRs + 2 > MaxAddrVGPRsPerBB) {
        LLVM_DEBUG(dbgs() << "  Skipping (VGPR budget exhausted): " << MI);
        continue;
      }

      LLVM_DEBUG(dbgs() << "  SAddr redef in window: " << MI);
      Register NewAddr;
      if (convertToVAddr(MI, MBB, NewAddr)) {
        GroupRegs.push_back(NewAddr);
        GroupLast = &MI;
        TotalAddrVGPRs += 2;
        Changed = true;

        if (SAddrToVAddrKillGroupSize > 0 &&
            GroupRegs.size() >= SAddrToVAddrKillGroupSize)
          FlushGroup();
      }
    }

    FlushGroup();
  }

  return Changed;
}
