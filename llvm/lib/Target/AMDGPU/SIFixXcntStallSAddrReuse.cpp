//===- SIFixXcntStallSAddrReuse.cpp - Fix xcnt stalls from SADDR reuse ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// Convert FLAT global SADDR memory ops to VADDR when the SGPR address pair
/// would be reused across multiple loads. Runs after SGPR RA (greedy) but
/// before VirtRegRewriter, so saddr operands are still virtual registers
/// with physical assignments available in VirtRegMap.
///
/// This avoids s_wait_xcnt stalls from XNACK replay hazards on reused.
//===----------------------------------------------------------------------===//

#include "SIFixXcntStallSAddrReuse.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "si-fix-xcnt-stall-saddr-reuse"

static cl::opt<unsigned> SAddrReuseWindow(
    "amdgpu-saddr-reuse-window", cl::init(30), cl::Hidden,
    cl::desc("Number of instructions to search for SAddr redefinition"));

STATISTIC(NumConverted, "Number of SADDR loads converted to VADDR");

namespace {

class SIFixXcntStallSAddrReuse {
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  LiveIntervals *LIS = nullptr;
  VirtRegMap *VRM = nullptr;

  MCRegister getPhysForSAddr(Register Reg) const;
  bool isSAddrRedefined(MachineInstr &MI, Register SAddr);
  MachineInstr *convertToVAddr(MachineInstr &MI, MachineBasicBlock &MBB,
                               Register &NewAddrReg);
  bool processMBB(MachineBasicBlock &MBB);

public:
  SIFixXcntStallSAddrReuse(LiveIntervals *LIS = nullptr,
                           VirtRegMap *VRM = nullptr)
      : LIS(LIS), VRM(VRM) {}
  bool run(MachineFunction &MF);
};

class SIFixXcntStallSAddrReuseLegacy : public MachineFunctionPass {
public:
  static char ID;
  SIFixXcntStallSAddrReuseLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    if (!MF.getSubtarget<GCNSubtarget>().hasWaitXcnt())
      return false;
    auto *LISWrapper = getAnalysisIfAvailable<LiveIntervalsWrapperPass>();
    LiveIntervals *LIS = LISWrapper ? &LISWrapper->getLIS() : nullptr;
    auto *VRMWrapper = getAnalysisIfAvailable<VirtRegMapWrapperLegacy>();
    VirtRegMap *VRM = VRMWrapper ? &VRMWrapper->getVRM() : nullptr;
    return SIFixXcntStallSAddrReuse(LIS, VRM).run(MF);
  }

  StringRef getPassName() const override {
    return "SI Fix Xcnt Stall SAddr Reuse";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addUsedIfAvailable<VirtRegMapWrapperLegacy>();
    AU.addUsedIfAvailable<LiveIntervalsWrapperPass>();
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
  auto *LIS = MFAM.getCachedResult<LiveIntervalsAnalysis>(MF);
  auto *VRM = MFAM.getCachedResult<VirtRegMapAnalysis>(MF);
  if (!SIFixXcntStallSAddrReuse(LIS, VRM).run(MF))
    return PreservedAnalyses::all();
  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  if (LIS)
    PA.preserve<LiveIntervalsAnalysis>();
  return PA;
}

static bool isEligible(const MachineInstr &MI, const SIInstrInfo &TII,
                       const SIRegisterInfo &TRI,
                       const MachineRegisterInfo &MRI, int &SAddrIdx) {
  if (!TII.isFLATGlobal(MI))
    return false;
  SAddrIdx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::saddr);
  if (SAddrIdx < 0)
    return false;
  Register SAddr = MI.getOperand(SAddrIdx).getReg();
  if (!SAddr.isVirtual() || !TRI.isSGPRReg(MRI, SAddr))
    return false;
  unsigned Size = TRI.getRegSizeInBits(*MRI.getRegClass(SAddr));
  return Size == 64;
}

MCRegister SIFixXcntStallSAddrReuse::getPhysForSAddr(Register Reg) const {
  if (!VRM || !Reg.isVirtual())
    return MCRegister();
  return VRM->getPhys(Reg);
}

bool SIFixXcntStallSAddrReuse::isSAddrRedefined(MachineInstr &MI,
                                                Register SAddr) {
  MCRegister Phys = getPhysForSAddr(SAddr);
  if (!Phys)
    return false;

  unsigned Count = 0;
  for (const MachineInstr &I : instructionsWithoutDebug(
           std::next(MachineBasicBlock::iterator(MI)), MI.getParent()->end())) {
    if (Count++ >= SAddrReuseWindow)
      return false;
    if (I.modifiesRegister(Phys, TRI))
      return true;
    for (const MachineOperand &MO : I.all_defs()) {
      if (!MO.getReg().isVirtual())
        continue;
      MCRegister DefPhys = getPhysForSAddr(MO.getReg());
      if (DefPhys && TRI->regsOverlap(Phys, DefPhys))
        return true;
    }
  }
  return false;
}

MachineInstr *SIFixXcntStallSAddrReuse::convertToVAddr(MachineInstr &MI,
                                                       MachineBasicBlock &MBB,
                                                       Register &NewAddrReg) {
  unsigned Opc = MI.getOpcode();
  int NewOpc = AMDGPU::getGlobalVaddrOp(Opc);
  if (NewOpc < 0)
    return nullptr;
  if (AMDGPU::getNamedOperandIdx(NewOpc, AMDGPU::OpName::vaddr) < 0)
    return nullptr;

  // VADDR form does not support scale_offset.
  int CPolIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::cpol);
  if (CPolIdx >= 0 && (MI.getOperand(CPolIdx).getImm() & AMDGPU::CPol::SCAL))
    return nullptr;

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
    if (auto *Def = MRI->getUniqueVRegDef(OldVAddrReg))
      VAddrIsZero = Def->isImplicitDef() ||
                    (Def->isMoveImmediate() && Def->getOperand(1).isImm() &&
                     Def->getOperand(1).getImm() == 0);
  }

  Register NewVAddr = MRI->createVirtualRegister(TRI->getVGPR64Class());

  if (VAddrIsZero) {
    auto Copy =
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), NewVAddr).addReg(SAddrReg);
    if (LIS)
      LIS->InsertMachineInstrInMaps(*Copy);
  } else {
    // new_vaddr = saddr + sext(old_vaddr_32)
    Register TmpVAddr = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    auto CopyVAddr =
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), TmpVAddr).add(VAddr);
    if (LIS) {
      LIS->InsertMachineInstrInMaps(*CopyVAddr);
      LIS->createAndComputeVirtRegInterval(TmpVAddr);
    }

    Register HiExt = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    auto Ashr = BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_ASHRREV_I32_e64), HiExt)
                    .addImm(31)
                    .addReg(TmpVAddr);
    if (LIS)
      LIS->InsertMachineInstrInMaps(*Ashr);

    Register LoV = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    Register HiV = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    // Copy full 64-bit SGPR to VGPR64 first to avoid subreg issues
    // with VirtRegRewriter (sub-ranges might not exist).
    Register SAddrV = MRI->createVirtualRegister(TRI->getVGPR64Class());
    auto CopyFull =
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), SAddrV).addReg(SAddrReg);
    if (LIS)
      LIS->InsertMachineInstrInMaps(*CopyFull);
    MachineInstrBuilder CopyLo =
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), LoV)
            .addReg(SAddrV, {}, AMDGPU::sub0);
    MachineInstrBuilder CopyHi =
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), HiV)
            .addReg(SAddrV, {}, AMDGPU::sub1);
    if (LIS) {
      LIS->InsertMachineInstrInMaps(*CopyLo);
      LIS->InsertMachineInstrInMaps(*CopyHi);
    }

    MCRegister VCC = TRI->getVCC();
    auto AddLo =
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_ADD_CO_U32_e64), NewVAddr)
            .addDef(VCC)
            .addReg(LoV)
            .addReg(TmpVAddr)
            .addImm(0);
    AddLo->getOperand(0).setSubReg(AMDGPU::sub0);
    AddLo->getOperand(0).setIsUndef(true);
    if (LIS)
      LIS->InsertMachineInstrInMaps(*AddLo);

    auto AddHi =
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_ADDC_U32_e64), NewVAddr)
            .addDef(VCC, RegState::Dead)
            .addReg(HiV)
            .addReg(HiExt)
            .addReg(VCC, RegState::Kill)
            .addImm(0);
    AddHi->getOperand(0).setSubReg(AMDGPU::sub1);
    if (LIS)
      LIS->InsertMachineInstrInMaps(*AddHi);

    if (LIS) {
      LIS->createAndComputeVirtRegInterval(SAddrV);
      LIS->createAndComputeVirtRegInterval(HiExt);
      LIS->createAndComputeVirtRegInterval(LoV);
      LIS->createAndComputeVirtRegInterval(HiV);
    }
  }

  if (LIS)
    LIS->createAndComputeVirtRegInterval(NewVAddr);

  MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, TII->get(NewOpc));
  for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
    if (i == (unsigned)OldSAddrIdx)
      continue;
    if (i == (unsigned)OldVAddrIdx)
      MIB.addReg(NewVAddr);
    else
      MIB.add(MI.getOperand(i));
  }
  MIB.cloneMemRefs(MI);

  if (LIS)
    LIS->ReplaceMachineInstrInMaps(MI, *MIB);

  LLVM_DEBUG(dbgs() << "  Converted: " << *MIB);
  MI.eraseFromParent();
  NewAddrReg = NewVAddr;
  ++NumConverted;
  return MIB.getInstr();
}

bool SIFixXcntStallSAddrReuse::processMBB(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
    int SAddrIdx;
    if (!isEligible(MI, *TII, *TRI, *MRI, SAddrIdx))
      continue;
    if (!isSAddrRedefined(MI, MI.getOperand(SAddrIdx).getReg()))
      continue;

    Register NewAddr;
    if (convertToVAddr(MI, MBB, NewAddr)) {
      if (VRM)
        VRM->grow();
      Changed = true;
    }
  }

  return Changed;
}

bool SIFixXcntStallSAddrReuse::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();

  if (VRM)
    VRM->grow();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= processMBB(MBB);
  return Changed;
}
