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
/// This avoids s_wait_xcnt stalls from XNACK replay hazards on reused
/// physical SGPR pairs.
//===----------------------------------------------------------------------===//

#include "SIFixXcntStallSAddrReuse.h"
#include "AMDGPU.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
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
    "amdgpu-saddr-reuse-window", cl::init(64), cl::Hidden,
    cl::desc("Instruction window for SAddr redefinition scan"));

static cl::opt<unsigned> SAddrKillDistance(
    "amdgpu-saddr-kill-distance",
    cl::desc("Instructions after each converted load to place KILL"),
    cl::init(16), cl::Hidden);

STATISTIC(NumConverted, "Number of SADDR loads converted to VADDR");
STATISTIC(NumSkippedPressure,
          "Number of SADDR loads skipped due to VGPR pressure");

namespace {

class SIFixXcntStallSAddrReuse {
  const GCNSubtarget *ST = nullptr;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  LiveIntervals *LIS = nullptr;
  VirtRegMap *VRM = nullptr;
  unsigned MaxVGPRs = 0;

  bool isSAddrRedefined(const MachineInstr &MI, Register SAddr);
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
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<LiveIntervalsAnalysis>();
  PA.preserve<VirtRegMapAnalysis>();
  return PA;
}

static bool isZeroVAddr(Register Reg, const MachineRegisterInfo &MRI) {
  if (!Reg.isVirtual())
    return false;
  const MachineInstr *Def = MRI.getUniqueVRegDef(Reg);
  return Def && Def->isMoveImmediate() && Def->getOperand(1).isImm() &&
         Def->getOperand(1).getImm() == 0;
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
  if (!SAddr.isVirtual())
    return false;
  return AMDGPU::SReg_64RegClass.hasSubClassEq(MRI.getRegClass(SAddr));
}

static bool canConvertToVAddr(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  int NewOpc = AMDGPU::getGlobalVaddrOp(Opc);
  if (NewOpc < 0)
    return false;
  if (AMDGPU::getNamedOperandIdx(NewOpc, AMDGPU::OpName::vaddr) < 0)
    return false;
  int CPolIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::cpol);
  if (CPolIdx >= 0 && (MI.getOperand(CPolIdx).getImm() & AMDGPU::CPol::SCAL))
    return false;
  return true;
}

bool SIFixXcntStallSAddrReuse::isSAddrRedefined(const MachineInstr &MI,
                                                Register SAddr) {
  MCRegister Phys = VRM->getPhys(SAddr);
  if (!Phys)
    return false;

  const MachineBasicBlock *MBB = MI.getParent();
  unsigned Count = 0;
  for (const MachineInstr &I : instructionsWithoutDebug(
           std::next(MachineBasicBlock::const_iterator(MI)), MBB->end())) {
    if (Count++ >= SAddrReuseWindow)
      return false;
    for (const MachineOperand &MO : I.all_defs()) {
      Register DefReg = MO.getReg();
      if (!DefReg.isVirtual())
        continue;
      if (TRI->regsOverlap(Phys, VRM->getPhys(DefReg)))
        return true;
    }
  }
  return false;
}

MachineInstr *SIFixXcntStallSAddrReuse::convertToVAddr(MachineInstr &MI,
                                                       MachineBasicBlock &MBB,
                                                       Register &NewAddrReg) {
  assert(canConvertToVAddr(MI));
  unsigned Opc = MI.getOpcode();
  int NewOpc = AMDGPU::getGlobalVaddrOp(Opc);
  int NewVAddrIdx = AMDGPU::getNamedOperandIdx(NewOpc, AMDGPU::OpName::vaddr);

  int OldSAddrIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::saddr);
  int OldVAddrIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vaddr);
  assert(OldSAddrIdx >= 0 && OldVAddrIdx >= 0);

  Register SAddrReg = MI.getOperand(OldSAddrIdx).getReg();
  Register OldVAddrReg = MI.getOperand(OldVAddrIdx).getReg();
  DebugLoc DL = MI.getDebugLoc();

  bool VAddrIsZero = isZeroVAddr(OldVAddrReg, *MRI);

  const TargetRegisterClass *VAddrRC =
      TII->getRegClass(TII->get(NewOpc), NewVAddrIdx);
  Register NewVAddr = MRI->createVirtualRegister(VAddrRC);

  if (VAddrIsZero) {
    auto Copy =
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), NewVAddr).addReg(SAddrReg);
    LIS->InsertMachineInstrInMaps(*Copy);
  } else {
    // new_vaddr = saddr + sext(old_vaddr_32)
    Register TmpVAddr = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    auto CopyVAddr = BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), TmpVAddr)
                         .add(MI.getOperand(OldVAddrIdx));

    Register HiExt = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    auto Ashr = BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_ASHRREV_I32_e64), HiExt)
                    .addImm(31)
                    .addReg(TmpVAddr);

    LIS->InsertMachineInstrInMaps(*CopyVAddr);
    LIS->InsertMachineInstrInMaps(*Ashr);

    Register SextVAddr = MRI->createVirtualRegister(VAddrRC);
    auto CopyLo = BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY))
                      .addDef(SextVAddr, RegState::Undef, AMDGPU::sub0)
                      .addReg(TmpVAddr);
    auto CopyHi = BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY))
                      .addDef(SextVAddr, {}, AMDGPU::sub1)
                      .addReg(HiExt);
    LIS->InsertMachineInstrInMaps(*CopyLo);
    LIS->InsertMachineInstrInMaps(*CopyHi);

    auto Add64 = BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_ADD_U64_e64), NewVAddr)
                     .addReg(SAddrReg)
                     .addReg(SextVAddr)
                     .addImm(0);
    LIS->InsertMachineInstrInMaps(*Add64);
    LIS->createAndComputeVirtRegInterval(SextVAddr);
    LIS->createAndComputeVirtRegInterval(TmpVAddr);
    LIS->createAndComputeVirtRegInterval(HiExt);
  }

  const MCInstrDesc &NewDesc = TII->get(NewOpc);
  MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, NewDesc);

  auto addOpIfPresent = [&](AMDGPU::OpName OpName) {
    int Idx = AMDGPU::getNamedOperandIdx(Opc, OpName);
    if (Idx != -1)
      MIB.add(MI.getOperand(Idx));
  };

  addOpIfPresent(AMDGPU::OpName::vdst);
  addOpIfPresent(AMDGPU::OpName::vdst_in);

  MIB.addReg(NewVAddr);

  addOpIfPresent(AMDGPU::OpName::vdata);

  addOpIfPresent(AMDGPU::OpName::offset);
  addOpIfPresent(AMDGPU::OpName::cpol);

  MIB.cloneMemRefs(MI);

  int NewVDst = AMDGPU::getNamedOperandIdx(NewOpc, AMDGPU::OpName::vdst);
  int NewVDstIn = AMDGPU::getNamedOperandIdx(NewOpc, AMDGPU::OpName::vdst_in);
  if (NewVDst != -1 && NewVDstIn != -1)
    MIB->tieOperands(NewVDst, NewVDstIn);

  LIS->ReplaceMachineInstrInMaps(MI, *MIB);
  LIS->createAndComputeVirtRegInterval(NewVAddr);

  LLVM_DEBUG(dbgs() << "  Converted: " << *MIB);
  MI.eraseFromParent();

  if (LIS->hasInterval(SAddrReg)) {
    LiveInterval &SAddrLI = LIS->getInterval(SAddrReg);
    if (LIS->shrinkToUses(&SAddrLI)) {
      SmallVector<LiveInterval *, 4> SplitLIs;
      LIS->splitSeparateComponents(SAddrLI, SplitLIs);
    }
  }
  if (OldVAddrReg.isVirtual() && LIS->hasInterval(OldVAddrReg)) {
    LiveInterval &VAddrLI = LIS->getInterval(OldVAddrReg);
    if (LIS->shrinkToUses(&VAddrLI)) {
      SmallVector<LiveInterval *, 4> SplitLIs;
      LIS->splitSeparateComponents(VAddrLI, SplitLIs);
    }
  }

  VRM->grow();

  NewAddrReg = NewVAddr;
  ++NumConverted;
  return MIB.getInstr();
}

bool SIFixXcntStallSAddrReuse::processMBB(MachineBasicBlock &MBB) {
  GCNDownwardRPTracker RPT(*LIS);
  SmallVector<MachineInstr *, 8> ToConvert;
  SmallVector<unsigned, 16> ActiveWindow;
  unsigned WindowFront = 0;
  unsigned InstIdx = 0;

  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    ++InstIdx;

    if (!RPT.getNext().isValid())
      RPT.reset(MI);
    else {
      RPT.advance(MachineBasicBlock::const_iterator(MI));
      RPT.advanceBeforeNext();
    }

    int SAddrIdx;
    if (!isEligible(MI, *TII, *TRI, *MRI, SAddrIdx))
      continue;
    if (!canConvertToVAddr(MI))
      continue;
    if (!isSAddrRedefined(MI, MI.getOperand(SAddrIdx).getReg()))
      continue;

    while (WindowFront < ActiveWindow.size() &&
           InstIdx > ActiveWindow[WindowFront] + SAddrKillDistance)
      ++WindowFront;

    unsigned ActiveCount = ActiveWindow.size() - WindowFront;
    unsigned BasePressure = RPT.getPressure().getVGPRNum(ST->hasGFX90AInsts());

    // *2: each active conversion holds a vreg_64 address.
    // +4: peak transient VGPRs from address math (sext + 64-bit add).
    unsigned AddedPressure = ActiveCount * 2 + 4;

    if (BasePressure + AddedPressure > MaxVGPRs) {
      LLVM_DEBUG(dbgs() << "  Skipping (pressure): Base=" << BasePressure
                        << " Added=" << AddedPressure
                        << " MaxVGPRs=" << MaxVGPRs << " " << MI);
      ++NumSkippedPressure;
      continue;
    }

    ToConvert.push_back(&MI);
    ActiveWindow.push_back(InstIdx);
  }

  if (ToConvert.empty())
    return false;

  SmallVector<std::pair<Register, MachineInstr *>, 8> Converted;
  for (MachineInstr *MI : ToConvert) {
    Register NewAddr;
    MachineInstr *NewMI = convertToVAddr(*MI, MBB, NewAddr);
    if (!NewMI)
      continue;
    Converted.push_back({NewAddr, NewMI});
  }

  for (auto &[Reg, NewMI] : Converted) {
    auto It = std::next(MachineBasicBlock::iterator(NewMI));
    unsigned Count = 0;
    while (It != MBB.end() && !It->isTerminator() &&
           Count < SAddrKillDistance) {
      if (!It->isDebugInstr())
        ++Count;
      ++It;
    }

    auto KillMI =
        BuildMI(MBB, It, NewMI->getDebugLoc(), TII->get(TargetOpcode::KILL))
            .addReg(Reg, RegState::Kill);

    LIS->InsertMachineInstrInMaps(*KillMI);
    SlotIndex KillIdx = LIS->getInstructionIndex(*KillMI).getRegSlot();
    if (LIS->hasInterval(Reg))
      LIS->extendToIndices(LIS->getInterval(Reg), {KillIdx});
  }

  return !Converted.empty();
}

bool SIFixXcntStallSAddrReuse::run(MachineFunction &MF) {
  if (!VRM || !LIS)
    return false;
  ST = &MF.getSubtarget<GCNSubtarget>();
  TII = ST->getInstrInfo();
  TRI = ST->getRegisterInfo();
  MRI = &MF.getRegInfo();

  auto *MFI = MF.getInfo<SIMachineFunctionInfo>();
  unsigned TargetOcc = MFI->getOccupancy();
  unsigned DynVGPRBlockSize = MFI->getDynamicVGPRBlockSize();
  MaxVGPRs = ST->getMaxNumVGPRs(TargetOcc, DynVGPRBlockSize);

  LLVM_DEBUG(dbgs() << "SIFixXcntStallSAddrReuse: TargetOcc=" << TargetOcc
                    << " MaxVGPRs=" << MaxVGPRs << "\n");

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= processMBB(MBB);
  return Changed;
}
