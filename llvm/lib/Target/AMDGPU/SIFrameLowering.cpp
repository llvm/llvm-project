//===----------------------- SIFrameLowering.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==-----------------------------------------------------------------------===//

#include "SIFrameLowering.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "frame-info"

static cl::opt<bool> EnableSpillVGPRToAGPR(
  "amdgpu-spill-vgpr-to-agpr",
  cl::desc("Enable spilling VGPRs to AGPRs"),
  cl::ReallyHidden,
  cl::init(true));

static constexpr unsigned SGPRBitSize = 32;
static constexpr unsigned SGPRByteSize = SGPRBitSize / 8;
static constexpr unsigned VGPRLaneBitSize = 32;
// FIXME: should be replaced by a constant defined elsewhere
static constexpr unsigned DW_ASPACE_AMDGPU_private_wave = 6;

// Find a scratch register matching \p RC which is unused and available
// throughout the function. On failure, returns a null register.
static MCRegister findUnusedRegister(MachineRegisterInfo &MRI,
                                     LivePhysRegs &LiveRegs,
                                     const TargetRegisterClass &RC) {
  for (MCRegister Reg : RC) {
    if (!MRI.isPhysRegUsed(Reg) && LiveRegs.available(MRI, Reg))
      return Reg;
  }
  return MCRegister();
}

static void encodeDwarfRegisterLocation(int DwarfReg, raw_ostream &OS) {
  assert(DwarfReg >= 0);
  if (DwarfReg < 32) {
    OS << uint8_t(dwarf::DW_OP_reg0 + DwarfReg);
  } else {
    OS << uint8_t(dwarf::DW_OP_regx);
    encodeULEB128(DwarfReg, OS);
  }
}

static MCCFIInstruction
createScaledCFAInPrivateWave(const GCNSubtarget &ST,
                             MCRegister DwarfStackPtrReg) {
  assert(ST.enableFlatScratch());

  // When flat scratch is used, the cfa is expressed in terms of private_lane
  // (address space 5), but the debugger only accepts addresses in terms of
  // private_wave (6). Override the cfa value using the expression
  // (wave_size*cfa_reg), which is equivalent to (cfa_reg << wave_size_log2)
  const unsigned WavefrontSizeLog2 = ST.getWavefrontSizeLog2();
  assert(WavefrontSizeLog2 < 32);

  SmallString<20> Block;
  raw_svector_ostream OSBlock(Block);
  encodeDwarfRegisterLocation(DwarfStackPtrReg, OSBlock);
  OSBlock << uint8_t(dwarf::DW_OP_deref_size) << uint8_t(4)
          << uint8_t(dwarf::DW_OP_lit0 + WavefrontSizeLog2)
          << uint8_t(dwarf::DW_OP_shl)
          << uint8_t(dwarf::DW_OP_lit0 + DW_ASPACE_AMDGPU_private_wave)
          << uint8_t(dwarf::DW_OP_LLVM_form_aspace_address);

  SmallString<20> CFIInst;
  raw_svector_ostream OSCFIInst(CFIInst);
  OSCFIInst << uint8_t(dwarf::DW_CFA_def_cfa_expression);
  encodeULEB128(Block.size(), OSCFIInst);
  OSCFIInst << Block;

  return MCCFIInstruction::createEscape(nullptr, OSCFIInst.str());
}

void SIFrameLowering::emitDefCFA(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 DebugLoc const &DL, Register StackPtrReg,
                                 bool AspaceAlreadyDefined,
                                 MachineInstr::MIFlag Flags) const {
  MachineFunction &MF = *MBB.getParent();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const MCRegisterInfo *MCRI = MF.getMMI().getContext().getRegisterInfo();

  MCRegister DwarfStackPtrReg = MCRI->getDwarfRegNum(StackPtrReg, false);
  MCCFIInstruction CFIInst =
      ST.enableFlatScratch()
          ? createScaledCFAInPrivateWave(ST, DwarfStackPtrReg)
          : (AspaceAlreadyDefined ? MCCFIInstruction::createLLVMDefAspaceCfa(
                                        nullptr, DwarfStackPtrReg, 0,
                                        DW_ASPACE_AMDGPU_private_wave)
                                  : MCCFIInstruction::createDefCfaRegister(
                                        nullptr, DwarfStackPtrReg));
  buildCFI(MBB, MBBI, DL, CFIInst, Flags);
}

// Find a scratch register that we can use in the prologue. We avoid using
// callee-save registers since they may appear to be free when this is called
// from canUseAsPrologue (during shrink wrapping), but then no longer be free
// when this is called from emitPrologue.
static MCRegister findScratchNonCalleeSaveRegister(MachineRegisterInfo &MRI,
                                                   LivePhysRegs &LiveRegs,
                                                   const TargetRegisterClass &RC,
                                                   bool Unused = false) {
  // Mark callee saved registers as used so we will not choose them.
  const MCPhysReg *CSRegs = MRI.getCalleeSavedRegs();
  for (unsigned i = 0; CSRegs[i]; ++i)
    LiveRegs.addReg(CSRegs[i]);

  // We are looking for a register that can be used throughout the entire
  // function, so any use is unacceptable.
  if (Unused)
    return findUnusedRegister(MRI, LiveRegs, RC);

  for (MCRegister Reg : RC) {
    if (LiveRegs.available(MRI, Reg))
      return Reg;
  }

  return MCRegister();
}

static void getVGPRSpillLaneOrTempRegister(
    MachineFunction &MF, LivePhysRegs &LiveRegs, Register SGPR,
    const TargetRegisterClass &RC = AMDGPU::SReg_32_XM0_XEXECRegClass,
    bool IncludeScratchCopy = true) {
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  MachineFrameInfo &FrameInfo = MF.getFrameInfo();

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  unsigned Size = TRI->getSpillSize(RC);
  Align Alignment = TRI->getSpillAlign(RC);

  // We need to save and restore the given SGPR.

  Register ScratchSGPR;
  // 1: Try to save the given register into an unused scratch SGPR. The LiveRegs
  // should have all the callee saved registers marked as used. For certain
  // cases we skip copy to scratch SGPR.
  if (IncludeScratchCopy)
    ScratchSGPR = findUnusedRegister(MF.getRegInfo(), LiveRegs, RC);

  if (!ScratchSGPR) {
    int FI = FrameInfo.CreateStackObject(Size, Alignment, true, nullptr,
                                         TargetStackID::SGPRSpill);

    if (TRI->spillSGPRToVGPR() &&
        MFI->allocateSGPRSpillToVGPRLane(MF, FI, /* IsPrologEpilog */ true)) {
      // 2: There's no free lane to spill, and no free register to save the
      // SGPR, so we're forced to take another VGPR to use for the spill.
      MFI->addToPrologEpilogSGPRSpills(
          SGPR, PrologEpilogSGPRSaveRestoreInfo(
                    SGPRSaveKind::SPILL_TO_VGPR_LANE, FI));

      LLVM_DEBUG(auto Spill = MFI->getSGPRSpillToPhysicalVGPRLanes(FI).front();
                 dbgs() << printReg(SGPR, TRI) << " requires fallback spill to "
                        << printReg(Spill.VGPR, TRI) << ':' << Spill.Lane
                        << '\n';);
    } else {
      // Remove dead <FI> index
      MF.getFrameInfo().RemoveStackObject(FI);
      // 3: If all else fails, spill the register to memory.
      FI = FrameInfo.CreateSpillStackObject(Size, Alignment);
      MFI->addToPrologEpilogSGPRSpills(
          SGPR,
          PrologEpilogSGPRSaveRestoreInfo(SGPRSaveKind::SPILL_TO_MEM, FI));
      LLVM_DEBUG(dbgs() << "Reserved FI " << FI << " for spilling "
                        << printReg(SGPR, TRI) << '\n');
    }
  } else {
    MFI->addToPrologEpilogSGPRSpills(
        SGPR, PrologEpilogSGPRSaveRestoreInfo(
                  SGPRSaveKind::COPY_TO_SCRATCH_SGPR, ScratchSGPR));
    LiveRegs.addReg(ScratchSGPR);
    LLVM_DEBUG(dbgs() << "Saving " << printReg(SGPR, TRI) << " with copy to "
                      << printReg(ScratchSGPR, TRI) << '\n');
  }
}

// We need to specially emit stack operations here because a different frame
// register is used than in the rest of the function, as getFrameRegister would
// use.
static void buildPrologSpill(const GCNSubtarget &ST, const SIRegisterInfo &TRI,
                             const SIMachineFunctionInfo &FuncInfo,
                             LivePhysRegs &LiveRegs, MachineFunction &MF,
                             MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator I, const DebugLoc &DL,
                             Register SpillReg, int FI, Register FrameReg,
                             int64_t DwordOff = 0) {
  unsigned Opc = ST.enableFlatScratch() ? AMDGPU::SCRATCH_STORE_DWORD_SADDR
                                        : AMDGPU::BUFFER_STORE_DWORD_OFFSET;

  MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  MachinePointerInfo PtrInfo = MachinePointerInfo::getFixedStack(MF, FI);
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      PtrInfo, MachineMemOperand::MOStore, FrameInfo.getObjectSize(FI),
      FrameInfo.getObjectAlign(FI));
  LiveRegs.addReg(SpillReg);
  bool IsKill = !MBB.isLiveIn(SpillReg);
  TRI.buildSpillLoadStore(MBB, I, DL, Opc, FI, SpillReg, IsKill, FrameReg,
                          DwordOff, MMO, nullptr, &LiveRegs);
  if (IsKill)
    LiveRegs.removeReg(SpillReg);
}

static void buildEpilogRestore(const GCNSubtarget &ST,
                               const SIRegisterInfo &TRI,
                               const SIMachineFunctionInfo &FuncInfo,
                               LivePhysRegs &LiveRegs, MachineFunction &MF,
                               MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I,
                               const DebugLoc &DL, Register SpillReg, int FI,
                               Register FrameReg, int64_t DwordOff = 0) {
  unsigned Opc = ST.enableFlatScratch() ? AMDGPU::SCRATCH_LOAD_DWORD_SADDR
                                        : AMDGPU::BUFFER_LOAD_DWORD_OFFSET;

  MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  MachinePointerInfo PtrInfo = MachinePointerInfo::getFixedStack(MF, FI);
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      PtrInfo, MachineMemOperand::MOLoad, FrameInfo.getObjectSize(FI),
      FrameInfo.getObjectAlign(FI));
  TRI.buildSpillLoadStore(MBB, I, DL, Opc, FI, SpillReg, false, FrameReg,
                          DwordOff, MMO, nullptr, &LiveRegs);
}

static void buildGitPtr(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                        const DebugLoc &DL, const SIInstrInfo *TII,
                        Register TargetReg) {
  MachineFunction *MF = MBB.getParent();
  const SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  const SIRegisterInfo *TRI = &TII->getRegisterInfo();
  const MCInstrDesc &SMovB32 = TII->get(AMDGPU::S_MOV_B32);
  Register TargetLo = TRI->getSubReg(TargetReg, AMDGPU::sub0);
  Register TargetHi = TRI->getSubReg(TargetReg, AMDGPU::sub1);

  if (MFI->getGITPtrHigh() != 0xffffffff) {
    BuildMI(MBB, I, DL, SMovB32, TargetHi)
        .addImm(MFI->getGITPtrHigh())
        .addReg(TargetReg, RegState::ImplicitDefine);
  } else {
    const MCInstrDesc &GetPC64 = TII->get(AMDGPU::S_GETPC_B64);
    BuildMI(MBB, I, DL, GetPC64, TargetReg);
  }
  Register GitPtrLo = MFI->getGITPtrLoReg(*MF);
  MF->getRegInfo().addLiveIn(GitPtrLo);
  MBB.addLiveIn(GitPtrLo);
  BuildMI(MBB, I, DL, SMovB32, TargetLo)
    .addReg(GitPtrLo);
}

static void initLiveRegs(LivePhysRegs &LiveRegs, const SIRegisterInfo &TRI,
                         const SIMachineFunctionInfo *FuncInfo,
                         MachineFunction &MF, MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI, bool IsProlog) {
  if (LiveRegs.empty()) {
    LiveRegs.init(TRI);
    if (IsProlog) {
      LiveRegs.addLiveIns(MBB);
    } else {
      // In epilog.
      LiveRegs.addLiveOuts(MBB);
      LiveRegs.stepBackward(*MBBI);
    }
  }
}

namespace llvm {

// SpillBuilder to save/restore special SGPR spills like the one needed for FP,
// BP, etc. These spills are delayed until the current function's frame is
// finalized. For a given register, the builder uses the
// PrologEpilogSGPRSaveRestoreInfo to decide the spill method.
class PrologEpilogSGPRSpillBuilder {
  MachineBasicBlock::iterator MI;
  MachineBasicBlock &MBB;
  MachineFunction &MF;
  const GCNSubtarget &ST;
  MachineFrameInfo &MFI;
  SIMachineFunctionInfo *FuncInfo;
  const SIInstrInfo *TII;
  const SIRegisterInfo &TRI;
  const MCRegisterInfo *MCRI;
  const SIFrameLowering *TFI;
  Register SuperReg;
  const PrologEpilogSGPRSaveRestoreInfo SI;
  LivePhysRegs &LiveRegs;
  const DebugLoc &DL;
  Register FrameReg;
  ArrayRef<int16_t> SplitParts;
  unsigned NumSubRegs;
  unsigned EltSize = 4;
  bool IsFramePtrPrologSpill;
  bool NeedsFrameMoves;

  bool isExec(Register Reg) const {
    return Reg == AMDGPU::EXEC_LO || Reg == AMDGPU::EXEC;
  }

  void saveToMemory(const int FI) const {
    MachineRegisterInfo &MRI = MF.getRegInfo();
    const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
    assert(!MFI.isDeadObjectIndex(FI));

    initLiveRegs(LiveRegs, TRI, FuncInfo, MF, MBB, MI, /*IsProlog*/ true);

    MCPhysReg TmpVGPR = findScratchNonCalleeSaveRegister(
        MRI, LiveRegs, AMDGPU::VGPR_32RegClass);
    if (!TmpVGPR)
      report_fatal_error("failed to find free scratch register");

    for (unsigned I = 0, DwordOff = 0; I < NumSubRegs; ++I) {
      Register SubReg = NumSubRegs == 1
                            ? SuperReg
                            : Register(TRI.getSubReg(SuperReg, SplitParts[I]));
      BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_MOV_B32_e32), TmpVGPR)
          .addReg(SubReg);

      buildPrologSpill(ST, TRI, *FuncInfo, LiveRegs, MF, MBB, MI, DL, TmpVGPR,
                       FI, FrameReg, DwordOff);
      if (NeedsFrameMoves) {
        if (isExec(SuperReg) && (I == NumSubRegs - 1))
          SubReg = AMDGPU::EXEC;
        else if (IsFramePtrPrologSpill)
          SubReg = FuncInfo->getFrameOffsetReg();

        // FIXME: CFI for EXEC needs a fix by accurately computing the spill
        // offset for both the low and high components.
        if (SubReg != AMDGPU::EXEC_LO)
          TFI->buildCFI(MBB, MI, DL,
                        MCCFIInstruction::createOffset(
                            nullptr, MCRI->getDwarfRegNum(SubReg, false),
                            MFI.getObjectOffset(FI) * ST.getWavefrontSize()));
      }
      DwordOff += 4;
    }
  }

  void saveToVGPRLane(const int FI) const {
    assert(!MFI.isDeadObjectIndex(FI));

    assert(MFI.getStackID(FI) == TargetStackID::SGPRSpill);
    ArrayRef<SIRegisterInfo::SpilledReg> Spill =
        FuncInfo->getSGPRSpillToPhysicalVGPRLanes(FI);
    assert(Spill.size() == NumSubRegs);

    for (unsigned I = 0; I < NumSubRegs; ++I) {
      Register SubReg = NumSubRegs == 1
                            ? SuperReg
                            : Register(TRI.getSubReg(SuperReg, SplitParts[I]));
      BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_WRITELANE_B32), Spill[I].VGPR)
          .addReg(SubReg)
          .addImm(Spill[I].Lane)
          .addReg(Spill[I].VGPR, RegState::Undef);
      if (NeedsFrameMoves) {
        if (isExec(SuperReg)) {
          if (I == NumSubRegs - 1)
            TFI->buildCFIForSGPRToVGPRSpill(MBB, MI, DL, AMDGPU::EXEC, Spill);
        } else if (IsFramePtrPrologSpill) {
          TFI->buildCFIForSGPRToVGPRSpill(MBB, MI, DL,
                                          FuncInfo->getFrameOffsetReg(),
                                          Spill[I].VGPR, Spill[I].Lane);
        } else {
          TFI->buildCFIForSGPRToVGPRSpill(MBB, MI, DL, SubReg, Spill[I].VGPR,
                                          Spill[I].Lane);
        }
      }
    }
  }

  void copyToScratchSGPR(Register DstReg) const {
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), DstReg)
        .addReg(SuperReg)
        .setMIFlag(MachineInstr::FrameSetup);
    if (NeedsFrameMoves) {
      const TargetRegisterClass *RC = TRI.getPhysRegBaseClass(DstReg);
      ArrayRef<int16_t> DstSplitParts = TRI.getRegSplitParts(RC, EltSize);
      unsigned DstNumSubRegs = DstSplitParts.empty() ? 1 : DstSplitParts.size();
      assert(NumSubRegs == DstNumSubRegs);
      for (unsigned I = 0; I < NumSubRegs; ++I) {
        Register SrcSubReg =
            NumSubRegs == 1 ? SuperReg
                            : Register(TRI.getSubReg(SuperReg, SplitParts[I]));
        Register DstSubReg =
            NumSubRegs == 1 ? DstReg
                            : Register(TRI.getSubReg(DstReg, DstSplitParts[I]));
        if (isExec(SuperReg)) {
          if (I == NumSubRegs - 1)
            TFI->buildCFIForRegToSGPRPairSpill(MBB, MI, DL, AMDGPU::EXEC,
                                               DstReg);
        } else {
          TFI->buildCFI(MBB, MI, DL,
                        MCCFIInstruction::createRegister(
                            nullptr, MCRI->getDwarfRegNum(SrcSubReg, false),
                            MCRI->getDwarfRegNum(DstSubReg, false)));
        }
      }
    }
  }

  void restoreFromMemory(const int FI) {
    MachineRegisterInfo &MRI = MF.getRegInfo();
    const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

    initLiveRegs(LiveRegs, TRI, FuncInfo, MF, MBB, MI, /*IsProlog*/ false);
    MCPhysReg TmpVGPR = findScratchNonCalleeSaveRegister(
        MRI, LiveRegs, AMDGPU::VGPR_32RegClass);
    if (!TmpVGPR)
      report_fatal_error("failed to find free scratch register");

    for (unsigned I = 0, DwordOff = 0; I < NumSubRegs; ++I) {
      Register SubReg = NumSubRegs == 1
                            ? SuperReg
                            : Register(TRI.getSubReg(SuperReg, SplitParts[I]));

      buildEpilogRestore(ST, TRI, *FuncInfo, LiveRegs, MF, MBB, MI, DL, TmpVGPR,
                         FI, FrameReg, DwordOff);
      BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_READFIRSTLANE_B32), SubReg)
          .addReg(TmpVGPR, RegState::Kill);
      DwordOff += 4;
    }
  }

  void restoreFromVGPRLane(const int FI) {
    assert(MFI.getStackID(FI) == TargetStackID::SGPRSpill);
    ArrayRef<SIRegisterInfo::SpilledReg> Spill =
        FuncInfo->getSGPRSpillToPhysicalVGPRLanes(FI);
    assert(Spill.size() == NumSubRegs);

    for (unsigned I = 0; I < NumSubRegs; ++I) {
      Register SubReg = NumSubRegs == 1
                            ? SuperReg
                            : Register(TRI.getSubReg(SuperReg, SplitParts[I]));
      BuildMI(MBB, MI, DL, TII->get(AMDGPU::V_READLANE_B32), SubReg)
          .addReg(Spill[I].VGPR)
          .addImm(Spill[I].Lane);
    }
  }

  void copyFromScratchSGPR(Register SrcReg) const {
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::COPY), SuperReg)
        .addReg(SrcReg)
        .setMIFlag(MachineInstr::FrameDestroy);
  }

public:
  PrologEpilogSGPRSpillBuilder(Register Reg,
                               const PrologEpilogSGPRSaveRestoreInfo SI,
                               MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MI,
                               const DebugLoc &DL, const SIInstrInfo *TII,
                               const SIRegisterInfo &TRI,
                               LivePhysRegs &LiveRegs, Register FrameReg,
                               bool IsFramePtrPrologSpill = false)
      : MI(MI), MBB(MBB), MF(*MBB.getParent()),
        ST(MF.getSubtarget<GCNSubtarget>()), MFI(MF.getFrameInfo()),
        FuncInfo(MF.getInfo<SIMachineFunctionInfo>()), TII(TII), TRI(TRI),
        MCRI(MF.getMMI().getContext().getRegisterInfo()),
        TFI(ST.getFrameLowering()), SuperReg(Reg), SI(SI), LiveRegs(LiveRegs),
        DL(DL), FrameReg(FrameReg),
        IsFramePtrPrologSpill(IsFramePtrPrologSpill) {
    const TargetRegisterClass *RC = TRI.getPhysRegBaseClass(SuperReg);
    SplitParts = TRI.getRegSplitParts(RC, EltSize);
    NumSubRegs = SplitParts.empty() ? 1 : SplitParts.size();

    // FIXME: Switch to using MF.needsFrameMoves() later.
    NeedsFrameMoves = true;

    assert(SuperReg != AMDGPU::M0 && "m0 should never spill");
  }

  void save() {
    switch (SI.getKind()) {
    case SGPRSaveKind::SPILL_TO_MEM:
      return saveToMemory(SI.getIndex());
    case SGPRSaveKind::SPILL_TO_VGPR_LANE:
      return saveToVGPRLane(SI.getIndex());
    case SGPRSaveKind::COPY_TO_SCRATCH_SGPR:
      return copyToScratchSGPR(SI.getReg());
    }
  }

  void restore() {
    switch (SI.getKind()) {
    case SGPRSaveKind::SPILL_TO_MEM:
      return restoreFromMemory(SI.getIndex());
    case SGPRSaveKind::SPILL_TO_VGPR_LANE:
      return restoreFromVGPRLane(SI.getIndex());
    case SGPRSaveKind::COPY_TO_SCRATCH_SGPR:
      return copyFromScratchSGPR(SI.getReg());
    }
  }
};

} // namespace llvm

// Emit flat scratch setup code, assuming `MFI->hasFlatScratchInit()`
void SIFrameLowering::emitEntryFunctionFlatScratchInit(
    MachineFunction &MF, MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
    const DebugLoc &DL, Register ScratchWaveOffsetReg) const {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = &TII->getRegisterInfo();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  // We don't need this if we only have spills since there is no user facing
  // scratch.

  // TODO: If we know we don't have flat instructions earlier, we can omit
  // this from the input registers.
  //
  // TODO: We only need to know if we access scratch space through a flat
  // pointer. Because we only detect if flat instructions are used at all,
  // this will be used more often than necessary on VI.

  Register FlatScrInitLo;
  Register FlatScrInitHi;

  if (ST.isAmdPalOS()) {
    // Extract the scratch offset from the descriptor in the GIT
    LivePhysRegs LiveRegs;
    LiveRegs.init(*TRI);
    LiveRegs.addLiveIns(MBB);

    // Find unused reg to load flat scratch init into
    MachineRegisterInfo &MRI = MF.getRegInfo();
    Register FlatScrInit = AMDGPU::NoRegister;
    ArrayRef<MCPhysReg> AllSGPR64s = TRI->getAllSGPR64(MF);
    unsigned NumPreloaded = (MFI->getNumPreloadedSGPRs() + 1) / 2;
    AllSGPR64s = AllSGPR64s.slice(
        std::min(static_cast<unsigned>(AllSGPR64s.size()), NumPreloaded));
    Register GITPtrLoReg = MFI->getGITPtrLoReg(MF);
    for (MCPhysReg Reg : AllSGPR64s) {
      if (LiveRegs.available(MRI, Reg) && MRI.isAllocatable(Reg) &&
          !TRI->isSubRegisterEq(Reg, GITPtrLoReg)) {
        FlatScrInit = Reg;
        break;
      }
    }
    assert(FlatScrInit && "Failed to find free register for scratch init");

    FlatScrInitLo = TRI->getSubReg(FlatScrInit, AMDGPU::sub0);
    FlatScrInitHi = TRI->getSubReg(FlatScrInit, AMDGPU::sub1);

    buildGitPtr(MBB, I, DL, TII, FlatScrInit);

    // We now have the GIT ptr - now get the scratch descriptor from the entry
    // at offset 0 (or offset 16 for a compute shader).
    MachinePointerInfo PtrInfo(AMDGPUAS::CONSTANT_ADDRESS);
    const MCInstrDesc &LoadDwordX2 = TII->get(AMDGPU::S_LOAD_DWORDX2_IMM);
    auto *MMO = MF.getMachineMemOperand(
        PtrInfo,
        MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant |
            MachineMemOperand::MODereferenceable,
        8, Align(4));
    unsigned Offset =
        MF.getFunction().getCallingConv() == CallingConv::AMDGPU_CS ? 16 : 0;
    const GCNSubtarget &Subtarget = MF.getSubtarget<GCNSubtarget>();
    unsigned EncodedOffset = AMDGPU::convertSMRDOffsetUnits(Subtarget, Offset);
    BuildMI(MBB, I, DL, LoadDwordX2, FlatScrInit)
        .addReg(FlatScrInit)
        .addImm(EncodedOffset) // offset
        .addImm(0)             // cpol
        .addMemOperand(MMO);

    // Mask the offset in [47:0] of the descriptor
    const MCInstrDesc &SAndB32 = TII->get(AMDGPU::S_AND_B32);
    auto And = BuildMI(MBB, I, DL, SAndB32, FlatScrInitHi)
        .addReg(FlatScrInitHi)
        .addImm(0xffff);
    And->getOperand(3).setIsDead(); // Mark SCC as dead.
  } else {
    Register FlatScratchInitReg =
        MFI->getPreloadedReg(AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT);
    assert(FlatScratchInitReg);

    MachineRegisterInfo &MRI = MF.getRegInfo();
    MRI.addLiveIn(FlatScratchInitReg);
    MBB.addLiveIn(FlatScratchInitReg);

    FlatScrInitLo = TRI->getSubReg(FlatScratchInitReg, AMDGPU::sub0);
    FlatScrInitHi = TRI->getSubReg(FlatScratchInitReg, AMDGPU::sub1);
  }

  // Do a 64-bit pointer add.
  if (ST.flatScratchIsPointer()) {
    if (ST.getGeneration() >= AMDGPUSubtarget::GFX10) {
      BuildMI(MBB, I, DL, TII->get(AMDGPU::S_ADD_U32), FlatScrInitLo)
        .addReg(FlatScrInitLo)
        .addReg(ScratchWaveOffsetReg);
      auto Addc = BuildMI(MBB, I, DL, TII->get(AMDGPU::S_ADDC_U32),
                          FlatScrInitHi)
        .addReg(FlatScrInitHi)
        .addImm(0);
      Addc->getOperand(3).setIsDead(); // Mark SCC as dead.

      BuildMI(MBB, I, DL, TII->get(AMDGPU::S_SETREG_B32)).
        addReg(FlatScrInitLo).
        addImm(int16_t(AMDGPU::Hwreg::ID_FLAT_SCR_LO |
                       (31 << AMDGPU::Hwreg::WIDTH_M1_SHIFT_)));
      BuildMI(MBB, I, DL, TII->get(AMDGPU::S_SETREG_B32)).
        addReg(FlatScrInitHi).
        addImm(int16_t(AMDGPU::Hwreg::ID_FLAT_SCR_HI |
                       (31 << AMDGPU::Hwreg::WIDTH_M1_SHIFT_)));
      return;
    }

    // For GFX9.
    BuildMI(MBB, I, DL, TII->get(AMDGPU::S_ADD_U32), AMDGPU::FLAT_SCR_LO)
      .addReg(FlatScrInitLo)
      .addReg(ScratchWaveOffsetReg);
    auto Addc = BuildMI(MBB, I, DL, TII->get(AMDGPU::S_ADDC_U32),
                        AMDGPU::FLAT_SCR_HI)
      .addReg(FlatScrInitHi)
      .addImm(0);
    Addc->getOperand(3).setIsDead(); // Mark SCC as dead.

    return;
  }

  assert(ST.getGeneration() < AMDGPUSubtarget::GFX9);

  // Copy the size in bytes.
  BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), AMDGPU::FLAT_SCR_LO)
    .addReg(FlatScrInitHi, RegState::Kill);

  // Add wave offset in bytes to private base offset.
  // See comment in AMDKernelCodeT.h for enable_sgpr_flat_scratch_init.
  BuildMI(MBB, I, DL, TII->get(AMDGPU::S_ADD_I32), FlatScrInitLo)
      .addReg(FlatScrInitLo)
      .addReg(ScratchWaveOffsetReg);

  // Convert offset to 256-byte units.
  auto LShr = BuildMI(MBB, I, DL, TII->get(AMDGPU::S_LSHR_B32),
                      AMDGPU::FLAT_SCR_HI)
    .addReg(FlatScrInitLo, RegState::Kill)
    .addImm(8);
  LShr->getOperand(3).setIsDead(); // Mark SCC as dead.
}

// Note SGPRSpill stack IDs should only be used for SGPR spilling to VGPRs, not
// memory. They should have been removed by now, except CFI Saved Reg spills.
static bool allStackObjectsAreDead(const MachineFunction &MF) {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();
  for (int I = MFI.getObjectIndexBegin(), E = MFI.getObjectIndexEnd();
       I != E; ++I) {
    if (!MFI.isDeadObjectIndex(I)) {
      // determineCalleeSaves() might have added the SGPRSpill stack IDs for
      // CFI saves into scratch VGPR, ignore them
      if (MFI.getStackID(I) == TargetStackID::SGPRSpill &&
          FuncInfo->checkIndexInPrologEpilogSGPRSpills(I)) {
        continue;
      }
      return false;
    }
  }

  return true;
}

// Shift down registers reserved for the scratch RSRC.
Register SIFrameLowering::getEntryFunctionReservedScratchRsrcReg(
    MachineFunction &MF) const {

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = &TII->getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  assert(MFI->isEntryFunction());

  Register ScratchRsrcReg = MFI->getScratchRSrcReg();

  if (!ScratchRsrcReg ||
      (!MRI.isPhysRegUsed(ScratchRsrcReg) && allStackObjectsAreDead(MF)))
    return Register();

  if (ST.hasSGPRInitBug() ||
      ScratchRsrcReg != TRI->reservedPrivateSegmentBufferReg(MF))
    return ScratchRsrcReg;

  // We reserved the last registers for this. Shift it down to the end of those
  // which were actually used.
  //
  // FIXME: It might be safer to use a pseudoregister before replacement.

  // FIXME: We should be able to eliminate unused input registers. We only
  // cannot do this for the resources required for scratch access. For now we
  // skip over user SGPRs and may leave unused holes.

  unsigned NumPreloaded = (MFI->getNumPreloadedSGPRs() + 3) / 4;
  ArrayRef<MCPhysReg> AllSGPR128s = TRI->getAllSGPR128(MF);
  AllSGPR128s = AllSGPR128s.slice(std::min(static_cast<unsigned>(AllSGPR128s.size()), NumPreloaded));

  // Skip the last N reserved elements because they should have already been
  // reserved for VCC etc.
  Register GITPtrLoReg = MFI->getGITPtrLoReg(MF);
  for (MCPhysReg Reg : AllSGPR128s) {
    // Pick the first unallocated one. Make sure we don't clobber the other
    // reserved input we needed. Also for PAL, make sure we don't clobber
    // the GIT pointer passed in SGPR0 or SGPR8.
    if (!MRI.isPhysRegUsed(Reg) && MRI.isAllocatable(Reg) &&
        !TRI->isSubRegisterEq(Reg, GITPtrLoReg)) {
      MRI.replaceRegWith(ScratchRsrcReg, Reg);
      MFI->setScratchRSrcReg(Reg);
      return Reg;
    }
  }

  return ScratchRsrcReg;
}

static unsigned getScratchScaleFactor(const GCNSubtarget &ST) {
  return ST.enableFlatScratch() ? 1 : ST.getWavefrontSize();
}

void SIFrameLowering::emitEntryFunctionPrologue(MachineFunction &MF,
                                                MachineBasicBlock &MBB) const {
  assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");

  // FIXME: If we only have SGPR spills, we won't actually be using scratch
  // memory since these spill to VGPRs. We should be cleaning up these unused
  // SGPR spill frame indices somewhere.

  // FIXME: We still have implicit uses on SGPR spill instructions in case they
  // need to spill to vector memory. It's likely that will not happen, but at
  // this point it appears we need the setup. This part of the prolog should be
  // emitted after frame indices are eliminated.

  // FIXME: Remove all of the isPhysRegUsed checks

  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = &TII->getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const Function &F = MF.getFunction();
  const MCRegisterInfo *MCRI = MF.getMMI().getContext().getRegisterInfo();
  MachineFrameInfo &FrameInfo = MF.getFrameInfo();

  assert(MFI->isEntryFunction());

  // Debug location must be unknown since the first debug location is used to
  // determine the end of the prologue.
  DebugLoc DL;
  MachineBasicBlock::iterator I = MBB.begin();

  // FIXME: Switch to using MF.needsFrameMoves() later
  const bool NeedsFrameMoves = true;

  if (NeedsFrameMoves) {
    // On entry the SP/FP are not set up, so we need to define the CFA in terms
    // of a literal location expression.
    static const char CFAEncodedInst[] = {
        dwarf::DW_CFA_def_cfa_expression,
        3, // length
        static_cast<char>(dwarf::DW_OP_lit0),
        static_cast<char>(dwarf::DW_OP_lit0 + DW_ASPACE_AMDGPU_private_wave),
        static_cast<char>(dwarf::DW_OP_LLVM_form_aspace_address)};
    buildCFI(MBB, I, DL,
             MCCFIInstruction::createEscape(
                 nullptr, StringRef(CFAEncodedInst, sizeof(CFAEncodedInst))));
    // Unwinding halts when the return address (PC) is undefined.
    buildCFI(MBB, I, DL,
             MCCFIInstruction::createUndefined(
                 nullptr, MCRI->getDwarfRegNum(AMDGPU::PC_REG, false)));
  }

  Register PreloadedScratchWaveOffsetReg = MFI->getPreloadedReg(
      AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET);

  // We need to do the replacement of the private segment buffer register even
  // if there are no stack objects. There could be stores to undef or a
  // constant without an associated object.
  //
  // This will return `Register()` in cases where there are no actual
  // uses of the SRSRC.
  Register ScratchRsrcReg;
  if (!ST.enableFlatScratch())
    ScratchRsrcReg = getEntryFunctionReservedScratchRsrcReg(MF);

  // Make the selected register live throughout the function.
  if (ScratchRsrcReg) {
    for (MachineBasicBlock &OtherBB : MF) {
      if (&OtherBB != &MBB) {
        OtherBB.addLiveIn(ScratchRsrcReg);
      }
    }
  }

  // Now that we have fixed the reserved SRSRC we need to locate the
  // (potentially) preloaded SRSRC.
  Register PreloadedScratchRsrcReg;
  if (ST.isAmdHsaOrMesa(F)) {
    PreloadedScratchRsrcReg =
        MFI->getPreloadedReg(AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER);
    if (ScratchRsrcReg && PreloadedScratchRsrcReg) {
      // We added live-ins during argument lowering, but since they were not
      // used they were deleted. We're adding the uses now, so add them back.
      MRI.addLiveIn(PreloadedScratchRsrcReg);
      MBB.addLiveIn(PreloadedScratchRsrcReg);
    }
  }

  // We found the SRSRC first because it needs four registers and has an
  // alignment requirement. If the SRSRC that we found is clobbering with
  // the scratch wave offset, which may be in a fixed SGPR or a free SGPR
  // chosen by SITargetLowering::allocateSystemSGPRs, COPY the scratch
  // wave offset to a free SGPR.
  Register ScratchWaveOffsetReg;
  if (PreloadedScratchWaveOffsetReg &&
      TRI->isSubRegisterEq(ScratchRsrcReg, PreloadedScratchWaveOffsetReg)) {
    ArrayRef<MCPhysReg> AllSGPRs = TRI->getAllSGPR32(MF);
    unsigned NumPreloaded = MFI->getNumPreloadedSGPRs();
    AllSGPRs = AllSGPRs.slice(
        std::min(static_cast<unsigned>(AllSGPRs.size()), NumPreloaded));
    Register GITPtrLoReg = MFI->getGITPtrLoReg(MF);
    for (MCPhysReg Reg : AllSGPRs) {
      if (!MRI.isPhysRegUsed(Reg) && MRI.isAllocatable(Reg) &&
          !TRI->isSubRegisterEq(ScratchRsrcReg, Reg) && GITPtrLoReg != Reg) {
        ScratchWaveOffsetReg = Reg;
        BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), ScratchWaveOffsetReg)
            .addReg(PreloadedScratchWaveOffsetReg, RegState::Kill);
        break;
      }
    }
  } else {
    ScratchWaveOffsetReg = PreloadedScratchWaveOffsetReg;
  }
  assert(ScratchWaveOffsetReg || !PreloadedScratchWaveOffsetReg);

  if (requiresStackPointerReference(MF)) {
    Register SPReg = MFI->getStackPtrOffsetReg();
    assert(SPReg != AMDGPU::SP_REG);
    BuildMI(MBB, I, DL, TII->get(AMDGPU::S_MOV_B32), SPReg)
        .addImm(FrameInfo.getStackSize() * getScratchScaleFactor(ST));
  }

  if (hasFP(MF)) {
    Register FPReg = MFI->getFrameOffsetReg();
    assert(FPReg != AMDGPU::FP_REG);
    BuildMI(MBB, I, DL, TII->get(AMDGPU::S_MOV_B32), FPReg).addImm(0);
  }

  bool NeedsFlatScratchInit =
      MFI->hasFlatScratchInit() &&
      (MRI.isPhysRegUsed(AMDGPU::FLAT_SCR) || FrameInfo.hasCalls() ||
       (!allStackObjectsAreDead(MF) && ST.enableFlatScratch()));

  if ((NeedsFlatScratchInit || ScratchRsrcReg) &&
      PreloadedScratchWaveOffsetReg && !ST.flatScratchIsArchitected()) {
    MRI.addLiveIn(PreloadedScratchWaveOffsetReg);
    MBB.addLiveIn(PreloadedScratchWaveOffsetReg);
  }

  if (NeedsFlatScratchInit) {
    emitEntryFunctionFlatScratchInit(MF, MBB, I, DL, ScratchWaveOffsetReg);
  }

  if (ScratchRsrcReg) {
    emitEntryFunctionScratchRsrcRegSetup(MF, MBB, I, DL,
                                         PreloadedScratchRsrcReg,
                                         ScratchRsrcReg, ScratchWaveOffsetReg);
  }
}

// Emit scratch RSRC setup code, assuming `ScratchRsrcReg != AMDGPU::NoReg`
void SIFrameLowering::emitEntryFunctionScratchRsrcRegSetup(
    MachineFunction &MF, MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
    const DebugLoc &DL, Register PreloadedScratchRsrcReg,
    Register ScratchRsrcReg, Register ScratchWaveOffsetReg) const {

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = &TII->getRegisterInfo();
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const Function &Fn = MF.getFunction();

  if (ST.isAmdPalOS()) {
    // The pointer to the GIT is formed from the offset passed in and either
    // the amdgpu-git-ptr-high function attribute or the top part of the PC
    Register Rsrc01 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub0_sub1);
    Register Rsrc03 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub3);

    buildGitPtr(MBB, I, DL, TII, Rsrc01);

    // We now have the GIT ptr - now get the scratch descriptor from the entry
    // at offset 0 (or offset 16 for a compute shader).
    MachinePointerInfo PtrInfo(AMDGPUAS::CONSTANT_ADDRESS);
    const MCInstrDesc &LoadDwordX4 = TII->get(AMDGPU::S_LOAD_DWORDX4_IMM);
    auto MMO = MF.getMachineMemOperand(PtrInfo,
                                       MachineMemOperand::MOLoad |
                                           MachineMemOperand::MOInvariant |
                                           MachineMemOperand::MODereferenceable,
                                       16, Align(4));
    unsigned Offset = Fn.getCallingConv() == CallingConv::AMDGPU_CS ? 16 : 0;
    const GCNSubtarget &Subtarget = MF.getSubtarget<GCNSubtarget>();
    unsigned EncodedOffset = AMDGPU::convertSMRDOffsetUnits(Subtarget, Offset);
    BuildMI(MBB, I, DL, LoadDwordX4, ScratchRsrcReg)
      .addReg(Rsrc01)
      .addImm(EncodedOffset) // offset
      .addImm(0) // cpol
      .addReg(ScratchRsrcReg, RegState::ImplicitDefine)
      .addMemOperand(MMO);

    // The driver will always set the SRD for wave 64 (bits 118:117 of
    // descriptor / bits 22:21 of third sub-reg will be 0b11)
    // If the shader is actually wave32 we have to modify the const_index_stride
    // field of the descriptor 3rd sub-reg (bits 22:21) to 0b10 (stride=32). The
    // reason the driver does this is that there can be cases where it presents
    // 2 shaders with different wave size (e.g. VsFs).
    // TODO: convert to using SCRATCH instructions or multiple SRD buffers
    if (ST.isWave32()) {
      const MCInstrDesc &SBitsetB32 = TII->get(AMDGPU::S_BITSET0_B32);
      BuildMI(MBB, I, DL, SBitsetB32, Rsrc03)
          .addImm(21)
          .addReg(Rsrc03);
    }
  } else if (ST.isMesaGfxShader(Fn) || !PreloadedScratchRsrcReg) {
    assert(!ST.isAmdHsaOrMesa(Fn));
    const MCInstrDesc &SMovB32 = TII->get(AMDGPU::S_MOV_B32);

    Register Rsrc2 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub2);
    Register Rsrc3 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub3);

    // Use relocations to get the pointer, and setup the other bits manually.
    uint64_t Rsrc23 = TII->getScratchRsrcWords23();

    if (MFI->hasImplicitBufferPtr()) {
      Register Rsrc01 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub0_sub1);

      if (AMDGPU::isCompute(MF.getFunction().getCallingConv())) {
        const MCInstrDesc &Mov64 = TII->get(AMDGPU::S_MOV_B64);

        BuildMI(MBB, I, DL, Mov64, Rsrc01)
          .addReg(MFI->getImplicitBufferPtrUserSGPR())
          .addReg(ScratchRsrcReg, RegState::ImplicitDefine);
      } else {
        const MCInstrDesc &LoadDwordX2 = TII->get(AMDGPU::S_LOAD_DWORDX2_IMM);

        MachinePointerInfo PtrInfo(AMDGPUAS::CONSTANT_ADDRESS);
        auto MMO = MF.getMachineMemOperand(
            PtrInfo,
            MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant |
                MachineMemOperand::MODereferenceable,
            8, Align(4));
        BuildMI(MBB, I, DL, LoadDwordX2, Rsrc01)
          .addReg(MFI->getImplicitBufferPtrUserSGPR())
          .addImm(0) // offset
          .addImm(0) // cpol
          .addMemOperand(MMO)
          .addReg(ScratchRsrcReg, RegState::ImplicitDefine);

        MF.getRegInfo().addLiveIn(MFI->getImplicitBufferPtrUserSGPR());
        MBB.addLiveIn(MFI->getImplicitBufferPtrUserSGPR());
      }
    } else {
      Register Rsrc0 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub0);
      Register Rsrc1 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub1);

      BuildMI(MBB, I, DL, SMovB32, Rsrc0)
        .addExternalSymbol("SCRATCH_RSRC_DWORD0")
        .addReg(ScratchRsrcReg, RegState::ImplicitDefine);

      BuildMI(MBB, I, DL, SMovB32, Rsrc1)
        .addExternalSymbol("SCRATCH_RSRC_DWORD1")
        .addReg(ScratchRsrcReg, RegState::ImplicitDefine);

    }

    BuildMI(MBB, I, DL, SMovB32, Rsrc2)
      .addImm(Rsrc23 & 0xffffffff)
      .addReg(ScratchRsrcReg, RegState::ImplicitDefine);

    BuildMI(MBB, I, DL, SMovB32, Rsrc3)
      .addImm(Rsrc23 >> 32)
      .addReg(ScratchRsrcReg, RegState::ImplicitDefine);
  } else if (ST.isAmdHsaOrMesa(Fn)) {
    assert(PreloadedScratchRsrcReg);

    if (ScratchRsrcReg != PreloadedScratchRsrcReg) {
      BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY), ScratchRsrcReg)
          .addReg(PreloadedScratchRsrcReg, RegState::Kill);
    }
  }

  // Add the scratch wave offset into the scratch RSRC.
  //
  // We only want to update the first 48 bits, which is the base address
  // pointer, without touching the adjacent 16 bits of flags. We know this add
  // cannot carry-out from bit 47, otherwise the scratch allocation would be
  // impossible to fit in the 48-bit global address space.
  //
  // TODO: Evaluate if it is better to just construct an SRD using the flat
  // scratch init and some constants rather than update the one we are passed.
  Register ScratchRsrcSub0 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub0);
  Register ScratchRsrcSub1 = TRI->getSubReg(ScratchRsrcReg, AMDGPU::sub1);

  // We cannot Kill ScratchWaveOffsetReg here because we allow it to be used in
  // the kernel body via inreg arguments.
  BuildMI(MBB, I, DL, TII->get(AMDGPU::S_ADD_U32), ScratchRsrcSub0)
      .addReg(ScratchRsrcSub0)
      .addReg(ScratchWaveOffsetReg)
      .addReg(ScratchRsrcReg, RegState::ImplicitDefine);
  auto Addc = BuildMI(MBB, I, DL, TII->get(AMDGPU::S_ADDC_U32), ScratchRsrcSub1)
      .addReg(ScratchRsrcSub1)
      .addImm(0)
      .addReg(ScratchRsrcReg, RegState::ImplicitDefine);
  Addc->getOperand(3).setIsDead(); // Mark SCC as dead.
}

bool SIFrameLowering::isSupportedStackID(TargetStackID::Value ID) const {
  switch (ID) {
  case TargetStackID::Default:
  case TargetStackID::NoAlloc:
  case TargetStackID::SGPRSpill:
    return true;
  case TargetStackID::ScalableVector:
  case TargetStackID::WasmLocal:
    return false;
  }
  llvm_unreachable("Invalid TargetStackID::Value");
}

void SIFrameLowering::emitPrologueEntryCFI(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator MBBI,
                                           const DebugLoc &DL) const {
  const MachineFunction &MF = *MBB.getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const MCRegisterInfo *MCRI = MF.getMMI().getContext().getRegisterInfo();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo &TRI = ST.getInstrInfo()->getRegisterInfo();
  Register StackPtrReg =
      MF.getInfo<SIMachineFunctionInfo>()->getStackPtrOffsetReg();

  emitDefCFA(MBB, MBBI, DL, StackPtrReg, /*AspaceAlreadyDefined=*/true,
             MachineInstr::FrameSetup);

  buildCFIForRegToSGPRPairSpill(MBB, MBBI, DL, AMDGPU::PC_REG,
                                TRI.getReturnAddressReg(MF));

  BitVector IsCalleeSaved(TRI.getNumRegs());
  const MCPhysReg *CSRegs = MRI.getCalleeSavedRegs();
  for (unsigned I = 0; CSRegs[I]; ++I) {
    IsCalleeSaved.set(CSRegs[I]);
  }
  auto ProcessReg = [&](MCPhysReg Reg) {
    if (IsCalleeSaved.test(Reg) || !MRI.isPhysRegModified(Reg))
      return;
    MCRegister DwarfReg = MCRI->getDwarfRegNum(Reg, false);
    buildCFI(MBB, MBBI, DL,
             MCCFIInstruction::createUndefined(nullptr, DwarfReg));
  };

  // Emit CFI rules for caller saved Arch VGPRs which are clobbered
  for_each(AMDGPU::VGPR_32RegClass.getRegisters(), ProcessReg);

  // Emit CFI rules for caller saved Accum VGPRs which are clobbered
  if (ST.hasMAIInsts()) {
    for_each(AMDGPU::AGPR_32RegClass.getRegisters(), ProcessReg);
  }

  // Emit CFI rules for caller saved SGPRs which are clobbered
  for_each(AMDGPU::SGPR_32RegClass.getRegisters(), ProcessReg);
}

// Activate only the inactive lanes when \p EnableInactiveLanes is true.
// Otherwise, activate all lanes. It returns the saved exec.
static Register buildScratchExecCopy(LivePhysRegs &LiveRegs,
                                     MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI,
                                     const DebugLoc &DL, bool IsProlog,
                                     bool EnableInactiveLanes) {
  Register ScratchExecCopy;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();

  initLiveRegs(LiveRegs, TRI, FuncInfo, MF, MBB, MBBI, IsProlog);

  ScratchExecCopy = findScratchNonCalleeSaveRegister(
      MRI, LiveRegs, *TRI.getWaveMaskRegClass());
  if (!ScratchExecCopy)
    report_fatal_error("failed to find free scratch register");

  LiveRegs.addReg(ScratchExecCopy);

  const unsigned SaveExecOpc =
      ST.isWave32() ? (EnableInactiveLanes ? AMDGPU::S_XOR_SAVEEXEC_B32
                                           : AMDGPU::S_OR_SAVEEXEC_B32)
                    : (EnableInactiveLanes ? AMDGPU::S_XOR_SAVEEXEC_B64
                                           : AMDGPU::S_OR_SAVEEXEC_B64);
  auto SaveExec =
      BuildMI(MBB, MBBI, DL, TII->get(SaveExecOpc), ScratchExecCopy).addImm(-1);
  SaveExec->getOperand(3).setIsDead(); // Mark SCC as dead.

  return ScratchExecCopy;
}

void SIFrameLowering::emitCSRSpillStores(MachineFunction &MF,
                                         MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MBBI,
                                         DebugLoc &DL, LivePhysRegs &LiveRegs,
                                         Register FrameReg,
                                         Register FramePtrRegScratchCopy,
                                         const bool NeedsFrameMoves) const {
  SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  const MCRegisterInfo *MCRI = MF.getMMI().getContext().getRegisterInfo();

  // Spill Whole-Wave Mode VGPRs. Save only the inactive lanes of the scratch
  // registers. However, save all lanes of callee-saved VGPRs. Due to this, we
  // might end up flipping the EXEC bits twice.
  Register ScratchExecCopy;
  SmallVector<std::pair<Register, int>, 2> WWMCalleeSavedRegs, WWMScratchRegs;
  FuncInfo->splitWWMSpillRegisters(MF, WWMCalleeSavedRegs, WWMScratchRegs);
  if (!WWMScratchRegs.empty())
    ScratchExecCopy =
        buildScratchExecCopy(LiveRegs, MF, MBB, MBBI, DL,
                             /*IsProlog*/ true, /*EnableInactiveLanes*/ true);

  auto StoreWWMRegisters =
      [&](SmallVectorImpl<std::pair<Register, int>> &WWMRegs) {
        for (const auto &Reg : WWMRegs) {
          Register VGPR = Reg.first;
          int FI = Reg.second;
          buildPrologSpill(ST, TRI, *FuncInfo, LiveRegs, MF, MBB, MBBI, DL,
                           VGPR, FI, FrameReg);
          if (NeedsFrameMoves)
            // We spill the entire VGPR, so we can get away with just cfi_offset
            buildCFI(MBB, MBBI, DL,
                     MCCFIInstruction::createOffset(
                         nullptr, MCRI->getDwarfRegNum(VGPR, false),
                         MFI.getObjectOffset(FI) * ST.getWavefrontSize()));
        }
      };

  StoreWWMRegisters(WWMScratchRegs);
  if (!WWMCalleeSavedRegs.empty()) {
    if (ScratchExecCopy) {
      unsigned MovOpc = ST.isWave32() ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64;
      MCRegister Exec = ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
      BuildMI(MBB, MBBI, DL, TII->get(MovOpc), Exec).addImm(-1);
    } else {
      ScratchExecCopy = buildScratchExecCopy(LiveRegs, MF, MBB, MBBI, DL,
                                             /*IsProlog*/ true,
                                             /*EnableInactiveLanes*/ false);
    }
  }

  StoreWWMRegisters(WWMCalleeSavedRegs);
  if (ScratchExecCopy) {
    // FIXME: Split block and make terminator.
    unsigned ExecMov = ST.isWave32() ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64;
    MCRegister Exec = ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
    BuildMI(MBB, MBBI, DL, TII->get(ExecMov), Exec)
        .addReg(ScratchExecCopy, RegState::Kill);
    LiveRegs.addReg(ScratchExecCopy);
  }

  Register FramePtrReg = FuncInfo->getFrameOffsetReg();

  for (const auto &Spill : FuncInfo->getPrologEpilogSGPRSpills()) {
    // Special handle FP spill:
    // Skip if FP is saved to a scratch SGPR, the save has already been emitted.
    // Otherwise, FP has been moved to a temporary register and spill it
    // instead.
    bool IsFramePtrPrologSpill = Spill.first == FramePtrReg ? true : false;
    Register Reg = IsFramePtrPrologSpill ? FramePtrRegScratchCopy : Spill.first;
    if (!Reg)
      continue;

    PrologEpilogSGPRSpillBuilder SB(Reg, Spill.second, MBB, MBBI, DL, TII, TRI,
                                    LiveRegs, FrameReg, IsFramePtrPrologSpill);
    SB.save();
  }

  // If a copy to scratch SGPR has been chosen for any of the SGPR spills, make
  // such scratch registers live throughout the function.
  SmallVector<Register, 1> ScratchSGPRs;
  FuncInfo->getAllScratchSGPRCopyDstRegs(ScratchSGPRs);
  if (!ScratchSGPRs.empty()) {
    for (MachineBasicBlock &MBB : MF) {
      for (MCPhysReg Reg : ScratchSGPRs)
        MBB.addLiveIn(Reg);

      MBB.sortUniqueLiveIns();
    }
    if (!LiveRegs.empty()) {
      for (MCPhysReg Reg : ScratchSGPRs)
        LiveRegs.addReg(Reg);
    }
  }

  // Remove the spill entry created for EXEC. It is needed only for CFISaves in
  // the prologue.
  if (TRI.isCFISavedRegsSpillEnabled())
    FuncInfo->removePrologEpilogSGPRSpillEntry(TRI.getExec());
}

void SIFrameLowering::emitCSRSpillRestores(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MBBI, DebugLoc &DL, LivePhysRegs &LiveRegs,
    Register FrameReg, Register FramePtrRegScratchCopy) const {
  const SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  Register FramePtrReg = FuncInfo->getFrameOffsetReg();

  for (const auto &Spill : FuncInfo->getPrologEpilogSGPRSpills()) {
    // Special handle FP restore:
    // Skip if FP needs to be restored from the scratch SGPR. Otherwise, restore
    // the FP value to a temporary register. The frame pointer should be
    // overwritten only at the end when all other spills are restored from
    // current frame.
    Register Reg =
        Spill.first == FramePtrReg ? FramePtrRegScratchCopy : Spill.first;
    if (!Reg)
      continue;

    PrologEpilogSGPRSpillBuilder SB(Reg, Spill.second, MBB, MBBI, DL, TII, TRI,
                                    LiveRegs, FrameReg);
    SB.restore();
  }

  // Restore Whole-Wave Mode VGPRs. Restore only the inactive lanes of the
  // scratch registers. However, restore all lanes of callee-saved VGPRs. Due to
  // this, we might end up flipping the EXEC bits twice.
  Register ScratchExecCopy;
  SmallVector<std::pair<Register, int>, 2> WWMCalleeSavedRegs, WWMScratchRegs;
  FuncInfo->splitWWMSpillRegisters(MF, WWMCalleeSavedRegs, WWMScratchRegs);
  if (!WWMScratchRegs.empty())
    ScratchExecCopy =
        buildScratchExecCopy(LiveRegs, MF, MBB, MBBI, DL,
                             /*IsProlog*/ false, /*EnableInactiveLanes*/ true);

  auto RestoreWWMRegisters =
      [&](SmallVectorImpl<std::pair<Register, int>> &WWMRegs) {
        for (const auto &Reg : WWMRegs) {
          Register VGPR = Reg.first;
          int FI = Reg.second;
          buildEpilogRestore(ST, TRI, *FuncInfo, LiveRegs, MF, MBB, MBBI, DL,
                             VGPR, FI, FrameReg);
        }
      };

  RestoreWWMRegisters(WWMScratchRegs);
  if (!WWMCalleeSavedRegs.empty()) {
    if (ScratchExecCopy) {
      unsigned MovOpc = ST.isWave32() ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64;
      MCRegister Exec = ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
      BuildMI(MBB, MBBI, DL, TII->get(MovOpc), Exec).addImm(-1);
    } else {
      ScratchExecCopy = buildScratchExecCopy(LiveRegs, MF, MBB, MBBI, DL,
                                             /*IsProlog*/ false,
                                             /*EnableInactiveLanes*/ false);
    }
  }

  RestoreWWMRegisters(WWMCalleeSavedRegs);
  if (ScratchExecCopy) {
    // FIXME: Split block and make terminator.
    unsigned ExecMov = ST.isWave32() ? AMDGPU::S_MOV_B32 : AMDGPU::S_MOV_B64;
    MCRegister Exec = ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC;
    BuildMI(MBB, MBBI, DL, TII->get(ExecMov), Exec)
        .addReg(ScratchExecCopy, RegState::Kill);
  }
}

void SIFrameLowering::emitPrologue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();
  if (FuncInfo->isEntryFunction()) {
    emitEntryFunctionPrologue(MF, MBB);
    return;
  }

  MachineFrameInfo &MFI = MF.getFrameInfo();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  const MCRegisterInfo *MCRI = MF.getMMI().getContext().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  Register StackPtrReg = FuncInfo->getStackPtrOffsetReg();
  Register FramePtrReg = FuncInfo->getFrameOffsetReg();
  Register BasePtrReg =
      TRI.hasBasePointer(MF) ? TRI.getBaseRegister() : Register();
  LivePhysRegs LiveRegs;

  MachineBasicBlock::iterator MBBI = MBB.begin();

  // DebugLoc must be unknown since the first instruction with DebugLoc is used
  // to determine the end of the prologue.
  DebugLoc DL;

  bool HasFP = false;
  bool HasBP = false;
  uint32_t NumBytes = MFI.getStackSize();
  uint32_t RoundedSize = NumBytes;

  // FIXME: Switch to using MF.needsFrameMoves() later
  const bool NeedsFrameMoves = true;

  if (NeedsFrameMoves)
    emitPrologueEntryCFI(MBB, MBBI, DL);

  if (TRI.hasStackRealignment(MF))
    HasFP = true;

  Register FramePtrRegScratchCopy;
  if (!HasFP && !hasFP(MF)) {
    // Emit the CSR spill stores with SP base register.
    emitCSRSpillStores(MF, MBB, MBBI, DL, LiveRegs, StackPtrReg,
                       FramePtrRegScratchCopy, NeedsFrameMoves);
  } else {
    // CSR spill stores will use FP as base register.
    Register SGPRForFPSaveRestoreCopy =
        FuncInfo->getScratchSGPRCopyDstReg(FramePtrReg);

    initLiveRegs(LiveRegs, TRI, FuncInfo, MF, MBB, MBBI, /*IsProlog*/ true);
    if (SGPRForFPSaveRestoreCopy) {
      // Copy FP to the scratch register now and emit the CFI entry. It avoids
      // the extra FP copy needed in the other two cases when FP is spilled to
      // memory or to a VGPR lane.
      PrologEpilogSGPRSpillBuilder SB(
          FramePtrReg,
          FuncInfo->getPrologEpilogSGPRSaveRestoreInfo(FramePtrReg), MBB, MBBI,
          DL, TII, TRI, LiveRegs, FramePtrReg,
          /*IsFramePtrPrologSpill*/ true);
      SB.save();
      LiveRegs.addReg(SGPRForFPSaveRestoreCopy);
    } else {
      // Copy FP into a new scratch register so that its previous value can be
      // spilled after setting up the new frame.
      FramePtrRegScratchCopy = findScratchNonCalleeSaveRegister(
          MRI, LiveRegs, AMDGPU::SReg_32_XM0_XEXECRegClass);
      if (!FramePtrRegScratchCopy)
        report_fatal_error("failed to find free scratch register");

      LiveRegs.addReg(FramePtrRegScratchCopy);
      BuildMI(MBB, MBBI, DL, TII->get(AMDGPU::COPY), FramePtrRegScratchCopy)
          .addReg(FramePtrReg);
    }
  }

  if (HasFP) { // Needs stack realignment.
    const unsigned Alignment = MFI.getMaxAlign().value();

    RoundedSize += Alignment;
    if (LiveRegs.empty()) {
      LiveRegs.init(TRI);
      LiveRegs.addLiveIns(MBB);
    }

    // s_add_i32 s33, s32, NumBytes
    // s_and_b32 s33, s33, 0b111...0000
    BuildMI(MBB, MBBI, DL, TII->get(AMDGPU::S_ADD_I32), FramePtrReg)
        .addReg(StackPtrReg)
        .addImm((Alignment - 1) * getScratchScaleFactor(ST))
        .setMIFlag(MachineInstr::FrameSetup);
    auto And = BuildMI(MBB, MBBI, DL, TII->get(AMDGPU::S_AND_B32), FramePtrReg)
        .addReg(FramePtrReg, RegState::Kill)
        .addImm(-Alignment * getScratchScaleFactor(ST))
        .setMIFlag(MachineInstr::FrameSetup);
    And->getOperand(3).setIsDead(); // Mark SCC as dead.
    FuncInfo->setIsStackRealigned(true);
  } else if ((HasFP = hasFP(MF))) {
    BuildMI(MBB, MBBI, DL, TII->get(AMDGPU::COPY), FramePtrReg)
        .addReg(StackPtrReg)
        .setMIFlag(MachineInstr::FrameSetup);
  }

  // If FP is used, emit the CSR spills with FP base register.
  if (HasFP) {
    emitCSRSpillStores(MF, MBB, MBBI, DL, LiveRegs, FramePtrReg,
                       FramePtrRegScratchCopy, NeedsFrameMoves);
    if (FramePtrRegScratchCopy)
      LiveRegs.removeReg(FramePtrRegScratchCopy);
  }
  // If we need a base pointer, set it up here. It's whatever the value of
  // the stack pointer is at this point. Any variable size objects will be
  // allocated after this, so we can still use the base pointer to reference
  // the incoming arguments.
  if ((HasBP = TRI.hasBasePointer(MF))) {
    BuildMI(MBB, MBBI, DL, TII->get(AMDGPU::COPY), BasePtrReg)
        .addReg(StackPtrReg)
        .setMIFlag(MachineInstr::FrameSetup);
  }

  if (HasFP) {
    if (NeedsFrameMoves)
      emitDefCFA(MBB, MBBI, DL, FramePtrReg, /*AspaceAlreadyDefined=*/false,
                 MachineInstr::FrameSetup);
  }

  if (HasFP && RoundedSize != 0) {
    auto Add = BuildMI(MBB, MBBI, DL, TII->get(AMDGPU::S_ADD_I32), StackPtrReg)
        .addReg(StackPtrReg)
        .addImm(RoundedSize * getScratchScaleFactor(ST))
        .setMIFlag(MachineInstr::FrameSetup);
    Add->getOperand(3).setIsDead(); // Mark SCC as dead.
  }

  bool FPSaved = FuncInfo->hasPrologEpilogSGPRSpillEntry(FramePtrReg);
  (void)FPSaved;
  assert((!HasFP || FPSaved) &&
         "Needed to save FP but didn't save it anywhere");

  // If we allow spilling to AGPRs we may have saved FP but then spill
  // everything into AGPRs instead of the stack.
  assert((HasFP || !FPSaved || EnableSpillVGPRToAGPR) &&
         "Saved FP but didn't need it");

  bool BPSaved = FuncInfo->hasPrologEpilogSGPRSpillEntry(BasePtrReg);
  (void)BPSaved;
  assert((!HasBP || BPSaved) &&
         "Needed to save BP but didn't save it anywhere");

  assert((HasBP || !BPSaved) && "Saved BP but didn't need it");
}

void SIFrameLowering::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  const SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();
  if (FuncInfo->isEntryFunction())
    return;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo &TRI = TII->getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  LivePhysRegs LiveRegs;
  // Get the insert location for the epilogue. If there were no terminators in
  // the block, get the last instruction.
  MachineBasicBlock::iterator MBBI = MBB.end();
  DebugLoc DL;
  if (!MBB.empty()) {
    MBBI = MBB.getLastNonDebugInstr();
    if (MBBI != MBB.end())
      DL = MBBI->getDebugLoc();

    MBBI = MBB.getFirstTerminator();
  }

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  uint32_t NumBytes = MFI.getStackSize();
  uint32_t RoundedSize = FuncInfo->isStackRealigned()
                             ? NumBytes + MFI.getMaxAlign().value()
                             : NumBytes;
  const Register StackPtrReg = FuncInfo->getStackPtrOffsetReg();
  Register FramePtrReg = FuncInfo->getFrameOffsetReg();
  bool FPSaved = FuncInfo->hasPrologEpilogSGPRSpillEntry(FramePtrReg);

  Register FramePtrRegScratchCopy;
  Register SGPRForFPSaveRestoreCopy =
      FuncInfo->getScratchSGPRCopyDstReg(FramePtrReg);
  if (FPSaved) {
    // CSR spill restores should use FP as base register. If
    // SGPRForFPSaveRestoreCopy is not true, restore the previous value of FP
    // into a new scratch register and copy to FP later when other registers are
    // restored from the current stack frame.
    initLiveRegs(LiveRegs, TRI, FuncInfo, MF, MBB, MBBI, /*IsProlog*/ false);
    if (SGPRForFPSaveRestoreCopy) {
      LiveRegs.addReg(SGPRForFPSaveRestoreCopy);
    } else {
      FramePtrRegScratchCopy = findScratchNonCalleeSaveRegister(
          MRI, LiveRegs, AMDGPU::SReg_32_XM0_XEXECRegClass);
      if (!FramePtrRegScratchCopy)
        report_fatal_error("failed to find free scratch register");

      LiveRegs.addReg(FramePtrRegScratchCopy);
    }

    emitCSRSpillRestores(MF, MBB, MBBI, DL, LiveRegs, FramePtrReg,
                         FramePtrRegScratchCopy);
  }

  if (RoundedSize != 0 && hasFP(MF)) {
    auto Add = BuildMI(MBB, MBBI, DL, TII->get(AMDGPU::S_ADD_I32), StackPtrReg)
        .addReg(StackPtrReg)
        .addImm(-static_cast<int64_t>(RoundedSize * getScratchScaleFactor(ST)))
        .setMIFlag(MachineInstr::FrameDestroy);
    Add->getOperand(3).setIsDead(); // Mark SCC as dead.
  }

  // FIXME: Switch to using MF.needsFrameMoves() later
  const bool NeedsFrameMoves = true;
  if (hasFP(MF)) {
    if (NeedsFrameMoves)
      emitDefCFA(MBB, MBBI, DL, StackPtrReg, /*AspaceAlreadyDefined=*/false,
                 MachineInstr::FrameDestroy);
  }

  if (FPSaved) {
    // Insert the copy to restore FP.
    Register SrcReg = SGPRForFPSaveRestoreCopy ? SGPRForFPSaveRestoreCopy
                                               : FramePtrRegScratchCopy;
    MachineInstrBuilder MIB =
        BuildMI(MBB, MBBI, DL, TII->get(AMDGPU::COPY), FramePtrReg)
            .addReg(SrcReg);
    if (SGPRForFPSaveRestoreCopy)
      MIB.setMIFlag(MachineInstr::FrameDestroy);
  } else {
    // Insert the CSR spill restores with SP as the base register.
    emitCSRSpillRestores(MF, MBB, MBBI, DL, LiveRegs, StackPtrReg,
                         FramePtrRegScratchCopy);
  }
}

#ifndef NDEBUG
static bool allSGPRSpillsAreDead(const MachineFunction &MF) {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();
  for (int I = MFI.getObjectIndexBegin(), E = MFI.getObjectIndexEnd();
       I != E; ++I) {
    if (!MFI.isDeadObjectIndex(I) &&
        MFI.getStackID(I) == TargetStackID::SGPRSpill &&
        !FuncInfo->checkIndexInPrologEpilogSGPRSpills(I)) {
      return false;
    }
  }

  return true;
}
#endif

StackOffset SIFrameLowering::getFrameIndexReference(const MachineFunction &MF,
                                                    int FI,
                                                    Register &FrameReg) const {
  const SIRegisterInfo *RI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();

  FrameReg = RI->getFrameRegister(MF);
  return StackOffset::getFixed(MF.getFrameInfo().getObjectOffset(FI));
}

void SIFrameLowering::processFunctionBeforeFrameFinalized(
  MachineFunction &MF,
  RegScavenger *RS) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();

  // Allocate spill slots for WWM reserved VGPRs.
  if (!FuncInfo->isEntryFunction()) {
    for (Register Reg : FuncInfo->getWWMReservedRegs()) {
      const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(Reg);
      FuncInfo->allocateWWMSpill(MF, Reg, TRI->getSpillSize(*RC),
                                 TRI->getSpillAlign(*RC));
    }
  }

  const bool SpillVGPRToAGPR = ST.hasMAIInsts() && FuncInfo->hasSpilledVGPRs()
                               && EnableSpillVGPRToAGPR;

  if (SpillVGPRToAGPR) {
    // To track the spill frame indices handled in this pass.
    BitVector SpillFIs(MFI.getObjectIndexEnd(), false);
    BitVector NonVGPRSpillFIs(MFI.getObjectIndexEnd(), false);

    bool SeenDbgInstr = false;

    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
        int FrameIndex;
        if (MI.isDebugInstr())
          SeenDbgInstr = true;

        if (TII->isVGPRSpill(MI)) {
          // Try to eliminate stack used by VGPR spills before frame
          // finalization.
          unsigned FIOp = AMDGPU::getNamedOperandIdx(MI.getOpcode(),
                                                     AMDGPU::OpName::vaddr);
          int FI = MI.getOperand(FIOp).getIndex();
          Register VReg =
            TII->getNamedOperand(MI, AMDGPU::OpName::vdata)->getReg();
          if (FuncInfo->allocateVGPRSpillToAGPR(MF, FI,
                                                TRI->isAGPR(MRI, VReg))) {
            assert(RS != nullptr);
            RS->enterBasicBlockEnd(MBB);
            RS->backward(MI);
            TRI->eliminateFrameIndex(MI, 0, FIOp, RS);
            SpillFIs.set(FI);
            continue;
          }
        } else if (TII->isStoreToStackSlot(MI, FrameIndex) ||
                   TII->isLoadFromStackSlot(MI, FrameIndex))
          if (!MFI.isFixedObjectIndex(FrameIndex))
            NonVGPRSpillFIs.set(FrameIndex);
      }
    }

    // Stack slot coloring may assign different objects to the same stack slot.
    // If not, then the VGPR to AGPR spill slot is dead.
    for (unsigned FI : SpillFIs.set_bits())
      if (!NonVGPRSpillFIs.test(FI))
        FuncInfo->setVGPRToAGPRSpillDead(FI);

    for (MachineBasicBlock &MBB : MF) {
      for (MCPhysReg Reg : FuncInfo->getVGPRSpillAGPRs())
        MBB.addLiveIn(Reg);

      for (MCPhysReg Reg : FuncInfo->getAGPRSpillVGPRs())
        MBB.addLiveIn(Reg);

      MBB.sortUniqueLiveIns();

      if (!SpillFIs.empty() && SeenDbgInstr) {
        // FIXME: The dead frame indices are replaced with a null register from
        // the debug value instructions. We should instead, update it with the
        // correct register value. But not sure the register value alone is
        for (MachineInstr &MI : MBB) {
          if (MI.isDebugValue() && MI.getOperand(0).isFI() &&
              !MFI.isFixedObjectIndex(MI.getOperand(0).getIndex()) &&
              SpillFIs[MI.getOperand(0).getIndex()]) {
            MI.getOperand(0).ChangeToRegister(Register(), false /*isDef*/);
          }
          // FIXME: Need to update expression to locate lane of VGPR to which
          // the SGPR was spilled.
          if (MI.isDebugDef() && MI.getDebugOperand(0).isFI() &&
              !MFI.isFixedObjectIndex(MI.getDebugOperand(0).getIndex()) &&
              SpillFIs[MI.getDebugOperand(0).getIndex()]) {
            MI.getDebugOperand(0).ChangeToRegister(Register(), false /*isDef*/);
          }
        }
      }
    }
  }

  // At this point we've already allocated all spilled SGPRs to VGPRs if we
  // can. Any remaining SGPR spills will go to memory, so move them back to the
  // default stack.
  bool HaveSGPRToVMemSpill =
      FuncInfo->removeDeadFrameIndices(MF, /*ResetSGPRSpillStackIDs*/ true);
  assert(allSGPRSpillsAreDead(MF) &&
         "SGPR spill should have been removed in SILowerSGPRSpills");

  // FIXME: The other checks should be redundant with allStackObjectsAreDead,
  // but currently hasNonSpillStackObjects is set only from source
  // allocas. Stack temps produced from legalization are not counted currently.
  if (!allStackObjectsAreDead(MF)) {
    assert(RS && "RegScavenger required if spilling");

    // Add an emergency spill slot
    RS->addScavengingFrameIndex(FuncInfo->getScavengeFI(MFI, *TRI));

    // If we are spilling SGPRs to memory with a large frame, we may need a
    // second VGPR emergency frame index.
    if (HaveSGPRToVMemSpill &&
        allocateScavengingFrameIndexesNearIncomingSP(MF)) {
      RS->addScavengingFrameIndex(MFI.CreateStackObject(4, Align(4), false));
    }
  }
}

void SIFrameLowering::processFunctionBeforeFrameIndicesReplaced(
    MachineFunction &MF, RegScavenger *RS) const {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();

  if (ST.hasMAIInsts() && !ST.hasGFX90AInsts()) {
    // On gfx908, we had initially reserved highest available VGPR for AGPR
    // copy. Now since we are done with RA, check if there exist an unused VGPR
    // which is lower than the eariler reserved VGPR before RA. If one exist,
    // use it for AGPR copy instead of one reserved before RA.
    Register VGPRForAGPRCopy = FuncInfo->getVGPRForAGPRCopy();
    Register UnusedLowVGPR =
        TRI->findUnusedRegister(MRI, &AMDGPU::VGPR_32RegClass, MF);
    if (UnusedLowVGPR && (TRI->getHWRegIndex(UnusedLowVGPR) <
                          TRI->getHWRegIndex(VGPRForAGPRCopy))) {
      // Call to setVGPRForAGPRCopy() should happen first before calling
      // freezeReservedRegs() so that getReservedRegs() can reserve this newly
      // identified VGPR (for AGPR copy).
      FuncInfo->setVGPRForAGPRCopy(UnusedLowVGPR);
      MRI.freezeReservedRegs(MF);
    }
  }
}

// The special SGPR spills like the one needed for FP, BP or any reserved
// registers delayed until frame lowering.
void SIFrameLowering::determinePrologEpilogSGPRSaves(
    MachineFunction &MF, BitVector &SavedVGPRs,
    bool NeedExecCopyReservedReg) const {
  MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  LivePhysRegs LiveRegs;
  LiveRegs.init(*TRI);
  // Initially mark callee saved registers as used so we will not choose them
  // while looking for scratch SGPRs.
  const MCPhysReg *CSRegs = MF.getRegInfo().getCalleeSavedRegs();
  for (unsigned I = 0; CSRegs[I]; ++I)
    LiveRegs.addReg(CSRegs[I]);

  const TargetRegisterClass &RC = ST.isWave32()
                                      ? AMDGPU::SReg_32_XM0_XEXECRegClass
                                      : AMDGPU::SGPR_64RegClass;

  if (NeedExecCopyReservedReg) {
    Register ReservedReg = MFI->getSGPRForEXECCopy();
    assert(ReservedReg && "Should have reserved an SGPR for EXEC copy.");
    Register UnusedScratchReg = findUnusedRegister(MRI, LiveRegs, RC);
    if (UnusedScratchReg) {
      // If found any unused scratch SGPR, reserve the register itself for Exec
      // copy and there is no need for any spills in that case.
      MFI->setSGPRForEXECCopy(UnusedScratchReg);
      LiveRegs.addReg(UnusedScratchReg);
    } else {
      // Needs spill.
      assert(!MFI->hasPrologEpilogSGPRSpillEntry(ReservedReg) &&
             "Re-reserving spill slot for EXEC copy register");
      getVGPRSpillLaneOrTempRegister(MF, LiveRegs, ReservedReg, RC,
                                     /*IncludeScratchCopy=*/false);
    }
  }

  if (TRI->isCFISavedRegsSpillEnabled()) {
    Register Exec = TRI->getExec();
    assert(!MFI->hasPrologEpilogSGPRSpillEntry(Exec) &&
           "Re-reserving spill slot for EXEC");
    // FIXME: Machine Copy Propagation currently optimizes away the EXEC copy to
    // the scratch as we emit it only in the prolog. This optimization should
    // not happen for frame related instructions. Until this is fixed ignore
    // copy to scratch SGPR.
    getVGPRSpillLaneOrTempRegister(MF, LiveRegs, Exec, RC,
                                   /*IncludeScratchCopy=*/false);
  }

  // hasFP only knows about stack objects that already exist. We're now
  // determining the stack slots that will be created, so we have to predict
  // them. Stack objects force FP usage with calls.
  //
  // Note a new VGPR CSR may be introduced if one is used for the spill, but we
  // don't want to report it here.
  //
  // FIXME: Is this really hasReservedCallFrame?
  const bool WillHaveFP =
      FrameInfo.hasCalls() && (SavedVGPRs.any() || !allStackObjectsAreDead(MF));

  if (WillHaveFP || hasFP(MF)) {
    Register FramePtrReg = MFI->getFrameOffsetReg();
    assert(!MFI->hasPrologEpilogSGPRSpillEntry(FramePtrReg) &&
           "Re-reserving spill slot for FP");
    getVGPRSpillLaneOrTempRegister(MF, LiveRegs, FramePtrReg);
  }

  if (TRI->hasBasePointer(MF)) {
    Register BasePtrReg = TRI->getBaseRegister();
    assert(!MFI->hasPrologEpilogSGPRSpillEntry(BasePtrReg) &&
           "Re-reserving spill slot for BP");
    getVGPRSpillLaneOrTempRegister(MF, LiveRegs, BasePtrReg);
  }
}

// Only report VGPRs to generic code.
void SIFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                           BitVector &SavedVGPRs,
                                           RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedVGPRs, RS);
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  if (MFI->isEntryFunction())
    return;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();
  bool NeedExecCopyReservedReg = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // WRITELANE instructions used for SGPR spills can overwrite the inactive
      // lanes of VGPRs and callee must spill and restore them even if they are
      // marked Caller-saved.

      // TODO: Handle this elsewhere at an early point. Walking through all MBBs
      // here would be a bad heuristic. A better way should be by calling
      // allocateWWMSpill during the regalloc pipeline whenever a physical
      // register is allocated for the intended virtual registers. That will
      // also help excluding the general use of WRITELANE/READLANE intrinsics
      // that won't really need any such special handling.
      if (MI.getOpcode() == AMDGPU::V_WRITELANE_B32)
        MFI->allocateWWMSpill(MF, MI.getOperand(0).getReg());
      else if (MI.getOpcode() == AMDGPU::V_READLANE_B32)
        MFI->allocateWWMSpill(MF, MI.getOperand(1).getReg());
      else if (TII->isWWMRegSpillOpcode(MI.getOpcode()))
        NeedExecCopyReservedReg = true;
    }
  }

  // Ignore the SGPRs the default implementation found.
  SavedVGPRs.clearBitsNotInMask(TRI->getAllVectorRegMask());

  // Do not save AGPRs prior to GFX90A because there was no easy way to do so.
  // In gfx908 there was do AGPR loads and stores and thus spilling also
  // require a temporary VGPR.
  if (!ST.hasGFX90AInsts())
    SavedVGPRs.clearBitsInMask(TRI->getAllAGPRRegMask());

  determinePrologEpilogSGPRSaves(MF, SavedVGPRs, NeedExecCopyReservedReg);

  // The Whole-Wave VGPRs need to be specially inserted in the prolog, so don't
  // allow the default insertion to handle them.
  for (auto &Reg : MFI->getWWMSpills())
    SavedVGPRs.reset(Reg.first);

  // Mark all lane VGPRs as BB LiveIns.
  for (MachineBasicBlock &MBB : MF) {
    for (auto &Reg : MFI->getWWMSpills())
      MBB.addLiveIn(Reg.first);

    MBB.sortUniqueLiveIns();
  }
}

void SIFrameLowering::determineCalleeSavesSGPR(MachineFunction &MF,
                                               BitVector &SavedRegs,
                                               RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  if (MFI->isEntryFunction())
    return;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();

  // The SP is specifically managed and we don't want extra spills of it.
  SavedRegs.reset(MFI->getStackPtrOffsetReg());

  const BitVector AllSavedRegs = SavedRegs;
  SavedRegs.clearBitsInMask(TRI->getAllVectorRegMask());

  // We have to anticipate introducing CSR VGPR spills or spill of caller
  // save VGPR reserved for SGPR spills as we now always create stack entry
  // for it, if we don't have any stack objects already, since we require a FP
  // if there is a call and stack. We will allocate a VGPR for SGPR spills if
  // there are any SGPR spills. Whether they are CSR spills or otherwise.
  MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  const bool WillHaveFP =
      FrameInfo.hasCalls() && (AllSavedRegs.any() || MFI->hasSpilledSGPRs());

  // FP will be specially managed like SP.
  if (WillHaveFP || hasFP(MF))
    SavedRegs.reset(MFI->getFrameOffsetReg());

  // Return address use with return instruction is hidden through the SI_RETURN
  // pseudo. Given that and since the IPRA computes actual register usage and
  // does not use CSR list, the clobbering of return address by function calls
  // (D117243) or otherwise (D120922) is ignored/not seen by the IPRA's register
  // usage collection. This will ensure save/restore of return address happens
  // in those scenarios.
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  Register RetAddrReg = TRI->getReturnAddressReg(MF);
  if (!MFI->isEntryFunction() &&
      (FrameInfo.hasCalls() || MRI.isPhysRegModified(RetAddrReg))) {
    SavedRegs.set(TRI->getSubReg(RetAddrReg, AMDGPU::sub0));
    SavedRegs.set(TRI->getSubReg(RetAddrReg, AMDGPU::sub1));
  }
}

bool SIFrameLowering::assignCalleeSavedSpillSlots(
    MachineFunction &MF, const TargetRegisterInfo *TRI,
    std::vector<CalleeSavedInfo> &CSI) const {
  if (CSI.empty())
    return true; // Early exit if no callee saved registers are modified!

  const SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *RI = ST.getRegisterInfo();
  Register FramePtrReg = FuncInfo->getFrameOffsetReg();
  Register BasePtrReg = RI->getBaseRegister();
  Register SGPRForFPSaveRestoreCopy =
      FuncInfo->getScratchSGPRCopyDstReg(FramePtrReg);
  Register SGPRForBPSaveRestoreCopy =
      FuncInfo->getScratchSGPRCopyDstReg(BasePtrReg);
  if (!SGPRForFPSaveRestoreCopy && !SGPRForBPSaveRestoreCopy)
    return false;

  unsigned NumModifiedRegs = 0;

  if (SGPRForFPSaveRestoreCopy)
    NumModifiedRegs++;
  if (SGPRForBPSaveRestoreCopy)
    NumModifiedRegs++;

  for (auto &CS : CSI) {
    if (CS.getReg() == FramePtrReg && SGPRForFPSaveRestoreCopy) {
      CS.setDstReg(SGPRForFPSaveRestoreCopy);
      if (--NumModifiedRegs)
        break;
    } else if (CS.getReg() == BasePtrReg && SGPRForBPSaveRestoreCopy) {
      CS.setDstReg(SGPRForBPSaveRestoreCopy);
      if (--NumModifiedRegs)
        break;
    }
  }

  return false;
}

bool SIFrameLowering::allocateScavengingFrameIndexesNearIncomingSP(
  const MachineFunction &MF) const {

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  uint64_t EstStackSize = MFI.estimateStackSize(MF);
  uint64_t MaxOffset = EstStackSize - 1;

  // We need the emergency stack slots to be allocated in range of the
  // MUBUF/flat scratch immediate offset from the base register, so assign these
  // first at the incoming SP position.
  //
  // TODO: We could try sorting the objects to find a hole in the first bytes
  // rather than allocating as close to possible. This could save a lot of space
  // on frames with alignment requirements.
  if (ST.enableFlatScratch()) {
    const SIInstrInfo *TII = ST.getInstrInfo();
    if (TII->isLegalFLATOffset(MaxOffset, AMDGPUAS::PRIVATE_ADDRESS,
                               SIInstrFlags::FlatScratch))
      return false;
  } else {
    if (SIInstrInfo::isLegalMUBUFImmOffset(MaxOffset))
      return false;
  }

  return true;
}

MachineBasicBlock::iterator SIFrameLowering::eliminateCallFramePseudoInstr(
  MachineFunction &MF,
  MachineBasicBlock &MBB,
  MachineBasicBlock::iterator I) const {
  int64_t Amount = I->getOperand(0).getImm();
  if (Amount == 0)
    return MBB.erase(I);

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const DebugLoc &DL = I->getDebugLoc();
  unsigned Opc = I->getOpcode();
  bool IsDestroy = Opc == TII->getCallFrameDestroyOpcode();
  uint64_t CalleePopAmount = IsDestroy ? I->getOperand(1).getImm() : 0;

  if (!hasReservedCallFrame(MF)) {
    Amount = alignTo(Amount, getStackAlign());
    assert(isUInt<32>(Amount) && "exceeded stack address space size");
    const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
    Register SPReg = MFI->getStackPtrOffsetReg();

    Amount *= getScratchScaleFactor(ST);
    if (IsDestroy)
      Amount = -Amount;
    auto Add = BuildMI(MBB, I, DL, TII->get(AMDGPU::S_ADD_I32), SPReg)
        .addReg(SPReg)
        .addImm(Amount);
    Add->getOperand(3).setIsDead(); // Mark SCC as dead.
  } else if (CalleePopAmount != 0) {
    llvm_unreachable("is this used?");
  }

  return MBB.erase(I);
}

/// Returns true if the frame will require a reference to the stack pointer.
///
/// This is the set of conditions common to setting up the stack pointer in a
/// kernel, and for using a frame pointer in a callable function.
///
/// FIXME: Should also check hasOpaqueSPAdjustment and if any inline asm
/// references SP.
static bool frameTriviallyRequiresSP(const MachineFrameInfo &MFI) {
  return MFI.hasVarSizedObjects() || MFI.hasStackMap() || MFI.hasPatchPoint();
}

// The FP for kernels is always known 0, so we never really need to setup an
// explicit register for it. However, DisableFramePointerElim will force us to
// use a register for it.
bool SIFrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  // For entry functions we can use an immediate offset in most cases, so the
  // presence of calls doesn't imply we need a distinct frame pointer.
  if (MFI.hasCalls() &&
      !MF.getInfo<SIMachineFunctionInfo>()->isEntryFunction()) {
    // All offsets are unsigned, so need to be addressed in the same direction
    // as stack growth.

    // FIXME: This function is pretty broken, since it can be called before the
    // frame layout is determined or CSR spills are inserted.
    return MFI.getStackSize() != 0;
  }

  return frameTriviallyRequiresSP(MFI) || MFI.isFrameAddressTaken() ||
         MF.getSubtarget<GCNSubtarget>().getRegisterInfo()->hasStackRealignment(
             MF) ||
         MF.getTarget().Options.DisableFramePointerElim(MF);
}

// This is essentially a reduced version of hasFP for entry functions. Since the
// stack pointer is known 0 on entry to kernels, we never really need an FP
// register. We may need to initialize the stack pointer depending on the frame
// properties, which logically overlaps many of the cases where an ordinary
// function would require an FP.
bool SIFrameLowering::requiresStackPointerReference(
    const MachineFunction &MF) const {
  // Callable functions always require a stack pointer reference.
  assert(MF.getInfo<SIMachineFunctionInfo>()->isEntryFunction() &&
         "only expected to call this for entry points");

  const MachineFrameInfo &MFI = MF.getFrameInfo();

  // Entry points ordinarily don't need to initialize SP. We have to set it up
  // for callees if there are any. Also note tail calls are impossible/don't
  // make any sense for kernels.
  if (MFI.hasCalls())
    return true;

  // We still need to initialize the SP if we're doing anything weird that
  // references the SP, like variable sized stack objects.
  return frameTriviallyRequiresSP(MFI);
}

bool SIFrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    const ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  MachineFunction &MF = *MBB.getParent();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo *RI = ST.getRegisterInfo();
  const SIInstrInfo *TII = ST.getInstrInfo();

  for (const CalleeSavedInfo &CS : CSI) {
    // Insert the spill to the stack frame.
    unsigned Reg = CS.getReg();

    if (CS.isSpilledToReg()) {
      TII->buildCopy(MBB, MBBI, DebugLoc(), CS.getDstReg(), Reg,
                     getKillRegState(true));
    } else {
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(
          Reg, Reg == RI->getReturnAddressReg(MF) ? MVT::i64 : MVT::i32);
      const MachineRegisterInfo &MRI = MF.getRegInfo();
      // If this value was already livein, we probably have a direct use of the
      // incoming register value, so don't kill at the spill point. This happens
      // since we pass some special inputs (workgroup IDs) in the callee saved
      // range.
      const bool IsLiveIn = MRI.isLiveIn(Reg);
      TII->storeRegToStackSlotCFI(MBB, MBBI, Reg, !IsLiveIn, CS.getFrameIdx(),
                                  RC, TRI);
    }
  }

  return true;
}

MachineInstr *SIFrameLowering::buildCFI(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI,
                                        const DebugLoc &DL,
                                        const MCCFIInstruction &CFIInst,
                                        MachineInstr::MIFlag flag) const {
  MachineFunction &MF = *MBB.getParent();
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  return BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(MF.addFrameInst(CFIInst))
      .setMIFlag(flag);
}

MachineInstr *SIFrameLowering::buildCFIForRegToRegSpill(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    const DebugLoc &DL, const Register Reg, const Register RegCopy) const {
  MachineFunction &MF = *MBB.getParent();
  const MCRegisterInfo &MCRI = *MF.getMMI().getContext().getRegisterInfo();
  return buildCFI(
      MBB, MBBI, DL,
      MCCFIInstruction::createRegister(nullptr, MCRI.getDwarfRegNum(Reg, false),
                                       MCRI.getDwarfRegNum(RegCopy, false)));
}

MachineInstr *SIFrameLowering::buildCFIForSGPRToVGPRSpill(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    const DebugLoc &DL, const Register SGPR, const Register VGPR,
    const int Lane) const {
  const MachineFunction &MF = *MBB.getParent();
  const MCRegisterInfo &MCRI = *MF.getMMI().getContext().getRegisterInfo();

  int DwarfSGPR = MCRI.getDwarfRegNum(SGPR, false);
  int DwarfVGPR = MCRI.getDwarfRegNum(VGPR, false);
  assert(DwarfSGPR != -1 && DwarfVGPR != -1);

  // CFI for an SGPR spilled to a single lane of a VGPR is implemented as an
  // expression(E) rule where E is a register location description referencing
  // a VGPR register location storage at a byte offset of the lane index
  // multiplied by the size of an SGPR (4 bytes). In other words we generate
  // the following DWARF:
  //
  // DW_CFA_expression: <SGPR>,
  //    (DW_OP_regx <VGPR>) (DW_OP_LLVM_offset_uconst <Lane>*4)
  //
  // The memory location description for the current CFA is pushed on the
  // stack before E is evaluated, but we choose not to drop it as it would
  // require a longer expression E and DWARF defines the result of the
  // evaulation to be the location description on the top of the stack (i.e. the
  // implictly pushed one is just ignored.)

  SmallString<20> Block;
  raw_svector_ostream OSBlock(Block);
  encodeDwarfRegisterLocation(DwarfVGPR, OSBlock);
  OSBlock << uint8_t(dwarf::DW_OP_LLVM_offset_uconst);
  encodeULEB128(Lane * SGPRByteSize, OSBlock);

  SmallString<20> CFIInst;
  raw_svector_ostream OSCFIInst(CFIInst);
  OSCFIInst << uint8_t(dwarf::DW_CFA_expression);
  encodeULEB128(DwarfSGPR, OSCFIInst);
  encodeULEB128(Block.size(), OSCFIInst);
  OSCFIInst << Block;

  return buildCFI(MBB, MBBI, DL,
           MCCFIInstruction::createEscape(nullptr, OSCFIInst.str()));
}

MachineInstr *SIFrameLowering::buildCFIForSGPRToVGPRSpill(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    const DebugLoc &DL, Register SGPR,
    ArrayRef<SIRegisterInfo::SpilledReg> VGPRSpills) const {
  const MachineFunction &MF = *MBB.getParent();
  const MCRegisterInfo &MCRI = *MF.getMMI().getContext().getRegisterInfo();

  int DwarfSGPR = MCRI.getDwarfRegNum(SGPR, false);
  assert(DwarfSGPR != -1);

  // CFI for an SGPR spilled to a multiple lanes of VGPRs is implemented as an
  // expression(E) rule where E is a composite location description
  // with multiple parts each referencing
  // VGPR register location storage with a bit offset of the lane index
  // multiplied by the size of an SGPR (32 bits). In other words we generate
  // the following DWARF:
  //
  // DW_CFA_expression: <SGPR>,
  //    (DW_OP_regx <VGPR[0]>) (DW_OP_bit_piece 32, <Lane[0]>*32)
  //    (DW_OP_regx <VGPR[1]>) (DW_OP_bit_piece 32, <Lane[1]>*32)
  //    ...
  //    (DW_OP_regx <VGPR[N]>) (DW_OP_bit_piece 32, <Lane[N]>*32)
  //
  // The memory location description for the current CFA is pushed on the
  // stack before E is evaluated, but we choose not to drop it as it would
  // require a longer expression E and DWARF defines the result of the
  // evaulation to be the location description on the top of the stack (i.e. the
  // implictly pushed one is just ignored.)

  SmallString<20> Block;
  raw_svector_ostream OSBlock(Block);
  // TODO: Detect when we can merge multiple adjacent pieces, or even reduce
  // this to a register location description (when all pieces are adjacent).
  for (SIRegisterInfo::SpilledReg Spill : VGPRSpills) {
    encodeDwarfRegisterLocation(MCRI.getDwarfRegNum(Spill.VGPR, false),
                                OSBlock);
    OSBlock << uint8_t(dwarf::DW_OP_bit_piece);
    encodeULEB128(SGPRBitSize, OSBlock);
    encodeULEB128(SGPRBitSize * Spill.Lane, OSBlock);
  }

  SmallString<20> CFIInst;
  raw_svector_ostream OSCFIInst(CFIInst);
  OSCFIInst << uint8_t(dwarf::DW_CFA_expression);
  encodeULEB128(DwarfSGPR, OSCFIInst);
  encodeULEB128(Block.size(), OSCFIInst);
  OSCFIInst << Block;

  return buildCFI(MBB, MBBI, DL,
                  MCCFIInstruction::createEscape(nullptr, OSCFIInst.str()));
}

MachineInstr *SIFrameLowering::buildCFIForSGPRToVMEMSpill(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    const DebugLoc &DL, unsigned SGPR, int64_t Offset) const {
  MachineFunction &MF = *MBB.getParent();
  const MCRegisterInfo &MCRI = *MF.getMMI().getContext().getRegisterInfo();
  return buildCFI(MBB, MBBI, DL,
                  llvm::MCCFIInstruction::createOffset(
                      nullptr, MCRI.getDwarfRegNum(SGPR, false), Offset));
}

MachineInstr *SIFrameLowering::buildCFIForVGPRToVMEMSpill(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    const DebugLoc &DL, unsigned VGPR, int64_t Offset) const {
  const MachineFunction &MF = *MBB.getParent();
  const MCRegisterInfo &MCRI = *MF.getMMI().getContext().getRegisterInfo();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  int DwarfVGPR = MCRI.getDwarfRegNum(VGPR, false);
  assert(DwarfVGPR != -1);

  SmallString<20> Block;
  raw_svector_ostream OSBlock(Block);
  encodeDwarfRegisterLocation(DwarfVGPR, OSBlock);
  OSBlock << uint8_t(dwarf::DW_OP_swap);
  OSBlock << uint8_t(dwarf::DW_OP_LLVM_offset_uconst);
  encodeULEB128(Offset, OSBlock);
  OSBlock << uint8_t(dwarf::DW_OP_LLVM_call_frame_entry_reg);
  encodeULEB128(MCRI.getDwarfRegNum(
                    ST.isWave32() ? AMDGPU::EXEC_LO : AMDGPU::EXEC, false),
                OSBlock);
  OSBlock << uint8_t(dwarf::DW_OP_deref_size);
  OSBlock << uint8_t(ST.getWavefrontSize() / 8);
  OSBlock << uint8_t(dwarf::DW_OP_LLVM_select_bit_piece);
  encodeULEB128(VGPRLaneBitSize, OSBlock);
  encodeULEB128(ST.getWavefrontSize(), OSBlock);

  SmallString<20> CFIInst;
  raw_svector_ostream OSCFIInst(CFIInst);
  OSCFIInst << uint8_t(dwarf::DW_CFA_expression);
  encodeULEB128(DwarfVGPR, OSCFIInst);
  encodeULEB128(Block.size(), OSCFIInst);
  OSCFIInst << Block;

  return buildCFI(MBB, MBBI, DL,
                  MCCFIInstruction::createEscape(nullptr, OSCFIInst.str()));
}

MachineInstr *SIFrameLowering::buildCFIForRegToSGPRPairSpill(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    const DebugLoc &DL, const Register Reg, const Register SGPRPair) const {
  const MachineFunction &MF = *MBB.getParent();
  const MCRegisterInfo &MCRI = *MF.getMMI().getContext().getRegisterInfo();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIRegisterInfo &TRI = ST.getInstrInfo()->getRegisterInfo();

  int SGPR0 = TRI.getSubReg(SGPRPair, AMDGPU::sub0);
  int SGPR1 = TRI.getSubReg(SGPRPair, AMDGPU::sub1);

  int DwarfReg = MCRI.getDwarfRegNum(Reg, false);
  int DwarfSGPR0 = MCRI.getDwarfRegNum(SGPR0, false);
  int DwarfSGPR1 = MCRI.getDwarfRegNum(SGPR1, false);
  assert(DwarfReg != -1 && DwarfSGPR0 != 1 && DwarfSGPR1 != 1);

  // CFI for a register spilled to a pair of SGPRs is implemented as an
  // expression(E) rule where E is a composite location description with
  // multiple parts each referencing SGPR register location storage with a bit
  // offset of 0. In other words we generate the following DWARF:
  //
  // DW_CFA_expression: <Reg>,
  //    (DW_OP_regx <SGPRPair[0]>) (DW_OP_piece 4)
  //    (DW_OP_regx <SGPRPair[1]>) (DW_OP_piece 4)
  //
  // The memory location description for the current CFA is pushed on the stack
  // before E is evaluated, but we choose not to drop it as it would require a
  // longer expression E and DWARF defines the result of the evaulation to be
  // the location description on the top of the stack (i.e. the implictly
  // pushed one is just ignored.)

  SmallString<10> Block;
  raw_svector_ostream OSBlock(Block);
  encodeDwarfRegisterLocation(DwarfSGPR0, OSBlock);
  OSBlock << uint8_t(dwarf::DW_OP_piece);
  encodeULEB128(SGPRByteSize, OSBlock);
  encodeDwarfRegisterLocation(DwarfSGPR1, OSBlock);
  OSBlock << uint8_t(dwarf::DW_OP_piece);
  encodeULEB128(SGPRByteSize, OSBlock);

  SmallString<20> CFIInst;
  raw_svector_ostream OSCFIInst(CFIInst);
  OSCFIInst << uint8_t(dwarf::DW_CFA_expression);
  encodeULEB128(DwarfReg, OSCFIInst);
  encodeULEB128(Block.size(), OSCFIInst);
  OSCFIInst << Block;

  return buildCFI(MBB, MBBI, DL,
                  MCCFIInstruction::createEscape(nullptr, OSCFIInst.str()));
}
