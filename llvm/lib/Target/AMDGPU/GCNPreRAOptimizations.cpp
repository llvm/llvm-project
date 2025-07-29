//===-- GCNPreRAOptimizations.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass combines split register tuple initialization into a single pseudo:
///
///   undef %0.sub1:sreg_64 = S_MOV_B32 1
///   %0.sub0:sreg_64 = S_MOV_B32 2
/// =>
///   %0:sreg_64 = S_MOV_B64_IMM_PSEUDO 0x200000001
///
/// This is to allow rematerialization of a value instead of spilling. It is
/// supposed to be done after register coalescer to allow it to do its job and
/// before actual register allocation to allow rematerialization.
///
/// Right now the pass only handles 64 bit SGPRs with immediate initializers,
/// although the same shall be possible with other register classes and
/// instructions if necessary.
///
/// This pass also adds register allocation hints to COPY.
/// The hints will be post-processed by SIRegisterInfo::getRegAllocationHints.
/// When using True16, we often see COPY moving a 16-bit value between a VGPR_32
/// and a VGPR_16. If we use the VGPR_16 that corresponds to the lo16 bits of
/// the VGPR_32, the COPY can be completely eliminated.
///
//===----------------------------------------------------------------------===//

#include "GCNPreRAOptimizations.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

#include "AMDGPURegisterBankInfo.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"

#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/InitializePasses.h"
#include <unordered_set>

#include "GCNSchedStrategy.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineScheduler.h"
using namespace llvm;

#define DEBUG_TYPE "amdgpu-pre-ra-optimizations"

namespace {

class GCNPreRAOptimizationsImpl {
private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;

  bool processReg(Register Reg);
  bool unpackInsts(MachineFunction &MF);
  bool createListOfPackedInstr(MachineInstr &BeginMI, std::unordered_set<MachineInstr *> &seen);
  bool isNeverCoissue(MachineInstr &MI, MachineFunction *MF) const;
  bool isUnpackingSupportedInstr(MachineInstr &MI) const;
  void insertMI(MachineInstr &I);
  SmallVector<MachineInstr *, 2> copyToVregAndInsertMI(MachineInstr &I,
                                                       unsigned SGPRSrcPos);
  SmallVector<MachineInstr *, 2>
  insertUnpackedMI(MachineInstr &I, MachineOperand &DstMO, MachineOperand &LoSrcMO1,
                   MachineOperand &LoSrcMO2, MachineOperand &HiSrcMO1, MachineOperand &HiSrcMO2,
                   bool isVreg_64);

public:
  GCNPreRAOptimizationsImpl(LiveIntervals *LS) : LIS(LS) {}
  bool run(MachineFunction &MF);
};

class GCNPreRAOptimizationsLegacy : public MachineFunctionPass {
public:
  static char ID;
  const MachineLoopInfo *MLI = nullptr;

  GCNPreRAOptimizationsLegacy() : MachineFunctionPass(ID) {
    initializeGCNPreRAOptimizationsLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Pre-RA optimizations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(GCNPreRAOptimizationsLegacy, DEBUG_TYPE,
                      "AMDGPU Pre-RA optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(GCNPreRAOptimizationsLegacy, DEBUG_TYPE,
                    "Pre-RA optimizations", false, false)

char GCNPreRAOptimizationsLegacy::ID = 0;

char &llvm::GCNPreRAOptimizationsID = GCNPreRAOptimizationsLegacy::ID;

FunctionPass *llvm::createGCNPreRAOptimizationsLegacyPass() {
  return new GCNPreRAOptimizationsLegacy();
}

bool GCNPreRAOptimizationsImpl::processReg(Register Reg) {
  MachineInstr *Def0 = nullptr;
  MachineInstr *Def1 = nullptr;
  uint64_t Init = 0;
  bool Changed = false;
  SmallSet<Register, 32> ModifiedRegs;
  bool IsAGPRDst = TRI->isAGPRClass(MRI->getRegClass(Reg));

  for (MachineInstr &I : MRI->def_instructions(Reg)) {
    switch (I.getOpcode()) {
    default:
      return false;
    case AMDGPU::V_ACCVGPR_WRITE_B32_e64:
      break;
    case AMDGPU::COPY: {
      // Some subtargets cannot do an AGPR to AGPR copy directly, and need an
      // intermdiate temporary VGPR register. Try to find the defining
      // accvgpr_write to avoid temporary registers.

      if (!IsAGPRDst)
        return false;

      Register SrcReg = I.getOperand(1).getReg();

      if (!SrcReg.isVirtual())
        break;

      // Check if source of copy is from another AGPR.
      bool IsAGPRSrc = TRI->isAGPRClass(MRI->getRegClass(SrcReg));
      if (!IsAGPRSrc)
        break;

      // def_instructions() does not look at subregs so it may give us a
      // different instruction that defines the same vreg but different subreg
      // so we have to manually check subreg.
      Register SrcSubReg = I.getOperand(1).getSubReg();
      for (auto &Def : MRI->def_instructions(SrcReg)) {
        if (SrcSubReg != Def.getOperand(0).getSubReg())
          continue;

        if (Def.getOpcode() == AMDGPU::V_ACCVGPR_WRITE_B32_e64) {
          MachineOperand DefSrcMO = Def.getOperand(1);

          // Immediates are not an issue and can be propagated in
          // postrapseudos pass. Only handle cases where defining
          // accvgpr_write source is a vreg.
          if (DefSrcMO.isReg() && DefSrcMO.getReg().isVirtual()) {
            // Propagate source reg of accvgpr write to this copy instruction
            I.getOperand(1).setReg(DefSrcMO.getReg());
            I.getOperand(1).setSubReg(DefSrcMO.getSubReg());

            // Reg uses were changed, collect unique set of registers to update
            // live intervals at the end.
            ModifiedRegs.insert(DefSrcMO.getReg());
            ModifiedRegs.insert(SrcReg);

            Changed = true;
          }

          // Found the defining accvgpr_write, stop looking any further.
          break;
        }
      }
      break;
    }
    case AMDGPU::S_MOV_B32:
      if (I.getOperand(0).getReg() != Reg || !I.getOperand(1).isImm() ||
          I.getNumOperands() != 2)
        return false;

      switch (I.getOperand(0).getSubReg()) {
      default:
        return false;
      case AMDGPU::sub0:
        if (Def0)
          return false;
        Def0 = &I;
        Init |= Lo_32(I.getOperand(1).getImm());
        break;
      case AMDGPU::sub1:
        if (Def1)
          return false;
        Def1 = &I;
        Init |= static_cast<uint64_t>(I.getOperand(1).getImm()) << 32;
        break;
      }
      break;
    }
  }

  // For AGPR reg, check if live intervals need to be updated.
  if (IsAGPRDst) {
    if (Changed) {
      for (Register RegToUpdate : ModifiedRegs) {
        LIS->removeInterval(RegToUpdate);
        LIS->createAndComputeVirtRegInterval(RegToUpdate);
      }
    }

    return Changed;
  }

  // For SGPR reg, check if we can combine instructions.
  if (!Def0 || !Def1 || Def0->getParent() != Def1->getParent())
    return Changed;

  LLVM_DEBUG(dbgs() << "Combining:\n  " << *Def0 << "  " << *Def1
                    << "    =>\n");

  if (SlotIndex::isEarlierInstr(LIS->getInstructionIndex(*Def1),
                                LIS->getInstructionIndex(*Def0)))
    std::swap(Def0, Def1);

  LIS->RemoveMachineInstrFromMaps(*Def0);
  LIS->RemoveMachineInstrFromMaps(*Def1);
  auto NewI = BuildMI(*Def0->getParent(), *Def0, Def0->getDebugLoc(),
                      TII->get(AMDGPU::S_MOV_B64_IMM_PSEUDO), Reg)
                  .addImm(Init);

  Def0->eraseFromParent();
  Def1->eraseFromParent();
  LIS->InsertMachineInstrInMaps(*NewI);
  LIS->removeInterval(Reg);
  LIS->createAndComputeVirtRegInterval(Reg);

  LLVM_DEBUG(dbgs() << "  " << *NewI);

  return true;
}

bool GCNPreRAOptimizationsImpl::isNeverCoissue(MachineInstr &MI, MachineFunction *MF) const {
  const GCNSubtarget &ST = MF->getSubtarget<GCNSubtarget>();
  // bool IsGFX942Only = ST.hasGFX940Insts() && !ST.hasGFX950Insts();
  // if (!IsGFX942Only)
  //   return false;

  if (!SIInstrInfo::isVALU(MI)){
    return false;
  }


  // V_COS, V_EXP, V_RCP, etc.
  if (SIInstrInfo::isTRANS(MI))
    return true;

  // DOT2, DOT2C, DOT4, etc.
  if (SIInstrInfo::isDOT(MI))
    return true;

  // MFMA, SMFMA
  if (SIInstrInfo::isMFMA(MI))
    return true;

  unsigned Opcode = MI.getOpcode();
  switch (Opcode) {
  case AMDGPU::V_CVT_PK_BF8_F32_e64:
  case AMDGPU::V_CVT_PK_FP8_F32_e64:
  case AMDGPU::V_MQSAD_PK_U16_U8_e64:
  case AMDGPU::V_MQSAD_U32_U8_e64:
  case AMDGPU::V_PK_ADD_F16:
  case AMDGPU::V_PK_ADD_F32:
  case AMDGPU::V_PK_ADD_I16:
  case AMDGPU::V_PK_ADD_U16:
  case AMDGPU::V_PK_ASHRREV_I16:
  case AMDGPU::V_PK_FMA_F16:
  case AMDGPU::V_PK_FMA_F32:
  case AMDGPU::V_PK_FMAC_F16_e32:
  case AMDGPU::V_PK_FMAC_F16_e64:
  case AMDGPU::V_PK_LSHLREV_B16:
  case AMDGPU::V_PK_LSHRREV_B16:
  case AMDGPU::V_PK_MAD_I16:
  case AMDGPU::V_PK_MAD_U16:
  case AMDGPU::V_PK_MAX_F16:
  case AMDGPU::V_PK_MAX_I16:
  case AMDGPU::V_PK_MAX_U16:
  case AMDGPU::V_PK_MIN_F16:
  case AMDGPU::V_PK_MIN_I16:
  case AMDGPU::V_PK_MIN_U16:
  case AMDGPU::V_PK_MOV_B32:
  case AMDGPU::V_PK_MUL_F16:
  case AMDGPU::V_PK_MUL_F32:
  case AMDGPU::V_PK_MUL_LO_U16:
  case AMDGPU::V_PK_SUB_I16:
  case AMDGPU::V_PK_SUB_U16:
  case AMDGPU::V_QSAD_PK_U16_U8_e64:
    return true;

  default:
    return false;

  }
}

bool GCNPreRAOptimizationsImpl::isUnpackingSupportedInstr(MachineInstr &MI) const {
  unsigned Opcode = MI.getOpcode();
  switch (Opcode) {
  case AMDGPU::V_PK_ADD_F16:
  case AMDGPU::V_PK_ADD_F32:
  case AMDGPU::V_PK_MUL_F16:
  case AMDGPU::V_PK_MUL_F32:
    return true;

  default:
    return false;

  }
}

SmallVector<MachineInstr *, 2>
GCNPreRAOptimizationsImpl::copyToVregAndInsertMI(MachineInstr &I,
                                                   unsigned SGPRSrcPos) {
  SmallVector<MachineInstr *, 2> MIList;

  MachineBasicBlock &MBB = *I.getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  MachineFunction &MF = *MBB.getParent();
  const DebugLoc &DL = I.getDebugLoc();

  Register TmpReg = MRI.createVirtualRegister(&AMDGPU::VReg_64_Align2RegClass);
  MachineInstr *CopySGPR1 =
      BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY))
          .addDef(TmpReg, RegState::Undef)
          .addReg(I.getOperand(SGPRSrcPos).getReg(), 0, AMDGPU::sub0);
  unsigned SubIdx = TRI->composeSubRegIndices(
      AMDGPU::sub0, CopySGPR1->getOperand(0).getSubReg());
  CopySGPR1->getOperand(0).setReg(CopySGPR1->getOperand(0).getReg());
  CopySGPR1->getOperand(0).setSubReg(SubIdx);
  LIS->InsertMachineInstrInMaps(*CopySGPR1);
  MIList.push_back(CopySGPR1);

  MachineInstr *CopySGPR2 =
      BuildMI(MBB, I, DL, TII->get(AMDGPU::COPY))
          .addDef(TmpReg)
          .addReg(I.getOperand(SGPRSrcPos).getReg(), 0, AMDGPU::sub1);
  SubIdx = TRI->composeSubRegIndices(AMDGPU::sub1,
                                     CopySGPR2->getOperand(0).getSubReg());
  CopySGPR2->getOperand(0).setReg(CopySGPR2->getOperand(0).getReg());
  CopySGPR2->getOperand(0).setSubReg(SubIdx);
  LIS->InsertMachineInstrInMaps(*CopySGPR2);
  MIList.push_back(CopySGPR2);
  return MIList;
}

bool GCNPreRAOptimizationsImpl::createListOfPackedInstr(
    MachineInstr &BeginMI, std::unordered_set<MachineInstr *> &seen) {
  auto *BB = BeginMI.getParent();
  auto *MF = BB->getParent();
  int NumInst = 0;

  auto E = BB->end();
  auto schedModel = TII->getSchedModel();
  const MCSchedClassDesc *schedClassDesc = schedModel.resolveSchedClass(&BeginMI);
  const int NumMFMACycles = schedModel.getWriteProcResBegin(schedClassDesc)->ReleaseAtCycle;
  int totalCyclesBetweenCandidates = 0;
  for (auto I = std::next(BeginMI.getIterator()); I != E; ++I) {
    MachineInstr &Instr = *I;
    const MCSchedClassDesc *instrSchedClassDesc = schedModel.resolveSchedClass(&Instr);
    totalCyclesBetweenCandidates += schedModel.getWriteProcResBegin(instrSchedClassDesc)->ReleaseAtCycle;
    if (Instr.isMetaInstruction())
      continue;

    if (Instr.isTerminator())
      return false;

    if (totalCyclesBetweenCandidates > NumMFMACycles)
      return false;

    if ((Instr.getOpcode() == AMDGPU::V_PK_MUL_F32) && isNeverCoissue(Instr, Instr.getParent()->getParent())) {
      totalCyclesBetweenCandidates += 1;
      seen.insert(&Instr);
    }
  }
  return true;
}

SmallVector<MachineInstr *, 2> GCNPreRAOptimizationsImpl::insertUnpackedMI(
    MachineInstr &I, MachineOperand &DstMO, MachineOperand &LoSrcMO1, MachineOperand &LoSrcMO2,
    MachineOperand &HiSrcMO1, MachineOperand &HiSrcMO2, bool isVreg_64) {

  SmallVector<MachineInstr *, 2> MIList;
  MachineBasicBlock &MBB = *I.getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  MachineFunction &MF = *MBB.getParent();
  const DebugLoc &DL = I.getDebugLoc();
  Register DstReg = DstMO.getReg();

  unsigned SrcSubIdx1 =
      TRI->composeSubRegIndices(LoSrcMO1.getSubReg(), AMDGPU::sub0);
  unsigned SrcSubIdx2 =
      TRI->composeSubRegIndices(LoSrcMO2.getSubReg(), AMDGPU::sub0);
  unsigned DestSubIdx =
      TRI->composeSubRegIndices(DstMO.getSubReg(), AMDGPU::sub0);

  const MCInstrDesc instrDesc = I.getDesc();

  int clampIdx = AMDGPU::getNamedOperandIdx(I.getOpcode(), AMDGPU::OpName::clamp);
  int64_t clampVal = I.getOperand(clampIdx).getImm();

  int src0_modifiers_Idx = AMDGPU::getNamedOperandIdx(I.getOpcode(), AMDGPU::OpName::src0_modifiers);
  int src1_modifiers_Idx = AMDGPU::getNamedOperandIdx(I.getOpcode(), AMDGPU::OpName::src1_modifiers);
  unsigned src0_Mods = I.getOperand(src0_modifiers_Idx).getImm();
  unsigned src1_Mods = I.getOperand(src1_modifiers_Idx).getImm();

  //don't worry about abs values. Packed instructions (VOP3P) do not support them
  unsigned Lo_src0_mods = 0;
  unsigned Lo_src1_mods = 0;

  MachineInstrBuilder Op0L_Op1L = BuildMI(MBB, I, DL, TII->get(AMDGPU::V_MUL_F32_e64));
  Op0L_Op1L.addDef(DstReg, 0, DestSubIdx); //vdst
  if (src0_Mods & SISrcMods::OP_SEL_0) {
    if (src0_Mods & SISrcMods::NEG) {
      Lo_src0_mods |= SISrcMods::NEG;
    }
    Op0L_Op1L.addImm(Lo_src0_mods); //src0_modifiers
    unsigned Src0SubIdx = TRI->composeSubRegIndices(LoSrcMO1.getSubReg(), AMDGPU::sub1);
    Op0L_Op1L.addReg(LoSrcMO1.getReg(), 0, Src0SubIdx); //src0
  }
  else {
    Op0L_Op1L.addImm(Lo_src0_mods); //src0_modifiers
    unsigned Src0SubIdx = TRI->composeSubRegIndices(LoSrcMO1.getSubReg(), AMDGPU::sub0);
    Op0L_Op1L.addReg(LoSrcMO1.getReg(), 0, Src0SubIdx); //src0 //if op_sel == 0, select register 0 of reg:sub0_sub1
  }

  if (src1_Mods & SISrcMods::OP_SEL_0) {
    if (src1_Mods & SISrcMods::NEG) {
      Lo_src1_mods |= SISrcMods::NEG;
    }
    Op0L_Op1L.addImm(Lo_src1_mods); //src0_modifiers
    unsigned Src1SubIdx = TRI->composeSubRegIndices(LoSrcMO2.getSubReg(), AMDGPU::sub1);
    Op0L_Op1L.addReg(LoSrcMO2.getReg(), 0, Src1SubIdx); //src0
  }
  else {
    Op0L_Op1L.addImm(Lo_src1_mods); //src0_modifiers
    unsigned Src1SubIdx = TRI->composeSubRegIndices(LoSrcMO2.getSubReg(), AMDGPU::sub0);
    Op0L_Op1L.addReg(LoSrcMO2.getReg(), 0, Src1SubIdx); //src0 //if op_sel_hi == 0, select register 0 of reg:sub0_sub1
  }
  Op0L_Op1L.addImm(clampVal); //clamp
  //packed instructions do not support output modifiers. safe to assign them 0 for this use case
  Op0L_Op1L.addImm(0); //omod

  if (isVreg_64) {
    Op0L_Op1L->getOperand(0).setIsUndef();
  }
  else {
    if (I.getOperand(0).isUndef()) {
      Op0L_Op1L->getOperand(0).setIsUndef();
    }
  }

  LIS->InsertMachineInstrInMaps(*Op0L_Op1L);

  SrcSubIdx1 =
      TRI->composeSubRegIndices(LoSrcMO1.getSubReg(), AMDGPU::sub1);
  SrcSubIdx2 =
      TRI->composeSubRegIndices(LoSrcMO2.getSubReg(), AMDGPU::sub1);
  DestSubIdx =
      TRI->composeSubRegIndices(DstMO.getSubReg(), AMDGPU::sub1);

  //don't worry about abs values. Packed instructions (VOP3P) do not support them
  unsigned Hi_src0_mods = 0;
  unsigned Hi_src1_mods = 0;

  MachineInstrBuilder Op0H_Op1H = BuildMI(MBB, I, DL, TII->get(AMDGPU::V_MUL_F32_e64));
  Op0H_Op1H.addDef(DstReg, 0, DestSubIdx); //vdst
  if (src0_Mods & SISrcMods::OP_SEL_1) {
    if (src0_Mods & SISrcMods::NEG_HI) {
      Hi_src0_mods |= SISrcMods::NEG;
    }
    Op0H_Op1H.addImm(Hi_src0_mods); //src0_modifiers
    unsigned Src0SubIdx = TRI->composeSubRegIndices(HiSrcMO1.getSubReg(), AMDGPU::sub1);
    Op0H_Op1H.addReg(HiSrcMO1.getReg(), 0, Src0SubIdx); //src0
  }
  else {
    Op0H_Op1H.addImm(Hi_src0_mods); //src0_modifiers
    unsigned Src0SubIdx = TRI->composeSubRegIndices(HiSrcMO1.getSubReg(), AMDGPU::sub0);
    Op0H_Op1H.addReg(HiSrcMO1.getReg(), 0, Src0SubIdx); //src0 //if op_sel_hi == 0, select register 0 of reg:sub0_sub1
  }

  if (src1_Mods & SISrcMods::OP_SEL_1) {
    if (src1_Mods & SISrcMods::NEG_HI) {
      Hi_src1_mods |= SISrcMods::NEG;
    }
    Op0H_Op1H.addImm(Hi_src1_mods); //src0_modifiers
    unsigned Src1SubIdx = TRI->composeSubRegIndices(HiSrcMO2.getSubReg(), AMDGPU::sub1);
    Op0H_Op1H.addReg(HiSrcMO2.getReg(), 0, Src1SubIdx); //src0
  }
  else {
    Op0H_Op1H.addImm(Hi_src1_mods); //src0_modifiers
    unsigned Src1SubIdx = TRI->composeSubRegIndices(HiSrcMO2.getSubReg(), AMDGPU::sub0);
    Op0H_Op1H.addReg(HiSrcMO2.getReg(), 0, Src1SubIdx); //src0 //if op_sel_hi == 0, select register 0 of reg:sub0_sub1
  }
  Op0H_Op1H.addImm(clampVal); //clamp
  //packed instructions do not support output modifiers. safe to assign them 0 for this use case
  Op0H_Op1H.addImm(0); //omod
  LIS->InsertMachineInstrInMaps(*Op0H_Op1H);

  if (I.getFlag(MachineInstr::MIFlag::NoFPExcept)) {
    Op0L_Op1L->setFlag(MachineInstr::MIFlag::NoFPExcept);
    Op0H_Op1H->setFlag(MachineInstr::MIFlag::NoFPExcept);
  }
  LIS->RemoveMachineInstrFromMaps(I);
  I.eraseFromParent();
  LIS->removeInterval(DstReg);
  LIS->createAndComputeVirtRegInterval(DstReg);
  MIList.push_back(Op0L_Op1L);
  MIList.push_back(Op0H_Op1H);
  return MIList;
}

void GCNPreRAOptimizationsImpl::insertMI(MachineInstr &I) {
  MachineBasicBlock &MBB = *I.getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  MachineFunction &MF = *MBB.getParent();

  Register DstReg = I.getOperand(0).getReg();
  Register SrcReg1 = I.getOperand(2).getReg();
  Register SrcReg2 = I.getOperand(4).getReg();

  MachineOperand &DstMO = I.getOperand(0);
  MachineOperand &SrcMO1 = I.getOperand(2);
  MachineOperand &SrcMO2 = I.getOperand(4);

  MachineBasicBlock::iterator MII = I;
  const DebugLoc &DL = I.getDebugLoc();
  const TargetRegisterClass *DstRC = MRI.getRegClass(I.getOperand(0).getReg());
  const TargetRegisterClass *Src0RC = MRI.getRegClass(I.getOperand(2).getReg());
  const TargetRegisterClass *Src1RC = MRI.getRegClass(I.getOperand(4).getReg());
  const TargetRegisterClass *Src0SubRC =
      TRI->getSubRegisterClass(Src0RC, AMDGPU::sub0);
  const TargetRegisterClass *SrcRC = TRI->getSubClassWithSubReg(Src0RC, 1);

  if ((Src1RC->getID() == AMDGPU::SGPR_64RegClassID) ||
      (Src0RC->getID() == AMDGPU::SGPR_64RegClassID)) {
    if (Src1RC->getID() == AMDGPU::SGPR_64RegClassID) {
      // try with sgpr32
      SmallVector<MachineInstr *, 2> copyInstrs = copyToVregAndInsertMI(I, 4);
      MachineInstr *CopySGPR1 = copyInstrs[0];
      MachineInstr *CopySGPR2 = copyInstrs[1];

      if (DstRC->getID() == AMDGPU::VReg_64_Align2RegClassID) {
        SmallVector<MachineInstr *, 2> unpackedInstrs = insertUnpackedMI(
            I, DstMO, SrcMO1, CopySGPR1->getOperand(0), SrcMO1,
            CopySGPR2->getOperand(0), true);
        unpackedInstrs[0]->addRegisterKilled(unpackedInstrs[0]->getOperand(2).getReg(), TRI);
        unpackedInstrs[1]->addRegisterKilled(unpackedInstrs[1]->getOperand(2).getReg(), TRI);
      } else {
        SmallVector<MachineInstr *, 2> unpackedInstrs = insertUnpackedMI(
            I, DstMO, SrcMO1, CopySGPR1->getOperand(0), SrcMO1,
            CopySGPR2->getOperand(0), false);
        unpackedInstrs[0]->addRegisterKilled(unpackedInstrs[0]->getOperand(2).getReg(), TRI);
        unpackedInstrs[1]->addRegisterKilled(unpackedInstrs[1]->getOperand(2).getReg(), TRI);
      }
    }
    else {
      SmallVector<MachineInstr *, 2> copyInstrs = copyToVregAndInsertMI(I, 2);
      MachineInstr *CopySGPR1 = copyInstrs[0];
      MachineInstr *CopySGPR2 = copyInstrs[1];

      if (DstRC->getID() == AMDGPU::VReg_64_Align2RegClassID) {
        SmallVector<MachineInstr *, 2> unpackedInstrs = insertUnpackedMI(
            I, DstMO, CopySGPR1->getOperand(0), SrcMO2, CopySGPR2->getOperand(0), SrcMO2, true);
        unpackedInstrs[0]->addRegisterKilled(unpackedInstrs[0]->getOperand(1).getReg(), TRI);
        unpackedInstrs[1]->addRegisterKilled(unpackedInstrs[1]->getOperand(1).getReg(), TRI);
      } else {
        SmallVector<MachineInstr *, 2> unpackedInstrs = insertUnpackedMI(
            I, DstMO, CopySGPR1->getOperand(0), SrcMO2, CopySGPR2->getOperand(0), SrcMO2, false);
        unpackedInstrs[0]->addRegisterKilled(unpackedInstrs[0]->getOperand(1).getReg(), TRI);
        unpackedInstrs[1]->addRegisterKilled(unpackedInstrs[1]->getOperand(1).getReg(), TRI);
      }
    }
    return;
  }

  if (DstRC->getID() == AMDGPU::VReg_512_Align2RegClassID) {
    SmallVector<MachineInstr *, 2> unpackedInstrs = insertUnpackedMI(
            I, DstMO, SrcMO1, SrcMO2, SrcMO1,
            SrcMO2, false);
  }
  else if (DstRC->getID() == AMDGPU::VReg_64_Align2RegClassID) {
    SmallVector<MachineInstr *, 2> unpackedInstrs = insertUnpackedMI(
            I, DstMO, SrcMO1, SrcMO2, SrcMO1,
            SrcMO2, true);
  }
  return;
}

bool GCNPreRAOptimizationsImpl::unpackInsts(MachineFunction &MF) {

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();

  auto schedModel = TII->getSchedModel();
  for (MachineBasicBlock &MBB : MF) {
    std::unordered_set<MachineInstr *> seen;
    for (MachineInstr &MI : MBB) {
      if (SIInstrInfo::isMFMA(MI)){
        createListOfPackedInstr(MI, seen);
      }

    }
    if (!seen.empty()) {
      for (MachineInstr *MI : seen) 
        insertMI(*MI);
    }
  }
  return true;
}

bool GCNPreRAOptimizationsLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  return GCNPreRAOptimizationsImpl(LIS).run(MF);
}

PreservedAnalyses
GCNPreRAOptimizationsPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  LiveIntervals *LIS = &MFAM.getResult<LiveIntervalsAnalysis>(MF);
  GCNPreRAOptimizationsImpl(LIS).run(MF);
  return PreservedAnalyses::all();
}

bool GCNPreRAOptimizationsImpl::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = ST.getRegisterInfo();

  bool Changed = false;

  Changed = unpackInsts(MF);
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (!LIS->hasInterval(Reg))
      continue;
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    if ((RC->MC->getSizeInBits() != 64 || !TRI->isSGPRClass(RC)) &&
        (ST.hasGFX90AInsts() || !TRI->isAGPRClass(RC)))
      continue;

    Changed |= processReg(Reg);
  }

  if (!ST.useRealTrue16Insts())
    return Changed;

  // Add RA hints to improve True16 COPY elimination.
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.getOpcode() != AMDGPU::COPY)
        continue;
      Register Dst = MI.getOperand(0).getReg();
      Register Src = MI.getOperand(1).getReg();
      if (Dst.isVirtual() &&
          MRI->getRegClass(Dst) == &AMDGPU::VGPR_16RegClass &&
          Src.isPhysical() &&
          TRI->getRegClassForReg(*MRI, Src) == &AMDGPU::VGPR_32RegClass)
        MRI->setRegAllocationHint(Dst, 0, TRI->getSubReg(Src, AMDGPU::lo16));
      if (Src.isVirtual() &&
          MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass &&
          Dst.isPhysical() &&
          TRI->getRegClassForReg(*MRI, Dst) == &AMDGPU::VGPR_32RegClass)
        MRI->setRegAllocationHint(Src, 0, TRI->getSubReg(Dst, AMDGPU::lo16));
      if (!Dst.isVirtual() || !Src.isVirtual())
        continue;
      if (MRI->getRegClass(Dst) == &AMDGPU::VGPR_32RegClass &&
          MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass) {
        MRI->setRegAllocationHint(Dst, AMDGPURI::Size32, Src);
        MRI->setRegAllocationHint(Src, AMDGPURI::Size16, Dst);
      }
      if (MRI->getRegClass(Dst) == &AMDGPU::VGPR_16RegClass &&
          MRI->getRegClass(Src) == &AMDGPU::VGPR_32RegClass)
        MRI->setRegAllocationHint(Dst, AMDGPURI::Size16, Src);
    }
  }

  return Changed;
}
