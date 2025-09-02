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
/// Additionally, this pass also unpacks packed instructions (V_PK_MUL_F32/F16,
/// V_PK_ADD_F32/F16, V_PK_FMA_F32) adjacent to MFMAs such that they can be
/// co-issued. This helps with overlapping MFMA and certain vector instructions
/// in machine schedules and is expected to improve performance. Only those
/// packed instructions are unpacked that are overlapped by the MFMA latency.
/// Rest should remain untouched.
/// TODO: Add support for F16 packed instructions
//===----------------------------------------------------------------------===//

#include "GCNPreRAOptimizations.h"
#include "AMDGPU.h"
#include "GCNSchedStrategy.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/InitializePasses.h"
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
  bool createListOfPackedInstr(MachineInstr &BeginMI,
                               SetVector<MachineInstr *> &InstrsToUnpack,
                               uint16_t NumMFMACycles);
  bool isUnpackingSupportedInstr(MachineInstr &MI) const;
  void processF32Unpacking(MachineInstr &I);
  uint16_t mapToUnpackedOpcode(MachineInstr &I);

  void insertUnpackedF32MI(MachineInstr &I, MachineOperand &DstMO,
                           MachineOperand &LoSrcMO1, MachineOperand &LoSrcMO2,
                           MachineOperand &HiSrcMO1, MachineOperand &HiSrcMO2,
                           bool isVreg_64);
  void processFMAF32Unpacking(MachineInstr &I);
  MachineInstrBuilder createUnpackedMI(MachineBasicBlock &MBB, MachineInstr &I,
                                       const DebugLoc &DL,
                                       uint16_t UnpackedOpcode, bool isHiBits,
                                       bool isFMA);
  bool hasReadWriteDependencies(const MachineInstr &PredMI,
                                const MachineInstr &SuccMI);

public:
  GCNPreRAOptimizationsImpl(LiveIntervals *LS) : LIS(LS) {}
  bool run(MachineFunction &MF);
};

class GCNPreRAOptimizationsLegacy : public MachineFunctionPass {
public:
  static char ID;

  GCNPreRAOptimizationsLegacy() : MachineFunctionPass(ID) {
    initializeGCNPreRAOptimizationsLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Pre-RA optimizations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
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

bool GCNPreRAOptimizationsImpl::isUnpackingSupportedInstr(
    MachineInstr &MI) const {
  unsigned Opcode = MI.getOpcode();
  switch (Opcode) {
  case AMDGPU::V_PK_ADD_F32:
  case AMDGPU::V_PK_MUL_F32:
    return (MI.getOperand(2).isReg() && MI.getOperand(4).isReg());
  case AMDGPU::V_PK_FMA_F32:
    return (MI.getOperand(2).isReg() && MI.getOperand(4).isReg() &&
            MI.getOperand(6).isReg());
  default:
    return false;
  }
}

uint16_t GCNPreRAOptimizationsImpl::mapToUnpackedOpcode(MachineInstr &I) {
  unsigned Opcode = I.getOpcode();
  // use 64 bit encoding to allow use of VOP3 instructions.
  // VOP3 instructions allow VOP3P source modifiers to be translated to VOP3
  // e32 instructions are VOP2 and don't allow source modifiers
  switch (Opcode) {
  case AMDGPU::V_PK_ADD_F32:
    return AMDGPU::V_ADD_F32_e64;
  case AMDGPU::V_PK_MUL_F32:
    return AMDGPU::V_MUL_F32_e64;
  case AMDGPU::V_PK_FMA_F32:
    return AMDGPU::V_FMA_F32_e64;
  default:
    return std::numeric_limits<uint16_t>::max();
  }
}

bool GCNPreRAOptimizationsImpl::hasReadWriteDependencies(
    const MachineInstr &PredMI, const MachineInstr &SuccMI) {
  for (const MachineOperand &Pred_Ops : PredMI.operands()) {
    if (!Pred_Ops.isReg() || !Pred_Ops.isDef())
      continue;
    Register Pred_Reg = Pred_Ops.getReg();
    if (!Pred_Reg.isValid())
      continue;
    for (const MachineOperand &Succ_Ops : SuccMI.operands()) {
      if (!Succ_Ops.isReg() || !Succ_Ops.isDef())
        continue;
      Register Succ_Reg = Succ_Ops.getReg();
      if (!Succ_Reg.isValid())
        continue;
      if ((Pred_Reg == Succ_Reg) || TRI->regsOverlap(Pred_Reg, Succ_Reg)) {
        return true;
      }
    }
  }
  return false;
}
bool GCNPreRAOptimizationsImpl::createListOfPackedInstr(
    MachineInstr &BeginMI, SetVector<MachineInstr *> &InstrsToUnpack,
    uint16_t NumMFMACycles) {
  auto *BB = BeginMI.getParent();
  auto *MF = BB->getParent();
  int NumInst = 0;
  auto E = BB->end();
  int TotalCyclesBetweenCandidates = 0;
  auto SchedModel = TII->getSchedModel();
  for (auto I = std::next(BeginMI.getIterator()); I != E; ++I) {
    MachineInstr &Instr = *I;
    const MCSchedClassDesc *InstrSchedClassDesc =
        SchedModel.resolveSchedClass(&Instr);
    TotalCyclesBetweenCandidates +=
        SchedModel.getWriteProcResBegin(InstrSchedClassDesc)->ReleaseAtCycle;

    if (Instr.isMetaInstruction())
      continue;
    if (Instr.isTerminator())
      return false;
    if (TotalCyclesBetweenCandidates > NumMFMACycles)
      return false;
    if ((isUnpackingSupportedInstr(Instr)) && TII->isNeverCoissue(Instr)) {
      if (hasReadWriteDependencies(BeginMI, Instr))
        return false;

      // if it is a packed instruction, we should subtract it's latency from the
      // overall latency calculation here, because the packed instruction will
      // be removed and replaced by 2 unpacked instructions
      TotalCyclesBetweenCandidates -=
          SchedModel.getWriteProcResBegin(InstrSchedClassDesc)->ReleaseAtCycle;
      // We're adding 2 to account for the extra latency added by unpacking into
      // 2 instructions. At the time of writing, the considered unpacked
      // instructions have latency of 1.
      // TODO: improve latency handling of possible inserted instructions
      TotalCyclesBetweenCandidates += 2;
      if (!(TotalCyclesBetweenCandidates >= NumMFMACycles))
        InstrsToUnpack.insert(&Instr);
    }
  }
  return true;
}

void GCNPreRAOptimizationsImpl::insertUnpackedF32MI(
    MachineInstr &I, MachineOperand &DstMO, MachineOperand &LoSrcMO1,
    MachineOperand &LoSrcMO2, MachineOperand &HiSrcMO1,
    MachineOperand &HiSrcMO2, bool IsVreg_64) {

  MachineBasicBlock &MBB = *I.getParent();
  const DebugLoc &DL = I.getDebugLoc();
  Register DstReg = DstMO.getReg();

  uint16_t UnpackedOpcode = mapToUnpackedOpcode(I);
  if (UnpackedOpcode == std::numeric_limits<uint16_t>::max())
    return;

  MachineInstrBuilder Op0L_Op1L =
      createUnpackedMI(MBB, I, DL, UnpackedOpcode, false, false);
  if (IsVreg_64) {
    Op0L_Op1L->getOperand(0).setIsUndef();
  } else if (DstMO.isUndef()) {
    Op0L_Op1L->getOperand(0).setIsUndef();
  }
  LIS->InsertMachineInstrInMaps(*Op0L_Op1L);

  MachineInstrBuilder Op0H_Op1H =
      createUnpackedMI(MBB, I, DL, UnpackedOpcode, true, false);
  LIS->InsertMachineInstrInMaps(*Op0H_Op1H);

  if (I.getFlag(MachineInstr::MIFlag::NoFPExcept)) {
    Op0L_Op1L->setFlag(MachineInstr::MIFlag::NoFPExcept);
    Op0H_Op1H->setFlag(MachineInstr::MIFlag::NoFPExcept);
  }
  if (I.getFlag(MachineInstr::MIFlag::FmContract)) {
    Op0L_Op1L->setFlag(MachineInstr::MIFlag::FmContract);
    Op0H_Op1H->setFlag(MachineInstr::MIFlag::FmContract);
  }

  LIS->RemoveMachineInstrFromMaps(I);
  I.eraseFromParent();
  LIS->removeInterval(DstReg);
  LIS->createAndComputeVirtRegInterval(DstReg);
  return;
}

void GCNPreRAOptimizationsImpl::processFMAF32Unpacking(MachineInstr &I) {
  MachineBasicBlock &MBB = *I.getParent();
  Register DstReg = I.getOperand(0).getReg();
  const DebugLoc &DL = I.getDebugLoc();
  const TargetRegisterClass *DstRC = MRI->getRegClass(I.getOperand(0).getReg());
  bool IsVReg64 = (DstRC->getID() == AMDGPU::VReg_64_Align2RegClassID);

  uint16_t UnpackedOpcode = mapToUnpackedOpcode(I);
  if (UnpackedOpcode == std::numeric_limits<uint16_t>::max())
    return;

  MachineInstrBuilder Op0L_Op1L =
      createUnpackedMI(MBB, I, DL, UnpackedOpcode, false, true);
  if (IsVReg64)
    Op0L_Op1L->getOperand(0).setIsUndef();
  else if (I.getOperand(0).isUndef()) {
    Op0L_Op1L->getOperand(0).setIsUndef();
  }
  LIS->InsertMachineInstrInMaps(*Op0L_Op1L);

  MachineInstrBuilder Op0H_Op1H =
      createUnpackedMI(MBB, I, DL, UnpackedOpcode, true, true);
  LIS->InsertMachineInstrInMaps(*Op0H_Op1H);

  if (I.getFlag(MachineInstr::MIFlag::NoFPExcept)) {
    Op0L_Op1L->setFlag(MachineInstr::MIFlag::NoFPExcept);
    Op0H_Op1H->setFlag(MachineInstr::MIFlag::NoFPExcept);
  }
  if (I.getFlag(MachineInstr::MIFlag::FmContract)) {
    Op0L_Op1L->setFlag(MachineInstr::MIFlag::FmContract);
    Op0H_Op1H->setFlag(MachineInstr::MIFlag::FmContract);
  }

  LIS->RemoveMachineInstrFromMaps(I);
  I.eraseFromParent();
  LIS->removeInterval(DstReg);
  LIS->createAndComputeVirtRegInterval(DstReg);
  return;
}

MachineInstrBuilder GCNPreRAOptimizationsImpl::createUnpackedMI(
    MachineBasicBlock &MBB, MachineInstr &I, const DebugLoc &DL,
    uint16_t UnpackedOpcode, bool isHiBits, bool isFMA) {
  MachineOperand &DstMO = I.getOperand(0);
  MachineOperand &SrcMO1 = I.getOperand(2);
  MachineOperand &SrcMO2 = I.getOperand(4);
  Register DstReg = DstMO.getReg();
  Register SrcReg1 = SrcMO1.getReg();
  Register SrcReg2 = SrcMO2.getReg();
  const TargetRegisterClass *DstRC = MRI->getRegClass(DstMO.getReg());
  unsigned DestSubIdx =
      isHiBits ? TRI->composeSubRegIndices(DstMO.getSubReg(), AMDGPU::sub1)
               : TRI->composeSubRegIndices(DstMO.getSubReg(), AMDGPU::sub0);
  int ClampIdx =
      AMDGPU::getNamedOperandIdx(I.getOpcode(), AMDGPU::OpName::clamp);
  int64_t ClampVal = I.getOperand(ClampIdx).getImm();
  int Src0_modifiers_Idx =
      AMDGPU::getNamedOperandIdx(I.getOpcode(), AMDGPU::OpName::src0_modifiers);
  int Src1_modifiers_Idx =
      AMDGPU::getNamedOperandIdx(I.getOpcode(), AMDGPU::OpName::src1_modifiers);

  unsigned Src0_Mods = I.getOperand(Src0_modifiers_Idx).getImm();
  unsigned Src1_Mods = I.getOperand(Src1_modifiers_Idx).getImm();
  // Packed instructions (VOP3P) do not support abs. It is okay to ignore them.
  unsigned New_Src0_Mods = 0;
  unsigned New_Src1_Mods = 0;

  unsigned NegModifier = isHiBits ? SISrcMods::NEG_HI : SISrcMods::NEG;
  unsigned OpSelModifier = isHiBits ? SISrcMods::OP_SEL_1 : SISrcMods::OP_SEL_0;

  MachineInstrBuilder NewMI = BuildMI(MBB, I, DL, TII->get(UnpackedOpcode));
  NewMI.addDef(DstReg, 0, DestSubIdx); // vdst
  if (Src0_Mods & NegModifier) {
    New_Src0_Mods |= SISrcMods::NEG;
  }
  NewMI.addImm(New_Src0_Mods); // src0_modifiers

  if (Src0_Mods & OpSelModifier) {
    unsigned Src0SubIdx =
        TRI->composeSubRegIndices(SrcMO1.getSubReg(), AMDGPU::sub1);
    NewMI.addReg(SrcMO1.getReg(), 0, Src0SubIdx); // src0
  } else {
    unsigned Src0SubIdx =
        TRI->composeSubRegIndices(SrcMO1.getSubReg(), AMDGPU::sub0);
    // if op_sel == 0, select register 0 of reg:sub0_sub1
    NewMI.addReg(SrcMO1.getReg(), 0, Src0SubIdx);
  }

  if (Src1_Mods & NegModifier) {
    New_Src1_Mods |= SISrcMods::NEG;
  }
  NewMI.addImm(New_Src1_Mods); // src1_modifiers
  if (Src1_Mods & OpSelModifier) {
    unsigned Src1SubIdx =
        TRI->composeSubRegIndices(SrcMO2.getSubReg(), AMDGPU::sub1);
    NewMI.addReg(SrcMO2.getReg(), 0, Src1SubIdx); // src0
  } else {
    // if op_sel_hi == 0, select register 0 of reg:sub0_sub1
    unsigned Src1SubIdx =
        TRI->composeSubRegIndices(SrcMO2.getSubReg(), AMDGPU::sub0);
    NewMI.addReg(SrcMO2.getReg(), 0, Src1SubIdx);
  }

  if (isFMA) {
    MachineOperand &SrcMO3 = I.getOperand(6);
    Register SrcReg3 = SrcMO3.getReg();
    int Src2_modifiers_Idx = AMDGPU::getNamedOperandIdx(
        I.getOpcode(), AMDGPU::OpName::src2_modifiers);
    unsigned Src2_Mods = I.getOperand(Src2_modifiers_Idx).getImm();
    unsigned New_Src2_Mods = 0;
    // If NEG or NEG_HI is true, we need to negate the corresponding 32 bit
    // lane.
    //  This is also true for NEG_HI as it shares the same bit position with
    //  ABS. But packed instructions do not support ABS. Therefore, NEG_HI must
    //  be translated to NEG source modifier for the higher 32 bits.
    //  Unpacked VOP3 instructions do support ABS, therefore we need to
    //  explicitly add the NEG modifier if present in the packed instruction
    if (Src2_Mods & NegModifier) {
      // New_Src2_Mods |= NegModifier;
      New_Src2_Mods |= SISrcMods::NEG;
    }
    NewMI.addImm(New_Src2_Mods); // src2_modifiers
    if (Src2_Mods & OpSelModifier) {
      unsigned Src2SubIdx =
          TRI->composeSubRegIndices(SrcMO3.getSubReg(), AMDGPU::sub1);
      NewMI.addReg(SrcMO3.getReg(), 0, Src2SubIdx);
    } else {
      unsigned Src2SubIdx =
          TRI->composeSubRegIndices(SrcMO3.getSubReg(), AMDGPU::sub0);
      // if op_sel_hi == 0, select register 0 of reg:sub0_sub1
      NewMI.addReg(SrcMO3.getReg(), 0, Src2SubIdx);
    }
  }
  NewMI.addImm(ClampVal); // clamp
  // packed instructions do not support output modifiers. safe to assign them 0
  // for this use case
  NewMI.addImm(0); // omod
  return NewMI;
}

void GCNPreRAOptimizationsImpl::processF32Unpacking(MachineInstr &I) {
  if (I.getOpcode() == AMDGPU::V_PK_FMA_F32) {
    processFMAF32Unpacking(I);
    return;
  }
  MachineBasicBlock &MBB = *I.getParent();

  MachineOperand &DstMO = I.getOperand(0);
  MachineOperand &SrcMO1 = I.getOperand(2);
  MachineOperand &SrcMO2 = I.getOperand(4);

  const DebugLoc &DL = I.getDebugLoc();
  const TargetRegisterClass *DstRC = MRI->getRegClass(I.getOperand(0).getReg());

  bool IsVReg64 = (DstRC->getID() == AMDGPU::VReg_64_Align2RegClassID);
  insertUnpackedF32MI(I, DstMO, SrcMO1, SrcMO2, SrcMO1, SrcMO2, IsVReg64);
  return;
}

bool GCNPreRAOptimizationsLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
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

  // Add RA hints to improve True16 COPY elimination.
  // Unpack packed instructions to overlap MFMAs. This allows the compiler to
  // co-issue unpacked instructions with MFMA
  for (MachineBasicBlock &MBB : MF) {
    SetVector<MachineInstr *> InstrsToUnpack;
    SetVector<MachineOperand *> WriteOperands;
    SetVector<MachineOperand *> ReadOperands;
    uint16_t NumMFMACycles = 0;
    auto SchedModel = TII->getSchedModel();
    for (MachineInstr &MI : MBB) {
      if (SIInstrInfo::isMFMA(MI)) {
        const MCSchedClassDesc *SchedClassDesc =
            SchedModel.resolveSchedClass(&MI);
        NumMFMACycles =
            SchedModel.getWriteProcResBegin(SchedClassDesc)->ReleaseAtCycle;
        createListOfPackedInstr(MI, InstrsToUnpack, NumMFMACycles);
      }
      if (ST.useRealTrue16Insts()) {
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

    if (!InstrsToUnpack.empty()) {
      for (MachineInstr *MI : InstrsToUnpack) {
        processF32Unpacking(*MI);
      }
    }
  }
  LIS->reanalyze(MF);
  return Changed;
}