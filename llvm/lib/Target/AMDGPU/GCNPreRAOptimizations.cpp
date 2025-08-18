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
/// Additionally, this pass also unpacks packed instructions (V_PK_MUL_F32 and
/// V_PK_ADD_F32) adjacent to MFMAs such that they can be co-issued. This helps
/// with overlapping MFMA and certain vector instructions in machine schedules
/// and is expected to improve performance.
/// Only those packed instructions are unpacked that are overlapped by the MFMA
/// latency. Rest should remain untouched.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNPreRAOptimizations.h"
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
  
  SmallVector<MachineInstr *, 2>
  insertUnpackedF32MI(MachineInstr &I, MachineOperand &DstMO,
                   MachineOperand &LoSrcMO1, MachineOperand &LoSrcMO2,
                   MachineOperand &HiSrcMO1, MachineOperand &HiSrcMO2,
                   bool isVreg_64);
  void processF16Unpacking(MachineInstr &I, uint16_t AvailableBudget);
  bool IsF16MaskSet;
  Register MaskLo; // mask to extract lower 16 bits for F16 packed instructions
  Register
      ShiftAmt; // mask to extract higher 16 bits from F16 packed instructions

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
  case AMDGPU::V_PK_MUL_F16:
  case AMDGPU::V_PK_ADD_F16:
    return true;

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
  case AMDGPU::V_PK_ADD_F16:
    return AMDGPU::V_ADD_F16_e64;
  case AMDGPU::V_PK_MUL_F16:
    return AMDGPU::V_MUL_F16_e64;
  default:
    return std::numeric_limits<uint16_t>::max();
  }
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
      if ((Instr.getOpcode() == AMDGPU::V_PK_MUL_F16) ||
          (Instr.getOpcode() == AMDGPU::V_PK_ADD_F16)) {
        // unpacking packed F16 instructions requires multiple instructions.
        // Instructions are issued to extract lower and higher bits for each
        // operand Instructions are then issued for 2 unpacked instructions, and
        // additional instructions to put them back into the original
        // destination register The following sequence of instructions are
        // issued

        // The next two are needed to move masks into vgprs. Ideally, immediates
        // should be used. However, if one of the source operands are
        // sgpr/sregs, then immediates are not allowed. Hence, the need to move
        // these into vgprs

        // vgpr_32 = V_MOV_B32_e32 65535
        // vgpr_32 = V_MOV_B32_e32 16

        // vgpr_32 = V_AND_B32_e32 sub1:sreg_64, vgpr_32
        // vgpr_32 = V_LSHRREV_B32_e64 vgpr_32, sub1:sreg_64
        // vgpr_32 = V_AND_B32_e32 vgpr_32, vgpr_32
        // vgpr_32 = V_LSHRREV_B32_e64 vgpr_32, vgpr_32
        // vgpr_32 = V_MUL_F16_e64 0, killed vgpr_32, 0, killed vgpr_32, 0, 0
        // vgpr_32 = V_MUL_F16_e64 0, killed vgpr_32, 0, killed vgpr_32, 0, 0
        // vgpr_32 = V_LSHLREV_B32_e64 vgpr_32, vgpr_32
        // dst_reg = V_OR_B32_e64 vgpr_32, vgpr_32

        // we need to issue the MOV instructions above only once. Once these are
        // issued, the IsF16MaskSet flag is set subsequent unpacking only needs
        // to issue the remaining instructions The number of latency cycles for
        // each instruction above is 1. It's hard coded into the code to reduce
        // code complexity.
        if (IsF16MaskSet)
          TotalCyclesBetweenCandidates += 7;
        else
          TotalCyclesBetweenCandidates += 9;
      } else
        TotalCyclesBetweenCandidates += 1;

      if (!(TotalCyclesBetweenCandidates > NumMFMACycles))
        InstrsToUnpack.insert(&Instr);
    }
  }
  return true;
}

SmallVector<MachineInstr *, 2> GCNPreRAOptimizationsImpl::insertUnpackedF32MI(
    MachineInstr &I, MachineOperand &DstMO, MachineOperand &LoSrcMO1,
    MachineOperand &LoSrcMO2, MachineOperand &HiSrcMO1,
    MachineOperand &HiSrcMO2, bool IsVreg_64) {

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

  const MCInstrDesc InstrDesc = I.getDesc();

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
  unsigned Lo_src0_mods = 0;
  unsigned Lo_src1_mods = 0;
  uint16_t UnpackedOpcode = mapToUnpackedOpcode(I);
  MachineInstrBuilder Op0L_Op1L = BuildMI(MBB, I, DL, TII->get(UnpackedOpcode));
  Op0L_Op1L.addDef(DstReg, 0, DestSubIdx); // vdst
  if (Src0_Mods & SISrcMods::OP_SEL_0) {
    if (Src0_Mods & SISrcMods::NEG) {
      Lo_src0_mods |= SISrcMods::NEG;
    }
    Op0L_Op1L.addImm(Lo_src0_mods); // src0_modifiers
    unsigned Src0SubIdx =
        TRI->composeSubRegIndices(LoSrcMO1.getSubReg(), AMDGPU::sub1);
    Op0L_Op1L.addReg(LoSrcMO1.getReg(), 0, Src0SubIdx); // src0
  } else {
    Op0L_Op1L.addImm(Lo_src0_mods); // src0_modifiers
    unsigned Src0SubIdx =
        TRI->composeSubRegIndices(LoSrcMO1.getSubReg(), AMDGPU::sub0);
    Op0L_Op1L.addReg(LoSrcMO1.getReg(), 0,
                     Src0SubIdx); // src0 //if op_sel == 0, select register 0 of
                                  // reg:sub0_sub1
  }
  if (Src1_Mods & SISrcMods::OP_SEL_0) {
    if (Src1_Mods & SISrcMods::NEG) {
      Lo_src1_mods |= SISrcMods::NEG;
    }
    Op0L_Op1L.addImm(Lo_src1_mods); // src0_modifiers
    unsigned Src1SubIdx =
        TRI->composeSubRegIndices(LoSrcMO2.getSubReg(), AMDGPU::sub1);
    Op0L_Op1L.addReg(LoSrcMO2.getReg(), 0, Src1SubIdx); // src0
  } else {
    Op0L_Op1L.addImm(Lo_src1_mods); // src0_modifiers
    unsigned Src1SubIdx =
        TRI->composeSubRegIndices(LoSrcMO2.getSubReg(), AMDGPU::sub0);
    Op0L_Op1L.addReg(LoSrcMO2.getReg(), 0,
                     Src1SubIdx); // src0 //if op_sel_hi == 0, select register 0
                                  // of reg:sub0_sub1
  }
  Op0L_Op1L.addImm(ClampVal); // clamp
  // packed instructions do not support output modifiers. safe to assign them 0
  // for this use case
  Op0L_Op1L.addImm(0); // omod

  if (IsVreg_64) {
    Op0L_Op1L->getOperand(0).setIsUndef();
  } else if (I.getOperand(0).isUndef()) {
    Op0L_Op1L->getOperand(0).setIsUndef();
  }

  LIS->InsertMachineInstrInMaps(*Op0L_Op1L);

  SrcSubIdx1 = TRI->composeSubRegIndices(LoSrcMO1.getSubReg(), AMDGPU::sub1);
  SrcSubIdx2 = TRI->composeSubRegIndices(LoSrcMO2.getSubReg(), AMDGPU::sub1);
  DestSubIdx = TRI->composeSubRegIndices(DstMO.getSubReg(), AMDGPU::sub1);

  // Packed instructions (VOP3P) do not support abs. It is okay to ignore them.
  unsigned Hi_src0_mods = 0;
  unsigned Hi_src1_mods = 0;

  MachineInstrBuilder Op0H_Op1H = BuildMI(MBB, I, DL, TII->get(UnpackedOpcode));
  Op0H_Op1H.addDef(DstReg, 0, DestSubIdx); // vdst
  if (Src0_Mods & SISrcMods::OP_SEL_1) {
    if (Src0_Mods & SISrcMods::NEG_HI) {
      Hi_src0_mods |= SISrcMods::NEG;
    }
    Op0H_Op1H.addImm(Hi_src0_mods); // src0_modifiers
    unsigned Src0SubIdx =
        TRI->composeSubRegIndices(HiSrcMO1.getSubReg(), AMDGPU::sub1);
    Op0H_Op1H.addReg(HiSrcMO1.getReg(), 0, Src0SubIdx); // src0
  } else {
    Op0H_Op1H.addImm(Hi_src0_mods); // src0_modifiers
    unsigned Src0SubIdx =
        TRI->composeSubRegIndices(HiSrcMO1.getSubReg(), AMDGPU::sub0);
    Op0H_Op1H.addReg(HiSrcMO1.getReg(), 0,
                     Src0SubIdx); // src0 //if op_sel_hi == 0, select register 0
                                  // of reg:sub0_sub1
  }

  if (Src1_Mods & SISrcMods::OP_SEL_1) {
    if (Src1_Mods & SISrcMods::NEG_HI) {
      Hi_src1_mods |= SISrcMods::NEG;
    }
    Op0H_Op1H.addImm(Hi_src1_mods); // src0_modifiers
    unsigned Src1SubIdx =
        TRI->composeSubRegIndices(HiSrcMO2.getSubReg(), AMDGPU::sub1);
    Op0H_Op1H.addReg(HiSrcMO2.getReg(), 0, Src1SubIdx); // src0
  } else {
    Op0H_Op1H.addImm(Hi_src1_mods); // src0_modifiers
    unsigned Src1SubIdx =
        TRI->composeSubRegIndices(HiSrcMO2.getSubReg(), AMDGPU::sub0);
    Op0H_Op1H.addReg(HiSrcMO2.getReg(), 0,
                     Src1SubIdx); // src0 //if op_sel_hi == 0, select register 0
                                  // of reg:sub0_sub1
  }
  Op0H_Op1H.addImm(ClampVal); // clamp
  // packed instructions do not support output modifiers. safe to assign them 0
  // for this use case
  Op0H_Op1H.addImm(0); // omod
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

void GCNPreRAOptimizationsImpl::processF32Unpacking(MachineInstr &I) {
  MachineBasicBlock &MBB = *I.getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  MachineFunction &MF = *MBB.getParent();

  Register DstReg = I.getOperand(0).getReg();
  Register SrcReg1 = I.getOperand(2).getReg();
  Register SrcReg2 = I.getOperand(4).getReg();
  MachineOperand &DstMO = I.getOperand(0);
  MachineOperand &SrcMO1 = I.getOperand(2);
  MachineOperand &SrcMO2 = I.getOperand(4);

  const DebugLoc &DL = I.getDebugLoc();
  const TargetRegisterClass *DstRC = MRI.getRegClass(I.getOperand(0).getReg());
  const TargetRegisterClass *Src0RC = MRI.getRegClass(I.getOperand(2).getReg());
  const TargetRegisterClass *Src1RC = MRI.getRegClass(I.getOperand(4).getReg());

  const TargetRegisterClass *Src0SubRC =
      TRI->getSubRegisterClass(Src0RC, AMDGPU::sub0);
  const TargetRegisterClass *SrcRC = TRI->getSubClassWithSubReg(Src0RC, 1);

  bool IsVReg64 = (DstRC->getID() == AMDGPU::VReg_64_Align2RegClassID);
  SmallVector<MachineInstr *, 2> UnpackedInstrs =
      insertUnpackedF32MI(I, DstMO, SrcMO1, SrcMO2, SrcMO1, SrcMO2, IsVReg64);
  return;
}

void GCNPreRAOptimizationsImpl::processF16Unpacking(MachineInstr &I,
                                                    uint16_t AvailableBudget) {
  MachineBasicBlock &MBB = *I.getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  MachineOperand &DstMO = I.getOperand(0);
  MachineOperand &SrcMO0 = I.getOperand(2);
  MachineOperand &SrcMO1 = I.getOperand(4);

  Register DstReg = DstMO.getReg();
  Register SrcReg0 = SrcMO0.getReg();
  Register SrcReg1 = SrcMO1.getReg();

  const DebugLoc &DL = I.getDebugLoc();

  const TargetRegisterClass *RC = &AMDGPU::VGPR_32RegClass;
  auto SchedModel = TII->getSchedModel();

  uint16_t AddlCyclesConsumed = 0;
  SetVector<MachineInstr *> ListOfNewInstructions;

  auto BuildImm = [&](uint32_t Val) -> std::pair<Register, uint16_t> {
    Register ImmReg = MRI.createVirtualRegister(RC);
    auto NewMI = BuildMI(MBB, I, DL, TII->get(AMDGPU::V_MOV_B32_e32), ImmReg)
                     .addImm(Val);
    LIS->InsertMachineInstrInMaps(*NewMI);
    const MCSchedClassDesc *SchedClassDesc =
        SchedModel.resolveSchedClass(NewMI);
    uint16_t LatencyCycles =
        SchedModel.getWriteProcResBegin(SchedClassDesc)->ReleaseAtCycle;
    return {ImmReg, LatencyCycles};
  };

  if (!IsF16MaskSet) {
    std::pair<Register, uint16_t> RegAndLatency = BuildImm(0x0000FFFF);
    MaskLo = RegAndLatency.first; // mask for lower 16 bits
    RegAndLatency = BuildImm(16);
    ShiftAmt = RegAndLatency.first; // mask for higher 16 bits
    IsF16MaskSet = true;
  }

  Register Src0_Lo = MRI.createVirtualRegister(RC);
  Register Src1_Lo = MRI.createVirtualRegister(RC);
  Register Src0_Hi = MRI.createVirtualRegister(RC);
  Register Src1_Hi = MRI.createVirtualRegister(RC);
  Register Input0 = MRI.createVirtualRegister(RC);
  Register Input1 = MRI.createVirtualRegister(RC);

  unsigned SubRegID = 0;
  if (SrcMO0.getSubReg())
    SubRegID = SrcMO0.getSubReg();

  int Src0_modifiers_Idx =
      AMDGPU::getNamedOperandIdx(I.getOpcode(), AMDGPU::OpName::src0_modifiers);
  int Src1_modifiers_Idx =
      AMDGPU::getNamedOperandIdx(I.getOpcode(), AMDGPU::OpName::src1_modifiers);
  unsigned Src0_Mods = I.getOperand(Src0_modifiers_Idx).getImm();
  unsigned Src1_Mods = I.getOperand(Src1_modifiers_Idx).getImm();
  int ClampIdx =
      AMDGPU::getNamedOperandIdx(I.getOpcode(), AMDGPU::OpName::clamp);
  int64_t ClampVal = I.getOperand(ClampIdx).getImm();

  // handle op_sel for src0
  if (Src0_Mods & SISrcMods::OP_SEL_0) {
    // if op_sel is set, select higher 16 bits and copy into lower 16 bits of
    // new vgpr
    MachineInstrBuilder LoInput0_MI =
        BuildMI(MBB, I, DL, TII->get(AMDGPU::V_LSHRREV_B32_e64), Src0_Lo)
            .addReg(ShiftAmt);
    if (SubRegID)
      LoInput0_MI.addReg(SrcReg0, 0, SubRegID);
    else
      LoInput0_MI.addReg(SrcReg0);
    LIS->InsertMachineInstrInMaps(*LoInput0_MI);
  } else {
    // if op_sel is not set, select lower 16 bits and copy into lower 16 bits of
    // new vgpr
    MachineInstrBuilder LoInput0_MI =
        BuildMI(MBB, I, DL, TII->get(AMDGPU::V_AND_B32_e32), Src0_Lo);
    if (SubRegID)
      LoInput0_MI.addReg(SrcReg0, 0, SubRegID);
    else
      LoInput0_MI.addReg(SrcReg0);
    LoInput0_MI.addReg(MaskLo);
    LIS->InsertMachineInstrInMaps(*LoInput0_MI);
  }

  // handle op_sel_hi for src0
  if (Src0_Mods & SISrcMods::OP_SEL_1) {
    // if op_sel_hi is set, select higher 16 bits and copy into lower 16 bits of
    // new vgpr
    MachineInstrBuilder HiInput0_MI =
        BuildMI(MBB, I, DL, TII->get(AMDGPU::V_LSHRREV_B32_e64), Src0_Hi)
            .addReg(ShiftAmt);
    if (SubRegID)
      HiInput0_MI.addReg(SrcReg0, 0, SubRegID);
    else
      HiInput0_MI.addReg(SrcReg0);
    LIS->InsertMachineInstrInMaps(*HiInput0_MI);
  } else {
    // if op_sel_hi is not set, select lower 16 bits and copy into lower 16 bits
    // of new vgpr
    MachineInstrBuilder HiInput0_MI =
        BuildMI(MBB, I, DL, TII->get(AMDGPU::V_AND_B32_e32), Src0_Hi);
    if (SubRegID)
      HiInput0_MI.addReg(SrcReg0, 0, SubRegID);
    else
      HiInput0_MI.addReg(SrcReg0);
    HiInput0_MI.addReg(MaskLo);
    LIS->InsertMachineInstrInMaps(*HiInput0_MI);
  }

  SubRegID = 0;
  if (SrcMO0.getSubReg())
    SubRegID = SrcMO1.getSubReg();
  // handle op_sel for src1
  if (Src1_Mods & SISrcMods::OP_SEL_0) {
    // if op_sel is set, select higher 16 bits and copy into lower 16 bits of
    // new vgpr
    MachineInstrBuilder LoInput1_MI =
        BuildMI(MBB, I, DL, TII->get(AMDGPU::V_LSHRREV_B32_e64), Src1_Lo)
            .addReg(ShiftAmt);
    if (SubRegID)
      LoInput1_MI.addReg(SrcReg1, 0, SubRegID);
    else
      LoInput1_MI.addReg(SrcReg1);
    LIS->InsertMachineInstrInMaps(*LoInput1_MI);
  } else {
    // if op_sel is not set, select lower 16 bits and copy into lower 16 bits of
    // new vgpr
    MachineInstrBuilder LoInput1_MI =
        BuildMI(MBB, I, DL, TII->get(AMDGPU::V_AND_B32_e32), Src1_Lo);
    if (SubRegID)
      LoInput1_MI.addReg(SrcReg1, 0, SubRegID);
    else
      LoInput1_MI.addReg(SrcReg1);
    LoInput1_MI.addReg(MaskLo);
    LIS->InsertMachineInstrInMaps(*LoInput1_MI);
  }

  // handle op_sel_hi for src1
  if (Src1_Mods & SISrcMods::OP_SEL_1) {
    // if op_sel_hi is set, select higher 16 bits and copy into lower 16 bits of
    // new vgpr
    MachineInstrBuilder HiInput1_MI =
        BuildMI(MBB, I, DL, TII->get(AMDGPU::V_LSHRREV_B32_e64), Src1_Hi)
            .addReg(ShiftAmt);
    if (SubRegID)
      HiInput1_MI.addReg(SrcReg1, 0, SubRegID);
    else
      HiInput1_MI.addReg(SrcReg1);
    LIS->InsertMachineInstrInMaps(*HiInput1_MI);
  } else {
    // if op_sel_hi is not set, select lower 16 bits and copy into lower 16 bits
    // of new vgpr
    MachineInstrBuilder HiInput1_MI =
        BuildMI(MBB, I, DL, TII->get(AMDGPU::V_AND_B32_e32), Src1_Hi);
    if (SubRegID)
      HiInput1_MI.addReg(SrcReg1, 0, SubRegID);
    else
      HiInput1_MI.addReg(SrcReg1);
    HiInput1_MI.addReg(MaskLo);
    LIS->InsertMachineInstrInMaps(*HiInput1_MI);
  }

  Register LoMul = MRI.createVirtualRegister(RC);
  Register HiMul = MRI.createVirtualRegister(RC);

  unsigned Lo_src0_mods = 0;
  unsigned Lo_src1_mods = 0;
  uint16_t UnpackedOpcode = mapToUnpackedOpcode(I);

  // Unpacked instructions
  MachineInstrBuilder LoMul_MI =
      BuildMI(MBB, I, DL, TII->get(UnpackedOpcode), LoMul);

  if (Src0_Mods & SISrcMods::NEG)
    Lo_src0_mods |= SISrcMods::NEG;

  LoMul_MI.addImm(Lo_src0_mods);            // src0_modifiers
  LoMul_MI.addReg(Src0_Lo, RegState::Kill); // src0

  if (Src1_Mods & SISrcMods::NEG)
    Lo_src1_mods |= SISrcMods::NEG;

  LoMul_MI.addImm(Lo_src1_mods);            // src1_modifiers
  LoMul_MI.addReg(Src1_Lo, RegState::Kill); // src1
  LoMul_MI.addImm(ClampVal);                // clamp
  // packed instructions do not support output modifiers. safe to assign them 0
  // for this use case
  LoMul_MI.addImm(0); // omod

  // unpacked instruction with VOP3 encoding for Hi bits
  unsigned Hi_src0_mods = 0;
  unsigned Hi_src1_mods = 0;

  MachineInstrBuilder HiMul_MI =
      BuildMI(MBB, I, DL, TII->get(UnpackedOpcode), HiMul);
  if (Src0_Mods & SISrcMods::NEG_HI)
    Hi_src0_mods |= SISrcMods::NEG_HI;

  HiMul_MI.addImm(Hi_src0_mods); // src0_modifiers
  HiMul_MI.addReg(Src0_Hi,
                  RegState::Kill); // select higher 16 bits if op_sel_hi is set

  if (Src1_Mods & SISrcMods::NEG_HI)
    Hi_src1_mods |= SISrcMods::NEG_HI;

  HiMul_MI.addImm(Hi_src1_mods); // src0_modifiers
  HiMul_MI.addReg(
      Src1_Hi,
      RegState::Kill); // select higher 16 bits from src1 if op_sel_hi is set
  HiMul_MI.addImm(ClampVal); // clamp
  // packed instructions do not support output modifiers. safe to assign them 0
  // for this use case
  HiMul_MI.addImm(0); // omod

  // Shift HiMul left by 16
  Register HiMulShifted = MRI.createVirtualRegister(RC);
  MachineInstrBuilder HiMulShifted_MI =
      BuildMI(MBB, I, DL, TII->get(AMDGPU::V_LSHLREV_B32_e64), HiMulShifted)
          .addReg(ShiftAmt)
          .addReg(HiMul);

  SubRegID = 0;
  if (DstMO.getSubReg())
    SubRegID = DstMO.getSubReg();
  // OR LoMul | (HiMul << 16)
  MachineInstrBuilder RewriteBackToDst_MI =
      BuildMI(MBB, I, DL, TII->get(AMDGPU::V_OR_B32_e64));
  if (SubRegID) {
    if (DstMO.isUndef()) {
      RewriteBackToDst_MI.addDef(DstReg, RegState::Undef, SubRegID);
    } else {
      RewriteBackToDst_MI.addDef(DstReg, 0, SubRegID);
    }
  } else {
    if (DstMO.isUndef()) {
      RewriteBackToDst_MI.addDef(DstReg, RegState::Undef);
    } else {
      RewriteBackToDst_MI.addDef(DstReg);
    }
  }
  RewriteBackToDst_MI.addReg(LoMul);
  RewriteBackToDst_MI.addReg(HiMulShifted);

  LIS->InsertMachineInstrInMaps(*LoMul_MI);
  LIS->InsertMachineInstrInMaps(*HiMul_MI);
  LIS->InsertMachineInstrInMaps(*HiMulShifted_MI);
  LIS->InsertMachineInstrInMaps(*RewriteBackToDst_MI);
  LIS->RemoveMachineInstrFromMaps(I);
  I.eraseFromParent();
  LIS->removeInterval(DstReg);
  LIS->createAndComputeVirtRegInterval(DstReg);
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
    IsF16MaskSet = false;
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
        if ((MI->getOpcode() == AMDGPU::V_PK_MUL_F16) ||
            (MI->getOpcode() == AMDGPU::V_PK_ADD_F16)) {
          processF16Unpacking(*MI, NumMFMACycles);
        } else {
          processF32Unpacking(*MI);
        }
      }
    }
  }
  return Changed;
}