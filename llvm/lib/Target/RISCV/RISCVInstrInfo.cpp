//===-- RISCVInstrInfo.cpp - RISC-V Instruction Information -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the RISC-V implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "RISCVInstrInfo.h"
#include "MCTargetDesc/RISCVMatInt.h"
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineCombinerPattern.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineTraceMetrics.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define GEN_CHECK_COMPRESS_INSTR
#include "RISCVGenCompressInstEmitter.inc"

#define GET_INSTRINFO_CTOR_DTOR
#define GET_INSTRINFO_NAMED_OPS
#include "RISCVGenInstrInfo.inc"

static cl::opt<bool> PreferWholeRegisterMove(
    "riscv-prefer-whole-register-move", cl::init(false), cl::Hidden,
    cl::desc("Prefer whole register move for vector registers."));

static cl::opt<MachineTraceStrategy> ForceMachineCombinerStrategy(
    "riscv-force-machine-combiner-strategy", cl::Hidden,
    cl::desc("Force machine combiner to use a specific strategy for machine "
             "trace metrics evaluation."),
    cl::init(MachineTraceStrategy::TS_NumStrategies),
    cl::values(clEnumValN(MachineTraceStrategy::TS_Local, "local",
                          "Local strategy."),
               clEnumValN(MachineTraceStrategy::TS_MinInstrCount, "min-instr",
                          "MinInstrCount strategy.")));

namespace llvm::RISCVVPseudosTable {

using namespace RISCV;

#define GET_RISCVVPseudosTable_IMPL
#include "RISCVGenSearchableTables.inc"

} // namespace llvm::RISCVVPseudosTable

RISCVInstrInfo::RISCVInstrInfo(RISCVSubtarget &STI)
    : RISCVGenInstrInfo(RISCV::ADJCALLSTACKDOWN, RISCV::ADJCALLSTACKUP),
      STI(STI) {}

MCInst RISCVInstrInfo::getNop() const {
  if (STI.hasStdExtCOrZca())
    return MCInstBuilder(RISCV::C_NOP);
  return MCInstBuilder(RISCV::ADDI)
      .addReg(RISCV::X0)
      .addReg(RISCV::X0)
      .addImm(0);
}

unsigned RISCVInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                             int &FrameIndex) const {
  unsigned Dummy;
  return isLoadFromStackSlot(MI, FrameIndex, Dummy);
}

unsigned RISCVInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                             int &FrameIndex,
                                             unsigned &MemBytes) const {
  switch (MI.getOpcode()) {
  default:
    return 0;
  case RISCV::LB:
  case RISCV::LBU:
    MemBytes = 1;
    break;
  case RISCV::LH:
  case RISCV::LHU:
  case RISCV::FLH:
    MemBytes = 2;
    break;
  case RISCV::LW:
  case RISCV::FLW:
  case RISCV::LWU:
    MemBytes = 4;
    break;
  case RISCV::LD:
  case RISCV::FLD:
    MemBytes = 8;
    break;
  }

  if (MI.getOperand(1).isFI() && MI.getOperand(2).isImm() &&
      MI.getOperand(2).getImm() == 0) {
    FrameIndex = MI.getOperand(1).getIndex();
    return MI.getOperand(0).getReg();
  }

  return 0;
}

unsigned RISCVInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                            int &FrameIndex) const {
  unsigned Dummy;
  return isStoreToStackSlot(MI, FrameIndex, Dummy);
}

unsigned RISCVInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                            int &FrameIndex,
                                            unsigned &MemBytes) const {
  switch (MI.getOpcode()) {
  default:
    return 0;
  case RISCV::SB:
    MemBytes = 1;
    break;
  case RISCV::SH:
  case RISCV::FSH:
    MemBytes = 2;
    break;
  case RISCV::SW:
  case RISCV::FSW:
    MemBytes = 4;
    break;
  case RISCV::SD:
  case RISCV::FSD:
    MemBytes = 8;
    break;
  }

  if (MI.getOperand(1).isFI() && MI.getOperand(2).isImm() &&
      MI.getOperand(2).getImm() == 0) {
    FrameIndex = MI.getOperand(1).getIndex();
    return MI.getOperand(0).getReg();
  }

  return 0;
}

static bool forwardCopyWillClobberTuple(unsigned DstReg, unsigned SrcReg,
                                        unsigned NumRegs) {
  return DstReg > SrcReg && (DstReg - SrcReg) < NumRegs;
}

static bool isConvertibleToVMV_V_V(const RISCVSubtarget &STI,
                                   const MachineBasicBlock &MBB,
                                   MachineBasicBlock::const_iterator MBBI,
                                   MachineBasicBlock::const_iterator &DefMBBI,
                                   RISCVII::VLMUL LMul) {
  if (PreferWholeRegisterMove)
    return false;

  assert(MBBI->getOpcode() == TargetOpcode::COPY &&
         "Unexpected COPY instruction.");
  Register SrcReg = MBBI->getOperand(1).getReg();
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();

  bool FoundDef = false;
  bool FirstVSetVLI = false;
  unsigned FirstSEW = 0;
  while (MBBI != MBB.begin()) {
    --MBBI;
    if (MBBI->isMetaInstruction())
      continue;

    if (MBBI->getOpcode() == RISCV::PseudoVSETVLI ||
        MBBI->getOpcode() == RISCV::PseudoVSETVLIX0 ||
        MBBI->getOpcode() == RISCV::PseudoVSETIVLI) {
      // There is a vsetvli between COPY and source define instruction.
      // vy = def_vop ...  (producing instruction)
      // ...
      // vsetvli
      // ...
      // vx = COPY vy
      if (!FoundDef) {
        if (!FirstVSetVLI) {
          FirstVSetVLI = true;
          unsigned FirstVType = MBBI->getOperand(2).getImm();
          RISCVII::VLMUL FirstLMul = RISCVVType::getVLMUL(FirstVType);
          FirstSEW = RISCVVType::getSEW(FirstVType);
          // The first encountered vsetvli must have the same lmul as the
          // register class of COPY.
          if (FirstLMul != LMul)
            return false;
        }
        // Only permit `vsetvli x0, x0, vtype` between COPY and the source
        // define instruction.
        if (MBBI->getOperand(0).getReg() != RISCV::X0)
          return false;
        if (MBBI->getOperand(1).isImm())
          return false;
        if (MBBI->getOperand(1).getReg() != RISCV::X0)
          return false;
        continue;
      }

      // MBBI is the first vsetvli before the producing instruction.
      unsigned VType = MBBI->getOperand(2).getImm();
      // If there is a vsetvli between COPY and the producing instruction.
      if (FirstVSetVLI) {
        // If SEW is different, return false.
        if (RISCVVType::getSEW(VType) != FirstSEW)
          return false;
      }

      // If the vsetvli is tail undisturbed, keep the whole register move.
      if (!RISCVVType::isTailAgnostic(VType))
        return false;

      // The checking is conservative. We only have register classes for
      // LMUL = 1/2/4/8. We should be able to convert vmv1r.v to vmv.v.v
      // for fractional LMUL operations. However, we could not use the vsetvli
      // lmul for widening operations. The result of widening operation is
      // 2 x LMUL.
      return LMul == RISCVVType::getVLMUL(VType);
    } else if (MBBI->isInlineAsm() || MBBI->isCall()) {
      return false;
    } else if (MBBI->getNumDefs()) {
      // Check all the instructions which will change VL.
      // For example, vleff has implicit def VL.
      if (MBBI->modifiesRegister(RISCV::VL))
        return false;

      // Only converting whole register copies to vmv.v.v when the defining
      // value appears in the explicit operands.
      for (const MachineOperand &MO : MBBI->explicit_operands()) {
        if (!MO.isReg() || !MO.isDef())
          continue;
        if (!FoundDef && TRI->regsOverlap(MO.getReg(), SrcReg)) {
          // We only permit the source of COPY has the same LMUL as the defined
          // operand.
          // There are cases we need to keep the whole register copy if the LMUL
          // is different.
          // For example,
          // $x0 = PseudoVSETIVLI 4, 73   // vsetivli zero, 4, e16,m2,ta,m
          // $v28m4 = PseudoVWADD_VV_M2 $v26m2, $v8m2
          // # The COPY may be created by vlmul_trunc intrinsic.
          // $v26m2 = COPY renamable $v28m2, implicit killed $v28m4
          //
          // After widening, the valid value will be 4 x e32 elements. If we
          // convert the COPY to vmv.v.v, it will only copy 4 x e16 elements.
          // FIXME: The COPY of subregister of Zvlsseg register will not be able
          // to convert to vmv.v.[v|i] under the constraint.
          if (MO.getReg() != SrcReg)
            return false;

          // In widening reduction instructions with LMUL_1 input vector case,
          // only checking the LMUL is insufficient due to reduction result is
          // always LMUL_1.
          // For example,
          // $x11 = PseudoVSETIVLI 1, 64 // vsetivli a1, 1, e8, m1, ta, mu
          // $v8m1 = PseudoVWREDSUM_VS_M1 $v26, $v27
          // $v26 = COPY killed renamable $v8
          // After widening, The valid value will be 1 x e16 elements. If we
          // convert the COPY to vmv.v.v, it will only copy 1 x e8 elements.
          uint64_t TSFlags = MBBI->getDesc().TSFlags;
          if (RISCVII::isRVVWideningReduction(TSFlags))
            return false;

          // If the producing instruction does not depend on vsetvli, do not
          // convert COPY to vmv.v.v. For example, VL1R_V or PseudoVRELOAD.
          if (!RISCVII::hasSEWOp(TSFlags) || !RISCVII::hasVLOp(TSFlags))
            return false;

          // Found the definition.
          FoundDef = true;
          DefMBBI = MBBI;
          break;
        }
      }
    }
  }

  return false;
}

void RISCVInstrInfo::copyPhysRegVector(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MBBI,
                                       const DebugLoc &DL, MCRegister DstReg,
                                       MCRegister SrcReg, bool KillSrc,
                                       unsigned Opc, unsigned NF) const {
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();

  RISCVII::VLMUL LMul;
  unsigned SubRegIdx;
  unsigned VVOpc, VIOpc;
  switch (Opc) {
  default:
    llvm_unreachable("Impossible LMUL for vector register copy.");
  case RISCV::VMV1R_V:
    LMul = RISCVII::LMUL_1;
    SubRegIdx = RISCV::sub_vrm1_0;
    VVOpc = RISCV::PseudoVMV_V_V_M1;
    VIOpc = RISCV::PseudoVMV_V_I_M1;
    break;
  case RISCV::VMV2R_V:
    LMul = RISCVII::LMUL_2;
    SubRegIdx = RISCV::sub_vrm2_0;
    VVOpc = RISCV::PseudoVMV_V_V_M2;
    VIOpc = RISCV::PseudoVMV_V_I_M2;
    break;
  case RISCV::VMV4R_V:
    LMul = RISCVII::LMUL_4;
    SubRegIdx = RISCV::sub_vrm4_0;
    VVOpc = RISCV::PseudoVMV_V_V_M4;
    VIOpc = RISCV::PseudoVMV_V_I_M4;
    break;
  case RISCV::VMV8R_V:
    assert(NF == 1);
    LMul = RISCVII::LMUL_8;
    SubRegIdx = RISCV::sub_vrm1_0; // There is no sub_vrm8_0.
    VVOpc = RISCV::PseudoVMV_V_V_M8;
    VIOpc = RISCV::PseudoVMV_V_I_M8;
    break;
  }

  bool UseVMV_V_V = false;
  bool UseVMV_V_I = false;
  MachineBasicBlock::const_iterator DefMBBI;
  if (isConvertibleToVMV_V_V(STI, MBB, MBBI, DefMBBI, LMul)) {
    UseVMV_V_V = true;
    Opc = VVOpc;

    if (DefMBBI->getOpcode() == VIOpc) {
      UseVMV_V_I = true;
      Opc = VIOpc;
    }
  }

  if (NF == 1) {
    auto MIB = BuildMI(MBB, MBBI, DL, get(Opc), DstReg);
    if (UseVMV_V_V)
      MIB.addReg(DstReg, RegState::Undef);
    if (UseVMV_V_I)
      MIB = MIB.add(DefMBBI->getOperand(2));
    else
      MIB = MIB.addReg(SrcReg, getKillRegState(KillSrc));
    if (UseVMV_V_V) {
      const MCInstrDesc &Desc = DefMBBI->getDesc();
      MIB.add(DefMBBI->getOperand(RISCVII::getVLOpNum(Desc)));  // AVL
      MIB.add(DefMBBI->getOperand(RISCVII::getSEWOpNum(Desc))); // SEW
      MIB.addImm(0);                                            // tu, mu
      MIB.addReg(RISCV::VL, RegState::Implicit);
      MIB.addReg(RISCV::VTYPE, RegState::Implicit);
    }
    return;
  }

  int I = 0, End = NF, Incr = 1;
  unsigned SrcEncoding = TRI->getEncodingValue(SrcReg);
  unsigned DstEncoding = TRI->getEncodingValue(DstReg);
  unsigned LMulVal;
  bool Fractional;
  std::tie(LMulVal, Fractional) = RISCVVType::decodeVLMUL(LMul);
  assert(!Fractional && "It is impossible be fractional lmul here.");
  if (forwardCopyWillClobberTuple(DstEncoding, SrcEncoding, NF * LMulVal)) {
    I = NF - 1;
    End = -1;
    Incr = -1;
  }

  for (; I != End; I += Incr) {
    auto MIB =
        BuildMI(MBB, MBBI, DL, get(Opc), TRI->getSubReg(DstReg, SubRegIdx + I));
    if (UseVMV_V_V)
      MIB.addReg(TRI->getSubReg(DstReg, SubRegIdx + I), RegState::Undef);
    if (UseVMV_V_I)
      MIB = MIB.add(DefMBBI->getOperand(2));
    else
      MIB = MIB.addReg(TRI->getSubReg(SrcReg, SubRegIdx + I),
                       getKillRegState(KillSrc));
    if (UseVMV_V_V) {
      const MCInstrDesc &Desc = DefMBBI->getDesc();
      MIB.add(DefMBBI->getOperand(RISCVII::getVLOpNum(Desc)));  // AVL
      MIB.add(DefMBBI->getOperand(RISCVII::getSEWOpNum(Desc))); // SEW
      MIB.addImm(0);                                            // tu, mu
      MIB.addReg(RISCV::VL, RegState::Implicit);
      MIB.addReg(RISCV::VTYPE, RegState::Implicit);
    }
  }
}

void RISCVInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 const DebugLoc &DL, MCRegister DstReg,
                                 MCRegister SrcReg, bool KillSrc) const {
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();

  if (RISCV::GPRRegClass.contains(DstReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(RISCV::ADDI), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addImm(0);
    return;
  }

  if (RISCV::GPRPF64RegClass.contains(DstReg, SrcReg)) {
    // Emit an ADDI for both parts of GPRPF64.
    BuildMI(MBB, MBBI, DL, get(RISCV::ADDI),
            TRI->getSubReg(DstReg, RISCV::sub_32))
        .addReg(TRI->getSubReg(SrcReg, RISCV::sub_32), getKillRegState(KillSrc))
        .addImm(0);
    BuildMI(MBB, MBBI, DL, get(RISCV::ADDI),
            TRI->getSubReg(DstReg, RISCV::sub_32_hi))
        .addReg(TRI->getSubReg(SrcReg, RISCV::sub_32_hi),
                getKillRegState(KillSrc))
        .addImm(0);
    return;
  }

  // Handle copy from csr
  if (RISCV::VCSRRegClass.contains(SrcReg) &&
      RISCV::GPRRegClass.contains(DstReg)) {
    BuildMI(MBB, MBBI, DL, get(RISCV::CSRRS), DstReg)
        .addImm(RISCVSysReg::lookupSysRegByName(TRI->getName(SrcReg))->Encoding)
        .addReg(RISCV::X0);
    return;
  }

  if (RISCV::FPR16RegClass.contains(DstReg, SrcReg)) {
    unsigned Opc;
    if (STI.hasStdExtZfh()) {
      Opc = RISCV::FSGNJ_H;
    } else {
      assert(STI.hasStdExtF() &&
             (STI.hasStdExtZfhmin() || STI.hasStdExtZfbfmin()) &&
             "Unexpected extensions");
      // Zfhmin/Zfbfmin doesn't have FSGNJ_H, replace FSGNJ_H with FSGNJ_S.
      DstReg = TRI->getMatchingSuperReg(DstReg, RISCV::sub_16,
                                        &RISCV::FPR32RegClass);
      SrcReg = TRI->getMatchingSuperReg(SrcReg, RISCV::sub_16,
                                        &RISCV::FPR32RegClass);
      Opc = RISCV::FSGNJ_S;
    }
    BuildMI(MBB, MBBI, DL, get(Opc), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (RISCV::FPR32RegClass.contains(DstReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(RISCV::FSGNJ_S), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (RISCV::FPR64RegClass.contains(DstReg, SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(RISCV::FSGNJ_D), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (RISCV::FPR32RegClass.contains(DstReg) &&
      RISCV::GPRRegClass.contains(SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(RISCV::FMV_W_X), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (RISCV::GPRRegClass.contains(DstReg) &&
      RISCV::FPR32RegClass.contains(SrcReg)) {
    BuildMI(MBB, MBBI, DL, get(RISCV::FMV_X_W), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (RISCV::FPR64RegClass.contains(DstReg) &&
      RISCV::GPRRegClass.contains(SrcReg)) {
    assert(STI.getXLen() == 64 && "Unexpected GPR size");
    BuildMI(MBB, MBBI, DL, get(RISCV::FMV_D_X), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  if (RISCV::GPRRegClass.contains(DstReg) &&
      RISCV::FPR64RegClass.contains(SrcReg)) {
    assert(STI.getXLen() == 64 && "Unexpected GPR size");
    BuildMI(MBB, MBBI, DL, get(RISCV::FMV_X_D), DstReg)
        .addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  // VR->VR copies.
  if (RISCV::VRRegClass.contains(DstReg, SrcReg)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RISCV::VMV1R_V);
    return;
  }

  if (RISCV::VRM2RegClass.contains(DstReg, SrcReg)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RISCV::VMV2R_V);
    return;
  }

  if (RISCV::VRM4RegClass.contains(DstReg, SrcReg)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RISCV::VMV4R_V);
    return;
  }

  if (RISCV::VRM8RegClass.contains(DstReg, SrcReg)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RISCV::VMV8R_V);
    return;
  }

  if (RISCV::VRN2M1RegClass.contains(DstReg, SrcReg)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RISCV::VMV1R_V,
                      /*NF=*/2);
    return;
  }

  if (RISCV::VRN2M2RegClass.contains(DstReg, SrcReg)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RISCV::VMV2R_V,
                      /*NF=*/2);
    return;
  }

  if (RISCV::VRN2M4RegClass.contains(DstReg, SrcReg)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RISCV::VMV4R_V,
                      /*NF=*/2);
    return;
  }

  if (RISCV::VRN3M1RegClass.contains(DstReg, SrcReg)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RISCV::VMV1R_V,
                      /*NF=*/3);
    return;
  }

  if (RISCV::VRN3M2RegClass.contains(DstReg, SrcReg)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RISCV::VMV2R_V,
                      /*NF=*/3);
    return;
  }

  if (RISCV::VRN4M1RegClass.contains(DstReg, SrcReg)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RISCV::VMV1R_V,
                      /*NF=*/4);
    return;
  }

  if (RISCV::VRN4M2RegClass.contains(DstReg, SrcReg)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RISCV::VMV2R_V,
                      /*NF=*/4);
    return;
  }

  if (RISCV::VRN5M1RegClass.contains(DstReg, SrcReg)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RISCV::VMV1R_V,
                      /*NF=*/5);
    return;
  }

  if (RISCV::VRN6M1RegClass.contains(DstReg, SrcReg)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RISCV::VMV1R_V,
                      /*NF=*/6);
    return;
  }

  if (RISCV::VRN7M1RegClass.contains(DstReg, SrcReg)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RISCV::VMV1R_V,
                      /*NF=*/7);
    return;
  }

  if (RISCV::VRN8M1RegClass.contains(DstReg, SrcReg)) {
    copyPhysRegVector(MBB, MBBI, DL, DstReg, SrcReg, KillSrc, RISCV::VMV1R_V,
                      /*NF=*/8);
    return;
  }

  llvm_unreachable("Impossible reg-to-reg copy");
}

void RISCVInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator I,
                                         Register SrcReg, bool IsKill, int FI,
                                         const TargetRegisterClass *RC,
                                         const TargetRegisterInfo *TRI,
                                         Register VReg) const {
  MachineFunction *MF = MBB.getParent();
  MachineFrameInfo &MFI = MF->getFrameInfo();

  unsigned Opcode;
  bool IsScalableVector = true;
  if (RISCV::GPRRegClass.hasSubClassEq(RC)) {
    Opcode = TRI->getRegSizeInBits(RISCV::GPRRegClass) == 32 ?
             RISCV::SW : RISCV::SD;
    IsScalableVector = false;
  } else if (RISCV::GPRPF64RegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::PseudoRV32ZdinxSD;
    IsScalableVector = false;
  } else if (RISCV::FPR16RegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::FSH;
    IsScalableVector = false;
  } else if (RISCV::FPR32RegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::FSW;
    IsScalableVector = false;
  } else if (RISCV::FPR64RegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::FSD;
    IsScalableVector = false;
  } else if (RISCV::VRRegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::VS1R_V;
  } else if (RISCV::VRM2RegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::VS2R_V;
  } else if (RISCV::VRM4RegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::VS4R_V;
  } else if (RISCV::VRM8RegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::VS8R_V;
  } else if (RISCV::VRN2M1RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVSPILL2_M1;
  else if (RISCV::VRN2M2RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVSPILL2_M2;
  else if (RISCV::VRN2M4RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVSPILL2_M4;
  else if (RISCV::VRN3M1RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVSPILL3_M1;
  else if (RISCV::VRN3M2RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVSPILL3_M2;
  else if (RISCV::VRN4M1RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVSPILL4_M1;
  else if (RISCV::VRN4M2RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVSPILL4_M2;
  else if (RISCV::VRN5M1RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVSPILL5_M1;
  else if (RISCV::VRN6M1RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVSPILL6_M1;
  else if (RISCV::VRN7M1RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVSPILL7_M1;
  else if (RISCV::VRN8M1RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVSPILL8_M1;
  else
    llvm_unreachable("Can't store this register to stack slot");

  if (IsScalableVector) {
    MachineMemOperand *MMO = MF->getMachineMemOperand(
        MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOStore,
        MemoryLocation::UnknownSize, MFI.getObjectAlign(FI));

    MFI.setStackID(FI, TargetStackID::ScalableVector);
    BuildMI(MBB, I, DebugLoc(), get(Opcode))
        .addReg(SrcReg, getKillRegState(IsKill))
        .addFrameIndex(FI)
        .addMemOperand(MMO);
  } else {
    MachineMemOperand *MMO = MF->getMachineMemOperand(
        MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOStore,
        MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

    BuildMI(MBB, I, DebugLoc(), get(Opcode))
        .addReg(SrcReg, getKillRegState(IsKill))
        .addFrameIndex(FI)
        .addImm(0)
        .addMemOperand(MMO);
  }
}

void RISCVInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator I,
                                          Register DstReg, int FI,
                                          const TargetRegisterClass *RC,
                                          const TargetRegisterInfo *TRI,
                                          Register VReg) const {
  MachineFunction *MF = MBB.getParent();
  MachineFrameInfo &MFI = MF->getFrameInfo();

  unsigned Opcode;
  bool IsScalableVector = true;
  if (RISCV::GPRRegClass.hasSubClassEq(RC)) {
    Opcode = TRI->getRegSizeInBits(RISCV::GPRRegClass) == 32 ?
             RISCV::LW : RISCV::LD;
    IsScalableVector = false;
  } else if (RISCV::GPRPF64RegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::PseudoRV32ZdinxLD;
    IsScalableVector = false;
  } else if (RISCV::FPR16RegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::FLH;
    IsScalableVector = false;
  } else if (RISCV::FPR32RegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::FLW;
    IsScalableVector = false;
  } else if (RISCV::FPR64RegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::FLD;
    IsScalableVector = false;
  } else if (RISCV::VRRegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::VL1RE8_V;
  } else if (RISCV::VRM2RegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::VL2RE8_V;
  } else if (RISCV::VRM4RegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::VL4RE8_V;
  } else if (RISCV::VRM8RegClass.hasSubClassEq(RC)) {
    Opcode = RISCV::VL8RE8_V;
  } else if (RISCV::VRN2M1RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVRELOAD2_M1;
  else if (RISCV::VRN2M2RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVRELOAD2_M2;
  else if (RISCV::VRN2M4RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVRELOAD2_M4;
  else if (RISCV::VRN3M1RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVRELOAD3_M1;
  else if (RISCV::VRN3M2RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVRELOAD3_M2;
  else if (RISCV::VRN4M1RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVRELOAD4_M1;
  else if (RISCV::VRN4M2RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVRELOAD4_M2;
  else if (RISCV::VRN5M1RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVRELOAD5_M1;
  else if (RISCV::VRN6M1RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVRELOAD6_M1;
  else if (RISCV::VRN7M1RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVRELOAD7_M1;
  else if (RISCV::VRN8M1RegClass.hasSubClassEq(RC))
    Opcode = RISCV::PseudoVRELOAD8_M1;
  else
    llvm_unreachable("Can't load this register from stack slot");

  if (IsScalableVector) {
    MachineMemOperand *MMO = MF->getMachineMemOperand(
        MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOLoad,
        MemoryLocation::UnknownSize, MFI.getObjectAlign(FI));

    MFI.setStackID(FI, TargetStackID::ScalableVector);
    BuildMI(MBB, I, DebugLoc(), get(Opcode), DstReg)
        .addFrameIndex(FI)
        .addMemOperand(MMO);
  } else {
    MachineMemOperand *MMO = MF->getMachineMemOperand(
        MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOLoad,
        MFI.getObjectSize(FI), MFI.getObjectAlign(FI));

    BuildMI(MBB, I, DebugLoc(), get(Opcode), DstReg)
        .addFrameIndex(FI)
        .addImm(0)
        .addMemOperand(MMO);
  }
}

MachineInstr *RISCVInstrInfo::foldMemoryOperandImpl(
    MachineFunction &MF, MachineInstr &MI, ArrayRef<unsigned> Ops,
    MachineBasicBlock::iterator InsertPt, int FrameIndex, LiveIntervals *LIS,
    VirtRegMap *VRM) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  // The below optimizations narrow the load so they are only valid for little
  // endian.
  // TODO: Support big endian by adding an offset into the frame object?
  if (MF.getDataLayout().isBigEndian())
    return nullptr;

  // Fold load from stack followed by sext.b/sext.h/sext.w/zext.b/zext.h/zext.w.
  if (Ops.size() != 1 || Ops[0] != 1)
   return nullptr;

  unsigned LoadOpc;
  switch (MI.getOpcode()) {
  default:
    if (RISCV::isSEXT_W(MI)) {
      LoadOpc = RISCV::LW;
      break;
    }
    if (RISCV::isZEXT_W(MI)) {
      LoadOpc = RISCV::LWU;
      break;
    }
    if (RISCV::isZEXT_B(MI)) {
      LoadOpc = RISCV::LBU;
      break;
    }
    return nullptr;
  case RISCV::SEXT_H:
    LoadOpc = RISCV::LH;
    break;
  case RISCV::SEXT_B:
    LoadOpc = RISCV::LB;
    break;
  case RISCV::ZEXT_H_RV32:
  case RISCV::ZEXT_H_RV64:
    LoadOpc = RISCV::LHU;
    break;
  }

  MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo::getFixedStack(MF, FrameIndex),
      MachineMemOperand::MOLoad, MFI.getObjectSize(FrameIndex),
      MFI.getObjectAlign(FrameIndex));

  Register DstReg = MI.getOperand(0).getReg();
  return BuildMI(*MI.getParent(), InsertPt, MI.getDebugLoc(), get(LoadOpc),
                 DstReg)
      .addFrameIndex(FrameIndex)
      .addImm(0)
      .addMemOperand(MMO);
}

void RISCVInstrInfo::movImm(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI,
                            const DebugLoc &DL, Register DstReg, uint64_t Val,
                            MachineInstr::MIFlag Flag, bool DstRenamable,
                            bool DstIsDead) const {
  Register SrcReg = RISCV::X0;

  if (!STI.is64Bit() && !isInt<32>(Val))
    report_fatal_error("Should only materialize 32-bit constants for RV32");

  RISCVMatInt::InstSeq Seq =
      RISCVMatInt::generateInstSeq(Val, STI.getFeatureBits());
  assert(!Seq.empty());

  bool SrcRenamable = false;
  unsigned Num = 0;

  for (const RISCVMatInt::Inst &Inst : Seq) {
    bool LastItem = ++Num == Seq.size();
    unsigned DstRegState = getDeadRegState(DstIsDead && LastItem) |
                           getRenamableRegState(DstRenamable);
    unsigned SrcRegState = getKillRegState(SrcReg != RISCV::X0) |
                           getRenamableRegState(SrcRenamable);
    switch (Inst.getOpndKind()) {
    case RISCVMatInt::Imm:
      BuildMI(MBB, MBBI, DL, get(Inst.getOpcode()))
          .addReg(DstReg, RegState::Define | DstRegState)
          .addImm(Inst.getImm())
          .setMIFlag(Flag);
      break;
    case RISCVMatInt::RegX0:
      BuildMI(MBB, MBBI, DL, get(Inst.getOpcode()))
          .addReg(DstReg, RegState::Define | DstRegState)
          .addReg(SrcReg, SrcRegState)
          .addReg(RISCV::X0)
          .setMIFlag(Flag);
      break;
    case RISCVMatInt::RegReg:
      BuildMI(MBB, MBBI, DL, get(Inst.getOpcode()))
          .addReg(DstReg, RegState::Define | DstRegState)
          .addReg(SrcReg, SrcRegState)
          .addReg(SrcReg, SrcRegState)
          .setMIFlag(Flag);
      break;
    case RISCVMatInt::RegImm:
      BuildMI(MBB, MBBI, DL, get(Inst.getOpcode()))
          .addReg(DstReg, RegState::Define | DstRegState)
          .addReg(SrcReg, SrcRegState)
          .addImm(Inst.getImm())
          .setMIFlag(Flag);
      break;
    }

    // Only the first instruction has X0 as its source.
    SrcReg = DstReg;
    SrcRenamable = DstRenamable;
  }
}

static RISCVCC::CondCode getCondFromBranchOpc(unsigned Opc) {
  switch (Opc) {
  default:
    return RISCVCC::COND_INVALID;
  case RISCV::BEQ:
    return RISCVCC::COND_EQ;
  case RISCV::BNE:
    return RISCVCC::COND_NE;
  case RISCV::BLT:
    return RISCVCC::COND_LT;
  case RISCV::BGE:
    return RISCVCC::COND_GE;
  case RISCV::BLTU:
    return RISCVCC::COND_LTU;
  case RISCV::BGEU:
    return RISCVCC::COND_GEU;
  }
}

// The contents of values added to Cond are not examined outside of
// RISCVInstrInfo, giving us flexibility in what to push to it. For RISCV, we
// push BranchOpcode, Reg1, Reg2.
static void parseCondBranch(MachineInstr &LastInst, MachineBasicBlock *&Target,
                            SmallVectorImpl<MachineOperand> &Cond) {
  // Block ends with fall-through condbranch.
  assert(LastInst.getDesc().isConditionalBranch() &&
         "Unknown conditional branch");
  Target = LastInst.getOperand(2).getMBB();
  unsigned CC = getCondFromBranchOpc(LastInst.getOpcode());
  Cond.push_back(MachineOperand::CreateImm(CC));
  Cond.push_back(LastInst.getOperand(0));
  Cond.push_back(LastInst.getOperand(1));
}

const MCInstrDesc &RISCVInstrInfo::getBrCond(RISCVCC::CondCode CC) const {
  switch (CC) {
  default:
    llvm_unreachable("Unknown condition code!");
  case RISCVCC::COND_EQ:
    return get(RISCV::BEQ);
  case RISCVCC::COND_NE:
    return get(RISCV::BNE);
  case RISCVCC::COND_LT:
    return get(RISCV::BLT);
  case RISCVCC::COND_GE:
    return get(RISCV::BGE);
  case RISCVCC::COND_LTU:
    return get(RISCV::BLTU);
  case RISCVCC::COND_GEU:
    return get(RISCV::BGEU);
  }
}

RISCVCC::CondCode RISCVCC::getOppositeBranchCondition(RISCVCC::CondCode CC) {
  switch (CC) {
  default:
    llvm_unreachable("Unrecognized conditional branch");
  case RISCVCC::COND_EQ:
    return RISCVCC::COND_NE;
  case RISCVCC::COND_NE:
    return RISCVCC::COND_EQ;
  case RISCVCC::COND_LT:
    return RISCVCC::COND_GE;
  case RISCVCC::COND_GE:
    return RISCVCC::COND_LT;
  case RISCVCC::COND_LTU:
    return RISCVCC::COND_GEU;
  case RISCVCC::COND_GEU:
    return RISCVCC::COND_LTU;
  }
}

bool RISCVInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                   MachineBasicBlock *&TBB,
                                   MachineBasicBlock *&FBB,
                                   SmallVectorImpl<MachineOperand> &Cond,
                                   bool AllowModify) const {
  TBB = FBB = nullptr;
  Cond.clear();

  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end() || !isUnpredicatedTerminator(*I))
    return false;

  // Count the number of terminators and find the first unconditional or
  // indirect branch.
  MachineBasicBlock::iterator FirstUncondOrIndirectBr = MBB.end();
  int NumTerminators = 0;
  for (auto J = I.getReverse(); J != MBB.rend() && isUnpredicatedTerminator(*J);
       J++) {
    NumTerminators++;
    if (J->getDesc().isUnconditionalBranch() ||
        J->getDesc().isIndirectBranch()) {
      FirstUncondOrIndirectBr = J.getReverse();
    }
  }

  // If AllowModify is true, we can erase any terminators after
  // FirstUncondOrIndirectBR.
  if (AllowModify && FirstUncondOrIndirectBr != MBB.end()) {
    while (std::next(FirstUncondOrIndirectBr) != MBB.end()) {
      std::next(FirstUncondOrIndirectBr)->eraseFromParent();
      NumTerminators--;
    }
    I = FirstUncondOrIndirectBr;
  }

  // We can't handle blocks that end in an indirect branch.
  if (I->getDesc().isIndirectBranch())
    return true;

  // We can't handle Generic branch opcodes from Global ISel.
  if (I->isPreISelOpcode())
    return true;

  // We can't handle blocks with more than 2 terminators.
  if (NumTerminators > 2)
    return true;

  // Handle a single unconditional branch.
  if (NumTerminators == 1 && I->getDesc().isUnconditionalBranch()) {
    TBB = getBranchDestBlock(*I);
    return false;
  }

  // Handle a single conditional branch.
  if (NumTerminators == 1 && I->getDesc().isConditionalBranch()) {
    parseCondBranch(*I, TBB, Cond);
    return false;
  }

  // Handle a conditional branch followed by an unconditional branch.
  if (NumTerminators == 2 && std::prev(I)->getDesc().isConditionalBranch() &&
      I->getDesc().isUnconditionalBranch()) {
    parseCondBranch(*std::prev(I), TBB, Cond);
    FBB = getBranchDestBlock(*I);
    return false;
  }

  // Otherwise, we can't handle this.
  return true;
}

unsigned RISCVInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                      int *BytesRemoved) const {
  if (BytesRemoved)
    *BytesRemoved = 0;
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end())
    return 0;

  if (!I->getDesc().isUnconditionalBranch() &&
      !I->getDesc().isConditionalBranch())
    return 0;

  // Remove the branch.
  if (BytesRemoved)
    *BytesRemoved += getInstSizeInBytes(*I);
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin())
    return 1;
  --I;
  if (!I->getDesc().isConditionalBranch())
    return 1;

  // Remove the branch.
  if (BytesRemoved)
    *BytesRemoved += getInstSizeInBytes(*I);
  I->eraseFromParent();
  return 2;
}

// Inserts a branch into the end of the specific MachineBasicBlock, returning
// the number of instructions inserted.
unsigned RISCVInstrInfo::insertBranch(
    MachineBasicBlock &MBB, MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, const DebugLoc &DL, int *BytesAdded) const {
  if (BytesAdded)
    *BytesAdded = 0;

  // Shouldn't be a fall through.
  assert(TBB && "insertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 3 || Cond.size() == 0) &&
         "RISC-V branch conditions have two components!");

  // Unconditional branch.
  if (Cond.empty()) {
    MachineInstr &MI = *BuildMI(&MBB, DL, get(RISCV::PseudoBR)).addMBB(TBB);
    if (BytesAdded)
      *BytesAdded += getInstSizeInBytes(MI);
    return 1;
  }

  // Either a one or two-way conditional branch.
  auto CC = static_cast<RISCVCC::CondCode>(Cond[0].getImm());
  MachineInstr &CondMI =
      *BuildMI(&MBB, DL, getBrCond(CC)).add(Cond[1]).add(Cond[2]).addMBB(TBB);
  if (BytesAdded)
    *BytesAdded += getInstSizeInBytes(CondMI);

  // One-way conditional branch.
  if (!FBB)
    return 1;

  // Two-way conditional branch.
  MachineInstr &MI = *BuildMI(&MBB, DL, get(RISCV::PseudoBR)).addMBB(FBB);
  if (BytesAdded)
    *BytesAdded += getInstSizeInBytes(MI);
  return 2;
}

void RISCVInstrInfo::insertIndirectBranch(MachineBasicBlock &MBB,
                                          MachineBasicBlock &DestBB,
                                          MachineBasicBlock &RestoreBB,
                                          const DebugLoc &DL, int64_t BrOffset,
                                          RegScavenger *RS) const {
  assert(RS && "RegScavenger required for long branching");
  assert(MBB.empty() &&
         "new block should be inserted for expanding unconditional branch");
  assert(MBB.pred_size() == 1);
  assert(RestoreBB.empty() &&
         "restore block should be inserted for restoring clobbered registers");

  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  RISCVMachineFunctionInfo *RVFI = MF->getInfo<RISCVMachineFunctionInfo>();
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();

  if (!isInt<32>(BrOffset))
    report_fatal_error(
        "Branch offsets outside of the signed 32-bit range not supported");

  // FIXME: A virtual register must be used initially, as the register
  // scavenger won't work with empty blocks (SIInstrInfo::insertIndirectBranch
  // uses the same workaround).
  Register ScratchReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
  auto II = MBB.end();
  // We may also update the jump target to RestoreBB later.
  MachineInstr &MI = *BuildMI(MBB, II, DL, get(RISCV::PseudoJump))
                          .addReg(ScratchReg, RegState::Define | RegState::Dead)
                          .addMBB(&DestBB, RISCVII::MO_CALL);

  RS->enterBasicBlockEnd(MBB);
  Register TmpGPR =
      RS->scavengeRegisterBackwards(RISCV::GPRRegClass, MI.getIterator(),
                                    /*RestoreAfter=*/false, /*SpAdj=*/0,
                                    /*AllowSpill=*/false);
  if (TmpGPR != RISCV::NoRegister)
    RS->setRegUsed(TmpGPR);
  else {
    // The case when there is no scavenged register needs special handling.

    // Pick s11 because it doesn't make a difference.
    TmpGPR = RISCV::X27;

    int FrameIndex = RVFI->getBranchRelaxationScratchFrameIndex();
    if (FrameIndex == -1)
      report_fatal_error("underestimated function size");

    storeRegToStackSlot(MBB, MI, TmpGPR, /*IsKill=*/true, FrameIndex,
                        &RISCV::GPRRegClass, TRI, Register());
    TRI->eliminateFrameIndex(std::prev(MI.getIterator()),
                             /*SpAdj=*/0, /*FIOperandNum=*/1);

    MI.getOperand(1).setMBB(&RestoreBB);

    loadRegFromStackSlot(RestoreBB, RestoreBB.end(), TmpGPR, FrameIndex,
                         &RISCV::GPRRegClass, TRI, Register());
    TRI->eliminateFrameIndex(RestoreBB.back(),
                             /*SpAdj=*/0, /*FIOperandNum=*/1);
  }

  MRI.replaceRegWith(ScratchReg, TmpGPR);
  MRI.clearVirtRegs();
}

bool RISCVInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  assert((Cond.size() == 3) && "Invalid branch condition!");
  auto CC = static_cast<RISCVCC::CondCode>(Cond[0].getImm());
  Cond[0].setImm(getOppositeBranchCondition(CC));
  return false;
}

MachineBasicBlock *
RISCVInstrInfo::getBranchDestBlock(const MachineInstr &MI) const {
  assert(MI.getDesc().isBranch() && "Unexpected opcode!");
  // The branch target is always the last operand.
  int NumOp = MI.getNumExplicitOperands();
  return MI.getOperand(NumOp - 1).getMBB();
}

bool RISCVInstrInfo::isBranchOffsetInRange(unsigned BranchOp,
                                           int64_t BrOffset) const {
  unsigned XLen = STI.getXLen();
  // Ideally we could determine the supported branch offset from the
  // RISCVII::FormMask, but this can't be used for Pseudo instructions like
  // PseudoBR.
  switch (BranchOp) {
  default:
    llvm_unreachable("Unexpected opcode!");
  case RISCV::BEQ:
  case RISCV::BNE:
  case RISCV::BLT:
  case RISCV::BGE:
  case RISCV::BLTU:
  case RISCV::BGEU:
    return isIntN(13, BrOffset);
  case RISCV::JAL:
  case RISCV::PseudoBR:
    return isIntN(21, BrOffset);
  case RISCV::PseudoJump:
    return isIntN(32, SignExtend64(BrOffset + 0x800, XLen));
  }
}

// If the operation has a predicated pseudo instruction, return the pseudo
// instruction opcode. Otherwise, return RISCV::INSTRUCTION_LIST_END.
// TODO: Support more operations.
unsigned getPredicatedOpcode(unsigned Opcode) {
  switch (Opcode) {
  case RISCV::ADD:   return RISCV::PseudoCCADD;   break;
  case RISCV::SUB:   return RISCV::PseudoCCSUB;   break;
  case RISCV::SLL:   return RISCV::PseudoCCSLL;   break;
  case RISCV::SRL:   return RISCV::PseudoCCSRL;   break;
  case RISCV::SRA:   return RISCV::PseudoCCSRA;   break;
  case RISCV::AND:   return RISCV::PseudoCCAND;   break;
  case RISCV::OR:    return RISCV::PseudoCCOR;    break;
  case RISCV::XOR:   return RISCV::PseudoCCXOR;   break;

  case RISCV::ADDI:  return RISCV::PseudoCCADDI;  break;
  case RISCV::SLLI:  return RISCV::PseudoCCSLLI;  break;
  case RISCV::SRLI:  return RISCV::PseudoCCSRLI;  break;
  case RISCV::SRAI:  return RISCV::PseudoCCSRAI;  break;
  case RISCV::ANDI:  return RISCV::PseudoCCANDI;  break;
  case RISCV::ORI:   return RISCV::PseudoCCORI;   break;
  case RISCV::XORI:  return RISCV::PseudoCCXORI;  break;

  case RISCV::ADDW:  return RISCV::PseudoCCADDW;  break;
  case RISCV::SUBW:  return RISCV::PseudoCCSUBW;  break;
  case RISCV::SLLW:  return RISCV::PseudoCCSLLW;  break;
  case RISCV::SRLW:  return RISCV::PseudoCCSRLW;  break;
  case RISCV::SRAW:  return RISCV::PseudoCCSRAW;  break;

  case RISCV::ADDIW: return RISCV::PseudoCCADDIW; break;
  case RISCV::SLLIW: return RISCV::PseudoCCSLLIW; break;
  case RISCV::SRLIW: return RISCV::PseudoCCSRLIW; break;
  case RISCV::SRAIW: return RISCV::PseudoCCSRAIW; break;
  }

  return RISCV::INSTRUCTION_LIST_END;
}

/// Identify instructions that can be folded into a CCMOV instruction, and
/// return the defining instruction.
static MachineInstr *canFoldAsPredicatedOp(Register Reg,
                                           const MachineRegisterInfo &MRI,
                                           const TargetInstrInfo *TII) {
  if (!Reg.isVirtual())
    return nullptr;
  if (!MRI.hasOneNonDBGUse(Reg))
    return nullptr;
  MachineInstr *MI = MRI.getVRegDef(Reg);
  if (!MI)
    return nullptr;
  // Check if MI can be predicated and folded into the CCMOV.
  if (getPredicatedOpcode(MI->getOpcode()) == RISCV::INSTRUCTION_LIST_END)
    return nullptr;
  // Don't predicate li idiom.
  if (MI->getOpcode() == RISCV::ADDI && MI->getOperand(1).isReg() &&
      MI->getOperand(1).getReg() == RISCV::X0)
    return nullptr;
  // Check if MI has any other defs or physreg uses.
  for (const MachineOperand &MO : llvm::drop_begin(MI->operands())) {
    // Reject frame index operands, PEI can't handle the predicated pseudos.
    if (MO.isFI() || MO.isCPI() || MO.isJTI())
      return nullptr;
    if (!MO.isReg())
      continue;
    // MI can't have any tied operands, that would conflict with predication.
    if (MO.isTied())
      return nullptr;
    if (MO.isDef())
      return nullptr;
    // Allow constant physregs.
    if (MO.getReg().isPhysical() && !MRI.isConstantPhysReg(MO.getReg()))
      return nullptr;
  }
  bool DontMoveAcrossStores = true;
  if (!MI->isSafeToMove(/* AliasAnalysis = */ nullptr, DontMoveAcrossStores))
    return nullptr;
  return MI;
}

bool RISCVInstrInfo::analyzeSelect(const MachineInstr &MI,
                                   SmallVectorImpl<MachineOperand> &Cond,
                                   unsigned &TrueOp, unsigned &FalseOp,
                                   bool &Optimizable) const {
  assert(MI.getOpcode() == RISCV::PseudoCCMOVGPR &&
         "Unknown select instruction");
  // CCMOV operands:
  // 0: Def.
  // 1: LHS of compare.
  // 2: RHS of compare.
  // 3: Condition code.
  // 4: False use.
  // 5: True use.
  TrueOp = 5;
  FalseOp = 4;
  Cond.push_back(MI.getOperand(1));
  Cond.push_back(MI.getOperand(2));
  Cond.push_back(MI.getOperand(3));
  // We can only fold when we support short forward branch opt.
  Optimizable = STI.hasShortForwardBranchOpt();
  return false;
}

MachineInstr *
RISCVInstrInfo::optimizeSelect(MachineInstr &MI,
                               SmallPtrSetImpl<MachineInstr *> &SeenMIs,
                               bool PreferFalse) const {
  assert(MI.getOpcode() == RISCV::PseudoCCMOVGPR &&
         "Unknown select instruction");
  if (!STI.hasShortForwardBranchOpt())
    return nullptr;

  MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
  MachineInstr *DefMI =
      canFoldAsPredicatedOp(MI.getOperand(5).getReg(), MRI, this);
  bool Invert = !DefMI;
  if (!DefMI)
    DefMI = canFoldAsPredicatedOp(MI.getOperand(4).getReg(), MRI, this);
  if (!DefMI)
    return nullptr;

  // Find new register class to use.
  MachineOperand FalseReg = MI.getOperand(Invert ? 5 : 4);
  Register DestReg = MI.getOperand(0).getReg();
  const TargetRegisterClass *PreviousClass = MRI.getRegClass(FalseReg.getReg());
  if (!MRI.constrainRegClass(DestReg, PreviousClass))
    return nullptr;

  unsigned PredOpc = getPredicatedOpcode(DefMI->getOpcode());
  assert(PredOpc != RISCV::INSTRUCTION_LIST_END && "Unexpected opcode!");

  // Create a new predicated version of DefMI.
  MachineInstrBuilder NewMI =
      BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), get(PredOpc), DestReg);

  // Copy the condition portion.
  NewMI.add(MI.getOperand(1));
  NewMI.add(MI.getOperand(2));

  // Add condition code, inverting if necessary.
  auto CC = static_cast<RISCVCC::CondCode>(MI.getOperand(3).getImm());
  if (Invert)
    CC = RISCVCC::getOppositeBranchCondition(CC);
  NewMI.addImm(CC);

  // Copy the false register.
  NewMI.add(FalseReg);

  // Copy all the DefMI operands.
  const MCInstrDesc &DefDesc = DefMI->getDesc();
  for (unsigned i = 1, e = DefDesc.getNumOperands(); i != e; ++i)
    NewMI.add(DefMI->getOperand(i));

  // Update SeenMIs set: register newly created MI and erase removed DefMI.
  SeenMIs.insert(NewMI);
  SeenMIs.erase(DefMI);

  // If MI is inside a loop, and DefMI is outside the loop, then kill flags on
  // DefMI would be invalid when tranferred inside the loop.  Checking for a
  // loop is expensive, but at least remove kill flags if they are in different
  // BBs.
  if (DefMI->getParent() != MI.getParent())
    NewMI->clearKillInfo();

  // The caller will erase MI, but not DefMI.
  DefMI->eraseFromParent();
  return NewMI;
}

unsigned RISCVInstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  if (MI.isMetaInstruction())
    return 0;

  unsigned Opcode = MI.getOpcode();

  if (Opcode == TargetOpcode::INLINEASM ||
      Opcode == TargetOpcode::INLINEASM_BR) {
    const MachineFunction &MF = *MI.getParent()->getParent();
    const auto &TM = static_cast<const RISCVTargetMachine &>(MF.getTarget());
    return getInlineAsmLength(MI.getOperand(0).getSymbolName(),
                              *TM.getMCAsmInfo());
  }

  if (!MI.memoperands_empty()) {
    MachineMemOperand *MMO = *(MI.memoperands_begin());
    const MachineFunction &MF = *MI.getParent()->getParent();
    const auto &ST = MF.getSubtarget<RISCVSubtarget>();
    if (ST.hasStdExtZihintntl() && MMO->isNonTemporal()) {
      if (ST.hasStdExtCOrZca() && ST.enableRVCHintInstrs()) {
        if (isCompressibleInst(MI, STI))
          return 4; // c.ntl.all + c.load/c.store
        return 6;   // c.ntl.all + load/store
      }
      return 8; // ntl.all + load/store
    }
  }

  if (Opcode == TargetOpcode::BUNDLE)
    return getInstBundleLength(MI);

  if (MI.getParent() && MI.getParent()->getParent()) {
    if (isCompressibleInst(MI, STI))
      return 2;
  }

  switch (Opcode) {
  case TargetOpcode::STACKMAP:
    // The upper bound for a stackmap intrinsic is the full length of its shadow
    return StackMapOpers(&MI).getNumPatchBytes();
  case TargetOpcode::PATCHPOINT:
    // The size of the patchpoint intrinsic is the number of bytes requested
    return PatchPointOpers(&MI).getNumPatchBytes();
  case TargetOpcode::STATEPOINT:
    // The size of the statepoint intrinsic is the number of bytes requested
    return StatepointOpers(&MI).getNumPatchBytes();
  default:
    return get(Opcode).getSize();
  }
}

unsigned RISCVInstrInfo::getInstBundleLength(const MachineInstr &MI) const {
  unsigned Size = 0;
  MachineBasicBlock::const_instr_iterator I = MI.getIterator();
  MachineBasicBlock::const_instr_iterator E = MI.getParent()->instr_end();
  while (++I != E && I->isInsideBundle()) {
    assert(!I->isBundle() && "No nested bundle!");
    Size += getInstSizeInBytes(*I);
  }
  return Size;
}

bool RISCVInstrInfo::isAsCheapAsAMove(const MachineInstr &MI) const {
  const unsigned Opcode = MI.getOpcode();
  switch (Opcode) {
  default:
    break;
  case RISCV::FSGNJ_D:
  case RISCV::FSGNJ_S:
  case RISCV::FSGNJ_H:
  case RISCV::FSGNJ_D_INX:
  case RISCV::FSGNJ_D_IN32X:
  case RISCV::FSGNJ_S_INX:
  case RISCV::FSGNJ_H_INX:
    // The canonical floating-point move is fsgnj rd, rs, rs.
    return MI.getOperand(1).isReg() && MI.getOperand(2).isReg() &&
           MI.getOperand(1).getReg() == MI.getOperand(2).getReg();
  case RISCV::ADDI:
  case RISCV::ORI:
  case RISCV::XORI:
    return (MI.getOperand(1).isReg() &&
            MI.getOperand(1).getReg() == RISCV::X0) ||
           (MI.getOperand(2).isImm() && MI.getOperand(2).getImm() == 0);
  }
  return MI.isAsCheapAsAMove();
}

std::optional<DestSourcePair>
RISCVInstrInfo::isCopyInstrImpl(const MachineInstr &MI) const {
  if (MI.isMoveReg())
    return DestSourcePair{MI.getOperand(0), MI.getOperand(1)};
  switch (MI.getOpcode()) {
  default:
    break;
  case RISCV::ADDI:
    // Operand 1 can be a frameindex but callers expect registers
    if (MI.getOperand(1).isReg() && MI.getOperand(2).isImm() &&
        MI.getOperand(2).getImm() == 0)
      return DestSourcePair{MI.getOperand(0), MI.getOperand(1)};
    break;
  case RISCV::FSGNJ_D:
  case RISCV::FSGNJ_S:
  case RISCV::FSGNJ_H:
  case RISCV::FSGNJ_D_INX:
  case RISCV::FSGNJ_D_IN32X:
  case RISCV::FSGNJ_S_INX:
  case RISCV::FSGNJ_H_INX:
    // The canonical floating-point move is fsgnj rd, rs, rs.
    if (MI.getOperand(1).isReg() && MI.getOperand(2).isReg() &&
        MI.getOperand(1).getReg() == MI.getOperand(2).getReg())
      return DestSourcePair{MI.getOperand(0), MI.getOperand(1)};
    break;
  }
  return std::nullopt;
}

MachineTraceStrategy RISCVInstrInfo::getMachineCombinerTraceStrategy() const {
  if (ForceMachineCombinerStrategy.getNumOccurrences() == 0) {
    // The option is unused. Choose Local strategy only for in-order cores. When
    // scheduling model is unspecified, use MinInstrCount strategy as more
    // generic one.
    const auto &SchedModel = STI.getSchedModel();
    return (!SchedModel.hasInstrSchedModel() || SchedModel.isOutOfOrder())
               ? MachineTraceStrategy::TS_MinInstrCount
               : MachineTraceStrategy::TS_Local;
  }
  // The strategy was forced by the option.
  return ForceMachineCombinerStrategy;
}

void RISCVInstrInfo::setSpecialOperandAttr(MachineInstr &OldMI1,
                                           MachineInstr &OldMI2,
                                           MachineInstr &NewMI1,
                                           MachineInstr &NewMI2) const {
  uint32_t IntersectedFlags = OldMI1.getFlags() & OldMI2.getFlags();
  NewMI1.setFlags(IntersectedFlags);
  NewMI2.setFlags(IntersectedFlags);
}

void RISCVInstrInfo::finalizeInsInstrs(
    MachineInstr &Root, MachineCombinerPattern &P,
    SmallVectorImpl<MachineInstr *> &InsInstrs) const {
  int16_t FrmOpIdx =
      RISCV::getNamedOperandIdx(Root.getOpcode(), RISCV::OpName::frm);
  if (FrmOpIdx < 0) {
    assert(all_of(InsInstrs,
                  [](MachineInstr *MI) {
                    return RISCV::getNamedOperandIdx(MI->getOpcode(),
                                                     RISCV::OpName::frm) < 0;
                  }) &&
           "New instructions require FRM whereas the old one does not have it");
    return;
  }

  const MachineOperand &FRM = Root.getOperand(FrmOpIdx);
  MachineFunction &MF = *Root.getMF();

  for (auto *NewMI : InsInstrs) {
    assert(static_cast<unsigned>(RISCV::getNamedOperandIdx(
               NewMI->getOpcode(), RISCV::OpName::frm)) ==
               NewMI->getNumOperands() &&
           "Instruction has unexpected number of operands");
    MachineInstrBuilder MIB(MF, NewMI);
    MIB.add(FRM);
    if (FRM.getImm() == RISCVFPRndMode::DYN)
      MIB.addUse(RISCV::FRM, RegState::Implicit);
  }
}

static bool isFADD(unsigned Opc) {
  switch (Opc) {
  default:
    return false;
  case RISCV::FADD_H:
  case RISCV::FADD_S:
  case RISCV::FADD_D:
    return true;
  }
}

static bool isFSUB(unsigned Opc) {
  switch (Opc) {
  default:
    return false;
  case RISCV::FSUB_H:
  case RISCV::FSUB_S:
  case RISCV::FSUB_D:
    return true;
  }
}

static bool isFMUL(unsigned Opc) {
  switch (Opc) {
  default:
    return false;
  case RISCV::FMUL_H:
  case RISCV::FMUL_S:
  case RISCV::FMUL_D:
    return true;
  }
}

bool RISCVInstrInfo::hasReassociableSibling(const MachineInstr &Inst,
                                            bool &Commuted) const {
  if (!TargetInstrInfo::hasReassociableSibling(Inst, Commuted))
    return false;

  const MachineRegisterInfo &MRI = Inst.getMF()->getRegInfo();
  unsigned OperandIdx = Commuted ? 2 : 1;
  const MachineInstr &Sibling =
      *MRI.getVRegDef(Inst.getOperand(OperandIdx).getReg());

  int16_t InstFrmOpIdx =
      RISCV::getNamedOperandIdx(Inst.getOpcode(), RISCV::OpName::frm);
  int16_t SiblingFrmOpIdx =
      RISCV::getNamedOperandIdx(Sibling.getOpcode(), RISCV::OpName::frm);

  return (InstFrmOpIdx < 0 && SiblingFrmOpIdx < 0) ||
         RISCV::hasEqualFRM(Inst, Sibling);
}

bool RISCVInstrInfo::isAssociativeAndCommutative(const MachineInstr &Inst,
                                                 bool Invert) const {
  unsigned Opc = Inst.getOpcode();
  if (Invert) {
    auto InverseOpcode = getInverseOpcode(Opc);
    if (!InverseOpcode)
      return false;
    Opc = *InverseOpcode;
  }

  if (isFADD(Opc) || isFMUL(Opc))
    return Inst.getFlag(MachineInstr::MIFlag::FmReassoc) &&
           Inst.getFlag(MachineInstr::MIFlag::FmNsz);

  switch (Opc) {
  default:
    return false;
  case RISCV::ADD:
  case RISCV::ADDW:
  case RISCV::AND:
  case RISCV::OR:
  case RISCV::XOR:
  // From RISC-V ISA spec, if both the high and low bits of the same product
  // are required, then the recommended code sequence is:
  //
  // MULH[[S]U] rdh, rs1, rs2
  // MUL        rdl, rs1, rs2
  // (source register specifiers must be in same order and rdh cannot be the
  //  same as rs1 or rs2)
  //
  // Microarchitectures can then fuse these into a single multiply operation
  // instead of performing two separate multiplies.
  // MachineCombiner may reassociate MUL operands and lose the fusion
  // opportunity.
  case RISCV::MUL:
  case RISCV::MULW:
  case RISCV::MIN:
  case RISCV::MINU:
  case RISCV::MAX:
  case RISCV::MAXU:
  case RISCV::FMIN_H:
  case RISCV::FMIN_S:
  case RISCV::FMIN_D:
  case RISCV::FMAX_H:
  case RISCV::FMAX_S:
  case RISCV::FMAX_D:
    return true;
  }

  return false;
}

std::optional<unsigned>
RISCVInstrInfo::getInverseOpcode(unsigned Opcode) const {
  switch (Opcode) {
  default:
    return std::nullopt;
  case RISCV::FADD_H:
    return RISCV::FSUB_H;
  case RISCV::FADD_S:
    return RISCV::FSUB_S;
  case RISCV::FADD_D:
    return RISCV::FSUB_D;
  case RISCV::FSUB_H:
    return RISCV::FADD_H;
  case RISCV::FSUB_S:
    return RISCV::FADD_S;
  case RISCV::FSUB_D:
    return RISCV::FADD_D;
  case RISCV::ADD:
    return RISCV::SUB;
  case RISCV::SUB:
    return RISCV::ADD;
  case RISCV::ADDW:
    return RISCV::SUBW;
  case RISCV::SUBW:
    return RISCV::ADDW;
  }
}

static bool canCombineFPFusedMultiply(const MachineInstr &Root,
                                      const MachineOperand &MO,
                                      bool DoRegPressureReduce) {
  if (!MO.isReg() || !MO.getReg().isVirtual())
    return false;
  const MachineRegisterInfo &MRI = Root.getMF()->getRegInfo();
  MachineInstr *MI = MRI.getVRegDef(MO.getReg());
  if (!MI || !isFMUL(MI->getOpcode()))
    return false;

  if (!Root.getFlag(MachineInstr::MIFlag::FmContract) ||
      !MI->getFlag(MachineInstr::MIFlag::FmContract))
    return false;

  // Try combining even if fmul has more than one use as it eliminates
  // dependency between fadd(fsub) and fmul. However, it can extend liveranges
  // for fmul operands, so reject the transformation in register pressure
  // reduction mode.
  if (DoRegPressureReduce && !MRI.hasOneNonDBGUse(MI->getOperand(0).getReg()))
    return false;

  // Do not combine instructions from different basic blocks.
  if (Root.getParent() != MI->getParent())
    return false;
  return RISCV::hasEqualFRM(Root, *MI);
}

static bool
getFPFusedMultiplyPatterns(MachineInstr &Root,
                           SmallVectorImpl<MachineCombinerPattern> &Patterns,
                           bool DoRegPressureReduce) {
  unsigned Opc = Root.getOpcode();
  bool IsFAdd = isFADD(Opc);
  if (!IsFAdd && !isFSUB(Opc))
    return false;
  bool Added = false;
  if (canCombineFPFusedMultiply(Root, Root.getOperand(1),
                                DoRegPressureReduce)) {
    Patterns.push_back(IsFAdd ? MachineCombinerPattern::FMADD_AX
                              : MachineCombinerPattern::FMSUB);
    Added = true;
  }
  if (canCombineFPFusedMultiply(Root, Root.getOperand(2),
                                DoRegPressureReduce)) {
    Patterns.push_back(IsFAdd ? MachineCombinerPattern::FMADD_XA
                              : MachineCombinerPattern::FNMSUB);
    Added = true;
  }
  return Added;
}

static bool getFPPatterns(MachineInstr &Root,
                          SmallVectorImpl<MachineCombinerPattern> &Patterns,
                          bool DoRegPressureReduce) {
  return getFPFusedMultiplyPatterns(Root, Patterns, DoRegPressureReduce);
}

bool RISCVInstrInfo::getMachineCombinerPatterns(
    MachineInstr &Root, SmallVectorImpl<MachineCombinerPattern> &Patterns,
    bool DoRegPressureReduce) const {

  if (getFPPatterns(Root, Patterns, DoRegPressureReduce))
    return true;

  return TargetInstrInfo::getMachineCombinerPatterns(Root, Patterns,
                                                     DoRegPressureReduce);
}

static unsigned getFPFusedMultiplyOpcode(unsigned RootOpc,
                                         MachineCombinerPattern Pattern) {
  switch (RootOpc) {
  default:
    llvm_unreachable("Unexpected opcode");
  case RISCV::FADD_H:
    return RISCV::FMADD_H;
  case RISCV::FADD_S:
    return RISCV::FMADD_S;
  case RISCV::FADD_D:
    return RISCV::FMADD_D;
  case RISCV::FSUB_H:
    return Pattern == MachineCombinerPattern::FMSUB ? RISCV::FMSUB_H
                                                    : RISCV::FNMSUB_H;
  case RISCV::FSUB_S:
    return Pattern == MachineCombinerPattern::FMSUB ? RISCV::FMSUB_S
                                                    : RISCV::FNMSUB_S;
  case RISCV::FSUB_D:
    return Pattern == MachineCombinerPattern::FMSUB ? RISCV::FMSUB_D
                                                    : RISCV::FNMSUB_D;
  }
}

static unsigned getAddendOperandIdx(MachineCombinerPattern Pattern) {
  switch (Pattern) {
  default:
    llvm_unreachable("Unexpected pattern");
  case MachineCombinerPattern::FMADD_AX:
  case MachineCombinerPattern::FMSUB:
    return 2;
  case MachineCombinerPattern::FMADD_XA:
  case MachineCombinerPattern::FNMSUB:
    return 1;
  }
}

static void combineFPFusedMultiply(MachineInstr &Root, MachineInstr &Prev,
                                   MachineCombinerPattern Pattern,
                                   SmallVectorImpl<MachineInstr *> &InsInstrs,
                                   SmallVectorImpl<MachineInstr *> &DelInstrs) {
  MachineFunction *MF = Root.getMF();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();

  MachineOperand &Mul1 = Prev.getOperand(1);
  MachineOperand &Mul2 = Prev.getOperand(2);
  MachineOperand &Dst = Root.getOperand(0);
  MachineOperand &Addend = Root.getOperand(getAddendOperandIdx(Pattern));

  Register DstReg = Dst.getReg();
  unsigned FusedOpc = getFPFusedMultiplyOpcode(Root.getOpcode(), Pattern);
  uint32_t IntersectedFlags = Root.getFlags() & Prev.getFlags();
  DebugLoc MergedLoc =
      DILocation::getMergedLocation(Root.getDebugLoc(), Prev.getDebugLoc());

  bool Mul1IsKill = Mul1.isKill();
  bool Mul2IsKill = Mul2.isKill();
  bool AddendIsKill = Addend.isKill();

  // We need to clear kill flags since we may be extending the live range past
  // a kill. If the mul had kill flags, we can preserve those since we know
  // where the previous range stopped.
  MRI.clearKillFlags(Mul1.getReg());
  MRI.clearKillFlags(Mul2.getReg());

  MachineInstrBuilder MIB =
      BuildMI(*MF, MergedLoc, TII->get(FusedOpc), DstReg)
          .addReg(Mul1.getReg(), getKillRegState(Mul1IsKill))
          .addReg(Mul2.getReg(), getKillRegState(Mul2IsKill))
          .addReg(Addend.getReg(), getKillRegState(AddendIsKill))
          .setMIFlags(IntersectedFlags);

  InsInstrs.push_back(MIB);
  if (MRI.hasOneNonDBGUse(Prev.getOperand(0).getReg()))
    DelInstrs.push_back(&Prev);
  DelInstrs.push_back(&Root);
}

void RISCVInstrInfo::genAlternativeCodeSequence(
    MachineInstr &Root, MachineCombinerPattern Pattern,
    SmallVectorImpl<MachineInstr *> &InsInstrs,
    SmallVectorImpl<MachineInstr *> &DelInstrs,
    DenseMap<unsigned, unsigned> &InstrIdxForVirtReg) const {
  MachineRegisterInfo &MRI = Root.getMF()->getRegInfo();
  switch (Pattern) {
  default:
    TargetInstrInfo::genAlternativeCodeSequence(Root, Pattern, InsInstrs,
                                                DelInstrs, InstrIdxForVirtReg);
    return;
  case MachineCombinerPattern::FMADD_AX:
  case MachineCombinerPattern::FMSUB: {
    MachineInstr &Prev = *MRI.getVRegDef(Root.getOperand(1).getReg());
    combineFPFusedMultiply(Root, Prev, Pattern, InsInstrs, DelInstrs);
    return;
  }
  case MachineCombinerPattern::FMADD_XA:
  case MachineCombinerPattern::FNMSUB: {
    MachineInstr &Prev = *MRI.getVRegDef(Root.getOperand(2).getReg());
    combineFPFusedMultiply(Root, Prev, Pattern, InsInstrs, DelInstrs);
    return;
  }
  }
}

bool RISCVInstrInfo::verifyInstruction(const MachineInstr &MI,
                                       StringRef &ErrInfo) const {
  MCInstrDesc const &Desc = MI.getDesc();

  for (const auto &[Index, Operand] : enumerate(Desc.operands())) {
    unsigned OpType = Operand.OperandType;
    if (OpType >= RISCVOp::OPERAND_FIRST_RISCV_IMM &&
        OpType <= RISCVOp::OPERAND_LAST_RISCV_IMM) {
      const MachineOperand &MO = MI.getOperand(Index);
      if (MO.isImm()) {
        int64_t Imm = MO.getImm();
        bool Ok;
        switch (OpType) {
        default:
          llvm_unreachable("Unexpected operand type");

          // clang-format off
#define CASE_OPERAND_UIMM(NUM)                                                 \
  case RISCVOp::OPERAND_UIMM##NUM:                                             \
    Ok = isUInt<NUM>(Imm);                                                     \
    break;
        CASE_OPERAND_UIMM(1)
        CASE_OPERAND_UIMM(2)
        CASE_OPERAND_UIMM(3)
        CASE_OPERAND_UIMM(4)
        CASE_OPERAND_UIMM(5)
        CASE_OPERAND_UIMM(6)
        CASE_OPERAND_UIMM(7)
        CASE_OPERAND_UIMM(8)
        CASE_OPERAND_UIMM(12)
        CASE_OPERAND_UIMM(20)
          // clang-format on
        case RISCVOp::OPERAND_UIMM2_LSB0:
          Ok = isShiftedUInt<1, 1>(Imm);
          break;
        case RISCVOp::OPERAND_UIMM7_LSB00:
          Ok = isShiftedUInt<5, 2>(Imm);
          break;
        case RISCVOp::OPERAND_UIMM8_LSB00:
          Ok = isShiftedUInt<6, 2>(Imm);
          break;
        case RISCVOp::OPERAND_UIMM8_LSB000:
          Ok = isShiftedUInt<5, 3>(Imm);
          break;
        case RISCVOp::OPERAND_UIMM8_GE32:
          Ok = isUInt<8>(Imm) && Imm >= 32;
          break;
        case RISCVOp::OPERAND_UIMM9_LSB000:
          Ok = isShiftedUInt<6, 3>(Imm);
          break;
        case RISCVOp::OPERAND_SIMM10_LSB0000_NONZERO:
          Ok = isShiftedInt<6, 4>(Imm) && (Imm != 0);
          break;
        case RISCVOp::OPERAND_UIMM10_LSB00_NONZERO:
          Ok = isShiftedUInt<8, 2>(Imm) && (Imm != 0);
          break;
        case RISCVOp::OPERAND_ZERO:
          Ok = Imm == 0;
          break;
        case RISCVOp::OPERAND_SIMM5:
          Ok = isInt<5>(Imm);
          break;
        case RISCVOp::OPERAND_SIMM5_PLUS1:
          Ok = (isInt<5>(Imm) && Imm != -16) || Imm == 16;
          break;
        case RISCVOp::OPERAND_SIMM6:
          Ok = isInt<6>(Imm);
          break;
        case RISCVOp::OPERAND_SIMM6_NONZERO:
          Ok = Imm != 0 && isInt<6>(Imm);
          break;
        case RISCVOp::OPERAND_VTYPEI10:
          Ok = isUInt<10>(Imm);
          break;
        case RISCVOp::OPERAND_VTYPEI11:
          Ok = isUInt<11>(Imm);
          break;
        case RISCVOp::OPERAND_SIMM12:
          Ok = isInt<12>(Imm);
          break;
        case RISCVOp::OPERAND_SIMM12_LSB00000:
          Ok = isShiftedInt<7, 5>(Imm);
          break;
        case RISCVOp::OPERAND_UIMMLOG2XLEN:
          Ok = STI.is64Bit() ? isUInt<6>(Imm) : isUInt<5>(Imm);
          break;
        case RISCVOp::OPERAND_UIMMLOG2XLEN_NONZERO:
          Ok = STI.is64Bit() ? isUInt<6>(Imm) : isUInt<5>(Imm);
          Ok = Ok && Imm != 0;
          break;
        case RISCVOp::OPERAND_CLUI_IMM:
          Ok = (isUInt<5>(Imm) && Imm != 0) ||
               (Imm >= 0xfffe0 && Imm <= 0xfffff);
          break;
        case RISCVOp::OPERAND_RVKRNUM:
          Ok = Imm >= 0 && Imm <= 10;
          break;
        case RISCVOp::OPERAND_RVKRNUM_0_7:
          Ok = Imm >= 0 && Imm <= 7;
          break;
        case RISCVOp::OPERAND_RVKRNUM_1_10:
          Ok = Imm >= 1 && Imm <= 10;
          break;
        case RISCVOp::OPERAND_RVKRNUM_2_14:
          Ok = Imm >= 2 && Imm <= 14;
          break;
        }
        if (!Ok) {
          ErrInfo = "Invalid immediate";
          return false;
        }
      }
    }
  }

  const uint64_t TSFlags = Desc.TSFlags;
  if (RISCVII::hasVLOp(TSFlags)) {
    const MachineOperand &Op = MI.getOperand(RISCVII::getVLOpNum(Desc));
    if (!Op.isImm() && !Op.isReg())  {
      ErrInfo = "Invalid operand type for VL operand";
      return false;
    }
    if (Op.isReg() && Op.getReg() != RISCV::NoRegister) {
      const MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
      auto *RC = MRI.getRegClass(Op.getReg());
      if (!RISCV::GPRRegClass.hasSubClassEq(RC)) {
        ErrInfo = "Invalid register class for VL operand";
        return false;
      }
    }
    if (!RISCVII::hasSEWOp(TSFlags)) {
      ErrInfo = "VL operand w/o SEW operand?";
      return false;
    }
  }
  if (RISCVII::hasSEWOp(TSFlags)) {
    unsigned OpIdx = RISCVII::getSEWOpNum(Desc);
    if (!MI.getOperand(OpIdx).isImm()) {
      ErrInfo = "SEW value expected to be an immediate";
      return false;
    }
    uint64_t Log2SEW = MI.getOperand(OpIdx).getImm();
    if (Log2SEW > 31) {
      ErrInfo = "Unexpected SEW value";
      return false;
    }
    unsigned SEW = Log2SEW ? 1 << Log2SEW : 8;
    if (!RISCVVType::isValidSEW(SEW)) {
      ErrInfo = "Unexpected SEW value";
      return false;
    }
  }
  if (RISCVII::hasVecPolicyOp(TSFlags)) {
    unsigned OpIdx = RISCVII::getVecPolicyOpNum(Desc);
    if (!MI.getOperand(OpIdx).isImm()) {
      ErrInfo = "Policy operand expected to be an immediate";
      return false;
    }
    uint64_t Policy = MI.getOperand(OpIdx).getImm();
    if (Policy > (RISCVII::TAIL_AGNOSTIC | RISCVII::MASK_AGNOSTIC)) {
      ErrInfo = "Invalid Policy Value";
      return false;
    }
    if (!RISCVII::hasVLOp(TSFlags)) {
      ErrInfo = "policy operand w/o VL operand?";
      return false;
    }

    // VecPolicy operands can only exist on instructions with passthru/merge
    // arguments. Note that not all arguments with passthru have vec policy
    // operands- some instructions have implicit policies.
    unsigned UseOpIdx;
    if (!MI.isRegTiedToUseOperand(0, &UseOpIdx)) {
      ErrInfo = "policy operand w/o tied operand?";
      return false;
    }
  }

  return true;
}

bool RISCVInstrInfo::canFoldIntoAddrMode(const MachineInstr &MemI, Register Reg,
                                         const MachineInstr &AddrI,
                                         ExtAddrMode &AM) const {
  switch (MemI.getOpcode()) {
  default:
    return false;
  case RISCV::LB:
  case RISCV::LBU:
  case RISCV::LH:
  case RISCV::LHU:
  case RISCV::LW:
  case RISCV::LWU:
  case RISCV::LD:
  case RISCV::FLH:
  case RISCV::FLW:
  case RISCV::FLD:
  case RISCV::SB:
  case RISCV::SH:
  case RISCV::SW:
  case RISCV::SD:
  case RISCV::FSH:
  case RISCV::FSW:
  case RISCV::FSD:
    break;
  }

  if (MemI.getOperand(0).getReg() == Reg)
    return false;

  if (AddrI.getOpcode() != RISCV::ADDI || !AddrI.getOperand(1).isReg() ||
      !AddrI.getOperand(2).isImm())
    return false;

  int64_t OldOffset = MemI.getOperand(2).getImm();
  int64_t Disp = AddrI.getOperand(2).getImm();
  int64_t NewOffset = OldOffset + Disp;
  if (!STI.is64Bit())
    NewOffset = SignExtend64<32>(NewOffset);

  if (!isInt<12>(NewOffset))
    return false;

  AM.BaseReg = AddrI.getOperand(1).getReg();
  AM.ScaledReg = 0;
  AM.Scale = 0;
  AM.Displacement = NewOffset;
  AM.Form = ExtAddrMode::Formula::Basic;
  return true;
}

MachineInstr *RISCVInstrInfo::emitLdStWithAddr(MachineInstr &MemI,
                                               const ExtAddrMode &AM) const {

  const DebugLoc &DL = MemI.getDebugLoc();
  MachineBasicBlock &MBB = *MemI.getParent();

  assert(AM.ScaledReg == 0 && AM.Scale == 0 &&
         "Addressing mode not supported for folding");

  return BuildMI(MBB, MemI, DL, get(MemI.getOpcode()))
      .addReg(MemI.getOperand(0).getReg(),
              MemI.mayLoad() ? RegState::Define : 0)
      .addReg(AM.BaseReg)
      .addImm(AM.Displacement)
      .setMemRefs(MemI.memoperands())
      .setMIFlags(MemI.getFlags());
}

// Return true if get the base operand, byte offset of an instruction and the
// memory width. Width is the size of memory that is being loaded/stored.
bool RISCVInstrInfo::getMemOperandWithOffsetWidth(
    const MachineInstr &LdSt, const MachineOperand *&BaseReg, int64_t &Offset,
    unsigned &Width, const TargetRegisterInfo *TRI) const {
  if (!LdSt.mayLoadOrStore())
    return false;

  // Here we assume the standard RISC-V ISA, which uses a base+offset
  // addressing mode. You'll need to relax these conditions to support custom
  // load/stores instructions.
  if (LdSt.getNumExplicitOperands() != 3)
    return false;
  if (!LdSt.getOperand(1).isReg() || !LdSt.getOperand(2).isImm())
    return false;

  if (!LdSt.hasOneMemOperand())
    return false;

  Width = (*LdSt.memoperands_begin())->getSize();
  BaseReg = &LdSt.getOperand(1);
  Offset = LdSt.getOperand(2).getImm();
  return true;
}

bool RISCVInstrInfo::areMemAccessesTriviallyDisjoint(
    const MachineInstr &MIa, const MachineInstr &MIb) const {
  assert(MIa.mayLoadOrStore() && "MIa must be a load or store.");
  assert(MIb.mayLoadOrStore() && "MIb must be a load or store.");

  if (MIa.hasUnmodeledSideEffects() || MIb.hasUnmodeledSideEffects() ||
      MIa.hasOrderedMemoryRef() || MIb.hasOrderedMemoryRef())
    return false;

  // Retrieve the base register, offset from the base register and width. Width
  // is the size of memory that is being loaded/stored (e.g. 1, 2, 4).  If
  // base registers are identical, and the offset of a lower memory access +
  // the width doesn't overlap the offset of a higher memory access,
  // then the memory accesses are different.
  const TargetRegisterInfo *TRI = STI.getRegisterInfo();
  const MachineOperand *BaseOpA = nullptr, *BaseOpB = nullptr;
  int64_t OffsetA = 0, OffsetB = 0;
  unsigned int WidthA = 0, WidthB = 0;
  if (getMemOperandWithOffsetWidth(MIa, BaseOpA, OffsetA, WidthA, TRI) &&
      getMemOperandWithOffsetWidth(MIb, BaseOpB, OffsetB, WidthB, TRI)) {
    if (BaseOpA->isIdenticalTo(*BaseOpB)) {
      int LowOffset = std::min(OffsetA, OffsetB);
      int HighOffset = std::max(OffsetA, OffsetB);
      int LowWidth = (LowOffset == OffsetA) ? WidthA : WidthB;
      if (LowOffset + LowWidth <= HighOffset)
        return true;
    }
  }
  return false;
}

std::pair<unsigned, unsigned>
RISCVInstrInfo::decomposeMachineOperandsTargetFlags(unsigned TF) const {
  const unsigned Mask = RISCVII::MO_DIRECT_FLAG_MASK;
  return std::make_pair(TF & Mask, TF & ~Mask);
}

ArrayRef<std::pair<unsigned, const char *>>
RISCVInstrInfo::getSerializableDirectMachineOperandTargetFlags() const {
  using namespace RISCVII;
  static const std::pair<unsigned, const char *> TargetFlags[] = {
      {MO_CALL, "riscv-call"},
      {MO_PLT, "riscv-plt"},
      {MO_LO, "riscv-lo"},
      {MO_HI, "riscv-hi"},
      {MO_PCREL_LO, "riscv-pcrel-lo"},
      {MO_PCREL_HI, "riscv-pcrel-hi"},
      {MO_GOT_HI, "riscv-got-hi"},
      {MO_TPREL_LO, "riscv-tprel-lo"},
      {MO_TPREL_HI, "riscv-tprel-hi"},
      {MO_TPREL_ADD, "riscv-tprel-add"},
      {MO_TLS_GOT_HI, "riscv-tls-got-hi"},
      {MO_TLS_GD_HI, "riscv-tls-gd-hi"}};
  return ArrayRef(TargetFlags);
}
bool RISCVInstrInfo::isFunctionSafeToOutlineFrom(
    MachineFunction &MF, bool OutlineFromLinkOnceODRs) const {
  const Function &F = MF.getFunction();

  // Can F be deduplicated by the linker? If it can, don't outline from it.
  if (!OutlineFromLinkOnceODRs && F.hasLinkOnceODRLinkage())
    return false;

  // Don't outline from functions with section markings; the program could
  // expect that all the code is in the named section.
  if (F.hasSection())
    return false;

  // It's safe to outline from MF.
  return true;
}

bool RISCVInstrInfo::isMBBSafeToOutlineFrom(MachineBasicBlock &MBB,
                                            unsigned &Flags) const {
  // More accurate safety checking is done in getOutliningCandidateInfo.
  return TargetInstrInfo::isMBBSafeToOutlineFrom(MBB, Flags);
}

// Enum values indicating how an outlined call should be constructed.
enum MachineOutlinerConstructionID {
  MachineOutlinerDefault
};

bool RISCVInstrInfo::shouldOutlineFromFunctionByDefault(
    MachineFunction &MF) const {
  return MF.getFunction().hasMinSize();
}

std::optional<outliner::OutlinedFunction>
RISCVInstrInfo::getOutliningCandidateInfo(
    std::vector<outliner::Candidate> &RepeatedSequenceLocs) const {

  // First we need to filter out candidates where the X5 register (IE t0) can't
  // be used to setup the function call.
  auto CannotInsertCall = [](outliner::Candidate &C) {
    const TargetRegisterInfo *TRI = C.getMF()->getSubtarget().getRegisterInfo();
    return !C.isAvailableAcrossAndOutOfSeq(RISCV::X5, *TRI);
  };

  llvm::erase_if(RepeatedSequenceLocs, CannotInsertCall);

  // If the sequence doesn't have enough candidates left, then we're done.
  if (RepeatedSequenceLocs.size() < 2)
    return std::nullopt;

  unsigned SequenceSize = 0;

  auto I = RepeatedSequenceLocs[0].front();
  auto E = std::next(RepeatedSequenceLocs[0].back());
  for (; I != E; ++I)
    SequenceSize += getInstSizeInBytes(*I);

  // call t0, function = 8 bytes.
  unsigned CallOverhead = 8;
  for (auto &C : RepeatedSequenceLocs)
    C.setCallInfo(MachineOutlinerDefault, CallOverhead);

  // jr t0 = 4 bytes, 2 bytes if compressed instructions are enabled.
  unsigned FrameOverhead = 4;
  if (RepeatedSequenceLocs[0]
          .getMF()
          ->getSubtarget<RISCVSubtarget>()
          .hasStdExtCOrZca())
    FrameOverhead = 2;

  return outliner::OutlinedFunction(RepeatedSequenceLocs, SequenceSize,
                                    FrameOverhead, MachineOutlinerDefault);
}

outliner::InstrType
RISCVInstrInfo::getOutliningTypeImpl(MachineBasicBlock::iterator &MBBI,
                                 unsigned Flags) const {
  MachineInstr &MI = *MBBI;
  MachineBasicBlock *MBB = MI.getParent();
  const TargetRegisterInfo *TRI =
      MBB->getParent()->getSubtarget().getRegisterInfo();
  const auto &F = MI.getMF()->getFunction();

  // We can manually strip out CFI instructions later.
  if (MI.isCFIInstruction())
    // If current function has exception handling code, we can't outline &
    // strip these CFI instructions since it may break .eh_frame section
    // needed in unwinding.
    return F.needsUnwindTableEntry() ? outliner::InstrType::Illegal
                                     : outliner::InstrType::Invisible;

  // We need support for tail calls to outlined functions before return
  // statements can be allowed.
  if (MI.isReturn())
    return outliner::InstrType::Illegal;

  // Don't allow modifying the X5 register which we use for return addresses for
  // these outlined functions.
  if (MI.modifiesRegister(RISCV::X5, TRI) ||
      MI.getDesc().hasImplicitDefOfPhysReg(RISCV::X5))
    return outliner::InstrType::Illegal;

  // Make sure the operands don't reference something unsafe.
  for (const auto &MO : MI.operands()) {

    // pcrel-hi and pcrel-lo can't put in separate sections, filter that out
    // if any possible.
    if (MO.getTargetFlags() == RISCVII::MO_PCREL_LO &&
        (MI.getMF()->getTarget().getFunctionSections() || F.hasComdat() ||
         F.hasSection()))
      return outliner::InstrType::Illegal;
  }

  return outliner::InstrType::Legal;
}

void RISCVInstrInfo::buildOutlinedFrame(
    MachineBasicBlock &MBB, MachineFunction &MF,
    const outliner::OutlinedFunction &OF) const {

  // Strip out any CFI instructions
  bool Changed = true;
  while (Changed) {
    Changed = false;
    auto I = MBB.begin();
    auto E = MBB.end();
    for (; I != E; ++I) {
      if (I->isCFIInstruction()) {
        I->removeFromParent();
        Changed = true;
        break;
      }
    }
  }

  MBB.addLiveIn(RISCV::X5);

  // Add in a return instruction to the end of the outlined frame.
  MBB.insert(MBB.end(), BuildMI(MF, DebugLoc(), get(RISCV::JALR))
      .addReg(RISCV::X0, RegState::Define)
      .addReg(RISCV::X5)
      .addImm(0));
}

MachineBasicBlock::iterator RISCVInstrInfo::insertOutlinedCall(
    Module &M, MachineBasicBlock &MBB, MachineBasicBlock::iterator &It,
    MachineFunction &MF, outliner::Candidate &C) const {

  // Add in a call instruction to the outlined function at the given location.
  It = MBB.insert(It,
                  BuildMI(MF, DebugLoc(), get(RISCV::PseudoCALLReg), RISCV::X5)
                      .addGlobalAddress(M.getNamedValue(MF.getName()), 0,
                                        RISCVII::MO_CALL));
  return It;
}

// MIR printer helper function to annotate Operands with a comment.
std::string RISCVInstrInfo::createMIROperandComment(
    const MachineInstr &MI, const MachineOperand &Op, unsigned OpIdx,
    const TargetRegisterInfo *TRI) const {
  // Print a generic comment for this operand if there is one.
  std::string GenericComment =
      TargetInstrInfo::createMIROperandComment(MI, Op, OpIdx, TRI);
  if (!GenericComment.empty())
    return GenericComment;

  // If not, we must have an immediate operand.
  if (!Op.isImm())
    return std::string();

  std::string Comment;
  raw_string_ostream OS(Comment);

  uint64_t TSFlags = MI.getDesc().TSFlags;

  // Print the full VType operand of vsetvli/vsetivli instructions, and the SEW
  // operand of vector codegen pseudos.
  if ((MI.getOpcode() == RISCV::VSETVLI || MI.getOpcode() == RISCV::VSETIVLI ||
       MI.getOpcode() == RISCV::PseudoVSETVLI ||
       MI.getOpcode() == RISCV::PseudoVSETIVLI ||
       MI.getOpcode() == RISCV::PseudoVSETVLIX0) &&
      OpIdx == 2) {
    unsigned Imm = MI.getOperand(OpIdx).getImm();
    RISCVVType::printVType(Imm, OS);
  } else if (RISCVII::hasSEWOp(TSFlags) &&
             OpIdx == RISCVII::getSEWOpNum(MI.getDesc())) {
    unsigned Log2SEW = MI.getOperand(OpIdx).getImm();
    unsigned SEW = Log2SEW ? 1 << Log2SEW : 8;
    assert(RISCVVType::isValidSEW(SEW) && "Unexpected SEW");
    OS << "e" << SEW;
  } else if (RISCVII::hasVecPolicyOp(TSFlags) &&
             OpIdx == RISCVII::getVecPolicyOpNum(MI.getDesc())) {
    unsigned Policy = MI.getOperand(OpIdx).getImm();
    assert(Policy <= (RISCVII::TAIL_AGNOSTIC | RISCVII::MASK_AGNOSTIC) &&
           "Invalid Policy Value");
    OS << (Policy & RISCVII::TAIL_AGNOSTIC ? "ta" : "tu") << ", "
       << (Policy & RISCVII::MASK_AGNOSTIC ? "ma" : "mu");
  }

  OS.flush();
  return Comment;
}

// clang-format off
#define CASE_VFMA_OPCODE_COMMON(OP, TYPE, LMUL)                                \
  RISCV::PseudoV##OP##_##TYPE##_##LMUL

#define CASE_VFMA_OPCODE_LMULS_M1(OP, TYPE)                                    \
  CASE_VFMA_OPCODE_COMMON(OP, TYPE, M1):                                       \
  case CASE_VFMA_OPCODE_COMMON(OP, TYPE, M2):                                  \
  case CASE_VFMA_OPCODE_COMMON(OP, TYPE, M4):                                  \
  case CASE_VFMA_OPCODE_COMMON(OP, TYPE, M8)

#define CASE_VFMA_OPCODE_LMULS_MF2(OP, TYPE)                                   \
  CASE_VFMA_OPCODE_COMMON(OP, TYPE, MF2):                                      \
  case CASE_VFMA_OPCODE_LMULS_M1(OP, TYPE)

#define CASE_VFMA_OPCODE_LMULS_MF4(OP, TYPE)                                   \
  CASE_VFMA_OPCODE_COMMON(OP, TYPE, MF4):                                      \
  case CASE_VFMA_OPCODE_LMULS_MF2(OP, TYPE)

#define CASE_VFMA_OPCODE_LMULS(OP, TYPE)                                       \
  CASE_VFMA_OPCODE_COMMON(OP, TYPE, MF8):                                      \
  case CASE_VFMA_OPCODE_LMULS_MF4(OP, TYPE)

#define CASE_VFMA_SPLATS(OP)                                                   \
  CASE_VFMA_OPCODE_LMULS_MF4(OP, VFPR16):                                      \
  case CASE_VFMA_OPCODE_LMULS_MF2(OP, VFPR32):                                 \
  case CASE_VFMA_OPCODE_LMULS_M1(OP, VFPR64)
// clang-format on

bool RISCVInstrInfo::findCommutedOpIndices(const MachineInstr &MI,
                                           unsigned &SrcOpIdx1,
                                           unsigned &SrcOpIdx2) const {
  const MCInstrDesc &Desc = MI.getDesc();
  if (!Desc.isCommutable())
    return false;

  switch (MI.getOpcode()) {
  case RISCV::TH_MVEQZ:
  case RISCV::TH_MVNEZ:
    // We can't commute operands if operand 2 (i.e., rs1 in
    // mveqz/mvnez rd,rs1,rs2) is the zero-register (as it is
    // not valid as the in/out-operand 1).
    if (MI.getOperand(2).getReg() == RISCV::X0)
      return false;
    // Operands 1 and 2 are commutable, if we switch the opcode.
    return fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2, 1, 2);
  case RISCV::TH_MULA:
  case RISCV::TH_MULAW:
  case RISCV::TH_MULAH:
  case RISCV::TH_MULS:
  case RISCV::TH_MULSW:
  case RISCV::TH_MULSH:
    // Operands 2 and 3 are commutable.
    return fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2, 2, 3);
  case RISCV::PseudoCCMOVGPR:
    // Operands 4 and 5 are commutable.
    return fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2, 4, 5);
  case CASE_VFMA_SPLATS(FMADD):
  case CASE_VFMA_SPLATS(FMSUB):
  case CASE_VFMA_SPLATS(FMACC):
  case CASE_VFMA_SPLATS(FMSAC):
  case CASE_VFMA_SPLATS(FNMADD):
  case CASE_VFMA_SPLATS(FNMSUB):
  case CASE_VFMA_SPLATS(FNMACC):
  case CASE_VFMA_SPLATS(FNMSAC):
  case CASE_VFMA_OPCODE_LMULS_MF4(FMACC, VV):
  case CASE_VFMA_OPCODE_LMULS_MF4(FMSAC, VV):
  case CASE_VFMA_OPCODE_LMULS_MF4(FNMACC, VV):
  case CASE_VFMA_OPCODE_LMULS_MF4(FNMSAC, VV):
  case CASE_VFMA_OPCODE_LMULS(MADD, VX):
  case CASE_VFMA_OPCODE_LMULS(NMSUB, VX):
  case CASE_VFMA_OPCODE_LMULS(MACC, VX):
  case CASE_VFMA_OPCODE_LMULS(NMSAC, VX):
  case CASE_VFMA_OPCODE_LMULS(MACC, VV):
  case CASE_VFMA_OPCODE_LMULS(NMSAC, VV): {
    // If the tail policy is undisturbed we can't commute.
    assert(RISCVII::hasVecPolicyOp(MI.getDesc().TSFlags));
    if ((MI.getOperand(MI.getNumExplicitOperands() - 1).getImm() & 1) == 0)
      return false;

    // For these instructions we can only swap operand 1 and operand 3 by
    // changing the opcode.
    unsigned CommutableOpIdx1 = 1;
    unsigned CommutableOpIdx2 = 3;
    if (!fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2, CommutableOpIdx1,
                              CommutableOpIdx2))
      return false;
    return true;
  }
  case CASE_VFMA_OPCODE_LMULS_MF4(FMADD, VV):
  case CASE_VFMA_OPCODE_LMULS_MF4(FMSUB, VV):
  case CASE_VFMA_OPCODE_LMULS_MF4(FNMADD, VV):
  case CASE_VFMA_OPCODE_LMULS_MF4(FNMSUB, VV):
  case CASE_VFMA_OPCODE_LMULS(MADD, VV):
  case CASE_VFMA_OPCODE_LMULS(NMSUB, VV): {
    // If the tail policy is undisturbed we can't commute.
    assert(RISCVII::hasVecPolicyOp(MI.getDesc().TSFlags));
    if ((MI.getOperand(MI.getNumExplicitOperands() - 1).getImm() & 1) == 0)
      return false;

    // For these instructions we have more freedom. We can commute with the
    // other multiplicand or with the addend/subtrahend/minuend.

    // Any fixed operand must be from source 1, 2 or 3.
    if (SrcOpIdx1 != CommuteAnyOperandIndex && SrcOpIdx1 > 3)
      return false;
    if (SrcOpIdx2 != CommuteAnyOperandIndex && SrcOpIdx2 > 3)
      return false;

    // It both ops are fixed one must be the tied source.
    if (SrcOpIdx1 != CommuteAnyOperandIndex &&
        SrcOpIdx2 != CommuteAnyOperandIndex && SrcOpIdx1 != 1 && SrcOpIdx2 != 1)
      return false;

    // Look for two different register operands assumed to be commutable
    // regardless of the FMA opcode. The FMA opcode is adjusted later if
    // needed.
    if (SrcOpIdx1 == CommuteAnyOperandIndex ||
        SrcOpIdx2 == CommuteAnyOperandIndex) {
      // At least one of operands to be commuted is not specified and
      // this method is free to choose appropriate commutable operands.
      unsigned CommutableOpIdx1 = SrcOpIdx1;
      if (SrcOpIdx1 == SrcOpIdx2) {
        // Both of operands are not fixed. Set one of commutable
        // operands to the tied source.
        CommutableOpIdx1 = 1;
      } else if (SrcOpIdx1 == CommuteAnyOperandIndex) {
        // Only one of the operands is not fixed.
        CommutableOpIdx1 = SrcOpIdx2;
      }

      // CommutableOpIdx1 is well defined now. Let's choose another commutable
      // operand and assign its index to CommutableOpIdx2.
      unsigned CommutableOpIdx2;
      if (CommutableOpIdx1 != 1) {
        // If we haven't already used the tied source, we must use it now.
        CommutableOpIdx2 = 1;
      } else {
        Register Op1Reg = MI.getOperand(CommutableOpIdx1).getReg();

        // The commuted operands should have different registers.
        // Otherwise, the commute transformation does not change anything and
        // is useless. We use this as a hint to make our decision.
        if (Op1Reg != MI.getOperand(2).getReg())
          CommutableOpIdx2 = 2;
        else
          CommutableOpIdx2 = 3;
      }

      // Assign the found pair of commutable indices to SrcOpIdx1 and
      // SrcOpIdx2 to return those values.
      if (!fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2, CommutableOpIdx1,
                                CommutableOpIdx2))
        return false;
    }

    return true;
  }
  }

  return TargetInstrInfo::findCommutedOpIndices(MI, SrcOpIdx1, SrcOpIdx2);
}

#define CASE_VFMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, LMUL)               \
  case RISCV::PseudoV##OLDOP##_##TYPE##_##LMUL:                                \
    Opc = RISCV::PseudoV##NEWOP##_##TYPE##_##LMUL;                             \
    break;

#define CASE_VFMA_CHANGE_OPCODE_LMULS_M1(OLDOP, NEWOP, TYPE)                   \
  CASE_VFMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, M1)                       \
  CASE_VFMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, M2)                       \
  CASE_VFMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, M4)                       \
  CASE_VFMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, M8)

#define CASE_VFMA_CHANGE_OPCODE_LMULS_MF2(OLDOP, NEWOP, TYPE)                  \
  CASE_VFMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, MF2)                      \
  CASE_VFMA_CHANGE_OPCODE_LMULS_M1(OLDOP, NEWOP, TYPE)

#define CASE_VFMA_CHANGE_OPCODE_LMULS_MF4(OLDOP, NEWOP, TYPE)                  \
  CASE_VFMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, MF4)                      \
  CASE_VFMA_CHANGE_OPCODE_LMULS_MF2(OLDOP, NEWOP, TYPE)

#define CASE_VFMA_CHANGE_OPCODE_LMULS(OLDOP, NEWOP, TYPE)                      \
  CASE_VFMA_CHANGE_OPCODE_COMMON(OLDOP, NEWOP, TYPE, MF8)                      \
  CASE_VFMA_CHANGE_OPCODE_LMULS_MF4(OLDOP, NEWOP, TYPE)

#define CASE_VFMA_CHANGE_OPCODE_SPLATS(OLDOP, NEWOP)                           \
  CASE_VFMA_CHANGE_OPCODE_LMULS_MF4(OLDOP, NEWOP, VFPR16)                      \
  CASE_VFMA_CHANGE_OPCODE_LMULS_MF2(OLDOP, NEWOP, VFPR32)                      \
  CASE_VFMA_CHANGE_OPCODE_LMULS_M1(OLDOP, NEWOP, VFPR64)

MachineInstr *RISCVInstrInfo::commuteInstructionImpl(MachineInstr &MI,
                                                     bool NewMI,
                                                     unsigned OpIdx1,
                                                     unsigned OpIdx2) const {
  auto cloneIfNew = [NewMI](MachineInstr &MI) -> MachineInstr & {
    if (NewMI)
      return *MI.getParent()->getParent()->CloneMachineInstr(&MI);
    return MI;
  };

  switch (MI.getOpcode()) {
  case RISCV::TH_MVEQZ:
  case RISCV::TH_MVNEZ: {
    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.setDesc(get(MI.getOpcode() == RISCV::TH_MVEQZ ? RISCV::TH_MVNEZ
                                                            : RISCV::TH_MVEQZ));
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, false, OpIdx1,
                                                   OpIdx2);
  }
  case RISCV::PseudoCCMOVGPR: {
    // CCMOV can be commuted by inverting the condition.
    auto CC = static_cast<RISCVCC::CondCode>(MI.getOperand(3).getImm());
    CC = RISCVCC::getOppositeBranchCondition(CC);
    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.getOperand(3).setImm(CC);
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI*/ false,
                                                   OpIdx1, OpIdx2);
  }
  case CASE_VFMA_SPLATS(FMACC):
  case CASE_VFMA_SPLATS(FMADD):
  case CASE_VFMA_SPLATS(FMSAC):
  case CASE_VFMA_SPLATS(FMSUB):
  case CASE_VFMA_SPLATS(FNMACC):
  case CASE_VFMA_SPLATS(FNMADD):
  case CASE_VFMA_SPLATS(FNMSAC):
  case CASE_VFMA_SPLATS(FNMSUB):
  case CASE_VFMA_OPCODE_LMULS_MF4(FMACC, VV):
  case CASE_VFMA_OPCODE_LMULS_MF4(FMSAC, VV):
  case CASE_VFMA_OPCODE_LMULS_MF4(FNMACC, VV):
  case CASE_VFMA_OPCODE_LMULS_MF4(FNMSAC, VV):
  case CASE_VFMA_OPCODE_LMULS(MADD, VX):
  case CASE_VFMA_OPCODE_LMULS(NMSUB, VX):
  case CASE_VFMA_OPCODE_LMULS(MACC, VX):
  case CASE_VFMA_OPCODE_LMULS(NMSAC, VX):
  case CASE_VFMA_OPCODE_LMULS(MACC, VV):
  case CASE_VFMA_OPCODE_LMULS(NMSAC, VV): {
    // It only make sense to toggle these between clobbering the
    // addend/subtrahend/minuend one of the multiplicands.
    assert((OpIdx1 == 1 || OpIdx2 == 1) && "Unexpected opcode index");
    assert((OpIdx1 == 3 || OpIdx2 == 3) && "Unexpected opcode index");
    unsigned Opc;
    switch (MI.getOpcode()) {
      default:
        llvm_unreachable("Unexpected opcode");
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FMACC, FMADD)
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FMADD, FMACC)
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FMSAC, FMSUB)
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FMSUB, FMSAC)
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FNMACC, FNMADD)
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FNMADD, FNMACC)
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FNMSAC, FNMSUB)
      CASE_VFMA_CHANGE_OPCODE_SPLATS(FNMSUB, FNMSAC)
      CASE_VFMA_CHANGE_OPCODE_LMULS_MF4(FMACC, FMADD, VV)
      CASE_VFMA_CHANGE_OPCODE_LMULS_MF4(FMSAC, FMSUB, VV)
      CASE_VFMA_CHANGE_OPCODE_LMULS_MF4(FNMACC, FNMADD, VV)
      CASE_VFMA_CHANGE_OPCODE_LMULS_MF4(FNMSAC, FNMSUB, VV)
      CASE_VFMA_CHANGE_OPCODE_LMULS(MACC, MADD, VX)
      CASE_VFMA_CHANGE_OPCODE_LMULS(MADD, MACC, VX)
      CASE_VFMA_CHANGE_OPCODE_LMULS(NMSAC, NMSUB, VX)
      CASE_VFMA_CHANGE_OPCODE_LMULS(NMSUB, NMSAC, VX)
      CASE_VFMA_CHANGE_OPCODE_LMULS(MACC, MADD, VV)
      CASE_VFMA_CHANGE_OPCODE_LMULS(NMSAC, NMSUB, VV)
    }

    auto &WorkingMI = cloneIfNew(MI);
    WorkingMI.setDesc(get(Opc));
    return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                   OpIdx1, OpIdx2);
  }
  case CASE_VFMA_OPCODE_LMULS_MF4(FMADD, VV):
  case CASE_VFMA_OPCODE_LMULS_MF4(FMSUB, VV):
  case CASE_VFMA_OPCODE_LMULS_MF4(FNMADD, VV):
  case CASE_VFMA_OPCODE_LMULS_MF4(FNMSUB, VV):
  case CASE_VFMA_OPCODE_LMULS(MADD, VV):
  case CASE_VFMA_OPCODE_LMULS(NMSUB, VV): {
    assert((OpIdx1 == 1 || OpIdx2 == 1) && "Unexpected opcode index");
    // If one of the operands, is the addend we need to change opcode.
    // Otherwise we're just swapping 2 of the multiplicands.
    if (OpIdx1 == 3 || OpIdx2 == 3) {
      unsigned Opc;
      switch (MI.getOpcode()) {
        default:
          llvm_unreachable("Unexpected opcode");
        CASE_VFMA_CHANGE_OPCODE_LMULS_MF4(FMADD, FMACC, VV)
        CASE_VFMA_CHANGE_OPCODE_LMULS_MF4(FMSUB, FMSAC, VV)
        CASE_VFMA_CHANGE_OPCODE_LMULS_MF4(FNMADD, FNMACC, VV)
        CASE_VFMA_CHANGE_OPCODE_LMULS_MF4(FNMSUB, FNMSAC, VV)
        CASE_VFMA_CHANGE_OPCODE_LMULS(MADD, MACC, VV)
        CASE_VFMA_CHANGE_OPCODE_LMULS(NMSUB, NMSAC, VV)
      }

      auto &WorkingMI = cloneIfNew(MI);
      WorkingMI.setDesc(get(Opc));
      return TargetInstrInfo::commuteInstructionImpl(WorkingMI, /*NewMI=*/false,
                                                     OpIdx1, OpIdx2);
    }
    // Let the default code handle it.
    break;
  }
  }

  return TargetInstrInfo::commuteInstructionImpl(MI, NewMI, OpIdx1, OpIdx2);
}

#undef CASE_VFMA_CHANGE_OPCODE_SPLATS
#undef CASE_VFMA_CHANGE_OPCODE_LMULS
#undef CASE_VFMA_CHANGE_OPCODE_COMMON
#undef CASE_VFMA_SPLATS
#undef CASE_VFMA_OPCODE_LMULS
#undef CASE_VFMA_OPCODE_COMMON

// clang-format off
#define CASE_WIDEOP_OPCODE_COMMON(OP, LMUL)                                    \
  RISCV::PseudoV##OP##_##LMUL##_TIED

#define CASE_WIDEOP_OPCODE_LMULS_MF4(OP)                                       \
  CASE_WIDEOP_OPCODE_COMMON(OP, MF4):                                          \
  case CASE_WIDEOP_OPCODE_COMMON(OP, MF2):                                     \
  case CASE_WIDEOP_OPCODE_COMMON(OP, M1):                                      \
  case CASE_WIDEOP_OPCODE_COMMON(OP, M2):                                      \
  case CASE_WIDEOP_OPCODE_COMMON(OP, M4)

#define CASE_WIDEOP_OPCODE_LMULS(OP)                                           \
  CASE_WIDEOP_OPCODE_COMMON(OP, MF8):                                          \
  case CASE_WIDEOP_OPCODE_LMULS_MF4(OP)
// clang-format on

#define CASE_WIDEOP_CHANGE_OPCODE_COMMON(OP, LMUL)                             \
  case RISCV::PseudoV##OP##_##LMUL##_TIED:                                     \
    NewOpc = RISCV::PseudoV##OP##_##LMUL;                                      \
    break;

#define CASE_WIDEOP_CHANGE_OPCODE_LMULS_MF4(OP)                                 \
  CASE_WIDEOP_CHANGE_OPCODE_COMMON(OP, MF4)                                    \
  CASE_WIDEOP_CHANGE_OPCODE_COMMON(OP, MF2)                                    \
  CASE_WIDEOP_CHANGE_OPCODE_COMMON(OP, M1)                                     \
  CASE_WIDEOP_CHANGE_OPCODE_COMMON(OP, M2)                                     \
  CASE_WIDEOP_CHANGE_OPCODE_COMMON(OP, M4)

#define CASE_WIDEOP_CHANGE_OPCODE_LMULS(OP)                                    \
  CASE_WIDEOP_CHANGE_OPCODE_COMMON(OP, MF8)                                    \
  CASE_WIDEOP_CHANGE_OPCODE_LMULS_MF4(OP)

MachineInstr *RISCVInstrInfo::convertToThreeAddress(MachineInstr &MI,
                                                    LiveVariables *LV,
                                                    LiveIntervals *LIS) const {
  MachineInstrBuilder MIB;
  switch (MI.getOpcode()) {
  default:
    return nullptr;
  case CASE_WIDEOP_OPCODE_LMULS_MF4(FWADD_WV):
  case CASE_WIDEOP_OPCODE_LMULS_MF4(FWSUB_WV): {
    assert(RISCVII::hasVecPolicyOp(MI.getDesc().TSFlags) &&
           MI.getNumExplicitOperands() == 7 &&
           "Expect 7 explicit operands rd, rs2, rs1, rm, vl, sew, policy");
    // If the tail policy is undisturbed we can't convert.
    if ((MI.getOperand(RISCVII::getVecPolicyOpNum(MI.getDesc())).getImm() &
         1) == 0)
      return nullptr;
    // clang-format off
    unsigned NewOpc;
    switch (MI.getOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode");
    CASE_WIDEOP_CHANGE_OPCODE_LMULS_MF4(FWADD_WV)
    CASE_WIDEOP_CHANGE_OPCODE_LMULS_MF4(FWSUB_WV)
    }
    // clang-format on

    MachineBasicBlock &MBB = *MI.getParent();
    MIB = BuildMI(MBB, MI, MI.getDebugLoc(), get(NewOpc))
              .add(MI.getOperand(0))
              .addReg(MI.getOperand(0).getReg(), RegState::Undef)
              .add(MI.getOperand(1))
              .add(MI.getOperand(2))
              .add(MI.getOperand(3))
              .add(MI.getOperand(4))
              .add(MI.getOperand(5))
              .add(MI.getOperand(6));
    break;
  }
  case CASE_WIDEOP_OPCODE_LMULS(WADD_WV):
  case CASE_WIDEOP_OPCODE_LMULS(WADDU_WV):
  case CASE_WIDEOP_OPCODE_LMULS(WSUB_WV):
  case CASE_WIDEOP_OPCODE_LMULS(WSUBU_WV): {
    // If the tail policy is undisturbed we can't convert.
    assert(RISCVII::hasVecPolicyOp(MI.getDesc().TSFlags) &&
           MI.getNumExplicitOperands() == 6);
    if ((MI.getOperand(5).getImm() & 1) == 0)
      return nullptr;

    // clang-format off
    unsigned NewOpc;
    switch (MI.getOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode");
    CASE_WIDEOP_CHANGE_OPCODE_LMULS(WADD_WV)
    CASE_WIDEOP_CHANGE_OPCODE_LMULS(WADDU_WV)
    CASE_WIDEOP_CHANGE_OPCODE_LMULS(WSUB_WV)
    CASE_WIDEOP_CHANGE_OPCODE_LMULS(WSUBU_WV)
    }
    // clang-format on

    MachineBasicBlock &MBB = *MI.getParent();
    MIB = BuildMI(MBB, MI, MI.getDebugLoc(), get(NewOpc))
              .add(MI.getOperand(0))
              .addReg(MI.getOperand(0).getReg(), RegState::Undef)
              .add(MI.getOperand(1))
              .add(MI.getOperand(2))
              .add(MI.getOperand(3))
              .add(MI.getOperand(4))
              .add(MI.getOperand(5));
    break;
  }
  }
  MIB.copyImplicitOps(MI);

  if (LV) {
    unsigned NumOps = MI.getNumOperands();
    for (unsigned I = 1; I < NumOps; ++I) {
      MachineOperand &Op = MI.getOperand(I);
      if (Op.isReg() && Op.isKill())
        LV->replaceKillInstruction(Op.getReg(), MI, *MIB);
    }
  }

  if (LIS) {
    SlotIndex Idx = LIS->ReplaceMachineInstrInMaps(MI, *MIB);

    if (MI.getOperand(0).isEarlyClobber()) {
      // Use operand 1 was tied to early-clobber def operand 0, so its live
      // interval could have ended at an early-clobber slot. Now they are not
      // tied we need to update it to the normal register slot.
      LiveInterval &LI = LIS->getInterval(MI.getOperand(1).getReg());
      LiveRange::Segment *S = LI.getSegmentContaining(Idx);
      if (S->end == Idx.getRegSlot(true))
        S->end = Idx.getRegSlot();
    }
  }

  return MIB;
}

#undef CASE_WIDEOP_CHANGE_OPCODE_LMULS
#undef CASE_WIDEOP_CHANGE_OPCODE_COMMON
#undef CASE_WIDEOP_OPCODE_LMULS
#undef CASE_WIDEOP_OPCODE_COMMON

void RISCVInstrInfo::getVLENFactoredAmount(MachineFunction &MF,
                                           MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator II,
                                           const DebugLoc &DL, Register DestReg,
                                           int64_t Amount,
                                           MachineInstr::MIFlag Flag) const {
  assert(Amount > 0 && "There is no need to get VLEN scaled value.");
  assert(Amount % 8 == 0 &&
         "Reserve the stack by the multiple of one vector size.");

  MachineRegisterInfo &MRI = MF.getRegInfo();
  int64_t NumOfVReg = Amount / 8;

  BuildMI(MBB, II, DL, get(RISCV::PseudoReadVLENB), DestReg).setMIFlag(Flag);
  assert(isInt<32>(NumOfVReg) &&
         "Expect the number of vector registers within 32-bits.");
  if (llvm::has_single_bit<uint32_t>(NumOfVReg)) {
    uint32_t ShiftAmount = Log2_32(NumOfVReg);
    if (ShiftAmount == 0)
      return;
    BuildMI(MBB, II, DL, get(RISCV::SLLI), DestReg)
        .addReg(DestReg, RegState::Kill)
        .addImm(ShiftAmount)
        .setMIFlag(Flag);
  } else if (STI.hasStdExtZba() &&
             ((NumOfVReg % 3 == 0 && isPowerOf2_64(NumOfVReg / 3)) ||
              (NumOfVReg % 5 == 0 && isPowerOf2_64(NumOfVReg / 5)) ||
              (NumOfVReg % 9 == 0 && isPowerOf2_64(NumOfVReg / 9)))) {
    // We can use Zba SHXADD+SLLI instructions for multiply in some cases.
    unsigned Opc;
    uint32_t ShiftAmount;
    if (NumOfVReg % 9 == 0) {
      Opc = RISCV::SH3ADD;
      ShiftAmount = Log2_64(NumOfVReg / 9);
    } else if (NumOfVReg % 5 == 0) {
      Opc = RISCV::SH2ADD;
      ShiftAmount = Log2_64(NumOfVReg / 5);
    } else if (NumOfVReg % 3 == 0) {
      Opc = RISCV::SH1ADD;
      ShiftAmount = Log2_64(NumOfVReg / 3);
    } else {
      llvm_unreachable("Unexpected number of vregs");
    }
    if (ShiftAmount)
      BuildMI(MBB, II, DL, get(RISCV::SLLI), DestReg)
          .addReg(DestReg, RegState::Kill)
          .addImm(ShiftAmount)
          .setMIFlag(Flag);
    BuildMI(MBB, II, DL, get(Opc), DestReg)
        .addReg(DestReg, RegState::Kill)
        .addReg(DestReg)
        .setMIFlag(Flag);
  } else if (llvm::has_single_bit<uint32_t>(NumOfVReg - 1)) {
    Register ScaledRegister = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    uint32_t ShiftAmount = Log2_32(NumOfVReg - 1);
    BuildMI(MBB, II, DL, get(RISCV::SLLI), ScaledRegister)
        .addReg(DestReg)
        .addImm(ShiftAmount)
        .setMIFlag(Flag);
    BuildMI(MBB, II, DL, get(RISCV::ADD), DestReg)
        .addReg(ScaledRegister, RegState::Kill)
        .addReg(DestReg, RegState::Kill)
        .setMIFlag(Flag);
  } else if (llvm::has_single_bit<uint32_t>(NumOfVReg + 1)) {
    Register ScaledRegister = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    uint32_t ShiftAmount = Log2_32(NumOfVReg + 1);
    BuildMI(MBB, II, DL, get(RISCV::SLLI), ScaledRegister)
        .addReg(DestReg)
        .addImm(ShiftAmount)
        .setMIFlag(Flag);
    BuildMI(MBB, II, DL, get(RISCV::SUB), DestReg)
        .addReg(ScaledRegister, RegState::Kill)
        .addReg(DestReg, RegState::Kill)
        .setMIFlag(Flag);
  } else {
    Register N = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    movImm(MBB, II, DL, N, NumOfVReg, Flag);
    if (!STI.hasStdExtM() && !STI.hasStdExtZmmul())
      MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported{
          MF.getFunction(),
          "M- or Zmmul-extension must be enabled to calculate the vscaled size/"
          "offset."});
    BuildMI(MBB, II, DL, get(RISCV::MUL), DestReg)
        .addReg(DestReg, RegState::Kill)
        .addReg(N, RegState::Kill)
        .setMIFlag(Flag);
  }
}

ArrayRef<std::pair<MachineMemOperand::Flags, const char *>>
RISCVInstrInfo::getSerializableMachineMemOperandTargetFlags() const {
  static const std::pair<MachineMemOperand::Flags, const char *> TargetFlags[] =
      {{MONontemporalBit0, "riscv-nontemporal-domain-bit-0"},
       {MONontemporalBit1, "riscv-nontemporal-domain-bit-1"}};
  return ArrayRef(TargetFlags);
}

// Returns true if this is the sext.w pattern, addiw rd, rs1, 0.
bool RISCV::isSEXT_W(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::ADDIW && MI.getOperand(1).isReg() &&
         MI.getOperand(2).isImm() && MI.getOperand(2).getImm() == 0;
}

// Returns true if this is the zext.w pattern, adduw rd, rs1, x0.
bool RISCV::isZEXT_W(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::ADD_UW && MI.getOperand(1).isReg() &&
         MI.getOperand(2).isReg() && MI.getOperand(2).getReg() == RISCV::X0;
}

// Returns true if this is the zext.b pattern, andi rd, rs1, 255.
bool RISCV::isZEXT_B(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::ANDI && MI.getOperand(1).isReg() &&
         MI.getOperand(2).isImm() && MI.getOperand(2).getImm() == 255;
}

static bool isRVVWholeLoadStore(unsigned Opcode) {
  switch (Opcode) {
  default:
    return false;
  case RISCV::VS1R_V:
  case RISCV::VS2R_V:
  case RISCV::VS4R_V:
  case RISCV::VS8R_V:
  case RISCV::VL1RE8_V:
  case RISCV::VL2RE8_V:
  case RISCV::VL4RE8_V:
  case RISCV::VL8RE8_V:
  case RISCV::VL1RE16_V:
  case RISCV::VL2RE16_V:
  case RISCV::VL4RE16_V:
  case RISCV::VL8RE16_V:
  case RISCV::VL1RE32_V:
  case RISCV::VL2RE32_V:
  case RISCV::VL4RE32_V:
  case RISCV::VL8RE32_V:
  case RISCV::VL1RE64_V:
  case RISCV::VL2RE64_V:
  case RISCV::VL4RE64_V:
  case RISCV::VL8RE64_V:
    return true;
  }
}

bool RISCV::isRVVSpill(const MachineInstr &MI) {
  // RVV lacks any support for immediate addressing for stack addresses, so be
  // conservative.
  unsigned Opcode = MI.getOpcode();
  if (!RISCVVPseudosTable::getPseudoInfo(Opcode) &&
      !isRVVWholeLoadStore(Opcode) && !isRVVSpillForZvlsseg(Opcode))
    return false;
  return true;
}

std::optional<std::pair<unsigned, unsigned>>
RISCV::isRVVSpillForZvlsseg(unsigned Opcode) {
  switch (Opcode) {
  default:
    return std::nullopt;
  case RISCV::PseudoVSPILL2_M1:
  case RISCV::PseudoVRELOAD2_M1:
    return std::make_pair(2u, 1u);
  case RISCV::PseudoVSPILL2_M2:
  case RISCV::PseudoVRELOAD2_M2:
    return std::make_pair(2u, 2u);
  case RISCV::PseudoVSPILL2_M4:
  case RISCV::PseudoVRELOAD2_M4:
    return std::make_pair(2u, 4u);
  case RISCV::PseudoVSPILL3_M1:
  case RISCV::PseudoVRELOAD3_M1:
    return std::make_pair(3u, 1u);
  case RISCV::PseudoVSPILL3_M2:
  case RISCV::PseudoVRELOAD3_M2:
    return std::make_pair(3u, 2u);
  case RISCV::PseudoVSPILL4_M1:
  case RISCV::PseudoVRELOAD4_M1:
    return std::make_pair(4u, 1u);
  case RISCV::PseudoVSPILL4_M2:
  case RISCV::PseudoVRELOAD4_M2:
    return std::make_pair(4u, 2u);
  case RISCV::PseudoVSPILL5_M1:
  case RISCV::PseudoVRELOAD5_M1:
    return std::make_pair(5u, 1u);
  case RISCV::PseudoVSPILL6_M1:
  case RISCV::PseudoVRELOAD6_M1:
    return std::make_pair(6u, 1u);
  case RISCV::PseudoVSPILL7_M1:
  case RISCV::PseudoVRELOAD7_M1:
    return std::make_pair(7u, 1u);
  case RISCV::PseudoVSPILL8_M1:
  case RISCV::PseudoVRELOAD8_M1:
    return std::make_pair(8u, 1u);
  }
}

bool RISCV::isFaultFirstLoad(const MachineInstr &MI) {
  return MI.getNumExplicitDefs() == 2 && MI.modifiesRegister(RISCV::VL) &&
         !MI.isInlineAsm();
}

bool RISCV::hasEqualFRM(const MachineInstr &MI1, const MachineInstr &MI2) {
  int16_t MI1FrmOpIdx =
      RISCV::getNamedOperandIdx(MI1.getOpcode(), RISCV::OpName::frm);
  int16_t MI2FrmOpIdx =
      RISCV::getNamedOperandIdx(MI2.getOpcode(), RISCV::OpName::frm);
  if (MI1FrmOpIdx < 0 || MI2FrmOpIdx < 0)
    return false;
  MachineOperand FrmOp1 = MI1.getOperand(MI1FrmOpIdx);
  MachineOperand FrmOp2 = MI2.getOperand(MI2FrmOpIdx);
  return FrmOp1.getImm() == FrmOp2.getImm();
}

std::optional<unsigned>
RISCV::getVectorLowDemandedScalarBits(uint16_t Opcode, unsigned Log2SEW) {
  // TODO: Handle Zvbb instructions
  switch (Opcode) {
  default:
    return std::nullopt;

  // 11.6. Vector Single-Width Shift Instructions
  case RISCV::VSLL_VX:
  case RISCV::VSRL_VX:
  case RISCV::VSRA_VX:
  // 12.4. Vector Single-Width Scaling Shift Instructions
  case RISCV::VSSRL_VX:
  case RISCV::VSSRA_VX:
    // Only the low lg2(SEW) bits of the shift-amount value are used.
    return Log2SEW;

  // 11.7 Vector Narrowing Integer Right Shift Instructions
  case RISCV::VNSRL_WX:
  case RISCV::VNSRA_WX:
  // 12.5. Vector Narrowing Fixed-Point Clip Instructions
  case RISCV::VNCLIPU_WX:
  case RISCV::VNCLIP_WX:
    // Only the low lg2(2*SEW) bits of the shift-amount value are used.
    return Log2SEW + 1;

  // 11.1. Vector Single-Width Integer Add and Subtract
  case RISCV::VADD_VX:
  case RISCV::VSUB_VX:
  case RISCV::VRSUB_VX:
  // 11.2. Vector Widening Integer Add/Subtract
  case RISCV::VWADDU_VX:
  case RISCV::VWSUBU_VX:
  case RISCV::VWADD_VX:
  case RISCV::VWSUB_VX:
  case RISCV::VWADDU_WX:
  case RISCV::VWSUBU_WX:
  case RISCV::VWADD_WX:
  case RISCV::VWSUB_WX:
  // 11.4. Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
  case RISCV::VADC_VXM:
  case RISCV::VADC_VIM:
  case RISCV::VMADC_VXM:
  case RISCV::VMADC_VIM:
  case RISCV::VMADC_VX:
  case RISCV::VSBC_VXM:
  case RISCV::VMSBC_VXM:
  case RISCV::VMSBC_VX:
  // 11.5 Vector Bitwise Logical Instructions
  case RISCV::VAND_VX:
  case RISCV::VOR_VX:
  case RISCV::VXOR_VX:
  // 11.8. Vector Integer Compare Instructions
  case RISCV::VMSEQ_VX:
  case RISCV::VMSNE_VX:
  case RISCV::VMSLTU_VX:
  case RISCV::VMSLT_VX:
  case RISCV::VMSLEU_VX:
  case RISCV::VMSLE_VX:
  case RISCV::VMSGTU_VX:
  case RISCV::VMSGT_VX:
  // 11.9. Vector Integer Min/Max Instructions
  case RISCV::VMINU_VX:
  case RISCV::VMIN_VX:
  case RISCV::VMAXU_VX:
  case RISCV::VMAX_VX:
  // 11.10. Vector Single-Width Integer Multiply Instructions
  case RISCV::VMUL_VX:
  case RISCV::VMULH_VX:
  case RISCV::VMULHU_VX:
  case RISCV::VMULHSU_VX:
  // 11.11. Vector Integer Divide Instructions
  case RISCV::VDIVU_VX:
  case RISCV::VDIV_VX:
  case RISCV::VREMU_VX:
  case RISCV::VREM_VX:
  // 11.12. Vector Widening Integer Multiply Instructions
  case RISCV::VWMUL_VX:
  case RISCV::VWMULU_VX:
  case RISCV::VWMULSU_VX:
  // 11.13. Vector Single-Width Integer Multiply-Add Instructions
  case RISCV::VMACC_VX:
  case RISCV::VNMSAC_VX:
  case RISCV::VMADD_VX:
  case RISCV::VNMSUB_VX:
  // 11.14. Vector Widening Integer Multiply-Add Instructions
  case RISCV::VWMACCU_VX:
  case RISCV::VWMACC_VX:
  case RISCV::VWMACCSU_VX:
  case RISCV::VWMACCUS_VX:
  // 11.15. Vector Integer Merge Instructions
  case RISCV::VMERGE_VXM:
  // 11.16. Vector Integer Move Instructions
  case RISCV::VMV_V_X:
  // 12.1. Vector Single-Width Saturating Add and Subtract
  case RISCV::VSADDU_VX:
  case RISCV::VSADD_VX:
  case RISCV::VSSUBU_VX:
  case RISCV::VSSUB_VX:
  // 12.2. Vector Single-Width Averaging Add and Subtract
  case RISCV::VAADDU_VX:
  case RISCV::VAADD_VX:
  case RISCV::VASUBU_VX:
  case RISCV::VASUB_VX:
  // 12.3. Vector Single-Width Fractional Multiply with Rounding and Saturation
  case RISCV::VSMUL_VX:
  // 16.1. Integer Scalar Move Instructions
  case RISCV::VMV_S_X:
    return 1U << Log2SEW;
  }
}

unsigned RISCV::getRVVMCOpcode(unsigned RVVPseudoOpcode) {
  const RISCVVPseudosTable::PseudoInfo *RVV =
      RISCVVPseudosTable::getPseudoInfo(RVVPseudoOpcode);
  if (!RVV)
    return 0;
  return RVV->BaseInstr;
}
