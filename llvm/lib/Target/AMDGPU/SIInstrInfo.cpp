//===-- SIInstrInfo.cpp - SI Instruction Information  ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief SI Implementation of TargetInstrInfo.
//
//===----------------------------------------------------------------------===//

#include "SIInstrInfo.h"
#include "AMDGPUTargetMachine.h"
#include "GCNHazardRecognizer.h"
#include "SIDefines.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/IR/Function.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

SIInstrInfo::SIInstrInfo(const SISubtarget &ST)
  : AMDGPUInstrInfo(ST), RI(), ST(ST) {}

//===----------------------------------------------------------------------===//
// TargetInstrInfo callbacks
//===----------------------------------------------------------------------===//

static unsigned getNumOperandsNoGlue(SDNode *Node) {
  unsigned N = Node->getNumOperands();
  while (N && Node->getOperand(N - 1).getValueType() == MVT::Glue)
    --N;
  return N;
}

static SDValue findChainOperand(SDNode *Load) {
  SDValue LastOp = Load->getOperand(getNumOperandsNoGlue(Load) - 1);
  assert(LastOp.getValueType() == MVT::Other && "Chain missing from load node");
  return LastOp;
}

/// \brief Returns true if both nodes have the same value for the given
///        operand \p Op, or if both nodes do not have this operand.
static bool nodesHaveSameOperandValue(SDNode *N0, SDNode* N1, unsigned OpName) {
  unsigned Opc0 = N0->getMachineOpcode();
  unsigned Opc1 = N1->getMachineOpcode();

  int Op0Idx = AMDGPU::getNamedOperandIdx(Opc0, OpName);
  int Op1Idx = AMDGPU::getNamedOperandIdx(Opc1, OpName);

  if (Op0Idx == -1 && Op1Idx == -1)
    return true;


  if ((Op0Idx == -1 && Op1Idx != -1) ||
      (Op1Idx == -1 && Op0Idx != -1))
    return false;

  // getNamedOperandIdx returns the index for the MachineInstr's operands,
  // which includes the result as the first operand. We are indexing into the
  // MachineSDNode's operands, so we need to skip the result operand to get
  // the real index.
  --Op0Idx;
  --Op1Idx;

  return N0->getOperand(Op0Idx) == N1->getOperand(Op1Idx);
}

bool SIInstrInfo::isReallyTriviallyReMaterializable(const MachineInstr &MI,
                                                    AliasAnalysis *AA) const {
  // TODO: The generic check fails for VALU instructions that should be
  // rematerializable due to implicit reads of exec. We really want all of the
  // generic logic for this except for this.
  switch (MI.getOpcode()) {
  case AMDGPU::V_MOV_B32_e32:
  case AMDGPU::V_MOV_B32_e64:
  case AMDGPU::V_MOV_B64_PSEUDO:
    return true;
  default:
    return false;
  }
}

bool SIInstrInfo::areLoadsFromSameBasePtr(SDNode *Load0, SDNode *Load1,
                                          int64_t &Offset0,
                                          int64_t &Offset1) const {
  if (!Load0->isMachineOpcode() || !Load1->isMachineOpcode())
    return false;

  unsigned Opc0 = Load0->getMachineOpcode();
  unsigned Opc1 = Load1->getMachineOpcode();

  // Make sure both are actually loads.
  if (!get(Opc0).mayLoad() || !get(Opc1).mayLoad())
    return false;

  if (isDS(Opc0) && isDS(Opc1)) {

    // FIXME: Handle this case:
    if (getNumOperandsNoGlue(Load0) != getNumOperandsNoGlue(Load1))
      return false;

    // Check base reg.
    if (Load0->getOperand(1) != Load1->getOperand(1))
      return false;

    // Check chain.
    if (findChainOperand(Load0) != findChainOperand(Load1))
      return false;

    // Skip read2 / write2 variants for simplicity.
    // TODO: We should report true if the used offsets are adjacent (excluded
    // st64 versions).
    if (AMDGPU::getNamedOperandIdx(Opc0, AMDGPU::OpName::data1) != -1 ||
        AMDGPU::getNamedOperandIdx(Opc1, AMDGPU::OpName::data1) != -1)
      return false;

    Offset0 = cast<ConstantSDNode>(Load0->getOperand(2))->getZExtValue();
    Offset1 = cast<ConstantSDNode>(Load1->getOperand(2))->getZExtValue();
    return true;
  }

  if (isSMRD(Opc0) && isSMRD(Opc1)) {
    assert(getNumOperandsNoGlue(Load0) == getNumOperandsNoGlue(Load1));

    // Check base reg.
    if (Load0->getOperand(0) != Load1->getOperand(0))
      return false;

    const ConstantSDNode *Load0Offset =
        dyn_cast<ConstantSDNode>(Load0->getOperand(1));
    const ConstantSDNode *Load1Offset =
        dyn_cast<ConstantSDNode>(Load1->getOperand(1));

    if (!Load0Offset || !Load1Offset)
      return false;

    // Check chain.
    if (findChainOperand(Load0) != findChainOperand(Load1))
      return false;

    Offset0 = Load0Offset->getZExtValue();
    Offset1 = Load1Offset->getZExtValue();
    return true;
  }

  // MUBUF and MTBUF can access the same addresses.
  if ((isMUBUF(Opc0) || isMTBUF(Opc0)) && (isMUBUF(Opc1) || isMTBUF(Opc1))) {

    // MUBUF and MTBUF have vaddr at different indices.
    if (!nodesHaveSameOperandValue(Load0, Load1, AMDGPU::OpName::soffset) ||
        findChainOperand(Load0) != findChainOperand(Load1) ||
        !nodesHaveSameOperandValue(Load0, Load1, AMDGPU::OpName::vaddr) ||
        !nodesHaveSameOperandValue(Load0, Load1, AMDGPU::OpName::srsrc))
      return false;

    int OffIdx0 = AMDGPU::getNamedOperandIdx(Opc0, AMDGPU::OpName::offset);
    int OffIdx1 = AMDGPU::getNamedOperandIdx(Opc1, AMDGPU::OpName::offset);

    if (OffIdx0 == -1 || OffIdx1 == -1)
      return false;

    // getNamedOperandIdx returns the index for MachineInstrs.  Since they
    // inlcude the output in the operand list, but SDNodes don't, we need to
    // subtract the index by one.
    --OffIdx0;
    --OffIdx1;

    SDValue Off0 = Load0->getOperand(OffIdx0);
    SDValue Off1 = Load1->getOperand(OffIdx1);

    // The offset might be a FrameIndexSDNode.
    if (!isa<ConstantSDNode>(Off0) || !isa<ConstantSDNode>(Off1))
      return false;

    Offset0 = cast<ConstantSDNode>(Off0)->getZExtValue();
    Offset1 = cast<ConstantSDNode>(Off1)->getZExtValue();
    return true;
  }

  return false;
}

static bool isStride64(unsigned Opc) {
  switch (Opc) {
  case AMDGPU::DS_READ2ST64_B32:
  case AMDGPU::DS_READ2ST64_B64:
  case AMDGPU::DS_WRITE2ST64_B32:
  case AMDGPU::DS_WRITE2ST64_B64:
    return true;
  default:
    return false;
  }
}

bool SIInstrInfo::getMemOpBaseRegImmOfs(MachineInstr &LdSt, unsigned &BaseReg,
                                        int64_t &Offset,
                                        const TargetRegisterInfo *TRI) const {
  unsigned Opc = LdSt.getOpcode();

  if (isDS(LdSt)) {
    const MachineOperand *OffsetImm =
        getNamedOperand(LdSt, AMDGPU::OpName::offset);
    if (OffsetImm) {
      // Normal, single offset LDS instruction.
      const MachineOperand *AddrReg =
          getNamedOperand(LdSt, AMDGPU::OpName::addr);

      BaseReg = AddrReg->getReg();
      Offset = OffsetImm->getImm();
      return true;
    }

    // The 2 offset instructions use offset0 and offset1 instead. We can treat
    // these as a load with a single offset if the 2 offsets are consecutive. We
    // will use this for some partially aligned loads.
    const MachineOperand *Offset0Imm =
        getNamedOperand(LdSt, AMDGPU::OpName::offset0);
    const MachineOperand *Offset1Imm =
        getNamedOperand(LdSt, AMDGPU::OpName::offset1);

    uint8_t Offset0 = Offset0Imm->getImm();
    uint8_t Offset1 = Offset1Imm->getImm();

    if (Offset1 > Offset0 && Offset1 - Offset0 == 1) {
      // Each of these offsets is in element sized units, so we need to convert
      // to bytes of the individual reads.

      unsigned EltSize;
      if (LdSt.mayLoad())
        EltSize = getOpRegClass(LdSt, 0)->getSize() / 2;
      else {
        assert(LdSt.mayStore());
        int Data0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::data0);
        EltSize = getOpRegClass(LdSt, Data0Idx)->getSize();
      }

      if (isStride64(Opc))
        EltSize *= 64;

      const MachineOperand *AddrReg =
          getNamedOperand(LdSt, AMDGPU::OpName::addr);
      BaseReg = AddrReg->getReg();
      Offset = EltSize * Offset0;
      return true;
    }

    return false;
  }

  if (isMUBUF(LdSt) || isMTBUF(LdSt)) {
    if (AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::soffset) != -1)
      return false;

    const MachineOperand *AddrReg =
        getNamedOperand(LdSt, AMDGPU::OpName::vaddr);
    if (!AddrReg)
      return false;

    const MachineOperand *OffsetImm =
        getNamedOperand(LdSt, AMDGPU::OpName::offset);
    BaseReg = AddrReg->getReg();
    Offset = OffsetImm->getImm();
    return true;
  }

  if (isSMRD(LdSt)) {
    const MachineOperand *OffsetImm =
        getNamedOperand(LdSt, AMDGPU::OpName::offset);
    if (!OffsetImm)
      return false;

    const MachineOperand *SBaseReg =
        getNamedOperand(LdSt, AMDGPU::OpName::sbase);
    BaseReg = SBaseReg->getReg();
    Offset = OffsetImm->getImm();
    return true;
  }

  if (isFLAT(LdSt)) {
    const MachineOperand *AddrReg = getNamedOperand(LdSt, AMDGPU::OpName::addr);
    BaseReg = AddrReg->getReg();
    Offset = 0;
    return true;
  }

  return false;
}

bool SIInstrInfo::shouldClusterMemOps(MachineInstr &FirstLdSt,
                                      MachineInstr &SecondLdSt,
                                      unsigned NumLoads) const {
  const MachineOperand *FirstDst = nullptr;
  const MachineOperand *SecondDst = nullptr;

  if (isDS(FirstLdSt) && isDS(SecondLdSt)) {
    FirstDst = getNamedOperand(FirstLdSt, AMDGPU::OpName::vdst);
    SecondDst = getNamedOperand(SecondLdSt, AMDGPU::OpName::vdst);
  }

  if (isSMRD(FirstLdSt) && isSMRD(SecondLdSt)) {
    FirstDst = getNamedOperand(FirstLdSt, AMDGPU::OpName::sdst);
    SecondDst = getNamedOperand(SecondLdSt, AMDGPU::OpName::sdst);
  }

  if ((isMUBUF(FirstLdSt) && isMUBUF(SecondLdSt)) ||
      (isMTBUF(FirstLdSt) && isMTBUF(SecondLdSt))) {
    FirstDst = getNamedOperand(FirstLdSt, AMDGPU::OpName::vdata);
    SecondDst = getNamedOperand(SecondLdSt, AMDGPU::OpName::vdata);
  }

  if (!FirstDst || !SecondDst)
    return false;

  // Try to limit clustering based on the total number of bytes loaded
  // rather than the number of instructions.  This is done to help reduce
  // register pressure.  The method used is somewhat inexact, though,
  // because it assumes that all loads in the cluster will load the
  // same number of bytes as FirstLdSt.

  // The unit of this value is bytes.
  // FIXME: This needs finer tuning.
  unsigned LoadClusterThreshold = 16;

  const MachineRegisterInfo &MRI =
      FirstLdSt.getParent()->getParent()->getRegInfo();
  const TargetRegisterClass *DstRC = MRI.getRegClass(FirstDst->getReg());

  return (NumLoads * DstRC->getSize()) <= LoadClusterThreshold;
}

void SIInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MI,
                              const DebugLoc &DL, unsigned DestReg,
                              unsigned SrcReg, bool KillSrc) const {

  static const int16_t Sub0_15[] = {
    AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2, AMDGPU::sub3,
    AMDGPU::sub4, AMDGPU::sub5, AMDGPU::sub6, AMDGPU::sub7,
    AMDGPU::sub8, AMDGPU::sub9, AMDGPU::sub10, AMDGPU::sub11,
    AMDGPU::sub12, AMDGPU::sub13, AMDGPU::sub14, AMDGPU::sub15,
  };

  static const int16_t Sub0_15_64[] = {
    AMDGPU::sub0_sub1, AMDGPU::sub2_sub3,
    AMDGPU::sub4_sub5, AMDGPU::sub6_sub7,
    AMDGPU::sub8_sub9, AMDGPU::sub10_sub11,
    AMDGPU::sub12_sub13, AMDGPU::sub14_sub15,
  };

  static const int16_t Sub0_7[] = {
    AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2, AMDGPU::sub3,
    AMDGPU::sub4, AMDGPU::sub5, AMDGPU::sub6, AMDGPU::sub7,
  };

  static const int16_t Sub0_7_64[] = {
    AMDGPU::sub0_sub1, AMDGPU::sub2_sub3,
    AMDGPU::sub4_sub5, AMDGPU::sub6_sub7,
  };

  static const int16_t Sub0_3[] = {
    AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2, AMDGPU::sub3,
  };

  static const int16_t Sub0_3_64[] = {
    AMDGPU::sub0_sub1, AMDGPU::sub2_sub3,
  };

  static const int16_t Sub0_2[] = {
    AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2,
  };

  static const int16_t Sub0_1[] = {
    AMDGPU::sub0, AMDGPU::sub1,
  };

  unsigned Opcode;
  ArrayRef<int16_t> SubIndices;

  if (AMDGPU::SReg_32RegClass.contains(DestReg)) {
    if (SrcReg == AMDGPU::SCC) {
      BuildMI(MBB, MI, DL, get(AMDGPU::S_CSELECT_B32), DestReg)
          .addImm(-1)
          .addImm(0);
      return;
    }

    assert(AMDGPU::SReg_32RegClass.contains(SrcReg));
    BuildMI(MBB, MI, DL, get(AMDGPU::S_MOV_B32), DestReg)
            .addReg(SrcReg, getKillRegState(KillSrc));
    return;

  } else if (AMDGPU::SReg_64RegClass.contains(DestReg)) {
    if (DestReg == AMDGPU::VCC) {
      if (AMDGPU::SReg_64RegClass.contains(SrcReg)) {
        BuildMI(MBB, MI, DL, get(AMDGPU::S_MOV_B64), AMDGPU::VCC)
          .addReg(SrcReg, getKillRegState(KillSrc));
      } else {
        // FIXME: Hack until VReg_1 removed.
        assert(AMDGPU::VGPR_32RegClass.contains(SrcReg));
        BuildMI(MBB, MI, DL, get(AMDGPU::V_CMP_NE_I32_e32))
          .addImm(0)
          .addReg(SrcReg, getKillRegState(KillSrc));
      }

      return;
    }

    assert(AMDGPU::SReg_64RegClass.contains(SrcReg));
    BuildMI(MBB, MI, DL, get(AMDGPU::S_MOV_B64), DestReg)
            .addReg(SrcReg, getKillRegState(KillSrc));
    return;

  } else if (DestReg == AMDGPU::SCC) {
    assert(AMDGPU::SReg_32RegClass.contains(SrcReg));
    BuildMI(MBB, MI, DL, get(AMDGPU::S_CMP_LG_U32))
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addImm(0);
    return;
  } else if (AMDGPU::SReg_128RegClass.contains(DestReg)) {
    assert(AMDGPU::SReg_128RegClass.contains(SrcReg));
    Opcode = AMDGPU::S_MOV_B64;
    SubIndices = Sub0_3_64;

  } else if (AMDGPU::SReg_256RegClass.contains(DestReg)) {
    assert(AMDGPU::SReg_256RegClass.contains(SrcReg));
    Opcode = AMDGPU::S_MOV_B64;
    SubIndices = Sub0_7_64;

  } else if (AMDGPU::SReg_512RegClass.contains(DestReg)) {
    assert(AMDGPU::SReg_512RegClass.contains(SrcReg));
    Opcode = AMDGPU::S_MOV_B64;
    SubIndices = Sub0_15_64;

  } else if (AMDGPU::VGPR_32RegClass.contains(DestReg)) {
    assert(AMDGPU::VGPR_32RegClass.contains(SrcReg) ||
           AMDGPU::SReg_32RegClass.contains(SrcReg));
    BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_e32), DestReg)
            .addReg(SrcReg, getKillRegState(KillSrc));
    return;

  } else if (AMDGPU::VReg_64RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_64RegClass.contains(SrcReg) ||
           AMDGPU::SReg_64RegClass.contains(SrcReg));
    Opcode = AMDGPU::V_MOV_B32_e32;
    SubIndices = Sub0_1;

  } else if (AMDGPU::VReg_96RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_96RegClass.contains(SrcReg));
    Opcode = AMDGPU::V_MOV_B32_e32;
    SubIndices = Sub0_2;

  } else if (AMDGPU::VReg_128RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_128RegClass.contains(SrcReg) ||
           AMDGPU::SReg_128RegClass.contains(SrcReg));
    Opcode = AMDGPU::V_MOV_B32_e32;
    SubIndices = Sub0_3;

  } else if (AMDGPU::VReg_256RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_256RegClass.contains(SrcReg) ||
           AMDGPU::SReg_256RegClass.contains(SrcReg));
    Opcode = AMDGPU::V_MOV_B32_e32;
    SubIndices = Sub0_7;

  } else if (AMDGPU::VReg_512RegClass.contains(DestReg)) {
    assert(AMDGPU::VReg_512RegClass.contains(SrcReg) ||
           AMDGPU::SReg_512RegClass.contains(SrcReg));
    Opcode = AMDGPU::V_MOV_B32_e32;
    SubIndices = Sub0_15;

  } else {
    llvm_unreachable("Can't copy register!");
  }

  bool Forward = RI.getHWRegIndex(DestReg) <= RI.getHWRegIndex(SrcReg);

  for (unsigned Idx = 0; Idx < SubIndices.size(); ++Idx) {
    unsigned SubIdx;
    if (Forward)
      SubIdx = SubIndices[Idx];
    else
      SubIdx = SubIndices[SubIndices.size() - Idx - 1];

    MachineInstrBuilder Builder = BuildMI(MBB, MI, DL,
      get(Opcode), RI.getSubReg(DestReg, SubIdx));

    Builder.addReg(RI.getSubReg(SrcReg, SubIdx));

    if (Idx == SubIndices.size() - 1)
      Builder.addReg(SrcReg, getKillRegState(KillSrc) | RegState::Implicit);

    if (Idx == 0)
      Builder.addReg(DestReg, RegState::Define | RegState::Implicit);

    Builder.addReg(SrcReg, RegState::Implicit);
  }
}

int SIInstrInfo::commuteOpcode(unsigned Opcode) const {
  int NewOpc;

  // Try to map original to commuted opcode
  NewOpc = AMDGPU::getCommuteRev(Opcode);
  if (NewOpc != -1)
    // Check if the commuted (REV) opcode exists on the target.
    return pseudoToMCOpcode(NewOpc) != -1 ? NewOpc : -1;

  // Try to map commuted to original opcode
  NewOpc = AMDGPU::getCommuteOrig(Opcode);
  if (NewOpc != -1)
    // Check if the original (non-REV) opcode exists on the target.
    return pseudoToMCOpcode(NewOpc) != -1 ? NewOpc : -1;

  return Opcode;
}

unsigned SIInstrInfo::getMovOpcode(const TargetRegisterClass *DstRC) const {

  if (DstRC->getSize() == 4) {
    return RI.isSGPRClass(DstRC) ? AMDGPU::S_MOV_B32 : AMDGPU::V_MOV_B32_e32;
  } else if (DstRC->getSize() == 8 && RI.isSGPRClass(DstRC)) {
    return AMDGPU::S_MOV_B64;
  } else if (DstRC->getSize() == 8 && !RI.isSGPRClass(DstRC)) {
    return  AMDGPU::V_MOV_B64_PSEUDO;
  }
  return AMDGPU::COPY;
}

static unsigned getSGPRSpillSaveOpcode(unsigned Size) {
  switch (Size) {
  case 4:
    return AMDGPU::SI_SPILL_S32_SAVE;
  case 8:
    return AMDGPU::SI_SPILL_S64_SAVE;
  case 16:
    return AMDGPU::SI_SPILL_S128_SAVE;
  case 32:
    return AMDGPU::SI_SPILL_S256_SAVE;
  case 64:
    return AMDGPU::SI_SPILL_S512_SAVE;
  default:
    llvm_unreachable("unknown register size");
  }
}

static unsigned getVGPRSpillSaveOpcode(unsigned Size) {
  switch (Size) {
  case 4:
    return AMDGPU::SI_SPILL_V32_SAVE;
  case 8:
    return AMDGPU::SI_SPILL_V64_SAVE;
  case 12:
    return AMDGPU::SI_SPILL_V96_SAVE;
  case 16:
    return AMDGPU::SI_SPILL_V128_SAVE;
  case 32:
    return AMDGPU::SI_SPILL_V256_SAVE;
  case 64:
    return AMDGPU::SI_SPILL_V512_SAVE;
  default:
    llvm_unreachable("unknown register size");
  }
}

void SIInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MI,
                                      unsigned SrcReg, bool isKill,
                                      int FrameIndex,
                                      const TargetRegisterClass *RC,
                                      const TargetRegisterInfo *TRI) const {
  MachineFunction *MF = MBB.getParent();
  SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  MachineFrameInfo &FrameInfo = MF->getFrameInfo();
  DebugLoc DL = MBB.findDebugLoc(MI);

  unsigned Size = FrameInfo.getObjectSize(FrameIndex);
  unsigned Align = FrameInfo.getObjectAlignment(FrameIndex);
  MachinePointerInfo PtrInfo
    = MachinePointerInfo::getFixedStack(*MF, FrameIndex);
  MachineMemOperand *MMO
    = MF->getMachineMemOperand(PtrInfo, MachineMemOperand::MOStore,
                               Size, Align);

  if (RI.isSGPRClass(RC)) {
    MFI->setHasSpilledSGPRs();

    // We are only allowed to create one new instruction when spilling
    // registers, so we need to use pseudo instruction for spilling SGPRs.
    const MCInstrDesc &OpDesc = get(getSGPRSpillSaveOpcode(RC->getSize()));

    // The SGPR spill/restore instructions only work on number sgprs, so we need
    // to make sure we are using the correct register class.
    if (TargetRegisterInfo::isVirtualRegister(SrcReg) && RC->getSize() == 4) {
      MachineRegisterInfo &MRI = MF->getRegInfo();
      MRI.constrainRegClass(SrcReg, &AMDGPU::SReg_32_XM0RegClass);
    }

    BuildMI(MBB, MI, DL, OpDesc)
      .addReg(SrcReg, getKillRegState(isKill)) // data
      .addFrameIndex(FrameIndex)               // addr
      .addMemOperand(MMO);

    return;
  }

  if (!ST.isVGPRSpillingEnabled(*MF->getFunction())) {
    LLVMContext &Ctx = MF->getFunction()->getContext();
    Ctx.emitError("SIInstrInfo::storeRegToStackSlot - Do not know how to"
                  " spill register");
    BuildMI(MBB, MI, DL, get(AMDGPU::KILL))
      .addReg(SrcReg);

    return;
  }

  assert(RI.hasVGPRs(RC) && "Only VGPR spilling expected");

  unsigned Opcode = getVGPRSpillSaveOpcode(RC->getSize());
  MFI->setHasSpilledVGPRs();
  BuildMI(MBB, MI, DL, get(Opcode))
    .addReg(SrcReg, getKillRegState(isKill)) // data
    .addFrameIndex(FrameIndex)               // addr
    .addReg(MFI->getScratchRSrcReg())        // scratch_rsrc
    .addReg(MFI->getScratchWaveOffsetReg())  // scratch_offset
    .addImm(0)                               // offset
    .addMemOperand(MMO);
}

static unsigned getSGPRSpillRestoreOpcode(unsigned Size) {
  switch (Size) {
  case 4:
    return AMDGPU::SI_SPILL_S32_RESTORE;
  case 8:
    return AMDGPU::SI_SPILL_S64_RESTORE;
  case 16:
    return AMDGPU::SI_SPILL_S128_RESTORE;
  case 32:
    return AMDGPU::SI_SPILL_S256_RESTORE;
  case 64:
    return AMDGPU::SI_SPILL_S512_RESTORE;
  default:
    llvm_unreachable("unknown register size");
  }
}

static unsigned getVGPRSpillRestoreOpcode(unsigned Size) {
  switch (Size) {
  case 4:
    return AMDGPU::SI_SPILL_V32_RESTORE;
  case 8:
    return AMDGPU::SI_SPILL_V64_RESTORE;
  case 12:
    return AMDGPU::SI_SPILL_V96_RESTORE;
  case 16:
    return AMDGPU::SI_SPILL_V128_RESTORE;
  case 32:
    return AMDGPU::SI_SPILL_V256_RESTORE;
  case 64:
    return AMDGPU::SI_SPILL_V512_RESTORE;
  default:
    llvm_unreachable("unknown register size");
  }
}

void SIInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator MI,
                                       unsigned DestReg, int FrameIndex,
                                       const TargetRegisterClass *RC,
                                       const TargetRegisterInfo *TRI) const {
  MachineFunction *MF = MBB.getParent();
  const SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  MachineFrameInfo &FrameInfo = MF->getFrameInfo();
  DebugLoc DL = MBB.findDebugLoc(MI);
  unsigned Align = FrameInfo.getObjectAlignment(FrameIndex);
  unsigned Size = FrameInfo.getObjectSize(FrameIndex);

  MachinePointerInfo PtrInfo
    = MachinePointerInfo::getFixedStack(*MF, FrameIndex);

  MachineMemOperand *MMO = MF->getMachineMemOperand(
    PtrInfo, MachineMemOperand::MOLoad, Size, Align);

  if (RI.isSGPRClass(RC)) {
    // FIXME: Maybe this should not include a memoperand because it will be
    // lowered to non-memory instructions.
    const MCInstrDesc &OpDesc = get(getSGPRSpillRestoreOpcode(RC->getSize()));
    if (TargetRegisterInfo::isVirtualRegister(DestReg) && RC->getSize() == 4) {
      MachineRegisterInfo &MRI = MF->getRegInfo();
      MRI.constrainRegClass(DestReg, &AMDGPU::SReg_32_XM0RegClass);
    }

    BuildMI(MBB, MI, DL, OpDesc, DestReg)
      .addFrameIndex(FrameIndex) // addr
      .addMemOperand(MMO);

    return;
  }

  if (!ST.isVGPRSpillingEnabled(*MF->getFunction())) {
    LLVMContext &Ctx = MF->getFunction()->getContext();
    Ctx.emitError("SIInstrInfo::loadRegFromStackSlot - Do not know how to"
                  " restore register");
    BuildMI(MBB, MI, DL, get(AMDGPU::IMPLICIT_DEF), DestReg);

    return;
  }

  assert(RI.hasVGPRs(RC) && "Only VGPR spilling expected");

  unsigned Opcode = getVGPRSpillRestoreOpcode(RC->getSize());
  BuildMI(MBB, MI, DL, get(Opcode), DestReg)
    .addFrameIndex(FrameIndex)              // vaddr
    .addReg(MFI->getScratchRSrcReg())       // scratch_rsrc
    .addReg(MFI->getScratchWaveOffsetReg()) // scratch_offset
    .addImm(0)                              // offset
    .addMemOperand(MMO);
}

/// \param @Offset Offset in bytes of the FrameIndex being spilled
unsigned SIInstrInfo::calculateLDSSpillAddress(
    MachineBasicBlock &MBB, MachineInstr &MI, RegScavenger *RS, unsigned TmpReg,
    unsigned FrameOffset, unsigned Size) const {
  MachineFunction *MF = MBB.getParent();
  SIMachineFunctionInfo *MFI = MF->getInfo<SIMachineFunctionInfo>();
  const SISubtarget &ST = MF->getSubtarget<SISubtarget>();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  DebugLoc DL = MBB.findDebugLoc(MI);
  unsigned WorkGroupSize = MFI->getMaxFlatWorkGroupSize();
  unsigned WavefrontSize = ST.getWavefrontSize();

  unsigned TIDReg = MFI->getTIDReg();
  if (!MFI->hasCalculatedTID()) {
    MachineBasicBlock &Entry = MBB.getParent()->front();
    MachineBasicBlock::iterator Insert = Entry.front();
    DebugLoc DL = Insert->getDebugLoc();

    TIDReg = RI.findUnusedRegister(MF->getRegInfo(), &AMDGPU::VGPR_32RegClass,
                                   *MF);
    if (TIDReg == AMDGPU::NoRegister)
      return TIDReg;

    if (!AMDGPU::isShader(MF->getFunction()->getCallingConv()) &&
        WorkGroupSize > WavefrontSize) {

      unsigned TIDIGXReg
        = TRI->getPreloadedValue(*MF, SIRegisterInfo::WORKGROUP_ID_X);
      unsigned TIDIGYReg
        = TRI->getPreloadedValue(*MF, SIRegisterInfo::WORKGROUP_ID_Y);
      unsigned TIDIGZReg
        = TRI->getPreloadedValue(*MF, SIRegisterInfo::WORKGROUP_ID_Z);
      unsigned InputPtrReg =
          TRI->getPreloadedValue(*MF, SIRegisterInfo::KERNARG_SEGMENT_PTR);
      for (unsigned Reg : {TIDIGXReg, TIDIGYReg, TIDIGZReg}) {
        if (!Entry.isLiveIn(Reg))
          Entry.addLiveIn(Reg);
      }

      RS->enterBasicBlock(Entry);
      // FIXME: Can we scavenge an SReg_64 and access the subregs?
      unsigned STmp0 = RS->scavengeRegister(&AMDGPU::SGPR_32RegClass, 0);
      unsigned STmp1 = RS->scavengeRegister(&AMDGPU::SGPR_32RegClass, 0);
      BuildMI(Entry, Insert, DL, get(AMDGPU::S_LOAD_DWORD_IMM), STmp0)
              .addReg(InputPtrReg)
              .addImm(SI::KernelInputOffsets::NGROUPS_Z);
      BuildMI(Entry, Insert, DL, get(AMDGPU::S_LOAD_DWORD_IMM), STmp1)
              .addReg(InputPtrReg)
              .addImm(SI::KernelInputOffsets::NGROUPS_Y);

      // NGROUPS.X * NGROUPS.Y
      BuildMI(Entry, Insert, DL, get(AMDGPU::S_MUL_I32), STmp1)
              .addReg(STmp1)
              .addReg(STmp0);
      // (NGROUPS.X * NGROUPS.Y) * TIDIG.X
      BuildMI(Entry, Insert, DL, get(AMDGPU::V_MUL_U32_U24_e32), TIDReg)
              .addReg(STmp1)
              .addReg(TIDIGXReg);
      // NGROUPS.Z * TIDIG.Y + (NGROUPS.X * NGROPUS.Y * TIDIG.X)
      BuildMI(Entry, Insert, DL, get(AMDGPU::V_MAD_U32_U24), TIDReg)
              .addReg(STmp0)
              .addReg(TIDIGYReg)
              .addReg(TIDReg);
      // (NGROUPS.Z * TIDIG.Y + (NGROUPS.X * NGROPUS.Y * TIDIG.X)) + TIDIG.Z
      BuildMI(Entry, Insert, DL, get(AMDGPU::V_ADD_I32_e32), TIDReg)
              .addReg(TIDReg)
              .addReg(TIDIGZReg);
    } else {
      // Get the wave id
      BuildMI(Entry, Insert, DL, get(AMDGPU::V_MBCNT_LO_U32_B32_e64),
              TIDReg)
              .addImm(-1)
              .addImm(0);

      BuildMI(Entry, Insert, DL, get(AMDGPU::V_MBCNT_HI_U32_B32_e64),
              TIDReg)
              .addImm(-1)
              .addReg(TIDReg);
    }

    BuildMI(Entry, Insert, DL, get(AMDGPU::V_LSHLREV_B32_e32),
            TIDReg)
            .addImm(2)
            .addReg(TIDReg);
    MFI->setTIDReg(TIDReg);
  }

  // Add FrameIndex to LDS offset
  unsigned LDSOffset = MFI->getLDSSize() + (FrameOffset * WorkGroupSize);
  BuildMI(MBB, MI, DL, get(AMDGPU::V_ADD_I32_e32), TmpReg)
          .addImm(LDSOffset)
          .addReg(TIDReg);

  return TmpReg;
}

void SIInstrInfo::insertWaitStates(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   int Count) const {
  DebugLoc DL = MBB.findDebugLoc(MI);
  while (Count > 0) {
    int Arg;
    if (Count >= 8)
      Arg = 7;
    else
      Arg = Count - 1;
    Count -= 8;
    BuildMI(MBB, MI, DL, get(AMDGPU::S_NOP))
            .addImm(Arg);
  }
}

void SIInstrInfo::insertNoop(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MI) const {
  insertWaitStates(MBB, MI, 1);
}

unsigned SIInstrInfo::getNumWaitStates(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  default: return 1; // FIXME: Do wait states equal cycles?

  case AMDGPU::S_NOP:
    return MI.getOperand(0).getImm() + 1;
  }
}

bool SIInstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MBB.findDebugLoc(MI);
  switch (MI.getOpcode()) {
  default: return AMDGPUInstrInfo::expandPostRAPseudo(MI);

  case AMDGPU::V_MOV_B64_PSEUDO: {
    unsigned Dst = MI.getOperand(0).getReg();
    unsigned DstLo = RI.getSubReg(Dst, AMDGPU::sub0);
    unsigned DstHi = RI.getSubReg(Dst, AMDGPU::sub1);

    const MachineOperand &SrcOp = MI.getOperand(1);
    // FIXME: Will this work for 64-bit floating point immediates?
    assert(!SrcOp.isFPImm());
    if (SrcOp.isImm()) {
      APInt Imm(64, SrcOp.getImm());
      BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_e32), DstLo)
        .addImm(Imm.getLoBits(32).getZExtValue())
        .addReg(Dst, RegState::Implicit | RegState::Define);
      BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_e32), DstHi)
        .addImm(Imm.getHiBits(32).getZExtValue())
        .addReg(Dst, RegState::Implicit | RegState::Define);
    } else {
      assert(SrcOp.isReg());
      BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_e32), DstLo)
        .addReg(RI.getSubReg(SrcOp.getReg(), AMDGPU::sub0))
        .addReg(Dst, RegState::Implicit | RegState::Define);
      BuildMI(MBB, MI, DL, get(AMDGPU::V_MOV_B32_e32), DstHi)
        .addReg(RI.getSubReg(SrcOp.getReg(), AMDGPU::sub1))
        .addReg(Dst, RegState::Implicit | RegState::Define);
    }
    MI.eraseFromParent();
    break;
  }
  case AMDGPU::SI_PC_ADD_REL_OFFSET: {
    MachineFunction &MF = *MBB.getParent();
    unsigned Reg = MI.getOperand(0).getReg();
    unsigned RegLo = RI.getSubReg(Reg, AMDGPU::sub0);
    unsigned RegHi = RI.getSubReg(Reg, AMDGPU::sub1);

    // Create a bundle so these instructions won't be re-ordered by the
    // post-RA scheduler.
    MIBundleBuilder Bundler(MBB, MI);
    Bundler.append(BuildMI(MF, DL, get(AMDGPU::S_GETPC_B64), Reg));

    // Add 32-bit offset from this instruction to the start of the
    // constant data.
    Bundler.append(BuildMI(MF, DL, get(AMDGPU::S_ADD_U32), RegLo)
                       .addReg(RegLo)
                       .addOperand(MI.getOperand(1)));
    Bundler.append(BuildMI(MF, DL, get(AMDGPU::S_ADDC_U32), RegHi)
                           .addReg(RegHi)
                           .addImm(0));

    llvm::finalizeBundle(MBB, Bundler.begin());

    MI.eraseFromParent();
    break;
  }
  }
  return true;
}

bool SIInstrInfo::swapSourceModifiers(MachineInstr &MI,
                                      MachineOperand &Src0,
                                      unsigned Src0OpName,
                                      MachineOperand &Src1,
                                      unsigned Src1OpName) const {
  MachineOperand *Src0Mods = getNamedOperand(MI, Src0OpName);
  if (!Src0Mods)
    return false;

  MachineOperand *Src1Mods = getNamedOperand(MI, Src1OpName);
  assert(Src1Mods &&
         "All commutable instructions have both src0 and src1 modifiers");

  int Src0ModsVal = Src0Mods->getImm();
  int Src1ModsVal = Src1Mods->getImm();

  Src1Mods->setImm(Src0ModsVal);
  Src0Mods->setImm(Src1ModsVal);
  return true;
}

static MachineInstr *swapRegAndNonRegOperand(MachineInstr &MI,
                                             MachineOperand &RegOp,
                                             MachineOperand &NonRegOp) {
  unsigned Reg = RegOp.getReg();
  unsigned SubReg = RegOp.getSubReg();
  bool IsKill = RegOp.isKill();
  bool IsDead = RegOp.isDead();
  bool IsUndef = RegOp.isUndef();
  bool IsDebug = RegOp.isDebug();

  if (NonRegOp.isImm())
    RegOp.ChangeToImmediate(NonRegOp.getImm());
  else if (NonRegOp.isFI())
    RegOp.ChangeToFrameIndex(NonRegOp.getIndex());
  else
    return nullptr;

  NonRegOp.ChangeToRegister(Reg, false, false, IsKill, IsDead, IsUndef, IsDebug);
  NonRegOp.setSubReg(SubReg);

  return &MI;
}

MachineInstr *SIInstrInfo::commuteInstructionImpl(MachineInstr &MI, bool NewMI,
                                                  unsigned Src0Idx,
                                                  unsigned Src1Idx) const {
  assert(!NewMI && "this should never be used");

  unsigned Opc = MI.getOpcode();
  int CommutedOpcode = commuteOpcode(Opc);
  if (CommutedOpcode == -1)
    return nullptr;

  assert(AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0) ==
           static_cast<int>(Src0Idx) &&
         AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1) ==
           static_cast<int>(Src1Idx) &&
         "inconsistency with findCommutedOpIndices");

  MachineOperand &Src0 = MI.getOperand(Src0Idx);
  MachineOperand &Src1 = MI.getOperand(Src1Idx);

  MachineInstr *CommutedMI = nullptr;
  if (Src0.isReg() && Src1.isReg()) {
    if (isOperandLegal(MI, Src1Idx, &Src0)) {
      // Be sure to copy the source modifiers to the right place.
      CommutedMI
        = TargetInstrInfo::commuteInstructionImpl(MI, NewMI, Src0Idx, Src1Idx);
    }

  } else if (Src0.isReg() && !Src1.isReg()) {
    // src0 should always be able to support any operand type, so no need to
    // check operand legality.
    CommutedMI = swapRegAndNonRegOperand(MI, Src0, Src1);
  } else if (!Src0.isReg() && Src1.isReg()) {
    if (isOperandLegal(MI, Src1Idx, &Src0))
      CommutedMI = swapRegAndNonRegOperand(MI, Src1, Src0);
  } else {
    // FIXME: Found two non registers to commute. This does happen.
    return nullptr;
  }


  if (CommutedMI) {
    swapSourceModifiers(MI, Src0, AMDGPU::OpName::src0_modifiers,
                        Src1, AMDGPU::OpName::src1_modifiers);

    CommutedMI->setDesc(get(CommutedOpcode));
  }

  return CommutedMI;
}

// This needs to be implemented because the source modifiers may be inserted
// between the true commutable operands, and the base
// TargetInstrInfo::commuteInstruction uses it.
bool SIInstrInfo::findCommutedOpIndices(MachineInstr &MI, unsigned &SrcOpIdx0,
                                        unsigned &SrcOpIdx1) const {
  if (!MI.isCommutable())
    return false;

  unsigned Opc = MI.getOpcode();
  int Src0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0);
  if (Src0Idx == -1)
    return false;

  int Src1Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1);
  if (Src1Idx == -1)
    return false;

  return fixCommutedOpIndices(SrcOpIdx0, SrcOpIdx1, Src0Idx, Src1Idx);
}

unsigned SIInstrInfo::getBranchOpcode(SIInstrInfo::BranchPredicate Cond) {
  switch (Cond) {
  case SIInstrInfo::SCC_TRUE:
    return AMDGPU::S_CBRANCH_SCC1;
  case SIInstrInfo::SCC_FALSE:
    return AMDGPU::S_CBRANCH_SCC0;
  case SIInstrInfo::VCCNZ:
    return AMDGPU::S_CBRANCH_VCCNZ;
  case SIInstrInfo::VCCZ:
    return AMDGPU::S_CBRANCH_VCCZ;
  case SIInstrInfo::EXECNZ:
    return AMDGPU::S_CBRANCH_EXECNZ;
  case SIInstrInfo::EXECZ:
    return AMDGPU::S_CBRANCH_EXECZ;
  default:
    llvm_unreachable("invalid branch predicate");
  }
}

SIInstrInfo::BranchPredicate SIInstrInfo::getBranchPredicate(unsigned Opcode) {
  switch (Opcode) {
  case AMDGPU::S_CBRANCH_SCC0:
    return SCC_FALSE;
  case AMDGPU::S_CBRANCH_SCC1:
    return SCC_TRUE;
  case AMDGPU::S_CBRANCH_VCCNZ:
    return VCCNZ;
  case AMDGPU::S_CBRANCH_VCCZ:
    return VCCZ;
  case AMDGPU::S_CBRANCH_EXECNZ:
    return EXECNZ;
  case AMDGPU::S_CBRANCH_EXECZ:
    return EXECZ;
  default:
    return INVALID_BR;
  }
}

bool SIInstrInfo::analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                                MachineBasicBlock *&FBB,
                                SmallVectorImpl<MachineOperand> &Cond,
                                bool AllowModify) const {
  MachineBasicBlock::iterator I = MBB.getFirstTerminator();

  if (I == MBB.end())
    return false;

  if (I->getOpcode() == AMDGPU::S_BRANCH) {
    // Unconditional Branch
    TBB = I->getOperand(0).getMBB();
    return false;
  }

  BranchPredicate Pred = getBranchPredicate(I->getOpcode());
  if (Pred == INVALID_BR)
    return true;

  MachineBasicBlock *CondBB = I->getOperand(0).getMBB();
  Cond.push_back(MachineOperand::CreateImm(Pred));

  ++I;

  if (I == MBB.end()) {
    // Conditional branch followed by fall-through.
    TBB = CondBB;
    return false;
  }

  if (I->getOpcode() == AMDGPU::S_BRANCH) {
    TBB = CondBB;
    FBB = I->getOperand(0).getMBB();
    return false;
  }

  return true;
}

unsigned SIInstrInfo::RemoveBranch(MachineBasicBlock &MBB,
                                   int *BytesRemoved) const {
  MachineBasicBlock::iterator I = MBB.getFirstTerminator();

  unsigned Count = 0;
  unsigned RemovedSize = 0;
  while (I != MBB.end()) {
    MachineBasicBlock::iterator Next = std::next(I);
    RemovedSize += getInstSizeInBytes(*I);
    I->eraseFromParent();
    ++Count;
    I = Next;
  }

  if (BytesRemoved)
    *BytesRemoved = RemovedSize;

  return Count;
}

unsigned SIInstrInfo::insertBranch(MachineBasicBlock &MBB,
                                   MachineBasicBlock *TBB,
                                   MachineBasicBlock *FBB,
                                   ArrayRef<MachineOperand> Cond,
                                   const DebugLoc &DL,
                                   int *BytesAdded) const {

  if (!FBB && Cond.empty()) {
    BuildMI(&MBB, DL, get(AMDGPU::S_BRANCH))
      .addMBB(TBB);
    if (BytesAdded)
      *BytesAdded = 4;
    return 1;
  }

  assert(TBB && Cond[0].isImm());

  unsigned Opcode
    = getBranchOpcode(static_cast<BranchPredicate>(Cond[0].getImm()));

  if (!FBB) {
    BuildMI(&MBB, DL, get(Opcode))
      .addMBB(TBB);

    if (BytesAdded)
      *BytesAdded = 4;
    return 1;
  }

  assert(TBB && FBB);

  BuildMI(&MBB, DL, get(Opcode))
    .addMBB(TBB);
  BuildMI(&MBB, DL, get(AMDGPU::S_BRANCH))
    .addMBB(FBB);

  if (BytesAdded)
      *BytesAdded = 8;

  return 2;
}

bool SIInstrInfo::ReverseBranchCondition(
  SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 1);
  Cond[0].setImm(-Cond[0].getImm());
  return false;
}

static void removeModOperands(MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  int Src0ModIdx = AMDGPU::getNamedOperandIdx(Opc,
                                              AMDGPU::OpName::src0_modifiers);
  int Src1ModIdx = AMDGPU::getNamedOperandIdx(Opc,
                                              AMDGPU::OpName::src1_modifiers);
  int Src2ModIdx = AMDGPU::getNamedOperandIdx(Opc,
                                              AMDGPU::OpName::src2_modifiers);

  MI.RemoveOperand(Src2ModIdx);
  MI.RemoveOperand(Src1ModIdx);
  MI.RemoveOperand(Src0ModIdx);
}

bool SIInstrInfo::FoldImmediate(MachineInstr &UseMI, MachineInstr &DefMI,
                                unsigned Reg, MachineRegisterInfo *MRI) const {
  if (!MRI->hasOneNonDBGUse(Reg))
    return false;

  unsigned Opc = UseMI.getOpcode();
  if (Opc == AMDGPU::COPY) {
    bool isVGPRCopy = RI.isVGPR(*MRI, UseMI.getOperand(0).getReg());
    switch (DefMI.getOpcode()) {
    default:
      return false;
    case AMDGPU::S_MOV_B64:
      // TODO: We could fold 64-bit immediates, but this get compilicated
      // when there are sub-registers.
      return false;

    case AMDGPU::V_MOV_B32_e32:
    case AMDGPU::S_MOV_B32:
      break;
    }
    unsigned NewOpc = isVGPRCopy ? AMDGPU::V_MOV_B32_e32 : AMDGPU::S_MOV_B32;
    const MachineOperand *ImmOp = getNamedOperand(DefMI, AMDGPU::OpName::src0);
    assert(ImmOp);
    // FIXME: We could handle FrameIndex values here.
    if (!ImmOp->isImm()) {
      return false;
    }
    UseMI.setDesc(get(NewOpc));
    UseMI.getOperand(1).ChangeToImmediate(ImmOp->getImm());
    UseMI.addImplicitDefUseOperands(*UseMI.getParent()->getParent());
    return true;
  }

  if (Opc == AMDGPU::V_MAD_F32 || Opc == AMDGPU::V_MAC_F32_e64) {
    // Don't fold if we are using source modifiers. The new VOP2 instructions
    // don't have them.
    if (hasModifiersSet(UseMI, AMDGPU::OpName::src0_modifiers) ||
        hasModifiersSet(UseMI, AMDGPU::OpName::src1_modifiers) ||
        hasModifiersSet(UseMI, AMDGPU::OpName::src2_modifiers)) {
      return false;
    }

    const MachineOperand &ImmOp = DefMI.getOperand(1);

    // If this is a free constant, there's no reason to do this.
    // TODO: We could fold this here instead of letting SIFoldOperands do it
    // later.
    if (isInlineConstant(ImmOp, 4))
      return false;

    MachineOperand *Src0 = getNamedOperand(UseMI, AMDGPU::OpName::src0);
    MachineOperand *Src1 = getNamedOperand(UseMI, AMDGPU::OpName::src1);
    MachineOperand *Src2 = getNamedOperand(UseMI, AMDGPU::OpName::src2);

    // Multiplied part is the constant: Use v_madmk_f32
    // We should only expect these to be on src0 due to canonicalizations.
    if (Src0->isReg() && Src0->getReg() == Reg) {
      if (!Src1->isReg() || RI.isSGPRClass(MRI->getRegClass(Src1->getReg())))
        return false;

      if (!Src2->isReg() || RI.isSGPRClass(MRI->getRegClass(Src2->getReg())))
        return false;

      // We need to swap operands 0 and 1 since madmk constant is at operand 1.

      const int64_t Imm = DefMI.getOperand(1).getImm();

      // FIXME: This would be a lot easier if we could return a new instruction
      // instead of having to modify in place.

      // Remove these first since they are at the end.
      UseMI.RemoveOperand(
          AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::omod));
      UseMI.RemoveOperand(
          AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::clamp));

      unsigned Src1Reg = Src1->getReg();
      unsigned Src1SubReg = Src1->getSubReg();
      Src0->setReg(Src1Reg);
      Src0->setSubReg(Src1SubReg);
      Src0->setIsKill(Src1->isKill());

      if (Opc == AMDGPU::V_MAC_F32_e64) {
        UseMI.untieRegOperand(
            AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src2));
      }

      Src1->ChangeToImmediate(Imm);

      removeModOperands(UseMI);
      UseMI.setDesc(get(AMDGPU::V_MADMK_F32));

      bool DeleteDef = MRI->hasOneNonDBGUse(Reg);
      if (DeleteDef)
        DefMI.eraseFromParent();

      return true;
    }

    // Added part is the constant: Use v_madak_f32
    if (Src2->isReg() && Src2->getReg() == Reg) {
      // Not allowed to use constant bus for another operand.
      // We can however allow an inline immediate as src0.
      if (!Src0->isImm() &&
          (Src0->isReg() && RI.isSGPRClass(MRI->getRegClass(Src0->getReg()))))
        return false;

      if (!Src1->isReg() || RI.isSGPRClass(MRI->getRegClass(Src1->getReg())))
        return false;

      const int64_t Imm = DefMI.getOperand(1).getImm();

      // FIXME: This would be a lot easier if we could return a new instruction
      // instead of having to modify in place.

      // Remove these first since they are at the end.
      UseMI.RemoveOperand(
          AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::omod));
      UseMI.RemoveOperand(
          AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::clamp));

      if (Opc == AMDGPU::V_MAC_F32_e64) {
        UseMI.untieRegOperand(
            AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src2));
      }

      // ChangingToImmediate adds Src2 back to the instruction.
      Src2->ChangeToImmediate(Imm);

      // These come before src2.
      removeModOperands(UseMI);
      UseMI.setDesc(get(AMDGPU::V_MADAK_F32));

      bool DeleteDef = MRI->hasOneNonDBGUse(Reg);
      if (DeleteDef)
        DefMI.eraseFromParent();

      return true;
    }
  }

  return false;
}

static bool offsetsDoNotOverlap(int WidthA, int OffsetA,
                                int WidthB, int OffsetB) {
  int LowOffset = OffsetA < OffsetB ? OffsetA : OffsetB;
  int HighOffset = OffsetA < OffsetB ? OffsetB : OffsetA;
  int LowWidth = (LowOffset == OffsetA) ? WidthA : WidthB;
  return LowOffset + LowWidth <= HighOffset;
}

bool SIInstrInfo::checkInstOffsetsDoNotOverlap(MachineInstr &MIa,
                                               MachineInstr &MIb) const {
  unsigned BaseReg0, BaseReg1;
  int64_t Offset0, Offset1;

  if (getMemOpBaseRegImmOfs(MIa, BaseReg0, Offset0, &RI) &&
      getMemOpBaseRegImmOfs(MIb, BaseReg1, Offset1, &RI)) {

    if (!MIa.hasOneMemOperand() || !MIb.hasOneMemOperand()) {
      // FIXME: Handle ds_read2 / ds_write2.
      return false;
    }
    unsigned Width0 = (*MIa.memoperands_begin())->getSize();
    unsigned Width1 = (*MIb.memoperands_begin())->getSize();
    if (BaseReg0 == BaseReg1 &&
        offsetsDoNotOverlap(Width0, Offset0, Width1, Offset1)) {
      return true;
    }
  }

  return false;
}

bool SIInstrInfo::areMemAccessesTriviallyDisjoint(MachineInstr &MIa,
                                                  MachineInstr &MIb,
                                                  AliasAnalysis *AA) const {
  assert((MIa.mayLoad() || MIa.mayStore()) &&
         "MIa must load from or modify a memory location");
  assert((MIb.mayLoad() || MIb.mayStore()) &&
         "MIb must load from or modify a memory location");

  if (MIa.hasUnmodeledSideEffects() || MIb.hasUnmodeledSideEffects())
    return false;

  // XXX - Can we relax this between address spaces?
  if (MIa.hasOrderedMemoryRef() || MIb.hasOrderedMemoryRef())
    return false;

  if (AA && MIa.hasOneMemOperand() && MIb.hasOneMemOperand()) {
    const MachineMemOperand *MMOa = *MIa.memoperands_begin();
    const MachineMemOperand *MMOb = *MIb.memoperands_begin();
    if (MMOa->getValue() && MMOb->getValue()) {
      MemoryLocation LocA(MMOa->getValue(), MMOa->getSize(), MMOa->getAAInfo());
      MemoryLocation LocB(MMOb->getValue(), MMOb->getSize(), MMOb->getAAInfo());
      if (!AA->alias(LocA, LocB))
        return true;
    }
  }

  // TODO: Should we check the address space from the MachineMemOperand? That
  // would allow us to distinguish objects we know don't alias based on the
  // underlying address space, even if it was lowered to a different one,
  // e.g. private accesses lowered to use MUBUF instructions on a scratch
  // buffer.
  if (isDS(MIa)) {
    if (isDS(MIb))
      return checkInstOffsetsDoNotOverlap(MIa, MIb);

    return !isFLAT(MIb);
  }

  if (isMUBUF(MIa) || isMTBUF(MIa)) {
    if (isMUBUF(MIb) || isMTBUF(MIb))
      return checkInstOffsetsDoNotOverlap(MIa, MIb);

    return !isFLAT(MIb) && !isSMRD(MIb);
  }

  if (isSMRD(MIa)) {
    if (isSMRD(MIb))
      return checkInstOffsetsDoNotOverlap(MIa, MIb);

    return !isFLAT(MIb) && !isMUBUF(MIa) && !isMTBUF(MIa);
  }

  if (isFLAT(MIa)) {
    if (isFLAT(MIb))
      return checkInstOffsetsDoNotOverlap(MIa, MIb);

    return false;
  }

  return false;
}

MachineInstr *SIInstrInfo::convertToThreeAddress(MachineFunction::iterator &MBB,
                                                 MachineInstr &MI,
                                                 LiveVariables *LV) const {

  switch (MI.getOpcode()) {
  default:
    return nullptr;
  case AMDGPU::V_MAC_F32_e64:
    break;
  case AMDGPU::V_MAC_F32_e32: {
    const MachineOperand *Src0 = getNamedOperand(MI, AMDGPU::OpName::src0);
    if (Src0->isImm() && !isInlineConstant(*Src0, 4))
      return nullptr;
    break;
  }
  }

  const MachineOperand *Dst = getNamedOperand(MI, AMDGPU::OpName::vdst);
  const MachineOperand *Src0 = getNamedOperand(MI, AMDGPU::OpName::src0);
  const MachineOperand *Src1 = getNamedOperand(MI, AMDGPU::OpName::src1);
  const MachineOperand *Src2 = getNamedOperand(MI, AMDGPU::OpName::src2);

  return BuildMI(*MBB, MI, MI.getDebugLoc(), get(AMDGPU::V_MAD_F32))
      .addOperand(*Dst)
      .addImm(0) // Src0 mods
      .addOperand(*Src0)
      .addImm(0) // Src1 mods
      .addOperand(*Src1)
      .addImm(0) // Src mods
      .addOperand(*Src2)
      .addImm(0)  // clamp
      .addImm(0); // omod
}

bool SIInstrInfo::isSchedulingBoundary(const MachineInstr &MI,
                                       const MachineBasicBlock *MBB,
                                       const MachineFunction &MF) const {
  // XXX - Do we want the SP check in the base implementation?

  // Target-independent instructions do not have an implicit-use of EXEC, even
  // when they operate on VGPRs. Treating EXEC modifications as scheduling
  // boundaries prevents incorrect movements of such instructions.
  return TargetInstrInfo::isSchedulingBoundary(MI, MBB, MF) ||
         MI.modifiesRegister(AMDGPU::EXEC, &RI);
}

bool SIInstrInfo::isInlineConstant(const APInt &Imm) const {
  int64_t SVal = Imm.getSExtValue();
  if (SVal >= -16 && SVal <= 64)
    return true;

  if (Imm.getBitWidth() == 64) {
    uint64_t Val = Imm.getZExtValue();
    return (DoubleToBits(0.0) == Val) ||
           (DoubleToBits(1.0) == Val) ||
           (DoubleToBits(-1.0) == Val) ||
           (DoubleToBits(0.5) == Val) ||
           (DoubleToBits(-0.5) == Val) ||
           (DoubleToBits(2.0) == Val) ||
           (DoubleToBits(-2.0) == Val) ||
           (DoubleToBits(4.0) == Val) ||
           (DoubleToBits(-4.0) == Val);
  }

  // The actual type of the operand does not seem to matter as long
  // as the bits match one of the inline immediate values.  For example:
  //
  // -nan has the hexadecimal encoding of 0xfffffffe which is -2 in decimal,
  // so it is a legal inline immediate.
  //
  // 1065353216 has the hexadecimal encoding 0x3f800000 which is 1.0f in
  // floating-point, so it is a legal inline immediate.
  uint32_t Val = Imm.getZExtValue();

  return (FloatToBits(0.0f) == Val) ||
         (FloatToBits(1.0f) == Val) ||
         (FloatToBits(-1.0f) == Val) ||
         (FloatToBits(0.5f) == Val) ||
         (FloatToBits(-0.5f) == Val) ||
         (FloatToBits(2.0f) == Val) ||
         (FloatToBits(-2.0f) == Val) ||
         (FloatToBits(4.0f) == Val) ||
         (FloatToBits(-4.0f) == Val);
}

bool SIInstrInfo::isInlineConstant(const MachineOperand &MO,
                                   unsigned OpSize) const {
  if (MO.isImm()) {
    // MachineOperand provides no way to tell the true operand size, since it
    // only records a 64-bit value. We need to know the size to determine if a
    // 32-bit floating point immediate bit pattern is legal for an integer
    // immediate. It would be for any 32-bit integer operand, but would not be
    // for a 64-bit one.

    unsigned BitSize = 8 * OpSize;
    return isInlineConstant(APInt(BitSize, MO.getImm(), true));
  }

  return false;
}

bool SIInstrInfo::isLiteralConstant(const MachineOperand &MO,
                                    unsigned OpSize) const {
  return MO.isImm() && !isInlineConstant(MO, OpSize);
}

bool SIInstrInfo::isLiteralConstantLike(const MachineOperand &MO,
                                        unsigned OpSize) const {
  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    return false;
  case MachineOperand::MO_Immediate:
    return !isInlineConstant(MO, OpSize);
  case MachineOperand::MO_FrameIndex:
  case MachineOperand::MO_MachineBasicBlock:
  case MachineOperand::MO_ExternalSymbol:
  case MachineOperand::MO_GlobalAddress:
  case MachineOperand::MO_MCSymbol:
    return true;
  default:
    llvm_unreachable("unexpected operand type");
  }
}

static bool compareMachineOp(const MachineOperand &Op0,
                             const MachineOperand &Op1) {
  if (Op0.getType() != Op1.getType())
    return false;

  switch (Op0.getType()) {
  case MachineOperand::MO_Register:
    return Op0.getReg() == Op1.getReg();
  case MachineOperand::MO_Immediate:
    return Op0.getImm() == Op1.getImm();
  default:
    llvm_unreachable("Didn't expect to be comparing these operand types");
  }
}

bool SIInstrInfo::isImmOperandLegal(const MachineInstr &MI, unsigned OpNo,
                                    const MachineOperand &MO) const {
  const MCOperandInfo &OpInfo = get(MI.getOpcode()).OpInfo[OpNo];

  assert(MO.isImm() || MO.isTargetIndex() || MO.isFI());

  if (OpInfo.OperandType == MCOI::OPERAND_IMMEDIATE)
    return true;

  if (OpInfo.RegClass < 0)
    return false;

  unsigned OpSize = RI.getRegClass(OpInfo.RegClass)->getSize();
  if (isLiteralConstant(MO, OpSize))
    return RI.opCanUseLiteralConstant(OpInfo.OperandType);

  return RI.opCanUseInlineConstant(OpInfo.OperandType);
}

bool SIInstrInfo::hasVALU32BitEncoding(unsigned Opcode) const {
  int Op32 = AMDGPU::getVOPe32(Opcode);
  if (Op32 == -1)
    return false;

  return pseudoToMCOpcode(Op32) != -1;
}

bool SIInstrInfo::hasModifiers(unsigned Opcode) const {
  // The src0_modifier operand is present on all instructions
  // that have modifiers.

  return AMDGPU::getNamedOperandIdx(Opcode,
                                    AMDGPU::OpName::src0_modifiers) != -1;
}

bool SIInstrInfo::hasModifiersSet(const MachineInstr &MI,
                                  unsigned OpName) const {
  const MachineOperand *Mods = getNamedOperand(MI, OpName);
  return Mods && Mods->getImm();
}

bool SIInstrInfo::usesConstantBus(const MachineRegisterInfo &MRI,
                                  const MachineOperand &MO,
                                  unsigned OpSize) const {
  // Literal constants use the constant bus.
  if (isLiteralConstant(MO, OpSize))
    return true;

  if (!MO.isReg() || !MO.isUse())
    return false;

  if (TargetRegisterInfo::isVirtualRegister(MO.getReg()))
    return RI.isSGPRClass(MRI.getRegClass(MO.getReg()));

  // FLAT_SCR is just an SGPR pair.
  if (!MO.isImplicit() && (MO.getReg() == AMDGPU::FLAT_SCR))
    return true;

  // EXEC register uses the constant bus.
  if (!MO.isImplicit() && MO.getReg() == AMDGPU::EXEC)
    return true;

  // SGPRs use the constant bus
  return (MO.getReg() == AMDGPU::VCC || MO.getReg() == AMDGPU::M0 ||
          (!MO.isImplicit() &&
           (AMDGPU::SGPR_32RegClass.contains(MO.getReg()) ||
            AMDGPU::SGPR_64RegClass.contains(MO.getReg()))));
}

static unsigned findImplicitSGPRRead(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.implicit_operands()) {
    // We only care about reads.
    if (MO.isDef())
      continue;

    switch (MO.getReg()) {
    case AMDGPU::VCC:
    case AMDGPU::M0:
    case AMDGPU::FLAT_SCR:
      return MO.getReg();

    default:
      break;
    }
  }

  return AMDGPU::NoRegister;
}

static bool shouldReadExec(const MachineInstr &MI) {
  if (SIInstrInfo::isVALU(MI)) {
    switch (MI.getOpcode()) {
    case AMDGPU::V_READLANE_B32:
    case AMDGPU::V_READLANE_B32_si:
    case AMDGPU::V_READLANE_B32_vi:
    case AMDGPU::V_WRITELANE_B32:
    case AMDGPU::V_WRITELANE_B32_si:
    case AMDGPU::V_WRITELANE_B32_vi:
      return false;
    }

    return true;
  }

  if (SIInstrInfo::isGenericOpcode(MI.getOpcode()) ||
      SIInstrInfo::isSALU(MI) ||
      SIInstrInfo::isSMRD(MI))
    return false;

  return true;
}

static bool isSubRegOf(const SIRegisterInfo &TRI,
                       const MachineOperand &SuperVec,
                       const MachineOperand &SubReg) {
  if (TargetRegisterInfo::isPhysicalRegister(SubReg.getReg()))
    return TRI.isSubRegister(SuperVec.getReg(), SubReg.getReg());

  return SubReg.getSubReg() != AMDGPU::NoSubRegister &&
         SubReg.getReg() == SuperVec.getReg();
}

bool SIInstrInfo::verifyInstruction(const MachineInstr &MI,
                                    StringRef &ErrInfo) const {
  uint16_t Opcode = MI.getOpcode();
  const MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
  int Src0Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src0);
  int Src1Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src1);
  int Src2Idx = AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src2);

  // Make sure the number of operands is correct.
  const MCInstrDesc &Desc = get(Opcode);
  if (!Desc.isVariadic() &&
      Desc.getNumOperands() != MI.getNumExplicitOperands()) {
    ErrInfo = "Instruction has wrong number of operands.";
    return false;
  }

  // Make sure the register classes are correct.
  for (int i = 0, e = Desc.getNumOperands(); i != e; ++i) {
    if (MI.getOperand(i).isFPImm()) {
      ErrInfo = "FPImm Machine Operands are not supported. ISel should bitcast "
                "all fp values to integers.";
      return false;
    }

    int RegClass = Desc.OpInfo[i].RegClass;

    switch (Desc.OpInfo[i].OperandType) {
    case MCOI::OPERAND_REGISTER:
      if (MI.getOperand(i).isImm()) {
        ErrInfo = "Illegal immediate value for operand.";
        return false;
      }
      break;
    case AMDGPU::OPERAND_REG_IMM32_INT:
    case AMDGPU::OPERAND_REG_IMM32_FP:
      break;
    case AMDGPU::OPERAND_REG_INLINE_C_INT:
    case AMDGPU::OPERAND_REG_INLINE_C_FP:
      if (isLiteralConstant(MI.getOperand(i),
                            RI.getRegClass(RegClass)->getSize())) {
        ErrInfo = "Illegal immediate value for operand.";
        return false;
      }
      break;
    case MCOI::OPERAND_IMMEDIATE:
    case AMDGPU::OPERAND_KIMM32:
      // Check if this operand is an immediate.
      // FrameIndex operands will be replaced by immediates, so they are
      // allowed.
      if (!MI.getOperand(i).isImm() && !MI.getOperand(i).isFI()) {
        ErrInfo = "Expected immediate, but got non-immediate";
        return false;
      }
      LLVM_FALLTHROUGH;
    default:
      continue;
    }

    if (!MI.getOperand(i).isReg())
      continue;

    if (RegClass != -1) {
      unsigned Reg = MI.getOperand(i).getReg();
      if (Reg == AMDGPU::NoRegister ||
          TargetRegisterInfo::isVirtualRegister(Reg))
        continue;

      const TargetRegisterClass *RC = RI.getRegClass(RegClass);
      if (!RC->contains(Reg)) {
        ErrInfo = "Operand has incorrect register class.";
        return false;
      }
    }
  }

  // Verify VOP*
  if (isVOP1(MI) || isVOP2(MI) || isVOP3(MI) || isVOPC(MI)) {
    // Only look at the true operands. Only a real operand can use the constant
    // bus, and we don't want to check pseudo-operands like the source modifier
    // flags.
    const int OpIndices[] = { Src0Idx, Src1Idx, Src2Idx };

    unsigned ConstantBusCount = 0;

    if (AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::imm) != -1)
      ++ConstantBusCount;

    unsigned SGPRUsed = findImplicitSGPRRead(MI);
    if (SGPRUsed != AMDGPU::NoRegister)
      ++ConstantBusCount;

    for (int OpIdx : OpIndices) {
      if (OpIdx == -1)
        break;
      const MachineOperand &MO = MI.getOperand(OpIdx);
      if (usesConstantBus(MRI, MO, getOpSize(Opcode, OpIdx))) {
        if (MO.isReg()) {
          if (MO.getReg() != SGPRUsed)
            ++ConstantBusCount;
          SGPRUsed = MO.getReg();
        } else {
          ++ConstantBusCount;
        }
      }
    }
    if (ConstantBusCount > 1) {
      ErrInfo = "VOP* instruction uses the constant bus more than once";
      return false;
    }
  }

  // Verify misc. restrictions on specific instructions.
  if (Desc.getOpcode() == AMDGPU::V_DIV_SCALE_F32 ||
      Desc.getOpcode() == AMDGPU::V_DIV_SCALE_F64) {
    const MachineOperand &Src0 = MI.getOperand(Src0Idx);
    const MachineOperand &Src1 = MI.getOperand(Src1Idx);
    const MachineOperand &Src2 = MI.getOperand(Src2Idx);
    if (Src0.isReg() && Src1.isReg() && Src2.isReg()) {
      if (!compareMachineOp(Src0, Src1) &&
          !compareMachineOp(Src0, Src2)) {
        ErrInfo = "v_div_scale_{f32|f64} require src0 = src1 or src2";
        return false;
      }
    }
  }

  if (Desc.getOpcode() == AMDGPU::V_MOVRELS_B32_e32 ||
      Desc.getOpcode() == AMDGPU::V_MOVRELS_B32_e64 ||
      Desc.getOpcode() == AMDGPU::V_MOVRELD_B32_e32 ||
      Desc.getOpcode() == AMDGPU::V_MOVRELD_B32_e64) {
    const bool IsDst = Desc.getOpcode() == AMDGPU::V_MOVRELD_B32_e32 ||
                       Desc.getOpcode() == AMDGPU::V_MOVRELD_B32_e64;

    const unsigned StaticNumOps = Desc.getNumOperands() +
      Desc.getNumImplicitUses();
    const unsigned NumImplicitOps = IsDst ? 2 : 1;

    if (MI.getNumOperands() != StaticNumOps + NumImplicitOps) {
      ErrInfo = "missing implicit register operands";
      return false;
    }

    const MachineOperand *Dst = getNamedOperand(MI, AMDGPU::OpName::vdst);
    if (IsDst) {
      if (!Dst->isUse()) {
        ErrInfo = "v_movreld_b32 vdst should be a use operand";
        return false;
      }

      unsigned UseOpIdx;
      if (!MI.isRegTiedToUseOperand(StaticNumOps, &UseOpIdx) ||
          UseOpIdx != StaticNumOps + 1) {
        ErrInfo = "movrel implicit operands should be tied";
        return false;
      }
    }

    const MachineOperand &Src0 = MI.getOperand(Src0Idx);
    const MachineOperand &ImpUse
      = MI.getOperand(StaticNumOps + NumImplicitOps - 1);
    if (!ImpUse.isReg() || !ImpUse.isUse() ||
        !isSubRegOf(RI, ImpUse, IsDst ? *Dst : Src0)) {
      ErrInfo = "src0 should be subreg of implicit vector use";
      return false;
    }
  }

  // Make sure we aren't losing exec uses in the td files. This mostly requires
  // being careful when using let Uses to try to add other use registers.
  if (shouldReadExec(MI)) {
    if (!MI.hasRegisterImplicitUseOperand(AMDGPU::EXEC)) {
      ErrInfo = "VALU instruction does not implicitly read exec mask";
      return false;
    }
  }

  return true;
}

unsigned SIInstrInfo::getVALUOp(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default: return AMDGPU::INSTRUCTION_LIST_END;
  case AMDGPU::REG_SEQUENCE: return AMDGPU::REG_SEQUENCE;
  case AMDGPU::COPY: return AMDGPU::COPY;
  case AMDGPU::PHI: return AMDGPU::PHI;
  case AMDGPU::INSERT_SUBREG: return AMDGPU::INSERT_SUBREG;
  case AMDGPU::S_MOV_B32:
    return MI.getOperand(1).isReg() ?
           AMDGPU::COPY : AMDGPU::V_MOV_B32_e32;
  case AMDGPU::S_ADD_I32:
  case AMDGPU::S_ADD_U32: return AMDGPU::V_ADD_I32_e32;
  case AMDGPU::S_ADDC_U32: return AMDGPU::V_ADDC_U32_e32;
  case AMDGPU::S_SUB_I32:
  case AMDGPU::S_SUB_U32: return AMDGPU::V_SUB_I32_e32;
  case AMDGPU::S_SUBB_U32: return AMDGPU::V_SUBB_U32_e32;
  case AMDGPU::S_MUL_I32: return AMDGPU::V_MUL_LO_I32;
  case AMDGPU::S_AND_B32: return AMDGPU::V_AND_B32_e64;
  case AMDGPU::S_OR_B32: return AMDGPU::V_OR_B32_e64;
  case AMDGPU::S_XOR_B32: return AMDGPU::V_XOR_B32_e64;
  case AMDGPU::S_MIN_I32: return AMDGPU::V_MIN_I32_e64;
  case AMDGPU::S_MIN_U32: return AMDGPU::V_MIN_U32_e64;
  case AMDGPU::S_MAX_I32: return AMDGPU::V_MAX_I32_e64;
  case AMDGPU::S_MAX_U32: return AMDGPU::V_MAX_U32_e64;
  case AMDGPU::S_ASHR_I32: return AMDGPU::V_ASHR_I32_e32;
  case AMDGPU::S_ASHR_I64: return AMDGPU::V_ASHR_I64;
  case AMDGPU::S_LSHL_B32: return AMDGPU::V_LSHL_B32_e32;
  case AMDGPU::S_LSHL_B64: return AMDGPU::V_LSHL_B64;
  case AMDGPU::S_LSHR_B32: return AMDGPU::V_LSHR_B32_e32;
  case AMDGPU::S_LSHR_B64: return AMDGPU::V_LSHR_B64;
  case AMDGPU::S_SEXT_I32_I8: return AMDGPU::V_BFE_I32;
  case AMDGPU::S_SEXT_I32_I16: return AMDGPU::V_BFE_I32;
  case AMDGPU::S_BFE_U32: return AMDGPU::V_BFE_U32;
  case AMDGPU::S_BFE_I32: return AMDGPU::V_BFE_I32;
  case AMDGPU::S_BFM_B32: return AMDGPU::V_BFM_B32_e64;
  case AMDGPU::S_BREV_B32: return AMDGPU::V_BFREV_B32_e32;
  case AMDGPU::S_NOT_B32: return AMDGPU::V_NOT_B32_e32;
  case AMDGPU::S_NOT_B64: return AMDGPU::V_NOT_B32_e32;
  case AMDGPU::S_CMP_EQ_I32: return AMDGPU::V_CMP_EQ_I32_e32;
  case AMDGPU::S_CMP_LG_I32: return AMDGPU::V_CMP_NE_I32_e32;
  case AMDGPU::S_CMP_GT_I32: return AMDGPU::V_CMP_GT_I32_e32;
  case AMDGPU::S_CMP_GE_I32: return AMDGPU::V_CMP_GE_I32_e32;
  case AMDGPU::S_CMP_LT_I32: return AMDGPU::V_CMP_LT_I32_e32;
  case AMDGPU::S_CMP_LE_I32: return AMDGPU::V_CMP_LE_I32_e32;
  case AMDGPU::S_CMP_EQ_U32: return AMDGPU::V_CMP_EQ_U32_e32;
  case AMDGPU::S_CMP_LG_U32: return AMDGPU::V_CMP_NE_U32_e32;
  case AMDGPU::S_CMP_GT_U32: return AMDGPU::V_CMP_GT_U32_e32;
  case AMDGPU::S_CMP_GE_U32: return AMDGPU::V_CMP_GE_U32_e32;
  case AMDGPU::S_CMP_LT_U32: return AMDGPU::V_CMP_LT_U32_e32;
  case AMDGPU::S_CMP_LE_U32: return AMDGPU::V_CMP_LE_U32_e32;
  case AMDGPU::S_BCNT1_I32_B32: return AMDGPU::V_BCNT_U32_B32_e64;
  case AMDGPU::S_FF1_I32_B32: return AMDGPU::V_FFBL_B32_e32;
  case AMDGPU::S_FLBIT_I32_B32: return AMDGPU::V_FFBH_U32_e32;
  case AMDGPU::S_FLBIT_I32: return AMDGPU::V_FFBH_I32_e64;
  case AMDGPU::S_CBRANCH_SCC0: return AMDGPU::S_CBRANCH_VCCZ;
  case AMDGPU::S_CBRANCH_SCC1: return AMDGPU::S_CBRANCH_VCCNZ;
  }
}

bool SIInstrInfo::isSALUOpSupportedOnVALU(const MachineInstr &MI) const {
  return getVALUOp(MI) != AMDGPU::INSTRUCTION_LIST_END;
}

const TargetRegisterClass *SIInstrInfo::getOpRegClass(const MachineInstr &MI,
                                                      unsigned OpNo) const {
  const MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
  const MCInstrDesc &Desc = get(MI.getOpcode());
  if (MI.isVariadic() || OpNo >= Desc.getNumOperands() ||
      Desc.OpInfo[OpNo].RegClass == -1) {
    unsigned Reg = MI.getOperand(OpNo).getReg();

    if (TargetRegisterInfo::isVirtualRegister(Reg))
      return MRI.getRegClass(Reg);
    return RI.getPhysRegClass(Reg);
  }

  unsigned RCID = Desc.OpInfo[OpNo].RegClass;
  return RI.getRegClass(RCID);
}

bool SIInstrInfo::canReadVGPR(const MachineInstr &MI, unsigned OpNo) const {
  switch (MI.getOpcode()) {
  case AMDGPU::COPY:
  case AMDGPU::REG_SEQUENCE:
  case AMDGPU::PHI:
  case AMDGPU::INSERT_SUBREG:
    return RI.hasVGPRs(getOpRegClass(MI, 0));
  default:
    return RI.hasVGPRs(getOpRegClass(MI, OpNo));
  }
}

void SIInstrInfo::legalizeOpWithMove(MachineInstr &MI, unsigned OpIdx) const {
  MachineBasicBlock::iterator I = MI;
  MachineBasicBlock *MBB = MI.getParent();
  MachineOperand &MO = MI.getOperand(OpIdx);
  MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();
  unsigned RCID = get(MI.getOpcode()).OpInfo[OpIdx].RegClass;
  const TargetRegisterClass *RC = RI.getRegClass(RCID);
  unsigned Opcode = AMDGPU::V_MOV_B32_e32;
  if (MO.isReg())
    Opcode = AMDGPU::COPY;
  else if (RI.isSGPRClass(RC))
    Opcode = AMDGPU::S_MOV_B32;

  const TargetRegisterClass *VRC = RI.getEquivalentVGPRClass(RC);
  if (RI.getCommonSubClass(&AMDGPU::VReg_64RegClass, VRC))
    VRC = &AMDGPU::VReg_64RegClass;
  else
    VRC = &AMDGPU::VGPR_32RegClass;

  unsigned Reg = MRI.createVirtualRegister(VRC);
  DebugLoc DL = MBB->findDebugLoc(I);
  BuildMI(*MI.getParent(), I, DL, get(Opcode), Reg).addOperand(MO);
  MO.ChangeToRegister(Reg, false);
}

unsigned SIInstrInfo::buildExtractSubReg(MachineBasicBlock::iterator MI,
                                         MachineRegisterInfo &MRI,
                                         MachineOperand &SuperReg,
                                         const TargetRegisterClass *SuperRC,
                                         unsigned SubIdx,
                                         const TargetRegisterClass *SubRC)
                                         const {
  MachineBasicBlock *MBB = MI->getParent();
  DebugLoc DL = MI->getDebugLoc();
  unsigned SubReg = MRI.createVirtualRegister(SubRC);

  if (SuperReg.getSubReg() == AMDGPU::NoSubRegister) {
    BuildMI(*MBB, MI, DL, get(TargetOpcode::COPY), SubReg)
      .addReg(SuperReg.getReg(), 0, SubIdx);
    return SubReg;
  }

  // Just in case the super register is itself a sub-register, copy it to a new
  // value so we don't need to worry about merging its subreg index with the
  // SubIdx passed to this function. The register coalescer should be able to
  // eliminate this extra copy.
  unsigned NewSuperReg = MRI.createVirtualRegister(SuperRC);

  BuildMI(*MBB, MI, DL, get(TargetOpcode::COPY), NewSuperReg)
    .addReg(SuperReg.getReg(), 0, SuperReg.getSubReg());

  BuildMI(*MBB, MI, DL, get(TargetOpcode::COPY), SubReg)
    .addReg(NewSuperReg, 0, SubIdx);

  return SubReg;
}

MachineOperand SIInstrInfo::buildExtractSubRegOrImm(
  MachineBasicBlock::iterator MII,
  MachineRegisterInfo &MRI,
  MachineOperand &Op,
  const TargetRegisterClass *SuperRC,
  unsigned SubIdx,
  const TargetRegisterClass *SubRC) const {
  if (Op.isImm()) {
    if (SubIdx == AMDGPU::sub0)
      return MachineOperand::CreateImm(static_cast<int32_t>(Op.getImm()));
    if (SubIdx == AMDGPU::sub1)
      return MachineOperand::CreateImm(static_cast<int32_t>(Op.getImm() >> 32));

    llvm_unreachable("Unhandled register index for immediate");
  }

  unsigned SubReg = buildExtractSubReg(MII, MRI, Op, SuperRC,
                                       SubIdx, SubRC);
  return MachineOperand::CreateReg(SubReg, false);
}

// Change the order of operands from (0, 1, 2) to (0, 2, 1)
void SIInstrInfo::swapOperands(MachineInstr &Inst) const {
  assert(Inst.getNumExplicitOperands() == 3);
  MachineOperand Op1 = Inst.getOperand(1);
  Inst.RemoveOperand(1);
  Inst.addOperand(Op1);
}

bool SIInstrInfo::isLegalRegOperand(const MachineRegisterInfo &MRI,
                                    const MCOperandInfo &OpInfo,
                                    const MachineOperand &MO) const {
  if (!MO.isReg())
    return false;

  unsigned Reg = MO.getReg();
  const TargetRegisterClass *RC =
    TargetRegisterInfo::isVirtualRegister(Reg) ?
    MRI.getRegClass(Reg) :
    RI.getPhysRegClass(Reg);

  const SIRegisterInfo *TRI =
      static_cast<const SIRegisterInfo*>(MRI.getTargetRegisterInfo());
  RC = TRI->getSubRegClass(RC, MO.getSubReg());

  // In order to be legal, the common sub-class must be equal to the
  // class of the current operand.  For example:
  //
  // v_mov_b32 s0 ; Operand defined as vsrc_b32
  //              ; RI.getCommonSubClass(s0,vsrc_b32) = sgpr ; LEGAL
  //
  // s_sendmsg 0, s0 ; Operand defined as m0reg
  //                 ; RI.getCommonSubClass(s0,m0reg) = m0reg ; NOT LEGAL

  return RI.getCommonSubClass(RC, RI.getRegClass(OpInfo.RegClass)) == RC;
}

bool SIInstrInfo::isLegalVSrcOperand(const MachineRegisterInfo &MRI,
                                     const MCOperandInfo &OpInfo,
                                     const MachineOperand &MO) const {
  if (MO.isReg())
    return isLegalRegOperand(MRI, OpInfo, MO);

  // Handle non-register types that are treated like immediates.
  assert(MO.isImm() || MO.isTargetIndex() || MO.isFI());
  return true;
}

bool SIInstrInfo::isOperandLegal(const MachineInstr &MI, unsigned OpIdx,
                                 const MachineOperand *MO) const {
  const MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
  const MCInstrDesc &InstDesc = MI.getDesc();
  const MCOperandInfo &OpInfo = InstDesc.OpInfo[OpIdx];
  const TargetRegisterClass *DefinedRC =
      OpInfo.RegClass != -1 ? RI.getRegClass(OpInfo.RegClass) : nullptr;
  if (!MO)
    MO = &MI.getOperand(OpIdx);

  if (isVALU(MI) && usesConstantBus(MRI, *MO, DefinedRC->getSize())) {

    RegSubRegPair SGPRUsed;
    if (MO->isReg())
      SGPRUsed = RegSubRegPair(MO->getReg(), MO->getSubReg());

    for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
      if (i == OpIdx)
        continue;
      const MachineOperand &Op = MI.getOperand(i);
      if (Op.isReg()) {
        if ((Op.getReg() != SGPRUsed.Reg || Op.getSubReg() != SGPRUsed.SubReg) &&
            usesConstantBus(MRI, Op, getOpSize(MI, i))) {
          return false;
        }
      } else if (InstDesc.OpInfo[i].OperandType == AMDGPU::OPERAND_KIMM32) {
        return false;
      }
    }
  }

  if (MO->isReg()) {
    assert(DefinedRC);
    return isLegalRegOperand(MRI, OpInfo, *MO);
  }

  // Handle non-register types that are treated like immediates.
  assert(MO->isImm() || MO->isTargetIndex() || MO->isFI());

  if (!DefinedRC) {
    // This operand expects an immediate.
    return true;
  }

  return isImmOperandLegal(MI, OpIdx, *MO);
}

void SIInstrInfo::legalizeOperandsVOP2(MachineRegisterInfo &MRI,
                                       MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();
  const MCInstrDesc &InstrDesc = get(Opc);

  int Src1Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1);
  MachineOperand &Src1 = MI.getOperand(Src1Idx);

  // If there is an implicit SGPR use such as VCC use for v_addc_u32/v_subb_u32
  // we need to only have one constant bus use.
  //
  // Note we do not need to worry about literal constants here. They are
  // disabled for the operand type for instructions because they will always
  // violate the one constant bus use rule.
  bool HasImplicitSGPR = findImplicitSGPRRead(MI) != AMDGPU::NoRegister;
  if (HasImplicitSGPR) {
    int Src0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0);
    MachineOperand &Src0 = MI.getOperand(Src0Idx);

    if (Src0.isReg() && RI.isSGPRReg(MRI, Src0.getReg()))
      legalizeOpWithMove(MI, Src0Idx);
  }

  // VOP2 src0 instructions support all operand types, so we don't need to check
  // their legality. If src1 is already legal, we don't need to do anything.
  if (isLegalRegOperand(MRI, InstrDesc.OpInfo[Src1Idx], Src1))
    return;

  // We do not use commuteInstruction here because it is too aggressive and will
  // commute if it is possible. We only want to commute here if it improves
  // legality. This can be called a fairly large number of times so don't waste
  // compile time pointlessly swapping and checking legality again.
  if (HasImplicitSGPR || !MI.isCommutable()) {
    legalizeOpWithMove(MI, Src1Idx);
    return;
  }

  int Src0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0);
  MachineOperand &Src0 = MI.getOperand(Src0Idx);

  // If src0 can be used as src1, commuting will make the operands legal.
  // Otherwise we have to give up and insert a move.
  //
  // TODO: Other immediate-like operand kinds could be commuted if there was a
  // MachineOperand::ChangeTo* for them.
  if ((!Src1.isImm() && !Src1.isReg()) ||
      !isLegalRegOperand(MRI, InstrDesc.OpInfo[Src1Idx], Src0)) {
    legalizeOpWithMove(MI, Src1Idx);
    return;
  }

  int CommutedOpc = commuteOpcode(MI);
  if (CommutedOpc == -1) {
    legalizeOpWithMove(MI, Src1Idx);
    return;
  }

  MI.setDesc(get(CommutedOpc));

  unsigned Src0Reg = Src0.getReg();
  unsigned Src0SubReg = Src0.getSubReg();
  bool Src0Kill = Src0.isKill();

  if (Src1.isImm())
    Src0.ChangeToImmediate(Src1.getImm());
  else if (Src1.isReg()) {
    Src0.ChangeToRegister(Src1.getReg(), false, false, Src1.isKill());
    Src0.setSubReg(Src1.getSubReg());
  } else
    llvm_unreachable("Should only have register or immediate operands");

  Src1.ChangeToRegister(Src0Reg, false, false, Src0Kill);
  Src1.setSubReg(Src0SubReg);
}

// Legalize VOP3 operands. Because all operand types are supported for any
// operand, and since literal constants are not allowed and should never be
// seen, we only need to worry about inserting copies if we use multiple SGPR
// operands.
void SIInstrInfo::legalizeOperandsVOP3(MachineRegisterInfo &MRI,
                                       MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();

  int VOP3Idx[3] = {
    AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0),
    AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1),
    AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src2)
  };

  // Find the one SGPR operand we are allowed to use.
  unsigned SGPRReg = findUsedSGPR(MI, VOP3Idx);

  for (unsigned i = 0; i < 3; ++i) {
    int Idx = VOP3Idx[i];
    if (Idx == -1)
      break;
    MachineOperand &MO = MI.getOperand(Idx);

    // We should never see a VOP3 instruction with an illegal immediate operand.
    if (!MO.isReg())
      continue;

    if (!RI.isSGPRClass(MRI.getRegClass(MO.getReg())))
      continue; // VGPRs are legal

    if (SGPRReg == AMDGPU::NoRegister || SGPRReg == MO.getReg()) {
      SGPRReg = MO.getReg();
      // We can use one SGPR in each VOP3 instruction.
      continue;
    }

    // If we make it this far, then the operand is not legal and we must
    // legalize it.
    legalizeOpWithMove(MI, Idx);
  }
}

unsigned SIInstrInfo::readlaneVGPRToSGPR(unsigned SrcReg, MachineInstr &UseMI,
                                         MachineRegisterInfo &MRI) const {
  const TargetRegisterClass *VRC = MRI.getRegClass(SrcReg);
  const TargetRegisterClass *SRC = RI.getEquivalentSGPRClass(VRC);
  unsigned DstReg = MRI.createVirtualRegister(SRC);
  unsigned SubRegs = VRC->getSize() / 4;

  SmallVector<unsigned, 8> SRegs;
  for (unsigned i = 0; i < SubRegs; ++i) {
    unsigned SGPR = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
    BuildMI(*UseMI.getParent(), UseMI, UseMI.getDebugLoc(),
            get(AMDGPU::V_READFIRSTLANE_B32), SGPR)
        .addReg(SrcReg, 0, RI.getSubRegFromChannel(i));
    SRegs.push_back(SGPR);
  }

  MachineInstrBuilder MIB =
      BuildMI(*UseMI.getParent(), UseMI, UseMI.getDebugLoc(),
              get(AMDGPU::REG_SEQUENCE), DstReg);
  for (unsigned i = 0; i < SubRegs; ++i) {
    MIB.addReg(SRegs[i]);
    MIB.addImm(RI.getSubRegFromChannel(i));
  }
  return DstReg;
}

void SIInstrInfo::legalizeOperandsSMRD(MachineRegisterInfo &MRI,
                                       MachineInstr &MI) const {

  // If the pointer is store in VGPRs, then we need to move them to
  // SGPRs using v_readfirstlane.  This is safe because we only select
  // loads with uniform pointers to SMRD instruction so we know the
  // pointer value is uniform.
  MachineOperand *SBase = getNamedOperand(MI, AMDGPU::OpName::sbase);
  if (SBase && !RI.isSGPRClass(MRI.getRegClass(SBase->getReg()))) {
      unsigned SGPR = readlaneVGPRToSGPR(SBase->getReg(), MI, MRI);
      SBase->setReg(SGPR);
  }
}

void SIInstrInfo::legalizeOperands(MachineInstr &MI) const {
  MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();

  // Legalize VOP2
  if (isVOP2(MI) || isVOPC(MI)) {
    legalizeOperandsVOP2(MRI, MI);
    return;
  }

  // Legalize VOP3
  if (isVOP3(MI)) {
    legalizeOperandsVOP3(MRI, MI);
    return;
  }

  // Legalize SMRD
  if (isSMRD(MI)) {
    legalizeOperandsSMRD(MRI, MI);
    return;
  }

  // Legalize REG_SEQUENCE and PHI
  // The register class of the operands much be the same type as the register
  // class of the output.
  if (MI.getOpcode() == AMDGPU::PHI) {
    const TargetRegisterClass *RC = nullptr, *SRC = nullptr, *VRC = nullptr;
    for (unsigned i = 1, e = MI.getNumOperands(); i != e; i += 2) {
      if (!MI.getOperand(i).isReg() ||
          !TargetRegisterInfo::isVirtualRegister(MI.getOperand(i).getReg()))
        continue;
      const TargetRegisterClass *OpRC =
          MRI.getRegClass(MI.getOperand(i).getReg());
      if (RI.hasVGPRs(OpRC)) {
        VRC = OpRC;
      } else {
        SRC = OpRC;
      }
    }

    // If any of the operands are VGPR registers, then they all most be
    // otherwise we will create illegal VGPR->SGPR copies when legalizing
    // them.
    if (VRC || !RI.isSGPRClass(getOpRegClass(MI, 0))) {
      if (!VRC) {
        assert(SRC);
        VRC = RI.getEquivalentVGPRClass(SRC);
      }
      RC = VRC;
    } else {
      RC = SRC;
    }

    // Update all the operands so they have the same type.
    for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2) {
      MachineOperand &Op = MI.getOperand(I);
      if (!Op.isReg() || !TargetRegisterInfo::isVirtualRegister(Op.getReg()))
        continue;
      unsigned DstReg = MRI.createVirtualRegister(RC);

      // MI is a PHI instruction.
      MachineBasicBlock *InsertBB = MI.getOperand(I + 1).getMBB();
      MachineBasicBlock::iterator Insert = InsertBB->getFirstTerminator();

      BuildMI(*InsertBB, Insert, MI.getDebugLoc(), get(AMDGPU::COPY), DstReg)
          .addOperand(Op);
      Op.setReg(DstReg);
    }
  }

  // REG_SEQUENCE doesn't really require operand legalization, but if one has a
  // VGPR dest type and SGPR sources, insert copies so all operands are
  // VGPRs. This seems to help operand folding / the register coalescer.
  if (MI.getOpcode() == AMDGPU::REG_SEQUENCE) {
    MachineBasicBlock *MBB = MI.getParent();
    const TargetRegisterClass *DstRC = getOpRegClass(MI, 0);
    if (RI.hasVGPRs(DstRC)) {
      // Update all the operands so they are VGPR register classes. These may
      // not be the same register class because REG_SEQUENCE supports mixing
      // subregister index types e.g. sub0_sub1 + sub2 + sub3
      for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2) {
        MachineOperand &Op = MI.getOperand(I);
        if (!Op.isReg() || !TargetRegisterInfo::isVirtualRegister(Op.getReg()))
          continue;

        const TargetRegisterClass *OpRC = MRI.getRegClass(Op.getReg());
        const TargetRegisterClass *VRC = RI.getEquivalentVGPRClass(OpRC);
        if (VRC == OpRC)
          continue;

        unsigned DstReg = MRI.createVirtualRegister(VRC);

        BuildMI(*MBB, MI, MI.getDebugLoc(), get(AMDGPU::COPY), DstReg)
            .addOperand(Op);

        Op.setReg(DstReg);
        Op.setIsKill();
      }
    }

    return;
  }

  // Legalize INSERT_SUBREG
  // src0 must have the same register class as dst
  if (MI.getOpcode() == AMDGPU::INSERT_SUBREG) {
    unsigned Dst = MI.getOperand(0).getReg();
    unsigned Src0 = MI.getOperand(1).getReg();
    const TargetRegisterClass *DstRC = MRI.getRegClass(Dst);
    const TargetRegisterClass *Src0RC = MRI.getRegClass(Src0);
    if (DstRC != Src0RC) {
      MachineBasicBlock &MBB = *MI.getParent();
      unsigned NewSrc0 = MRI.createVirtualRegister(DstRC);
      BuildMI(MBB, MI, MI.getDebugLoc(), get(AMDGPU::COPY), NewSrc0)
          .addReg(Src0);
      MI.getOperand(1).setReg(NewSrc0);
    }
    return;
  }

  // Legalize MIMG
  if (isMIMG(MI)) {
    MachineOperand *SRsrc = getNamedOperand(MI, AMDGPU::OpName::srsrc);
    if (SRsrc && !RI.isSGPRClass(MRI.getRegClass(SRsrc->getReg()))) {
      unsigned SGPR = readlaneVGPRToSGPR(SRsrc->getReg(), MI, MRI);
      SRsrc->setReg(SGPR);
    }

    MachineOperand *SSamp = getNamedOperand(MI, AMDGPU::OpName::ssamp);
    if (SSamp && !RI.isSGPRClass(MRI.getRegClass(SSamp->getReg()))) {
      unsigned SGPR = readlaneVGPRToSGPR(SSamp->getReg(), MI, MRI);
      SSamp->setReg(SGPR);
    }
    return;
  }

  // Legalize MUBUF* instructions
  // FIXME: If we start using the non-addr64 instructions for compute, we
  // may need to legalize them here.
  int SRsrcIdx =
      AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::srsrc);
  if (SRsrcIdx != -1) {
    // We have an MUBUF instruction
    MachineOperand *SRsrc = &MI.getOperand(SRsrcIdx);
    unsigned SRsrcRC = get(MI.getOpcode()).OpInfo[SRsrcIdx].RegClass;
    if (RI.getCommonSubClass(MRI.getRegClass(SRsrc->getReg()),
                                             RI.getRegClass(SRsrcRC))) {
      // The operands are legal.
      // FIXME: We may need to legalize operands besided srsrc.
      return;
    }

    MachineBasicBlock &MBB = *MI.getParent();

    // Extract the ptr from the resource descriptor.
    unsigned SRsrcPtr = buildExtractSubReg(MI, MRI, *SRsrc,
      &AMDGPU::VReg_128RegClass, AMDGPU::sub0_sub1, &AMDGPU::VReg_64RegClass);

    // Create an empty resource descriptor
    unsigned Zero64 = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);
    unsigned SRsrcFormatLo = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
    unsigned SRsrcFormatHi = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
    unsigned NewSRsrc = MRI.createVirtualRegister(&AMDGPU::SReg_128RegClass);
    uint64_t RsrcDataFormat = getDefaultRsrcDataFormat();

    // Zero64 = 0
    BuildMI(MBB, MI, MI.getDebugLoc(), get(AMDGPU::S_MOV_B64), Zero64)
        .addImm(0);

    // SRsrcFormatLo = RSRC_DATA_FORMAT{31-0}
    BuildMI(MBB, MI, MI.getDebugLoc(), get(AMDGPU::S_MOV_B32), SRsrcFormatLo)
        .addImm(RsrcDataFormat & 0xFFFFFFFF);

    // SRsrcFormatHi = RSRC_DATA_FORMAT{63-32}
    BuildMI(MBB, MI, MI.getDebugLoc(), get(AMDGPU::S_MOV_B32), SRsrcFormatHi)
        .addImm(RsrcDataFormat >> 32);

    // NewSRsrc = {Zero64, SRsrcFormat}
    BuildMI(MBB, MI, MI.getDebugLoc(), get(AMDGPU::REG_SEQUENCE), NewSRsrc)
        .addReg(Zero64)
        .addImm(AMDGPU::sub0_sub1)
        .addReg(SRsrcFormatLo)
        .addImm(AMDGPU::sub2)
        .addReg(SRsrcFormatHi)
        .addImm(AMDGPU::sub3);

    MachineOperand *VAddr = getNamedOperand(MI, AMDGPU::OpName::vaddr);
    unsigned NewVAddr = MRI.createVirtualRegister(&AMDGPU::VReg_64RegClass);
    if (VAddr) {
      // This is already an ADDR64 instruction so we need to add the pointer
      // extracted from the resource descriptor to the current value of VAddr.
      unsigned NewVAddrLo = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
      unsigned NewVAddrHi = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);

      // NewVaddrLo = SRsrcPtr:sub0 + VAddr:sub0
      DebugLoc DL = MI.getDebugLoc();
      BuildMI(MBB, MI, DL, get(AMDGPU::V_ADD_I32_e32), NewVAddrLo)
        .addReg(SRsrcPtr, 0, AMDGPU::sub0)
        .addReg(VAddr->getReg(), 0, AMDGPU::sub0);

      // NewVaddrHi = SRsrcPtr:sub1 + VAddr:sub1
      BuildMI(MBB, MI, DL, get(AMDGPU::V_ADDC_U32_e32), NewVAddrHi)
        .addReg(SRsrcPtr, 0, AMDGPU::sub1)
        .addReg(VAddr->getReg(), 0, AMDGPU::sub1);

      // NewVaddr = {NewVaddrHi, NewVaddrLo}
      BuildMI(MBB, MI, MI.getDebugLoc(), get(AMDGPU::REG_SEQUENCE), NewVAddr)
          .addReg(NewVAddrLo)
          .addImm(AMDGPU::sub0)
          .addReg(NewVAddrHi)
          .addImm(AMDGPU::sub1);
    } else {
      // This instructions is the _OFFSET variant, so we need to convert it to
      // ADDR64.
      assert(MBB.getParent()->getSubtarget<SISubtarget>().getGeneration()
             < SISubtarget::VOLCANIC_ISLANDS &&
             "FIXME: Need to emit flat atomics here");

      MachineOperand *VData = getNamedOperand(MI, AMDGPU::OpName::vdata);
      MachineOperand *Offset = getNamedOperand(MI, AMDGPU::OpName::offset);
      MachineOperand *SOffset = getNamedOperand(MI, AMDGPU::OpName::soffset);
      unsigned Addr64Opcode = AMDGPU::getAddr64Inst(MI.getOpcode());

      // Atomics rith return have have an additional tied operand and are
      // missing some of the special bits.
      MachineOperand *VDataIn = getNamedOperand(MI, AMDGPU::OpName::vdata_in);
      MachineInstr *Addr64;

      if (!VDataIn) {
        // Regular buffer load / store.
        MachineInstrBuilder MIB =
            BuildMI(MBB, MI, MI.getDebugLoc(), get(Addr64Opcode))
                .addOperand(*VData)
                .addReg(AMDGPU::NoRegister) // Dummy value for vaddr.
                // This will be replaced later
                // with the new value of vaddr.
                .addOperand(*SRsrc)
                .addOperand(*SOffset)
                .addOperand(*Offset);

        // Atomics do not have this operand.
        if (const MachineOperand *GLC =
                getNamedOperand(MI, AMDGPU::OpName::glc)) {
          MIB.addImm(GLC->getImm());
        }

        MIB.addImm(getNamedImmOperand(MI, AMDGPU::OpName::slc));

        if (const MachineOperand *TFE =
                getNamedOperand(MI, AMDGPU::OpName::tfe)) {
          MIB.addImm(TFE->getImm());
        }

        MIB.setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
        Addr64 = MIB;
      } else {
        // Atomics with return.
        Addr64 = BuildMI(MBB, MI, MI.getDebugLoc(), get(Addr64Opcode))
                     .addOperand(*VData)
                     .addOperand(*VDataIn)
                     .addReg(AMDGPU::NoRegister) // Dummy value for vaddr.
                     // This will be replaced later
                     // with the new value of vaddr.
                     .addOperand(*SRsrc)
                     .addOperand(*SOffset)
                     .addOperand(*Offset)
                     .addImm(getNamedImmOperand(MI, AMDGPU::OpName::slc))
                     .setMemRefs(MI.memoperands_begin(), MI.memoperands_end());
      }

      MI.removeFromParent();

      // NewVaddr = {NewVaddrHi, NewVaddrLo}
      BuildMI(MBB, Addr64, Addr64->getDebugLoc(), get(AMDGPU::REG_SEQUENCE),
              NewVAddr)
          .addReg(SRsrcPtr, 0, AMDGPU::sub0)
          .addImm(AMDGPU::sub0)
          .addReg(SRsrcPtr, 0, AMDGPU::sub1)
          .addImm(AMDGPU::sub1);

      VAddr = getNamedOperand(*Addr64, AMDGPU::OpName::vaddr);
      SRsrc = getNamedOperand(*Addr64, AMDGPU::OpName::srsrc);
    }

    // Update the instruction to use NewVaddr
    VAddr->setReg(NewVAddr);
    // Update the instruction to use NewSRsrc
    SRsrc->setReg(NewSRsrc);
  }
}

void SIInstrInfo::moveToVALU(MachineInstr &TopInst) const {
  SmallVector<MachineInstr *, 128> Worklist;
  Worklist.push_back(&TopInst);

  while (!Worklist.empty()) {
    MachineInstr &Inst = *Worklist.pop_back_val();
    MachineBasicBlock *MBB = Inst.getParent();
    MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();

    unsigned Opcode = Inst.getOpcode();
    unsigned NewOpcode = getVALUOp(Inst);

    // Handle some special cases
    switch (Opcode) {
    default:
      break;
    case AMDGPU::S_AND_B64:
      splitScalar64BitBinaryOp(Worklist, Inst, AMDGPU::V_AND_B32_e64);
      Inst.eraseFromParent();
      continue;

    case AMDGPU::S_OR_B64:
      splitScalar64BitBinaryOp(Worklist, Inst, AMDGPU::V_OR_B32_e64);
      Inst.eraseFromParent();
      continue;

    case AMDGPU::S_XOR_B64:
      splitScalar64BitBinaryOp(Worklist, Inst, AMDGPU::V_XOR_B32_e64);
      Inst.eraseFromParent();
      continue;

    case AMDGPU::S_NOT_B64:
      splitScalar64BitUnaryOp(Worklist, Inst, AMDGPU::V_NOT_B32_e32);
      Inst.eraseFromParent();
      continue;

    case AMDGPU::S_BCNT1_I32_B64:
      splitScalar64BitBCNT(Worklist, Inst);
      Inst.eraseFromParent();
      continue;

    case AMDGPU::S_BFE_I64: {
      splitScalar64BitBFE(Worklist, Inst);
      Inst.eraseFromParent();
      continue;
    }

    case AMDGPU::S_LSHL_B32:
      if (ST.getGeneration() >= SISubtarget::VOLCANIC_ISLANDS) {
        NewOpcode = AMDGPU::V_LSHLREV_B32_e64;
        swapOperands(Inst);
      }
      break;
    case AMDGPU::S_ASHR_I32:
      if (ST.getGeneration() >= SISubtarget::VOLCANIC_ISLANDS) {
        NewOpcode = AMDGPU::V_ASHRREV_I32_e64;
        swapOperands(Inst);
      }
      break;
    case AMDGPU::S_LSHR_B32:
      if (ST.getGeneration() >= SISubtarget::VOLCANIC_ISLANDS) {
        NewOpcode = AMDGPU::V_LSHRREV_B32_e64;
        swapOperands(Inst);
      }
      break;
    case AMDGPU::S_LSHL_B64:
      if (ST.getGeneration() >= SISubtarget::VOLCANIC_ISLANDS) {
        NewOpcode = AMDGPU::V_LSHLREV_B64;
        swapOperands(Inst);
      }
      break;
    case AMDGPU::S_ASHR_I64:
      if (ST.getGeneration() >= SISubtarget::VOLCANIC_ISLANDS) {
        NewOpcode = AMDGPU::V_ASHRREV_I64;
        swapOperands(Inst);
      }
      break;
    case AMDGPU::S_LSHR_B64:
      if (ST.getGeneration() >= SISubtarget::VOLCANIC_ISLANDS) {
        NewOpcode = AMDGPU::V_LSHRREV_B64;
        swapOperands(Inst);
      }
      break;

    case AMDGPU::S_ABS_I32:
      lowerScalarAbs(Worklist, Inst);
      Inst.eraseFromParent();
      continue;

    case AMDGPU::S_CBRANCH_SCC0:
    case AMDGPU::S_CBRANCH_SCC1:
      // Clear unused bits of vcc
      BuildMI(*MBB, Inst, Inst.getDebugLoc(), get(AMDGPU::S_AND_B64),
              AMDGPU::VCC)
          .addReg(AMDGPU::EXEC)
          .addReg(AMDGPU::VCC);
      break;

    case AMDGPU::S_BFE_U64:
    case AMDGPU::S_BFM_B64:
      llvm_unreachable("Moving this op to VALU not implemented");
    }

    if (NewOpcode == AMDGPU::INSTRUCTION_LIST_END) {
      // We cannot move this instruction to the VALU, so we should try to
      // legalize its operands instead.
      legalizeOperands(Inst);
      continue;
    }

    // Use the new VALU Opcode.
    const MCInstrDesc &NewDesc = get(NewOpcode);
    Inst.setDesc(NewDesc);

    // Remove any references to SCC. Vector instructions can't read from it, and
    // We're just about to add the implicit use / defs of VCC, and we don't want
    // both.
    for (unsigned i = Inst.getNumOperands() - 1; i > 0; --i) {
      MachineOperand &Op = Inst.getOperand(i);
      if (Op.isReg() && Op.getReg() == AMDGPU::SCC) {
        Inst.RemoveOperand(i);
        addSCCDefUsersToVALUWorklist(Inst, Worklist);
      }
    }

    if (Opcode == AMDGPU::S_SEXT_I32_I8 || Opcode == AMDGPU::S_SEXT_I32_I16) {
      // We are converting these to a BFE, so we need to add the missing
      // operands for the size and offset.
      unsigned Size = (Opcode == AMDGPU::S_SEXT_I32_I8) ? 8 : 16;
      Inst.addOperand(MachineOperand::CreateImm(0));
      Inst.addOperand(MachineOperand::CreateImm(Size));

    } else if (Opcode == AMDGPU::S_BCNT1_I32_B32) {
      // The VALU version adds the second operand to the result, so insert an
      // extra 0 operand.
      Inst.addOperand(MachineOperand::CreateImm(0));
    }

    Inst.addImplicitDefUseOperands(*Inst.getParent()->getParent());

    if (Opcode == AMDGPU::S_BFE_I32 || Opcode == AMDGPU::S_BFE_U32) {
      const MachineOperand &OffsetWidthOp = Inst.getOperand(2);
      // If we need to move this to VGPRs, we need to unpack the second operand
      // back into the 2 separate ones for bit offset and width.
      assert(OffsetWidthOp.isImm() &&
             "Scalar BFE is only implemented for constant width and offset");
      uint32_t Imm = OffsetWidthOp.getImm();

      uint32_t Offset = Imm & 0x3f; // Extract bits [5:0].
      uint32_t BitWidth = (Imm & 0x7f0000) >> 16; // Extract bits [22:16].
      Inst.RemoveOperand(2);                     // Remove old immediate.
      Inst.addOperand(MachineOperand::CreateImm(Offset));
      Inst.addOperand(MachineOperand::CreateImm(BitWidth));
    }

    bool HasDst = Inst.getOperand(0).isReg() && Inst.getOperand(0).isDef();
    unsigned NewDstReg = AMDGPU::NoRegister;
    if (HasDst) {
      // Update the destination register class.
      const TargetRegisterClass *NewDstRC = getDestEquivalentVGPRClass(Inst);
      if (!NewDstRC)
        continue;

      unsigned DstReg = Inst.getOperand(0).getReg();
      NewDstReg = MRI.createVirtualRegister(NewDstRC);
      MRI.replaceRegWith(DstReg, NewDstReg);
    }

    // Legalize the operands
    legalizeOperands(Inst);

    if (HasDst)
     addUsersToMoveToVALUWorklist(NewDstReg, MRI, Worklist);
  }
}

void SIInstrInfo::lowerScalarAbs(SmallVectorImpl<MachineInstr *> &Worklist,
                                 MachineInstr &Inst) const {
  MachineBasicBlock &MBB = *Inst.getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  MachineBasicBlock::iterator MII = Inst;
  DebugLoc DL = Inst.getDebugLoc();

  MachineOperand &Dest = Inst.getOperand(0);
  MachineOperand &Src = Inst.getOperand(1);
  unsigned TmpReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  unsigned ResultReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);

  BuildMI(MBB, MII, DL, get(AMDGPU::V_SUB_I32_e32), TmpReg)
    .addImm(0)
    .addReg(Src.getReg());

  BuildMI(MBB, MII, DL, get(AMDGPU::V_MAX_I32_e64), ResultReg)
    .addReg(Src.getReg())
    .addReg(TmpReg);

  MRI.replaceRegWith(Dest.getReg(), ResultReg);
  addUsersToMoveToVALUWorklist(ResultReg, MRI, Worklist);
}

void SIInstrInfo::splitScalar64BitUnaryOp(
    SmallVectorImpl<MachineInstr *> &Worklist, MachineInstr &Inst,
    unsigned Opcode) const {
  MachineBasicBlock &MBB = *Inst.getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  MachineOperand &Dest = Inst.getOperand(0);
  MachineOperand &Src0 = Inst.getOperand(1);
  DebugLoc DL = Inst.getDebugLoc();

  MachineBasicBlock::iterator MII = Inst;

  const MCInstrDesc &InstDesc = get(Opcode);
  const TargetRegisterClass *Src0RC = Src0.isReg() ?
    MRI.getRegClass(Src0.getReg()) :
    &AMDGPU::SGPR_32RegClass;

  const TargetRegisterClass *Src0SubRC = RI.getSubRegClass(Src0RC, AMDGPU::sub0);

  MachineOperand SrcReg0Sub0 = buildExtractSubRegOrImm(MII, MRI, Src0, Src0RC,
                                                       AMDGPU::sub0, Src0SubRC);

  const TargetRegisterClass *DestRC = MRI.getRegClass(Dest.getReg());
  const TargetRegisterClass *NewDestRC = RI.getEquivalentVGPRClass(DestRC);
  const TargetRegisterClass *NewDestSubRC = RI.getSubRegClass(NewDestRC, AMDGPU::sub0);

  unsigned DestSub0 = MRI.createVirtualRegister(NewDestSubRC);
  BuildMI(MBB, MII, DL, InstDesc, DestSub0)
    .addOperand(SrcReg0Sub0);

  MachineOperand SrcReg0Sub1 = buildExtractSubRegOrImm(MII, MRI, Src0, Src0RC,
                                                       AMDGPU::sub1, Src0SubRC);

  unsigned DestSub1 = MRI.createVirtualRegister(NewDestSubRC);
  BuildMI(MBB, MII, DL, InstDesc, DestSub1)
    .addOperand(SrcReg0Sub1);

  unsigned FullDestReg = MRI.createVirtualRegister(NewDestRC);
  BuildMI(MBB, MII, DL, get(TargetOpcode::REG_SEQUENCE), FullDestReg)
    .addReg(DestSub0)
    .addImm(AMDGPU::sub0)
    .addReg(DestSub1)
    .addImm(AMDGPU::sub1);

  MRI.replaceRegWith(Dest.getReg(), FullDestReg);

  // We don't need to legalizeOperands here because for a single operand, src0
  // will support any kind of input.

  // Move all users of this moved value.
  addUsersToMoveToVALUWorklist(FullDestReg, MRI, Worklist);
}

void SIInstrInfo::splitScalar64BitBinaryOp(
    SmallVectorImpl<MachineInstr *> &Worklist, MachineInstr &Inst,
    unsigned Opcode) const {
  MachineBasicBlock &MBB = *Inst.getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  MachineOperand &Dest = Inst.getOperand(0);
  MachineOperand &Src0 = Inst.getOperand(1);
  MachineOperand &Src1 = Inst.getOperand(2);
  DebugLoc DL = Inst.getDebugLoc();

  MachineBasicBlock::iterator MII = Inst;

  const MCInstrDesc &InstDesc = get(Opcode);
  const TargetRegisterClass *Src0RC = Src0.isReg() ?
    MRI.getRegClass(Src0.getReg()) :
    &AMDGPU::SGPR_32RegClass;

  const TargetRegisterClass *Src0SubRC = RI.getSubRegClass(Src0RC, AMDGPU::sub0);
  const TargetRegisterClass *Src1RC = Src1.isReg() ?
    MRI.getRegClass(Src1.getReg()) :
    &AMDGPU::SGPR_32RegClass;

  const TargetRegisterClass *Src1SubRC = RI.getSubRegClass(Src1RC, AMDGPU::sub0);

  MachineOperand SrcReg0Sub0 = buildExtractSubRegOrImm(MII, MRI, Src0, Src0RC,
                                                       AMDGPU::sub0, Src0SubRC);
  MachineOperand SrcReg1Sub0 = buildExtractSubRegOrImm(MII, MRI, Src1, Src1RC,
                                                       AMDGPU::sub0, Src1SubRC);

  const TargetRegisterClass *DestRC = MRI.getRegClass(Dest.getReg());
  const TargetRegisterClass *NewDestRC = RI.getEquivalentVGPRClass(DestRC);
  const TargetRegisterClass *NewDestSubRC = RI.getSubRegClass(NewDestRC, AMDGPU::sub0);

  unsigned DestSub0 = MRI.createVirtualRegister(NewDestSubRC);
  MachineInstr &LoHalf = *BuildMI(MBB, MII, DL, InstDesc, DestSub0)
                              .addOperand(SrcReg0Sub0)
                              .addOperand(SrcReg1Sub0);

  MachineOperand SrcReg0Sub1 = buildExtractSubRegOrImm(MII, MRI, Src0, Src0RC,
                                                       AMDGPU::sub1, Src0SubRC);
  MachineOperand SrcReg1Sub1 = buildExtractSubRegOrImm(MII, MRI, Src1, Src1RC,
                                                       AMDGPU::sub1, Src1SubRC);

  unsigned DestSub1 = MRI.createVirtualRegister(NewDestSubRC);
  MachineInstr &HiHalf = *BuildMI(MBB, MII, DL, InstDesc, DestSub1)
                              .addOperand(SrcReg0Sub1)
                              .addOperand(SrcReg1Sub1);

  unsigned FullDestReg = MRI.createVirtualRegister(NewDestRC);
  BuildMI(MBB, MII, DL, get(TargetOpcode::REG_SEQUENCE), FullDestReg)
    .addReg(DestSub0)
    .addImm(AMDGPU::sub0)
    .addReg(DestSub1)
    .addImm(AMDGPU::sub1);

  MRI.replaceRegWith(Dest.getReg(), FullDestReg);

  // Try to legalize the operands in case we need to swap the order to keep it
  // valid.
  legalizeOperands(LoHalf);
  legalizeOperands(HiHalf);

  // Move all users of this moved vlaue.
  addUsersToMoveToVALUWorklist(FullDestReg, MRI, Worklist);
}

void SIInstrInfo::splitScalar64BitBCNT(
    SmallVectorImpl<MachineInstr *> &Worklist, MachineInstr &Inst) const {
  MachineBasicBlock &MBB = *Inst.getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();

  MachineBasicBlock::iterator MII = Inst;
  DebugLoc DL = Inst.getDebugLoc();

  MachineOperand &Dest = Inst.getOperand(0);
  MachineOperand &Src = Inst.getOperand(1);

  const MCInstrDesc &InstDesc = get(AMDGPU::V_BCNT_U32_B32_e64);
  const TargetRegisterClass *SrcRC = Src.isReg() ?
    MRI.getRegClass(Src.getReg()) :
    &AMDGPU::SGPR_32RegClass;

  unsigned MidReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  unsigned ResultReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);

  const TargetRegisterClass *SrcSubRC = RI.getSubRegClass(SrcRC, AMDGPU::sub0);

  MachineOperand SrcRegSub0 = buildExtractSubRegOrImm(MII, MRI, Src, SrcRC,
                                                      AMDGPU::sub0, SrcSubRC);
  MachineOperand SrcRegSub1 = buildExtractSubRegOrImm(MII, MRI, Src, SrcRC,
                                                      AMDGPU::sub1, SrcSubRC);

  BuildMI(MBB, MII, DL, InstDesc, MidReg)
    .addOperand(SrcRegSub0)
    .addImm(0);

  BuildMI(MBB, MII, DL, InstDesc, ResultReg)
    .addOperand(SrcRegSub1)
    .addReg(MidReg);

  MRI.replaceRegWith(Dest.getReg(), ResultReg);

  // We don't need to legalize operands here. src0 for etiher instruction can be
  // an SGPR, and the second input is unused or determined here.
  addUsersToMoveToVALUWorklist(ResultReg, MRI, Worklist);
}

void SIInstrInfo::splitScalar64BitBFE(SmallVectorImpl<MachineInstr *> &Worklist,
                                      MachineInstr &Inst) const {
  MachineBasicBlock &MBB = *Inst.getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  MachineBasicBlock::iterator MII = Inst;
  DebugLoc DL = Inst.getDebugLoc();

  MachineOperand &Dest = Inst.getOperand(0);
  uint32_t Imm = Inst.getOperand(2).getImm();
  uint32_t Offset = Imm & 0x3f; // Extract bits [5:0].
  uint32_t BitWidth = (Imm & 0x7f0000) >> 16; // Extract bits [22:16].

  (void) Offset;

  // Only sext_inreg cases handled.
  assert(Inst.getOpcode() == AMDGPU::S_BFE_I64 && BitWidth <= 32 &&
         Offset == 0 && "Not implemented");

  if (BitWidth < 32) {
    unsigned MidRegLo = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    unsigned MidRegHi = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    unsigned ResultReg = MRI.createVirtualRegister(&AMDGPU::VReg_64RegClass);

    BuildMI(MBB, MII, DL, get(AMDGPU::V_BFE_I32), MidRegLo)
        .addReg(Inst.getOperand(1).getReg(), 0, AMDGPU::sub0)
        .addImm(0)
        .addImm(BitWidth);

    BuildMI(MBB, MII, DL, get(AMDGPU::V_ASHRREV_I32_e32), MidRegHi)
      .addImm(31)
      .addReg(MidRegLo);

    BuildMI(MBB, MII, DL, get(TargetOpcode::REG_SEQUENCE), ResultReg)
      .addReg(MidRegLo)
      .addImm(AMDGPU::sub0)
      .addReg(MidRegHi)
      .addImm(AMDGPU::sub1);

    MRI.replaceRegWith(Dest.getReg(), ResultReg);
    addUsersToMoveToVALUWorklist(ResultReg, MRI, Worklist);
    return;
  }

  MachineOperand &Src = Inst.getOperand(1);
  unsigned TmpReg = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  unsigned ResultReg = MRI.createVirtualRegister(&AMDGPU::VReg_64RegClass);

  BuildMI(MBB, MII, DL, get(AMDGPU::V_ASHRREV_I32_e64), TmpReg)
    .addImm(31)
    .addReg(Src.getReg(), 0, AMDGPU::sub0);

  BuildMI(MBB, MII, DL, get(TargetOpcode::REG_SEQUENCE), ResultReg)
    .addReg(Src.getReg(), 0, AMDGPU::sub0)
    .addImm(AMDGPU::sub0)
    .addReg(TmpReg)
    .addImm(AMDGPU::sub1);

  MRI.replaceRegWith(Dest.getReg(), ResultReg);
  addUsersToMoveToVALUWorklist(ResultReg, MRI, Worklist);
}

void SIInstrInfo::addUsersToMoveToVALUWorklist(
  unsigned DstReg,
  MachineRegisterInfo &MRI,
  SmallVectorImpl<MachineInstr *> &Worklist) const {
  for (MachineRegisterInfo::use_iterator I = MRI.use_begin(DstReg),
         E = MRI.use_end(); I != E; ++I) {
    MachineInstr &UseMI = *I->getParent();
    if (!canReadVGPR(UseMI, I.getOperandNo())) {
      Worklist.push_back(&UseMI);
    }
  }
}

void SIInstrInfo::addSCCDefUsersToVALUWorklist(
    MachineInstr &SCCDefInst, SmallVectorImpl<MachineInstr *> &Worklist) const {
  // This assumes that all the users of SCC are in the same block
  // as the SCC def.
  for (MachineInstr &MI :
       llvm::make_range(MachineBasicBlock::iterator(SCCDefInst),
                        SCCDefInst.getParent()->end())) {
    // Exit if we find another SCC def.
    if (MI.findRegisterDefOperandIdx(AMDGPU::SCC) != -1)
      return;

    if (MI.findRegisterUseOperandIdx(AMDGPU::SCC) != -1)
      Worklist.push_back(&MI);
  }
}

const TargetRegisterClass *SIInstrInfo::getDestEquivalentVGPRClass(
  const MachineInstr &Inst) const {
  const TargetRegisterClass *NewDstRC = getOpRegClass(Inst, 0);

  switch (Inst.getOpcode()) {
  // For target instructions, getOpRegClass just returns the virtual register
  // class associated with the operand, so we need to find an equivalent VGPR
  // register class in order to move the instruction to the VALU.
  case AMDGPU::COPY:
  case AMDGPU::PHI:
  case AMDGPU::REG_SEQUENCE:
  case AMDGPU::INSERT_SUBREG:
    if (RI.hasVGPRs(NewDstRC))
      return nullptr;

    NewDstRC = RI.getEquivalentVGPRClass(NewDstRC);
    if (!NewDstRC)
      return nullptr;
    return NewDstRC;
  default:
    return NewDstRC;
  }
}

// Find the one SGPR operand we are allowed to use.
unsigned SIInstrInfo::findUsedSGPR(const MachineInstr &MI,
                                   int OpIndices[3]) const {
  const MCInstrDesc &Desc = MI.getDesc();

  // Find the one SGPR operand we are allowed to use.
  //
  // First we need to consider the instruction's operand requirements before
  // legalizing. Some operands are required to be SGPRs, such as implicit uses
  // of VCC, but we are still bound by the constant bus requirement to only use
  // one.
  //
  // If the operand's class is an SGPR, we can never move it.

  unsigned SGPRReg = findImplicitSGPRRead(MI);
  if (SGPRReg != AMDGPU::NoRegister)
    return SGPRReg;

  unsigned UsedSGPRs[3] = { AMDGPU::NoRegister };
  const MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();

  for (unsigned i = 0; i < 3; ++i) {
    int Idx = OpIndices[i];
    if (Idx == -1)
      break;

    const MachineOperand &MO = MI.getOperand(Idx);
    if (!MO.isReg())
      continue;

    // Is this operand statically required to be an SGPR based on the operand
    // constraints?
    const TargetRegisterClass *OpRC = RI.getRegClass(Desc.OpInfo[Idx].RegClass);
    bool IsRequiredSGPR = RI.isSGPRClass(OpRC);
    if (IsRequiredSGPR)
      return MO.getReg();

    // If this could be a VGPR or an SGPR, Check the dynamic register class.
    unsigned Reg = MO.getReg();
    const TargetRegisterClass *RegRC = MRI.getRegClass(Reg);
    if (RI.isSGPRClass(RegRC))
      UsedSGPRs[i] = Reg;
  }

  // We don't have a required SGPR operand, so we have a bit more freedom in
  // selecting operands to move.

  // Try to select the most used SGPR. If an SGPR is equal to one of the
  // others, we choose that.
  //
  // e.g.
  // V_FMA_F32 v0, s0, s0, s0 -> No moves
  // V_FMA_F32 v0, s0, s1, s0 -> Move s1

  // TODO: If some of the operands are 64-bit SGPRs and some 32, we should
  // prefer those.

  if (UsedSGPRs[0] != AMDGPU::NoRegister) {
    if (UsedSGPRs[0] == UsedSGPRs[1] || UsedSGPRs[0] == UsedSGPRs[2])
      SGPRReg = UsedSGPRs[0];
  }

  if (SGPRReg == AMDGPU::NoRegister && UsedSGPRs[1] != AMDGPU::NoRegister) {
    if (UsedSGPRs[1] == UsedSGPRs[2])
      SGPRReg = UsedSGPRs[1];
  }

  return SGPRReg;
}

MachineOperand *SIInstrInfo::getNamedOperand(MachineInstr &MI,
                                             unsigned OperandName) const {
  int Idx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), OperandName);
  if (Idx == -1)
    return nullptr;

  return &MI.getOperand(Idx);
}

uint64_t SIInstrInfo::getDefaultRsrcDataFormat() const {
  uint64_t RsrcDataFormat = AMDGPU::RSRC_DATA_FORMAT;
  if (ST.isAmdHsaOS()) {
    RsrcDataFormat |= (1ULL << 56);

    if (ST.getGeneration() >= SISubtarget::VOLCANIC_ISLANDS)
      // Set MTYPE = 2
      RsrcDataFormat |= (2ULL << 59);
  }

  return RsrcDataFormat;
}

uint64_t SIInstrInfo::getScratchRsrcWords23() const {
  uint64_t Rsrc23 = getDefaultRsrcDataFormat() |
                    AMDGPU::RSRC_TID_ENABLE |
                    0xffffffff; // Size;

  uint64_t EltSizeValue = Log2_32(ST.getMaxPrivateElementSize()) - 1;

  Rsrc23 |= (EltSizeValue << AMDGPU::RSRC_ELEMENT_SIZE_SHIFT) |
            // IndexStride = 64
            (UINT64_C(3) << AMDGPU::RSRC_INDEX_STRIDE_SHIFT);

  // If TID_ENABLE is set, DATA_FORMAT specifies stride bits [14:17].
  // Clear them unless we want a huge stride.
  if (ST.getGeneration() >= SISubtarget::VOLCANIC_ISLANDS)
    Rsrc23 &= ~AMDGPU::RSRC_DATA_FORMAT;

  return Rsrc23;
}

bool SIInstrInfo::isLowLatencyInstruction(const MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();

  return isSMRD(Opc);
}

bool SIInstrInfo::isHighLatencyInstruction(const MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();

  return isMUBUF(Opc) || isMTBUF(Opc) || isMIMG(Opc);
}

unsigned SIInstrInfo::isStackAccess(const MachineInstr &MI,
                                    int &FrameIndex) const {
  const MachineOperand *Addr = getNamedOperand(MI, AMDGPU::OpName::vaddr);
  if (!Addr || !Addr->isFI())
    return AMDGPU::NoRegister;

  assert(!MI.memoperands_empty() &&
         (*MI.memoperands_begin())->getAddrSpace() == AMDGPUAS::PRIVATE_ADDRESS);

  FrameIndex = Addr->getIndex();
  return getNamedOperand(MI, AMDGPU::OpName::vdata)->getReg();
}

unsigned SIInstrInfo::isSGPRStackAccess(const MachineInstr &MI,
                                        int &FrameIndex) const {
  const MachineOperand *Addr = getNamedOperand(MI, AMDGPU::OpName::addr);
  assert(Addr && Addr->isFI());
  FrameIndex = Addr->getIndex();
  return getNamedOperand(MI, AMDGPU::OpName::data)->getReg();
}

unsigned SIInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                          int &FrameIndex) const {

  if (!MI.mayLoad())
    return AMDGPU::NoRegister;

  if (isMUBUF(MI) || isVGPRSpill(MI))
    return isStackAccess(MI, FrameIndex);

  if (isSGPRSpill(MI))
    return isSGPRStackAccess(MI, FrameIndex);

  return AMDGPU::NoRegister;
}

unsigned SIInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                         int &FrameIndex) const {
  if (!MI.mayStore())
    return AMDGPU::NoRegister;

  if (isMUBUF(MI) || isVGPRSpill(MI))
    return isStackAccess(MI, FrameIndex);

  if (isSGPRSpill(MI))
    return isSGPRStackAccess(MI, FrameIndex);

  return AMDGPU::NoRegister;
}

unsigned SIInstrInfo::getInstSizeInBytes(const MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();
  const MCInstrDesc &Desc = getMCOpcodeFromPseudo(Opc);
  unsigned DescSize = Desc.getSize();

  // If we have a definitive size, we can use it. Otherwise we need to inspect
  // the operands to know the size.
  if (DescSize != 0)
    return DescSize;

  // 4-byte instructions may have a 32-bit literal encoded after them. Check
  // operands that coud ever be literals.
  if (isVALU(MI) || isSALU(MI)) {
    int Src0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0);
    if (Src0Idx == -1)
      return 4; // No operands.

    if (isLiteralConstantLike(MI.getOperand(Src0Idx), getOpSize(MI, Src0Idx)))
      return 8;

    int Src1Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1);
    if (Src1Idx == -1)
      return 4;

    if (isLiteralConstantLike(MI.getOperand(Src1Idx), getOpSize(MI, Src1Idx)))
      return 8;

    return 4;
  }

  switch (Opc) {
  case TargetOpcode::IMPLICIT_DEF:
  case TargetOpcode::KILL:
  case TargetOpcode::DBG_VALUE:
  case TargetOpcode::BUNDLE:
  case TargetOpcode::EH_LABEL:
    return 0;
  case TargetOpcode::INLINEASM: {
    const MachineFunction *MF = MI.getParent()->getParent();
    const char *AsmStr = MI.getOperand(0).getSymbolName();
    return getInlineAsmLength(AsmStr, *MF->getTarget().getMCAsmInfo());
  }
  default:
    llvm_unreachable("unable to find instruction size");
  }
}

ArrayRef<std::pair<int, const char *>>
SIInstrInfo::getSerializableTargetIndices() const {
  static const std::pair<int, const char *> TargetIndices[] = {
      {AMDGPU::TI_CONSTDATA_START, "amdgpu-constdata-start"},
      {AMDGPU::TI_SCRATCH_RSRC_DWORD0, "amdgpu-scratch-rsrc-dword0"},
      {AMDGPU::TI_SCRATCH_RSRC_DWORD1, "amdgpu-scratch-rsrc-dword1"},
      {AMDGPU::TI_SCRATCH_RSRC_DWORD2, "amdgpu-scratch-rsrc-dword2"},
      {AMDGPU::TI_SCRATCH_RSRC_DWORD3, "amdgpu-scratch-rsrc-dword3"}};
  return makeArrayRef(TargetIndices);
}

/// This is used by the post-RA scheduler (SchedulePostRAList.cpp).  The
/// post-RA version of misched uses CreateTargetMIHazardRecognizer.
ScheduleHazardRecognizer *
SIInstrInfo::CreateTargetPostRAHazardRecognizer(const InstrItineraryData *II,
                                            const ScheduleDAG *DAG) const {
  return new GCNHazardRecognizer(DAG->MF);
}

/// This is the hazard recognizer used at -O0 by the PostRAHazardRecognizer
/// pass.
ScheduleHazardRecognizer *
SIInstrInfo::CreateTargetPostRAHazardRecognizer(const MachineFunction &MF) const {
  return new GCNHazardRecognizer(MF);
}
