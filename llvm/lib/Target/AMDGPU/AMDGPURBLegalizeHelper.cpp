//===-- AMDGPURBLegalizeHelper.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Implements actual lowering algorithms for each ID that can be used in
/// Rule.OperandMapping. Similar to legalizer helper but with register banks.
//
//===----------------------------------------------------------------------===//

#include "AMDGPURBLegalizeHelper.h"
#include "AMDGPUGlobalISelUtils.h"
#include "AMDGPUInstrInfo.h"

using namespace llvm;
using namespace AMDGPU;

bool RegBankLegalizeHelper::findRuleAndApplyMapping(MachineInstr &MI) {
  const SetOfRulesForOpcode &RuleSet = RBLRules.getRulesForOpc(MI);
  const RegBankLLTMapping &Mapping = RuleSet.findMappingForMI(MI, MRI, MUI);

  SmallSet<Register, 4> WaterfallSGPRs;
  unsigned OpIdx = 0;
  if (Mapping.DstOpMapping.size() > 0) {
    B.setInsertPt(*MI.getParent(), std::next(MI.getIterator()));
    applyMappingDst(MI, OpIdx, Mapping.DstOpMapping);
  }
  if (Mapping.SrcOpMapping.size() > 0) {
    B.setInstr(MI);
    applyMappingSrc(MI, OpIdx, Mapping.SrcOpMapping, WaterfallSGPRs);
  }

  lower(MI, Mapping, WaterfallSGPRs);
  return true;
}

void RegBankLegalizeHelper::lower(MachineInstr &MI,
                                  const RegBankLLTMapping &Mapping,
                                  SmallSet<Register, 4> &WaterfallSGPRs) {

  switch (Mapping.LoweringMethod) {
  case DoNotLower:
    return;
  case UniExtToSel: {
    LLT Ty = MRI.getType(MI.getOperand(0).getReg());
    auto True =
        B.buildConstant(createSgpr(Ty), MI.getOpcode() == G_SEXT ? -1 : 1);
    auto False = B.buildConstant(createSgpr(Ty), 0);
    // Input to G_{Z|S}EXT is 'Legalizer legal' S1. Most common case is compare.
    // We are making select here. S1 cond was already 'any-extended to S32' +
    // 'AND with 1 to clean high bits' by Sgpr32AExtBoolInReg.
    B.buildSelect(MI.getOperand(0).getReg(), MI.getOperand(1).getReg(), True,
                  False);
    MI.eraseFromParent();
    return;
  }
  case Ext32To64: {
    const RegisterBank *RB = getRegBank(MI.getOperand(0).getReg());
    Register Hi = MRI.createVirtualRegister({RB, S32});

    if (MI.getOpcode() == AMDGPU::G_ZEXT) {
      B.buildConstant(Hi, 0);
    } else {
      Register ShiftAmt = MRI.createVirtualRegister({RB, S32});
      // Replicate sign bit from 32-bit extended part.
      B.buildConstant(ShiftAmt, 31);
      B.buildAShr(Hi, MI.getOperand(1).getReg(), ShiftAmt);
    }

    B.buildMergeLikeInstr(MI.getOperand(0).getReg(),
                          {MI.getOperand(1).getReg(), Hi});
    MI.eraseFromParent();
    return;
  }
  case UniCstExt: {
    uint64_t ConstVal = MI.getOperand(1).getCImm()->getZExtValue();
    B.buildConstant(MI.getOperand(0).getReg(), ConstVal);

    MI.eraseFromParent();
    return;
  }
  case VgprToVccCopy: {
    Register Src = MI.getOperand(1).getReg();
    LLT Ty = MRI.getType(Src);
    // Take lowest bit from each lane and put it in lane mask.
    // Lowering via compare, but we need to clean high bits first as compare
    // compares all bits in register.
    Register BoolSrc = createVgpr(Ty);
    if (Ty == S64) {
      auto Src64 = B.buildUnmerge({createVgpr(S32), createVgpr(S32)}, Src);
      auto One = B.buildConstant(createVgpr(S32), 1);
      auto AndLo = B.buildAnd(createVgpr(S32), Src64.getReg(0), One);
      auto Zero = B.buildConstant(createVgpr(S32), 0);
      auto AndHi = B.buildAnd(createVgpr(S32), Src64.getReg(1), Zero);
      B.buildMergeLikeInstr(BoolSrc, {AndLo, AndHi});
    } else {
      assert(Ty == S32 || Ty == S16);
      auto One = B.buildConstant(createVgpr(Ty), 1);
      B.buildAnd(BoolSrc, Src, One);
    }
    auto Zero = B.buildConstant(createVgpr(Ty), 0);
    B.buildICmp(CmpInst::ICMP_NE, MI.getOperand(0).getReg(), BoolSrc, Zero);
    MI.eraseFromParent();
    return;
  }
  case SplitTo32: {
    auto Op1 = B.buildUnmerge({createVgpr(S32), createVgpr(S32)},
                              MI.getOperand(1).getReg());
    auto Op2 = B.buildUnmerge({createVgpr(S32), createVgpr(S32)},
                              MI.getOperand(2).getReg());
    auto ResLo = B.buildInstr(MI.getOpcode(), {createVgpr(S32)},
                              {Op1.getReg(0), Op2.getReg(0)});
    auto ResHi = B.buildInstr(MI.getOpcode(), {createVgpr(S32)},
                              {Op1.getReg(1), Op2.getReg(1)});
    B.buildMergeLikeInstr(MI.getOperand(0).getReg(), {ResLo, ResHi});
    MI.eraseFromParent();
    break;
  }
  }

  // TODO: executeInWaterfallLoop(... WaterfallSGPRs)
}

LLT RegBankLegalizeHelper::getTyFromID(RegBankLLTMapingApplyID ID) {
  switch (ID) {
  case Vcc:
  case UniInVcc:
    return LLT::scalar(1);
  case Sgpr16:
    return LLT::scalar(16);
  case Sgpr32:
  case Sgpr32Trunc:
  case Sgpr32AExt:
  case Sgpr32AExtBoolInReg:
  case Sgpr32SExt:
  case UniInVgprS32:
  case Vgpr32:
    return LLT::scalar(32);
  case Sgpr64:
  case Vgpr64:
    return LLT::scalar(64);

  case SgprV4S32:
  case VgprV4S32:
  case UniInVgprV4S32:
    return LLT::fixed_vector(4, 32);
  case VgprP1:
    return LLT::pointer(1, 64);
  default:
    return LLT();
  }
}

const RegisterBank *
RegBankLegalizeHelper::getRBFromID(RegBankLLTMapingApplyID ID) {
  switch (ID) {
  case Vcc:
    return VccRB;

  case Sgpr16:
  case Sgpr32:
  case Sgpr64:
  case SgprV4S32:
  case UniInVcc:
  case UniInVgprS32:
  case UniInVgprV4S32:
  case Sgpr32Trunc:
  case Sgpr32AExt:
  case Sgpr32AExtBoolInReg:
  case Sgpr32SExt:
    return SgprRB;

  case Vgpr32:
  case Vgpr64:
  case VgprP1:
  case VgprV4S32:
    return VgprRB;

  default:
    return nullptr;
  }
}

void RegBankLegalizeHelper::applyMappingDst(
    MachineInstr &MI, unsigned &OpIdx,
    const SmallVectorImpl<RegBankLLTMapingApplyID> &MethodIDs) {
  // Defs start from operand 0
  for (; OpIdx < MethodIDs.size(); ++OpIdx) {
    if (MethodIDs[OpIdx] == None)
      continue;
    MachineOperand &Op = MI.getOperand(OpIdx);
    Register Reg = Op.getReg();
    LLT Ty = MRI.getType(Reg);
    const RegisterBank *RB = getRegBank(Reg);

    switch (MethodIDs[OpIdx]) {
    // vcc, sgpr and vgpr scalars, pointers and vectors
    case Vcc:
    case Sgpr16:
    case Sgpr32:
    case Sgpr64:
    case SgprV4S32:
    case Vgpr32:
    case Vgpr64:
    case VgprP1:
    case VgprV4S32: {
      assert(Ty == getTyFromID(MethodIDs[OpIdx]));
      assert(RB == getRBFromID(MethodIDs[OpIdx]));
      break;
    }

    // uniform in vcc/vgpr: scalars and vectors
    case UniInVcc: {
      assert(Ty == S1);
      assert(RB == SgprRB);
      Op.setReg(createVcc());
      auto CopyS32_Vcc =
          B.buildInstr(G_COPY_SCC_VCC, {createSgpr(S32)}, {Op.getReg()});
      B.buildTrunc(Reg, CopyS32_Vcc);
      break;
    }
    case UniInVgprS32:
    case UniInVgprV4S32: {
      assert(Ty == getTyFromID(MethodIDs[OpIdx]));
      assert(RB == SgprRB);
      AMDGPU::buildReadAnyLaneDst(B, MI, RBI);
      break;
    }

    // sgpr trunc
    case Sgpr32Trunc: {
      assert(Ty.getSizeInBits() < 32);
      assert(RB == SgprRB);
      Op.setReg(createSgpr(S32));
      B.buildTrunc(Reg, Op.getReg());
      break;
    }
    case Invalid: {
      MI.dump();
      llvm_unreachable("missing fast rule for MI");
    }

    default:
      llvm_unreachable("ID not supported");
    }
  }
}

void RegBankLegalizeHelper::applyMappingSrc(
    MachineInstr &MI, unsigned &OpIdx,
    const SmallVectorImpl<RegBankLLTMapingApplyID> &MethodIDs,
    SmallSet<Register, 4> &SGPRWaterfallOperandRegs) {
  for (unsigned i = 0; i < MethodIDs.size(); ++OpIdx, ++i) {
    if (MethodIDs[i] == None || MethodIDs[i] == IntrId || MethodIDs[i] == Imm)
      continue;

    MachineOperand &Op = MI.getOperand(OpIdx);
    Register Reg = Op.getReg();
    LLT Ty = MRI.getType(Reg);
    const RegisterBank *RB = getRegBank(Reg);

    switch (MethodIDs[i]) {
    case Vcc: {
      assert(Ty == S1);
      assert(RB == VccRB || RB == SgprRB);

      if (RB == SgprRB) {
        auto Aext = B.buildAnyExt(createSgpr(S32), Reg);
        auto CopyVcc_Scc = B.buildInstr(G_COPY_VCC_SCC, {createVcc()}, {Aext});
        Op.setReg(CopyVcc_Scc.getReg(0));
      }
      break;
    }

    // sgpr scalars, pointers and vectors
    case Sgpr16:
    case Sgpr32:
    case Sgpr64:
    case SgprV4S32: {
      assert(Ty == getTyFromID(MethodIDs[i]));
      assert(RB == getRBFromID(MethodIDs[i]));
      break;
    }

    // vgpr scalars, pointers and vectors
    case Vgpr32:
    case Vgpr64:
    case VgprP1:
    case VgprV4S32: {
      assert(Ty == getTyFromID(MethodIDs[i]));
      if (RB != VgprRB) {
        auto CopyToVgpr =
            B.buildCopy(createVgpr(getTyFromID(MethodIDs[i])), Reg);
        Op.setReg(CopyToVgpr.getReg(0));
      }
      break;
    }

    // sgpr and vgpr scalars with extend
    case Sgpr32AExt: {
      // Note: this ext allows S1, and it is meant to be combined away.
      assert(Ty.getSizeInBits() < 32);
      assert(RB == SgprRB);
      auto Aext = B.buildAnyExt(createSgpr(S32), Reg);
      Op.setReg(Aext.getReg(0));
      break;
    }
    case Sgpr32AExtBoolInReg: {
      // Note: this ext allows S1, and it is meant to be combined away.
      assert(Ty.getSizeInBits() == 1);
      assert(RB == SgprRB);
      auto Aext = B.buildAnyExt(createSgpr(S32), Reg);
      // Zext SgprS1 is not legal, this instruction is most of times meant to be
      // combined away in RB combiner, so do not make AND with 1.
      auto Cst1 = B.buildConstant(createSgpr(S32), 1);
      auto BoolInReg = B.buildAnd(createSgpr(S32), Aext, Cst1);
      Op.setReg(BoolInReg.getReg(0));
      break;
    }
    case Sgpr32SExt: {
      assert(1 < Ty.getSizeInBits() && Ty.getSizeInBits() < 32);
      assert(RB == SgprRB);
      auto Sext = B.buildSExt(createSgpr(S32), Reg);
      Op.setReg(Sext.getReg(0));
      break;
    }
    default:
      llvm_unreachable("ID not supported");
    }
  }
}

void RegBankLegalizeHelper::applyMappingPHI(MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  LLT Ty = MRI.getType(Dst);

  LLT S32 = LLT::scalar(32);
  if (Ty == LLT::scalar(1) && MUI.isUniform(Dst)) {
    B.setInsertPt(*MI.getParent(), MI.getParent()->getFirstNonPHI());

    Register NewDst = createSgpr(S32);
    B.buildTrunc(Dst, NewDst);
    MI.getOperand(0).setReg(NewDst);

    for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
      Register UseReg = MI.getOperand(i).getReg();

      auto DefMI = MRI.getVRegDef(UseReg)->getIterator();
      MachineBasicBlock *DefMBB = DefMI->getParent();

      B.setInsertPt(*DefMBB, DefMBB->SkipPHIsAndLabels(std::next(DefMI)));

      Register NewUseReg = createSgpr(S32);
      B.buildAnyExt(NewUseReg, UseReg);
      MI.getOperand(i).setReg(NewUseReg);
    }

    return;
  }

  // ALL divergent i1 phis should be already lowered and inst-selected into PHI
  // with sgpr reg class and S1 LLT.
  // Note: this includes divergent phis that don't require lowering.
  if (Ty == LLT::scalar(1) && MUI.isDivergent(Dst)) {
    llvm_unreachable("Make sure to run AMDGPUGlobalISelDivergenceLowering "
                     "before RB-legalize to lower lane mask(vcc) phis\n");
  }

  // We accept all types that can fit in some register class.
  // Uniform G_PHIs have all sgpr registers.
  // Divergent G_PHIs have vgpr dst but inputs can be sgpr or vgpr.
  if (Ty == LLT::scalar(32)) {
    return;
  }

  MI.dump();
  llvm_unreachable("type not supported\n");
}

bool operandsHaveRB(MachineInstr &MI, const RegisterBank *RB,
                    MachineRegisterInfo &MRI, unsigned StartOpIdx,
                    unsigned EndOpIdx) {
  for (unsigned i = StartOpIdx; i <= EndOpIdx; ++i) {
    if (MRI.getRegBankOrNull(MI.getOperand(i).getReg()) != RB)
      return false;
  }
  return true;
}

void RegBankLegalizeHelper::applyMappingTrivial(MachineInstr &MI) {
  const RegisterBank *RB = getRegBank(MI.getOperand(0).getReg());
  // Put RB on all registers
  unsigned NumDefs = MI.getNumDefs();
  unsigned NumOperands = MI.getNumOperands();

  assert(operandsHaveRB(MI, RB, MRI, 0, NumDefs - 1));
  if (RB->getID() == AMDGPU::SGPRRegBankID)
    assert(operandsHaveRB(MI, RB, MRI, NumDefs, NumOperands - 1));

  if (RB->getID() == AMDGPU::VGPRRegBankID) {
    for (unsigned i = NumDefs; i < NumOperands; ++i) {
      Register Reg = MI.getOperand(i).getReg();
      if (getRegBank(Reg) != RB) {
        B.setInstr(MI);
        auto Copy = B.buildCopy(createVgpr(MRI.getType(Reg)), Reg);
        MI.getOperand(i).setReg(Copy.getReg(0));
      }
    }
  }
}
