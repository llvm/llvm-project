//===-- AMDGPURegBankLegalizeHelper.cpp -----------------------------------===//
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

#include "AMDGPURegBankLegalizeHelper.h"
#include "AMDGPUGlobalISelUtils.h"
#include "AMDGPUInstrInfo.h"
#include "AMDGPURegisterBankInfo.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"

#define DEBUG_TYPE "amdgpu-regbanklegalize"

using namespace llvm;
using namespace AMDGPU;

RegBankLegalizeHelper::RegBankLegalizeHelper(
    MachineIRBuilder &B, const MachineUniformityInfo &MUI,
    const RegisterBankInfo &RBI, const RegBankLegalizeRules &RBLRules)
    : B(B), MRI(*B.getMRI()), MUI(MUI), RBI(RBI), RBLRules(RBLRules),
      SgprRB(&RBI.getRegBank(AMDGPU::SGPRRegBankID)),
      VgprRB(&RBI.getRegBank(AMDGPU::VGPRRegBankID)),
      VccRB(&RBI.getRegBank(AMDGPU::VCCRegBankID)) {}

void RegBankLegalizeHelper::findRuleAndApplyMapping(MachineInstr &MI) {
  const SetOfRulesForOpcode &RuleSet = RBLRules.getRulesForOpc(MI);
  const RegBankLLTMapping &Mapping = RuleSet.findMappingForMI(MI, MRI, MUI);

  SmallSet<Register, 4> WaterfallSgprs;
  unsigned OpIdx = 0;
  if (Mapping.DstOpMapping.size() > 0) {
    B.setInsertPt(*MI.getParent(), std::next(MI.getIterator()));
    applyMappingDst(MI, OpIdx, Mapping.DstOpMapping);
  }
  if (Mapping.SrcOpMapping.size() > 0) {
    B.setInstr(MI);
    applyMappingSrc(MI, OpIdx, Mapping.SrcOpMapping, WaterfallSgprs);
  }

  lower(MI, Mapping, WaterfallSgprs);
}

void RegBankLegalizeHelper::lower(MachineInstr &MI,
                                  const RegBankLLTMapping &Mapping,
                                  SmallSet<Register, 4> &WaterfallSgprs) {

  switch (Mapping.LoweringMethod) {
  case DoNotLower:
    return;
  case UniExtToSel: {
    LLT Ty = MRI.getType(MI.getOperand(0).getReg());
    auto True = B.buildConstant({SgprRB, Ty},
                                MI.getOpcode() == AMDGPU::G_SEXT ? -1 : 1);
    auto False = B.buildConstant({SgprRB, Ty}, 0);
    // Input to G_{Z|S}EXT is 'Legalizer legal' S1. Most common case is compare.
    // We are making select here. S1 cond was already 'any-extended to S32' +
    // 'AND with 1 to clean high bits' by Sgpr32AExtBoolInReg.
    B.buildSelect(MI.getOperand(0).getReg(), MI.getOperand(1).getReg(), True,
                  False);
    MI.eraseFromParent();
    return;
  }
  case Ext32To64: {
    const RegisterBank *RB = MRI.getRegBank(MI.getOperand(0).getReg());
    MachineInstrBuilder Hi;

    if (MI.getOpcode() == AMDGPU::G_ZEXT) {
      Hi = B.buildConstant({RB, S32}, 0);
    } else {
      // Replicate sign bit from 32-bit extended part.
      auto ShiftAmt = B.buildConstant({RB, S32}, 31);
      Hi = B.buildAShr({RB, S32}, MI.getOperand(1).getReg(), ShiftAmt);
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
    Register BoolSrc = MRI.createVirtualRegister({VgprRB, Ty});
    if (Ty == S64) {
      auto Src64 = B.buildUnmerge({VgprRB, Ty}, Src);
      auto One = B.buildConstant(VgprRB_S32, 1);
      auto AndLo = B.buildAnd(VgprRB_S32, Src64.getReg(0), One);
      auto Zero = B.buildConstant(VgprRB_S32, 0);
      auto AndHi = B.buildAnd(VgprRB_S32, Src64.getReg(1), Zero);
      B.buildMergeLikeInstr(BoolSrc, {AndLo, AndHi});
    } else {
      assert(Ty == S32 || Ty == S16);
      auto One = B.buildConstant({VgprRB, Ty}, 1);
      B.buildAnd(BoolSrc, Src, One);
    }
    auto Zero = B.buildConstant({VgprRB, Ty}, 0);
    B.buildICmp(CmpInst::ICMP_NE, MI.getOperand(0).getReg(), BoolSrc, Zero);
    MI.eraseFromParent();
    return;
  }
  case SplitTo32: {
    auto Op1 = B.buildUnmerge(VgprRB_S32, MI.getOperand(1).getReg());
    auto Op2 = B.buildUnmerge(VgprRB_S32, MI.getOperand(2).getReg());
    unsigned Opc = MI.getOpcode();
    auto Lo = B.buildInstr(Opc, {VgprRB_S32}, {Op1.getReg(0), Op2.getReg(0)});
    auto Hi = B.buildInstr(Opc, {VgprRB_S32}, {Op1.getReg(1), Op2.getReg(1)});
    B.buildMergeLikeInstr(MI.getOperand(0).getReg(), {Lo, Hi});
    MI.eraseFromParent();
    break;
  }
  }

  // TODO: executeInWaterfallLoop(... WaterfallSgprs)
}

LLT RegBankLegalizeHelper::getTyFromID(RegBankLLTMappingApplyID ID) {
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
RegBankLegalizeHelper::getRegBankFromID(RegBankLLTMappingApplyID ID) {
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
    const SmallVectorImpl<RegBankLLTMappingApplyID> &MethodIDs) {
  // Defs start from operand 0
  for (; OpIdx < MethodIDs.size(); ++OpIdx) {
    if (MethodIDs[OpIdx] == None)
      continue;
    MachineOperand &Op = MI.getOperand(OpIdx);
    Register Reg = Op.getReg();
    LLT Ty = MRI.getType(Reg);
    [[maybe_unused]] const RegisterBank *RB = MRI.getRegBank(Reg);

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
      assert(RB == getRegBankFromID(MethodIDs[OpIdx]));
      break;
    }
    // uniform in vcc/vgpr: scalars and vectors
    case UniInVcc: {
      assert(Ty == S1);
      assert(RB == SgprRB);
      Register NewDst = MRI.createVirtualRegister(VccRB_S1);
      Op.setReg(NewDst);
      auto CopyS32_Vcc =
          B.buildInstr(AMDGPU::G_AMDGPU_COPY_SCC_VCC, {SgprRB_S32}, {NewDst});
      B.buildTrunc(Reg, CopyS32_Vcc);
      break;
    }
    case UniInVgprS32:
    case UniInVgprV4S32: {
      assert(Ty == getTyFromID(MethodIDs[OpIdx]));
      assert(RB == SgprRB);
      Register NewVgprDst = MRI.createVirtualRegister({VgprRB, Ty});
      Op.setReg(NewVgprDst);
      buildReadAnyLane(B, Reg, NewVgprDst, RBI);
      break;
    }
    // sgpr trunc
    case Sgpr32Trunc: {
      assert(Ty.getSizeInBits() < 32);
      assert(RB == SgprRB);
      Register NewDst = MRI.createVirtualRegister(SgprRB_S32);
      Op.setReg(NewDst);
      B.buildTrunc(Reg, NewDst);
      break;
    }
    case InvalidMapping: {
      LLVM_DEBUG(dbgs() << "Instruction with Invalid mapping: "; MI.dump(););
      llvm_unreachable("missing fast rule for MI");
    }
    default:
      llvm_unreachable("ID not supported");
    }
  }
}

void RegBankLegalizeHelper::applyMappingSrc(
    MachineInstr &MI, unsigned &OpIdx,
    const SmallVectorImpl<RegBankLLTMappingApplyID> &MethodIDs,
    SmallSet<Register, 4> &SgprWaterfallOperandRegs) {
  for (unsigned i = 0; i < MethodIDs.size(); ++OpIdx, ++i) {
    if (MethodIDs[i] == None || MethodIDs[i] == IntrId || MethodIDs[i] == Imm)
      continue;

    MachineOperand &Op = MI.getOperand(OpIdx);
    Register Reg = Op.getReg();
    LLT Ty = MRI.getType(Reg);
    const RegisterBank *RB = MRI.getRegBank(Reg);

    switch (MethodIDs[i]) {
    case Vcc: {
      assert(Ty == S1);
      assert(RB == VccRB || RB == SgprRB);
      if (RB == SgprRB) {
        auto Aext = B.buildAnyExt(SgprRB_S32, Reg);
        auto CopyVcc_Scc =
            B.buildInstr(AMDGPU::G_AMDGPU_COPY_VCC_SCC, {VccRB_S1}, {Aext});
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
      assert(RB == getRegBankFromID(MethodIDs[i]));
      break;
    }
    // vgpr scalars, pointers and vectors
    case Vgpr32:
    case Vgpr64:
    case VgprP1:
    case VgprV4S32: {
      assert(Ty == getTyFromID(MethodIDs[i]));
      if (RB != VgprRB) {
        auto CopyToVgpr = B.buildCopy({VgprRB, Ty}, Reg);
        Op.setReg(CopyToVgpr.getReg(0));
      }
      break;
    }
    // sgpr and vgpr scalars with extend
    case Sgpr32AExt: {
      // Note: this ext allows S1, and it is meant to be combined away.
      assert(Ty.getSizeInBits() < 32);
      assert(RB == SgprRB);
      auto Aext = B.buildAnyExt(SgprRB_S32, Reg);
      Op.setReg(Aext.getReg(0));
      break;
    }
    case Sgpr32AExtBoolInReg: {
      // Note: this ext allows S1, and it is meant to be combined away.
      assert(Ty.getSizeInBits() == 1);
      assert(RB == SgprRB);
      auto Aext = B.buildAnyExt(SgprRB_S32, Reg);
      // Zext SgprS1 is not legal, this instruction is most of times meant to be
      // combined away in RB combiner, so do not make AND with 1.
      auto Cst1 = B.buildConstant(SgprRB_S32, 1);
      auto BoolInReg = B.buildAnd(SgprRB_S32, Aext, Cst1);
      Op.setReg(BoolInReg.getReg(0));
      break;
    }
    case Sgpr32SExt: {
      assert(1 < Ty.getSizeInBits() && Ty.getSizeInBits() < 32);
      assert(RB == SgprRB);
      auto Sext = B.buildSExt(SgprRB_S32, Reg);
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

  if (Ty == LLT::scalar(1) && MUI.isUniform(Dst)) {
    B.setInsertPt(*MI.getParent(), MI.getParent()->getFirstNonPHI());

    Register NewDst = MRI.createVirtualRegister(SgprRB_S32);
    MI.getOperand(0).setReg(NewDst);
    B.buildTrunc(Dst, NewDst);

    for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
      Register UseReg = MI.getOperand(i).getReg();

      auto DefMI = MRI.getVRegDef(UseReg)->getIterator();
      MachineBasicBlock *DefMBB = DefMI->getParent();

      B.setInsertPt(*DefMBB, DefMBB->SkipPHIsAndLabels(std::next(DefMI)));

      auto NewUse = B.buildAnyExt(SgprRB_S32, UseReg);
      MI.getOperand(i).setReg(NewUse.getReg(0));
    }

    return;
  }

  // ALL divergent i1 phis should be already lowered and inst-selected into PHI
  // with sgpr reg class and S1 LLT.
  // Note: this includes divergent phis that don't require lowering.
  if (Ty == LLT::scalar(1) && MUI.isDivergent(Dst)) {
    LLVM_DEBUG(dbgs() << "Divergent S1 G_PHI: "; MI.dump(););
    llvm_unreachable("Make sure to run AMDGPUGlobalISelDivergenceLowering "
                     "before RegBankLegalize to lower lane mask(vcc) phis");
  }

  // We accept all types that can fit in some register class.
  // Uniform G_PHIs have all sgpr registers.
  // Divergent G_PHIs have vgpr dst but inputs can be sgpr or vgpr.
  if (Ty == LLT::scalar(32)) {
    return;
  }

  LLVM_DEBUG(dbgs() << "G_PHI not handled: "; MI.dump(););
  llvm_unreachable("type not supported");
}

[[maybe_unused]] static bool verifyRegBankOnOperands(MachineInstr &MI,
                                                     const RegisterBank *RB,
                                                     MachineRegisterInfo &MRI,
                                                     unsigned StartOpIdx,
                                                     unsigned EndOpIdx) {
  for (unsigned i = StartOpIdx; i <= EndOpIdx; ++i) {
    if (MRI.getRegBankOrNull(MI.getOperand(i).getReg()) != RB)
      return false;
  }
  return true;
}

void RegBankLegalizeHelper::applyMappingTrivial(MachineInstr &MI) {
  const RegisterBank *RB = MRI.getRegBank(MI.getOperand(0).getReg());
  // Put RB on all registers
  unsigned NumDefs = MI.getNumDefs();
  unsigned NumOperands = MI.getNumOperands();

  assert(verifyRegBankOnOperands(MI, RB, MRI, 0, NumDefs - 1));
  if (RB == SgprRB)
    assert(verifyRegBankOnOperands(MI, RB, MRI, NumDefs, NumOperands - 1));

  if (RB == VgprRB) {
    B.setInstr(MI);
    for (unsigned i = NumDefs; i < NumOperands; ++i) {
      Register Reg = MI.getOperand(i).getReg();
      if (MRI.getRegBank(Reg) != RB) {
        auto Copy = B.buildCopy({VgprRB, MRI.getType(Reg)}, Reg);
        MI.getOperand(i).setReg(Copy.getReg(0));
      }
    }
  }
}
