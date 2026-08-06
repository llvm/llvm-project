//===-- lib/CodeGen/GlobalISel/PISAPostLegalizerCombiner.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PISAMCTargetDesc.h"
#include "PISA.h"
#include "PISALegalizerInfo.h"
#include "PISATargetMachine.h"
#include "PISAUtils.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutor.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/GlobalISel/GISelValueTracking.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IntrinsicsPISA.h"
#include "llvm/IR/PISAIntrinsicUtils.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Target/TargetMachine.h"

#define GET_GICOMBINER_DEPS
#include "PISAGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "pisa-postlegalizer-combiner"

using namespace llvm;
using namespace llvm::MIPatternMatch;

namespace {

#define GET_GICOMBINER_TYPES
#include "PISAGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_TYPES

// integer types
constexpr ElementCount EC0 = ElementCount::getFixed(0);
constexpr LLT I16 = LLT(LLT::Kind::INTEGER, EC0, 16);
constexpr LLT I32 = LLT(LLT::Kind::INTEGER, EC0, 32);

class PISAPostLegalizerCombinerImpl : public Combiner {
protected:
  const PISAPostLegalizerCombinerImplRuleConfig &RuleConfig;
  const PISASubtarget &STI;

  // TODO: Make CombinerHelper methods const.
  mutable CombinerHelper Helper;

public:
  PISAPostLegalizerCombinerImpl(
      MachineFunction &MF, CombinerInfo &CInfo, GISelValueTracking &KB,
      GISelCSEInfo *CSEInfo,
      const PISAPostLegalizerCombinerImplRuleConfig &RuleConfig,
      const PISASubtarget &STI, MachineDominatorTree *MDT,
      const LegalizerInfo *LI);

  static const char *getName() { return "PISAGenPostLegalizeGICombiner"; }

  bool tryCombineAllImpl(MachineInstr &MI) const;
  bool tryCombineAll(MachineInstr &I) const override;

  void applyBuildVectorWithConstants(MachineInstr &MI) const;

  bool matchCompareSelect(MachineInstr *MI) const;
  void applyCompareSelect(MachineInstr *MI) const;

  bool
  matchFCmpInvertedCond(MachineInstr &MI,
                        std::tuple<MachineInstr *, Register> &MatchInfo) const;
  void
  applyFCmpInvertedCond(MachineInstr &MI,
                        std::tuple<MachineInstr *, Register> &MatchInfo) const;

  bool
  matchShlAddToMad(MachineInstr &MI,
                   std::function<void(MachineIRBuilder &)> &MatchInfo) const;

  bool
  matchFDivToRcpFMul(MachineInstr &MI,
                     std::function<void(MachineIRBuilder &)> &MatchInfo) const;
  void applyShiftOfConstants(MachineInstr *MI) const;

  bool matchRedundantMovesPost(MachineInstr &MI) const;
  void applyRedundantMovesPost(MachineInstr &MI) const;

  bool matchExtractAllToBuildVector(MachineInstr &MI,
                                    Register &Replacement) const;
  void applyExtractAllToBuildVector(MachineInstr &MI,
                                    Register Replacement) const;
  bool
  matchOrAndToBfi(MachineInstr &MI,
                  std::function<void(MachineIRBuilder &)> &MatchInfo) const;
  bool
  matchShiftSubToBfe(MachineInstr &MI,
                     std::function<void(MachineIRBuilder &)> &MatchInfo) const;

  bool matchAndBitfieldToBitfield(
      MachineInstr &MI,
      std::function<void(MachineIRBuilder &)> &MatchInfo) const;

  bool matchV2i1ZextToBfeBfi(MachineInstr &BuildVectorMI,
                             Register &BitcastInput) const;
  void applyV2i1ZextToBfeBfi(MachineInstr &MI, Register BitcastInput) const;

  bool matchAndSelect(MachineInstr &MI, Register &Replacement) const;
  void applyAndSelect(MachineInstr &MI, Register Replacement) const;

  bool matchCmpAndAllOnes(MachineInstr &MI, GISelValueTracking *VT,
                          Register &MatchInfo) const;
  void applyCmpAndAllOnes(MachineInstr &MI, Register &MatchInfo) const;

  // Fix G_SHL/G_LSHR/G_ASHR where the shift amount is not i32.
  // PISA legalizes shifts only as {I16,I32}, {I32,I32}, {I64,I32}; rules like
  // mul_to_shl can introduce a shift with the value type as the amount type.
  bool matchFixIllegalShiftAmt(MachineInstr &MI) const;
  void applyFixIllegalShiftAmt(MachineInstr &MI) const;

  bool matchSinkTrunc(MachineInstr &MI,
                      std::tuple<Register, int64_t> &MatchInfo) const;
  void applySinkTrunc(MachineInstr &MI,
                      std::tuple<Register, int64_t> &MatchInfo) const;

  bool matchTruncTrunc(MachineInstr &MI, Register &MatchInfo) const;
  void applyTruncTrunc(MachineInstr &MI, Register &MatchInfo) const;

  bool matchAbsRedMaxToRedAbsMax(
      MachineInstr &MI,
      std::function<void(MachineIRBuilder &)> &MatchInfo) const;

  struct Reg2MatchInfo {
    Register Reg0, Reg1;
  };
  bool matchBuildRegFrom2(MachineInstr &MI, Reg2MatchInfo &MatchInfo) const;
  void applyBuildRegFrom2(MachineInstr &MI, Reg2MatchInfo &MatchInfo) const;

  bool matchBuildVectorFromUnmergeLanes(
      MachineInstr &MI, std::tuple<Register, int64_t> &MatchInfo) const;
  void applyBuildVectorFromUnmergeLanes(
      MachineInstr &MI, std::tuple<Register, int64_t> &MatchInfo) const;

  bool
  matchBuildVectorConcatSubvectors(MachineInstr &MI,
                                   SmallVector<Register, 4> &MatchInfo) const;
  void
  applyBuildVectorConcatSubvectors(MachineInstr &MI,
                                   SmallVector<Register, 4> &MatchInfo) const;

  bool
  matchShiftTrueFalse(MachineInstr &MI,
                      std::function<void(MachineIRBuilder &)> &MatchInfo) const;

  bool matchAddInt8Reduction(
      MachineInstr &MI,
      std::function<void(MachineIRBuilder &)> &MatchInfo) const;

  bool matchUnmergeBitcastBuildVectorToBitcast(MachineInstr &MI,
                                               Register &MatchInfo) const;
  void applyUnmergeBitcastBuildVectorToBitcast(MachineInstr &MI,
                                               Register &MatchInfo) const;

  bool matchRedundantFence(MachineInstr &MI, MachineInstr *&PrevFence) const;
  void applyRedundantFence(MachineInstr &MI, MachineInstr *&PrevFence) const;

  bool matchMinMaxIdentityFold(MachineInstr &MI, Register &MatchInfo) const;

  bool
  matchExtractSubvectorBuildVector(MachineInstr &MI,
                                   SmallVector<Register, 8> &MatchInfo) const;
  void
  applyExtractSubvectorBuildVector(MachineInstr &MI,
                                   SmallVector<Register, 8> &MatchInfo) const;

  bool matchMergeAdjacentFences(MachineInstr &MI,
                                MachineInstr *&PrevFence) const;
  void applyMergeAdjacentFences(MachineInstr &MI,
                                MachineInstr *&PrevFence) const;

  bool
  matchExtractSubvectorPartial(MachineInstr &MI,
                               SmallVector<MachineInstr *, 8> &MatchInfo) const;
  void applyExtractSubvectorPartial(
      MachineInstr &MI, const SmallVector<MachineInstr *, 8> &MatchInfo) const;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#include "PISAGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
};

#define GET_GICOMBINER_IMPL
#include "PISAGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_IMPL

PISAPostLegalizerCombinerImpl::PISAPostLegalizerCombinerImpl(
    MachineFunction &MF, CombinerInfo &CInfo, GISelValueTracking &KB,
    GISelCSEInfo *CSEInfo,
    const PISAPostLegalizerCombinerImplRuleConfig &RuleConfig,
    const PISASubtarget &STI, MachineDominatorTree *MDT,
    const LegalizerInfo *LI)
    : Combiner(MF, CInfo, &KB, CSEInfo), RuleConfig(RuleConfig), STI(STI),
      Helper(Observer, B, /*IsPreLegalize=*/false, &KB, MDT, LI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "PISAGenPostLegalizeGICombiner.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

bool PISAPostLegalizerCombinerImpl::tryCombineAll(MachineInstr &MI) const {
  return tryCombineAllImpl(MI);
}

void PISAPostLegalizerCombinerImpl::applyBuildVectorWithConstants(
    MachineInstr &MI) const {
  auto EltSize = MRI.getType(MI.getOperand(1).getReg()).getSizeInBits();
  auto Size = EltSize * (MI.getNumOperands() - 1);
  assert((Size <= 64) && "vector size too large");

  auto NewReg = MRI.createGenericVirtualRegister(LLT::integer(Size));
  uint64_t Value = 0;

  for (unsigned I = MI.getNumOperands() - 1; I > 0; I--) {
    auto Reg = MI.getOperand(I).getReg();
    auto CValue = getAnyConstantVRegValWithLookThrough(Reg, MRI);
    assert(CValue.has_value() && "expected const vreg val");
    APInt IValue = CValue->Value;
    Value <<= EltSize;
    Value |= IValue.getZExtValue();
  }

  B.buildConstant(NewReg, Value);
  B.buildBitcast(MI.getOperand(0), NewReg);
  MI.eraseFromParent();
}

// i1 C = G_CMP ne i? A, 0
// i? S = G_SELECT i1 C, i? LHS, i? RHS
// => S = sel.? LHS, RHS, A
// ... or ...
// i1 C = G_CMP eq i? A, 0
// i? S = G_SELECT i1 C, i? LHS, i? RHS
// => S = sel.? RHS, LHS, A
bool PISAPostLegalizerCombinerImpl::matchCompareSelect(MachineInstr *MI) const {
  auto *SelectMI = MI;
  auto *CmpMI = getDefIgnoringCopies(SelectMI->getOperand(1).getReg(), MRI);
  if (CmpMI->getOpcode() == TargetOpcode::G_ICMP) {
    auto CC = CmpMI->getOperand(1).getPredicate();
    auto CReg = CmpMI->getOperand(3).getReg();
    auto CVal = getIConstantVRegValWithLookThrough(CReg, MRI);
    if (CVal.has_value() && (CVal->Value == 0) &&
        ((CC == CmpInst::ICMP_EQ) || (CC == CmpInst::ICMP_NE))) {
      auto CmpTy = MRI.getType(CmpMI->getOperand(2).getReg());
      auto SelTy = MRI.getType(SelectMI->getOperand(2).getReg());
      if ((CmpTy.isScalar() && !CmpTy.isPointer()) &&
          (SelTy.isScalar() && !SelTy.isPointer()) &&
          CmpTy.getScalarSizeInBits() == SelTy.getScalarSizeInBits()) {
        return true;
      }
    }
  }
  return false;
}
void PISAPostLegalizerCombinerImpl::applyCompareSelect(MachineInstr *MI) const {
  auto *SelectMI = MI;
  auto *CmpMI = getDefIgnoringCopies(SelectMI->getOperand(1).getReg(), MRI);
  auto CC = CmpMI->getOperand(1).getPredicate();
  auto DstReg = SelectMI->getOperand(0).getReg();
  auto MIB = B.buildInstr(PISA::G_PISA_SELECT)
                 .addDef(DstReg)
                 .add(CmpMI->getOperand(2));
  if (CC == CmpInst::ICMP_NE)
    MIB.add(SelectMI->getOperand(2)).add(SelectMI->getOperand(3));
  else
    MIB.add(SelectMI->getOperand(3)).add(SelectMI->getOperand(2));
  MI->eraseFromParent();
}

// %9:_(s1) = G_FCMP floatpred(oge), %1:reg32b(s32), %2:reg32b
// %22:_(s32) = G_CONSTANT i32 0
// %23:_(s32) = G_CONSTANT i32 -1
// %10:_(s32) = G_SELECT %9:_(s1), %23:_, %22:_
// %11:_(s32) = G_CONSTANT i32 -1
// %12:_(s32) = G_XOR %10:_, %11:_
// %13:_(s32) = G_CONSTANT i32 -1
// %6:_(s1) = G_ICMP intpred(eq), %12:_(s32), %13:_
// G_BRCOND %6:_(s1), %bb.4
// G_BR %bb.3
// => %9:_(s1) = G_FCMP floatpred(oge), %1:reg32b(s32), %2:reg32b
// => G_BRCOND %6:_(s1), %bb.3
// => G_BR %bb.4
bool PISAPostLegalizerCombinerImpl::matchFCmpInvertedCond(
    MachineInstr &MI, std::tuple<MachineInstr *, Register> &MatchInfo) const {
  assert(MI.getOpcode() == TargetOpcode::G_BR);

  MachineInstr *BrCondMI;
  MachineBasicBlock *MBB = MI.getParent();
  MachineBasicBlock::iterator BrIt(MI);

  if (BrIt == MBB->begin())
    return false;
  assert(std::next(BrIt) == MBB->end() && "expected G_BR to be a terminator");

  // G_BRCOND %6:_(s1), %bb.4
  BrCondMI = &*std::prev(BrIt);
  if (BrCondMI->getOpcode() != TargetOpcode::G_BRCOND)
    return false;

  auto *CmpMI = MRI.getVRegDef(BrCondMI->getOperand(0).getReg());

  Register FCC;
  CmpInst::Predicate Pred;
  // Match either:
  //   icmp eq (xor (select fcc, -1, 0), -1), -1  =>  NOT fcc
  //   icmp eq (select fcc, 0, -1), -1             =>  NOT fcc
  //   icmp eq (select fcc, -1, 0), 0              =>  NOT fcc (produced by
  //     not_cmp_fold simplifying the xor form above)
  bool Matched =
      mi_match(
          CmpMI, MRI,
          m_GICmp(m_Pred(Pred),
                  m_GXor(m_GISelect(m_Reg(FCC), m_AllOnesInt(), m_ZeroInt()),
                         m_AllOnesInt()),
                  m_AllOnesInt())) ||
      mi_match(CmpMI, MRI,
               m_GICmp(m_Pred(Pred),
                       m_GISelect(m_Reg(FCC), m_ZeroInt(), m_AllOnesInt()),
                       m_AllOnesInt())) ||
      mi_match(CmpMI, MRI,
               m_GICmp(m_Pred(Pred),
                       m_GISelect(m_Reg(FCC), m_AllOnesInt(), m_ZeroInt()),
                       m_ZeroInt()));
  if (!Matched || Pred != CmpInst::ICMP_EQ)
    return false;

  // %9:_(s1) = G_FCMP floatpred(oge), %1:reg32b(s32), %2:reg32b
  auto *FCmpMI = MRI.getVRegDef(FCC);
  if (FCmpMI->getOpcode() != TargetOpcode::G_FCMP)
    return false;

  auto FCCTy = MRI.getType(FCC);
  if (!(FCCTy.isScalar() && FCCTy.getScalarSizeInBits() == 1))
    return false;

  MatchInfo = std::make_tuple(BrCondMI, FCC);
  return true;
}
void PISAPostLegalizerCombinerImpl::applyFCmpInvertedCond(
    MachineInstr &MI, std::tuple<MachineInstr *, Register> &MatchInfo) const {
  MachineInstr *BrCond;
  Register FCC;
  std::tie(BrCond, FCC) = MatchInfo;
  auto *BrCondBB = BrCond->getOperand(1).getMBB();
  auto *BrBB = MI.getOperand(0).getMBB();

  Observer.changingInstr(MI);
  MI.getOperand(0).setMBB(BrCondBB);
  Observer.changedInstr(MI);

  Observer.changingInstr(*BrCond);
  BrCond->getOperand(0).setReg(FCC);
  BrCond->getOperand(1).setMBB(BrBB);
  Observer.changedInstr(*BrCond);
}

//%2:registers(s32) = G_SHL %0:reg32b, 2
//%4:registers(s32) = G_ADD %2:registers, %3:registers
// => %4:registers(s32) = G_PISA_SMAD %0:reg32b, 4, %3:registers
bool PISAPostLegalizerCombinerImpl::matchShlAddToMad(
    MachineInstr &AddMI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {

  Register MulReg, AddReg;
  APInt Shift;

  if (!mi_match(&AddMI, MRI,
                m_GAdd(m_GShl(m_Reg(MulReg), m_ICst(Shift)), m_Reg(AddReg))))
    return false;

  auto Ty = MRI.getType(MulReg);
  if (Shift.getZExtValue() >= Ty.getSizeInBits())
    return false;

  MatchInfo = [Ty, &AddMI, AddReg, MulReg,
               Shift = std::move(Shift)](MachineIRBuilder &B) {
    auto MulOp2 = B.buildConstant(
        Ty, APInt::getOneBitSet(Ty.getSizeInBits(), Shift.getZExtValue()));
    B.buildIntrinsic(Intrinsic::pisa_smad, AddMI.getOperand(0).getReg())
        .addUse(MulReg)
        .addUse(MulOp2.getReg(0))
        .addUse(AddReg);
  };
  return true;
}

// arcp G_FDIV %0, %1 that has at least one G_FADD user
// =>
// %rcp = arcp G_INTRINSIC intrinsic(@llvm.pisa.frcp), %1
// %result = arcp G_FMUL %0, %rcp
//
// This transformation exposes G_FMUL for potential FMA fusion with the G_FADD
// users.
bool PISAPostLegalizerCombinerImpl::matchFDivToRcpFMul(
    MachineInstr &MI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {
  // Check if this is a floating-point divide with FmArcp flag
  if (MI.getOpcode() != TargetOpcode::G_FDIV)
    return false;

  // Must have the FmArcp flag to allow reciprocal approximation
  if (!MI.getFlag(MachineInstr::FmArcp))
    return false;

  Register DstReg = MI.getOperand(0).getReg();
  Register Src0Reg = MI.getOperand(1).getReg();
  Register Src1Reg = MI.getOperand(2).getReg();
  LLT DstTy = MRI.getType(DstReg);

  // Only handle 32-bit float for now
  if (!DstTy.isScalar() || DstTy.getSizeInBits() != 32)
    return false;

  // Check if the result of this FDIV is used by at least one FADD/FSUB
  // This ensures we only apply the transformation when FMA fusion is possible
  if (!llvm::any_of(MRI.use_instructions(DstReg), [](const MachineInstr &Use) {
        return Use.getOpcode() == TargetOpcode::G_FADD ||
               Use.getOpcode() == TargetOpcode::G_FSUB;
      }))
    return false;

  uint16_t Flags = MI.getFlags();

  MatchInfo = [=](MachineIRBuilder &B) {
    auto Rcp = B.buildIntrinsic(Intrinsic::pisa_frcp, {DstTy})
                   .addUse(Src1Reg)
                   .setMIFlags(Flags);

    // A / B -> A * RCP(B)
    B.buildFMul(DstReg, Src0Reg, Rcp, Flags);
  };
  return true;
}

void PISAPostLegalizerCombinerImpl::applyShiftOfConstants(
    MachineInstr *MI) const {
  auto [SrcMI, SrcRegIdx] =
      PISA::getDefIgnoringBitcasts(MI->getOperand(1).getReg(), MRI);
  auto [ShiftMI, ShiftRegIdx] =
      PISA::getDefIgnoringBitcasts(MI->getOperand(2).getReg(), MRI);

  const auto *CImm = SrcMI->getOperand(1).getCImm();
  auto Shift = ShiftMI->getOperand(1).getCImm()->getZExtValue();
  int64_t NewVal = 0;
  switch (MI->getOpcode()) {
  default:
    assert(0 && "unhandled shift opcode");
    break;
  case TargetOpcode::G_SHL:
    NewVal = CImm->getZExtValue() << Shift;
    break;
  case TargetOpcode::G_ASHR:
    NewVal = CImm->getSExtValue() >> Shift;
    break;
  case TargetOpcode::G_LSHR:
    NewVal = CImm->getZExtValue() >> Shift;
    break;
  }
  B.buildConstant(MI->getOperand(0), NewVal);
  MI->eraseFromParent();
}

// A(<2x16>) = G_BITCAST ARG(32)
// B(16), C(16) = G_UNMERGE_VALUES A(<2x16)
// D(<2x16>) = G_BUILD_VECTOR B, C
// E(32) = G_BITCAST D(<2x16>)
// => E(32) = COPY ARG(32)
bool PISAPostLegalizerCombinerImpl::matchRedundantMovesPost(
    MachineInstr &MI) const {
  auto &BitcastMI = MI;
  auto DstReg = BitcastMI.getOperand(0).getReg();
  auto SrcReg = BitcastMI.getOperand(1).getReg();
  if (MRI.getType(DstReg).isVector() || !MRI.getType(SrcReg).isVector())
    return false;

  auto &BuildVecMI = *getDefIgnoringCopies(SrcReg, MRI);
  if (BuildVecMI.getOpcode() != TargetOpcode::G_BUILD_VECTOR)
    return false;

  Register UnmergeReg = 0;
  for (unsigned I = 1; I < BuildVecMI.getNumOperands(); I++) {
    auto &UnmergeMI =
        *getDefIgnoringCopies(BuildVecMI.getOperand(I).getReg(), MRI);
    if (UnmergeMI.getOpcode() != TargetOpcode::G_UNMERGE_VALUES)
      return false;
    if (I == 1) {
      UnmergeReg =
          UnmergeMI.getOperand(UnmergeMI.getNumOperands() - 1).getReg();
    } else if (UnmergeReg !=
               UnmergeMI.getOperand(UnmergeMI.getNumOperands() - 1).getReg()) {
      // must extract indices from the same vector
      return false;
    }
    // order to extracted operands should match
    auto VecEltReg = BuildVecMI.getOperand(I).getReg();
    if (UnmergeMI.getOperand(I - 1).getReg() != VecEltReg)
      return false;
  }

  auto &SrcBitcastMI = *getDefIgnoringCopies(UnmergeReg, MRI);
  if (SrcBitcastMI.getOpcode() != TargetOpcode::G_BITCAST)
    return false;

  auto SrcBitcastReg = SrcBitcastMI.getOperand(1).getReg();
  if (MRI.getType(DstReg).getSizeInBits() !=
      MRI.getType(SrcBitcastReg).getSizeInBits())
    return false;
  return true;
}

void PISAPostLegalizerCombinerImpl::applyRedundantMovesPost(
    MachineInstr &MI) const {
  auto &BitcastMI = MI;
  auto DstReg = BitcastMI.getOperand(0).getReg();
  auto SrcReg = BitcastMI.getOperand(1).getReg();
  auto &BuildVecMI = *getDefIgnoringCopies(SrcReg, MRI);
  auto &UnmergeMI =
      *getDefIgnoringCopies(BuildVecMI.getOperand(1).getReg(), MRI);
  auto &SrcBitcastMI = *getDefIgnoringCopies(
      UnmergeMI.getOperand(UnmergeMI.getNumOperands() - 1).getReg(), MRI);
  auto SrcBitcastReg = SrcBitcastMI.getOperand(1).getReg();
  B.buildCopy(DstReg, SrcBitcastReg);
  MI.eraseFromParent();
}

// A(s16) = G_EXTRACT_VECTOR_ELT ARG(<N x s16), 0
// B(s16) = G_EXTRACT_VECTOR_ELT ARG(<N x s16), 1
// ...
// X(s16) = G_EXTRACT_VECTOR_ELT ARG(<N x s16>), N-1
// Y(<N x s16>) = G_BUILD_VECTOR A, B, ..., X
// => Y(<N x s16>) = COPY ARG(<N x s16>)
//
// Also matches the equivalent unpack-via-unmerge idiom:
//   A, B, ..., X = G_UNMERGE_VALUES ARG(<N x s16>)
//   Y(<N x s16>) = G_BUILD_VECTOR A, B, ..., X
//   => Y(<N x s16>) = COPY ARG(<N x s16>)
bool PISAPostLegalizerCombinerImpl::matchExtractAllToBuildVector(
    MachineInstr &MI, Register &Replacement) const {
  auto &BuildVecMI = MI;
  auto DstReg = BuildVecMI.getOperand(0).getReg();
  auto SrcReg = BuildVecMI.getOperand(1).getReg();

  if (MRI.getType(SrcReg).isVector())
    return false;

  // Branch A: all operands are G_EXTRACT_VECTOR_ELT from the same vector with
  // sequential constant indices 0..N-1.
  Register SrcVecReg;
  bool AllExtracts = true;
  for (unsigned I = 1; I < BuildVecMI.getNumOperands(); I++) {
    auto EltReg = BuildVecMI.getOperand(I).getReg();
    auto &ExtractMI = *getDefIgnoringCopies(EltReg, MRI);
    if (ExtractMI.getOpcode() != TargetOpcode::G_EXTRACT_VECTOR_ELT) {
      AllExtracts = false;
      break;
    }

    if (I == 1)
      SrcVecReg = ExtractMI.getOperand(1).getReg();
    else if (SrcVecReg != ExtractMI.getOperand(1).getReg()) {
      AllExtracts = false;
      break;
    }

    auto IndexReg = ExtractMI.getOperand(2).getReg();
    auto CValue = getIConstantVRegValWithLookThrough(IndexReg, MRI);
    if (!CValue.has_value() || CValue->Value != (I - 1)) {
      AllExtracts = false;
      break;
    }
  }
  if (AllExtracts) {
    if (MRI.getType(DstReg) != MRI.getType(SrcVecReg))
      return false;
    Replacement = SrcVecReg;
    return true;
  }

  // Branch B: all operands are the sequential defs of a single G_UNMERGE_VALUES
  // whose source vector type matches the G_BUILD_VECTOR dest type.
  unsigned NumElts = BuildVecMI.getNumOperands() - 1;
  auto FirstDefAndReg =
      getDefSrcRegIgnoringCopies(BuildVecMI.getOperand(1).getReg(), MRI);
  if (!FirstDefAndReg)
    return false;
  auto *UnmergeMI = FirstDefAndReg->MI;
  if (UnmergeMI->getOpcode() != TargetOpcode::G_UNMERGE_VALUES)
    return false;
  if (UnmergeMI->getNumOperands() - 1 != NumElts)
    return false; // G_UNMERGE_VALUES produces a different number of elements
  auto UnmergeSrcReg = UnmergeMI->getOperand(NumElts).getReg();
  if (MRI.getType(DstReg) != MRI.getType(UnmergeSrcReg))
    return false;

  for (unsigned I = 0; I < NumElts; I++) {
    auto DefAndReg =
        getDefSrcRegIgnoringCopies(BuildVecMI.getOperand(I + 1).getReg(), MRI);
    if (!DefAndReg || DefAndReg->MI != UnmergeMI)
      return false; // different producer
    if (DefAndReg->Reg != UnmergeMI->getOperand(I).getReg())
      return false; // out-of-order element pickup
  }

  Replacement = UnmergeSrcReg;
  return true;
}
void PISAPostLegalizerCombinerImpl::applyExtractAllToBuildVector(
    MachineInstr &MI, Register Replacement) const {
  auto &BuildVecMI = MI;
  auto DstReg = BuildVecMI.getOperand(0).getReg();
  B.buildCopy(DstReg, Replacement);
  MI.eraseFromParent();
}

// This function tries to match following pattern:
//   mask = ((1 << width) - 1) << offset;
//   dst = ((data << offset) & mask) | (base & ~mask);
//     -> bfi %data, %base, %width, %offset
// Currently only the variant with constant masks is supported.
bool PISAPostLegalizerCombinerImpl::matchOrAndToBfi(
    MachineInstr &OrMI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {
  // BFI supports only 32 bits of width, start with this simple check to
  // exclude unsupported shifts early.
  auto BitWidth = MRI.getType(OrMI.getOperand(0).getReg()).getSizeInBits();
  if (BitWidth != 32)
    return false;

  int64_t MaskA, MaskB;
  Register A, B;
  if (!mi_match(OrMI, MRI,
                m_GOr(m_GAnd(m_Reg(A), m_ICst(MaskA)),
                      m_GAnd(m_Reg(B), m_ICst(MaskB)))))
    return false;
  if (MaskA != ~MaskB)
    return false;

  Register ShiftedData, Base;
  unsigned Offset = 0, Width = 0;
  if (isShiftedMask_32(MaskA, Offset, Width)) {
    ShiftedData = A;
    Base = B;
  } else if (isShiftedMask_32(MaskB, Offset, Width)) {
    ShiftedData = B;
    Base = A;
  } else {
    return false;
  }

  auto [ShiftedDataMI, ShiftedDataRegIdx] =
      PISA::getDefIgnoringBitcasts(ShiftedData, MRI);

  Register Data;
  if (!mi_match(ShiftedDataMI, MRI,
                m_GShl(m_Reg(Data), m_SpecificICst(Offset))))
    return false;

  MatchInfo = [Data, Base, Width, Offset, &OrMI](MachineIRBuilder &B) {
    auto WidthReg = B.buildConstant(I32, Width);
    auto OffsetReg = B.buildConstant(I32, Offset);
    B.buildIntrinsic(Intrinsic::pisa_bfi, {OrMI.getOperand(0)})
        .addUse(Base)
        .addUse(Data)
        .addUse(WidthReg.getReg(0))
        .addUse(OffsetReg.getReg(0));
  };
  return true;
}

// This function tries to match following pattern:
//   diff = bitwidth - width;
//   shift = diff - offset
//   dst = ((data << shift) >> diff);
//     -> bfe %data, %width, %offset
bool PISAPostLegalizerCombinerImpl::matchShiftSubToBfe(
    MachineInstr &ShrMI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {
  // BFE supports only 32 bits of width, start with this simple check to
  // exclude unsupported shifts early.
  auto BitWidth = MRI.getType(ShrMI.getOperand(0).getReg()).getSizeInBits();
  if (BitWidth != 32)
    return false;

  auto MakeApplyHandler = [&ShrMI](Register Data, Register Width,
                                   Register Offset) {
    return [&ShrMI, Data, Width, Offset](MachineIRBuilder &B) {
      B.buildIntrinsic(Intrinsic::pisa_ubfe, {ShrMI.getOperand(0)})
          .addUse(Data)
          .addUse(Width)
          .addUse(Offset);
    };
  };

  constexpr int64_t ShiftInstMask = 0x1f;

  Register Data, Diff, Offset;
  if (mi_match(ShrMI, MRI,
               m_GLShr(m_GShl(m_Reg(Data), m_GSub(m_Reg(Diff), m_Reg(Offset))),
                       m_Reg(Diff)))) {
    Register Width;
    if (mi_match(Diff, MRI,
                 m_GSub(m_SpecificICst((int64_t)BitWidth), m_Reg(Width))) &&
        VT->getKnownBits(Width).countMaxTrailingOnes() <= 6 &&
        VT->getKnownBits(Offset).countMaxTrailingOnes() <= 5) {
      // s32 %bitwidth  = G_CONSTANT i32 32
      // s32 %andwidth  = G_AND i32 %width, <=63
      // s32 %andoffset = G_AND i32 %offset, <=31
      // s32 %diff      = G_SUB i32 %bitwidth, %andwidth
      // s32 %shift     = G_SUB i32 %diff, %andoffset
      // s32 %shl       = G_SHL i32 %data, %shift
      // s32 %shr       = G_LSHR i32 %shl, %diff
      // =>
      // bfe %data %andwidth %andoffset
      MatchInfo = MakeApplyHandler(Data, Width, Offset);
      return true;
    }
  } else if (mi_match(
                 ShrMI, MRI,
                 m_GLShr(m_GShl(m_Reg(Data),
                                m_GAnd(m_GSub(m_Reg(Diff), m_Reg(Offset)),
                                       m_SpecificICst(ShiftInstMask))),
                         m_GAnd(m_Reg(Diff), m_SpecificICst(ShiftInstMask))))) {
    Register Width;
    if (mi_match(Diff, MRI,
                 m_GSub(m_SpecificICst((int64_t)BitWidth), m_Reg(Width)))) {
      if (VT->getKnownBits(Width).countMaxTrailingOnes() <= 5) {
        // s32 %bitwidth = G_CONSTANT i32 32
        // s32 %andwidth = G_AND i32 %width, <=31
        // s32 %diff     = G_SUB i32 %bitwidth, %andwidth
        // s32 %shift    = G_SUB i32 %diff, %offset
        // s32 %andshift = G_AND i32 %shift, 31
        // s32 %shl      = G_SHL i32 %data, %andshift
        // s32 %anddiff  = G_AND i32 %diff, 31
        // s32 %shr      = G_LSHR i32 %shl, %anddiff
        // =>
        // s32 %andwidth = G_AND i32 %width, <=31
        // bfe %data %andwidth %offset
        MatchInfo = MakeApplyHandler(Data, Width, Offset);
        return true;
      }
      // Masking %diff with 0x1f is equivalent to masking %width with 0x1f.
      // We have to explicitly mask width with 0x1f to preserve behavior as
      // bfe masks width with 0x3f by default.
      // s32 %bitwidth = G_CONSTANT i32 32
      // s32 %diff     = G_SUB i32 %bitwidth, %width
      // s32 %shift    = G_SUB i32 %diff, %offset
      // s32 %andshift = G_AND i32 %shift, 31
      // s32 %shl      = G_SHL i32 %data, %andshift
      // s32 %anddiff  = G_AND i32 %diff, 31
      // s32 %shr      = G_LSHR i32 %shl, %anddiff
      // =>
      // s32 %andwidth = G_AND i32 %width, 31
      // bfe %data %andwidth %offset
      auto WidthType = MRI.getType(Width);
      MatchInfo = [Data, Width, WidthType, Offset,
                   &ShrMI](MachineIRBuilder &B) {
        auto Mask = B.buildConstant(WidthType, ShiftInstMask);
        auto MaskedWidth = B.buildAnd(WidthType, Width, Mask);
        B.buildIntrinsic(Intrinsic::pisa_ubfe, {ShrMI.getOperand(0)})
            .addUse(Data)
            .addUse(MaskedWidth->getOperand(0).getReg())
            .addUse(Offset);
      };
      return true;
    }
  }
  return false;
}

// s32 %maskedwidth = G_AND %width, 63
// s32 %maskedoffset = G_AND %offset, 31
// bfe %data %maskedwidth %maskedoffset
// =>
// bfe %data %width %offset
bool PISAPostLegalizerCombinerImpl::matchAndBitfieldToBitfield(
    MachineInstr &MI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {
  Register Data, Width, Offset;
  Intrinsic::ID II;
  if (auto *MIIntrinsic = dyn_cast<GIntrinsic>(&MI)) {
    II = MIIntrinsic->getIntrinsicID();
    if (II == Intrinsic::pisa_ubfe) {
      Data = MI.getOperand(2).getReg();
      Width = MI.getOperand(3).getReg();
      Offset = MI.getOperand(4).getReg();
    } else if (II == Intrinsic::pisa_bfi) {
      Data = MI.getOperand(2).getReg();
      Width = MI.getOperand(4).getReg();
      Offset = MI.getOperand(5).getReg();
    } else {
      return false;
    }
  } else {
    // G_[SU]BFX, convert it to pisa_[su]bfe
    II = MI.getOpcode() == TargetOpcode::G_UBFX ? Intrinsic::pisa_ubfe
                                                : Intrinsic::pisa_sbfe;
    Data = MI.getOperand(1).getReg();
    Width = MI.getOperand(3).getReg();
    Offset = MI.getOperand(2).getReg();
  }

  Register UnmaskedWidth{}, UnmaskedOffset{};
  if (VT->getKnownBits(Width).countMaxTrailingOnes() >= 6 &&
      mi_match(Width, MRI, m_GAnd(m_Reg(UnmaskedWidth), m_Reg()))) {
    Width = UnmaskedWidth;
  }
  if (VT->getKnownBits(Offset).countMaxTrailingOnes() >= 5 &&
      mi_match(Offset, MRI, m_GAnd(m_Reg(UnmaskedOffset), m_Reg()))) {
    Offset = UnmaskedOffset;
  }
  if (!UnmaskedWidth.isValid() && !UnmaskedOffset.isValid())
    return false; // No match, exit.

  if (II == Intrinsic::pisa_ubfe || II == Intrinsic::pisa_sbfe) {
    MatchInfo = [&MI, II, Data, Width, Offset](MachineIRBuilder &B) {
      B.buildIntrinsic(II, {MI.getOperand(0)})
          .addUse(Data)
          .addUse(Width)
          .addUse(Offset);
    };
  } else {
    MatchInfo = [&MI, Data, Width, Offset](MachineIRBuilder &B) {
      B.buildIntrinsic(Intrinsic::pisa_bfi, {MI.getOperand(0)})
          .addUse(Data)
          .addUse(MI.getOperand(3).getReg())
          .addUse(Width)
          .addUse(Offset);
    };
  }
  return true;
}

// This transforms an unoptimal pattern for fcmp + zext i1->i8 on <2 x i8>
// result types (e.g. from code that uses native i8 element vectors):
//  %14:_(<2 x s16>) = G_BITCAST %CmpRes:_(s32)
//  %20:_(s16), %21:_(s16) = G_UNMERGE_VALUES %14:_(<2 x s16>)
//  %22:_(s8) = G_TRUNC %20:_(s16)
//  %23:_(s8) = G_TRUNC %21:_(s16)
//  %17:_(<2 x s8>) = G_BUILD_VECTOR %22:_(s8), %23:_(s8)
//  %28:_(s16) = G_CONSTANT i16 257
//  %19:_(<2 x s8>) = G_BITCAST %28:_(s16)
//  %24:_(s16) = G_BITCAST %17:_(<2 x s8>)
//  %25:_(s16) = G_BITCAST %19:_(<2 x s8>)
//  %26:_(s16) = G_AND %24:_, %25:_
//  %res:_(<2 x s8>) = G_BITCAST %26:_(s16)
// => bfe %1, %CmpRes, 1, 0
//    bfe %2, %CmpRes, 1, 16
//    bfi %3, %1, %2, 8
//    trunc.16.32 %res, %3
// If the only user of the result is a G_STORE, we modify it to store
// an s16 instead of <2xs8>. Otherwise, we insert a G_BITCAST.
// Note: sub-byte types (i1, i4) are now promoted to i16, so this pattern
// applies only to code that explicitly uses native <2 x i8> element types.
bool PISAPostLegalizerCombinerImpl::matchV2i1ZextToBfeBfi(
    MachineInstr &BitcastMI, Register &BitcastInput) const {

  auto VectorElementSize =
      MRI.getType(BitcastMI.getOperand(0).getReg()).getScalarSizeInBits();
  // We only support an i8 zext
  if (VectorElementSize != 8)
    return false;

  // Match automatically up to the G_AND instructions.
  // The mask constant 257 (= 0x0101 = {1,1} as two i8 lanes) may appear in
  // two forms depending on whether build_vector_with_constants has already
  // folded G_BUILD_VECTOR(i8 1, i8 1) -> G_CONSTANT i16 257:
  //   original: G_BITCAST(G_BITCAST(G_CONSTANT i?? 257))
  //   folded:   G_CONSTANT i16 257  (bare constant, no bitcasts)
  Register UpperCmpRes, LowerCmpRes;
  bool Matched =
      mi_match(BitcastMI, MRI,
               m_GBitcast(m_GAnd(
                   m_GBitcast(m_GBuildVector(m_GTrunc(m_Reg(UpperCmpRes)),
                                             m_GTrunc(m_Reg(LowerCmpRes)))),
                   m_GBitcast(m_GBitcast(m_SpecificICst(257)))))) ||
      mi_match(BitcastMI, MRI,
               m_GBitcast(m_GAnd(
                   m_GBitcast(m_GBuildVector(m_GTrunc(m_Reg(UpperCmpRes)),
                                             m_GTrunc(m_Reg(LowerCmpRes)))),
                   m_SpecificICst(257))));
  if (!Matched)
    return false;

  /*
    Verify that
     - the extracted halves of the comparison result are part of
       G_UNMERGE_VALUES instructions
     - the extracted halves are at the correct position in the unmerge
       instructions, e.g. the UpperCmpRes reg should be at index 0
     - both G_UNMERGE_VALUES instructions use the same input
    Note that this way, we both support two unmerges with two unused values (see
    code example), and one unmerge instruction with both values being used
    (i.e. UpperUnmerge =?= LowerUnmerge)
  */
  auto *UpperUnmergeInstr = MRI.getUniqueVRegDef(UpperCmpRes);
  auto *LowerUnmergeInstr = MRI.getUniqueVRegDef(LowerCmpRes);
  if (!UpperUnmergeInstr || !LowerUnmergeInstr ||
      UpperUnmergeInstr->getOpcode() != TargetOpcode::G_UNMERGE_VALUES ||
      LowerUnmergeInstr->getOpcode() != TargetOpcode::G_UNMERGE_VALUES)
    return false;

  if (UpperUnmergeInstr->getOperand(0).getReg() != UpperCmpRes)
    return false;

  if (LowerUnmergeInstr->getOperand(1).getReg() != LowerCmpRes)
    return false;

  auto BitcastRes = UpperUnmergeInstr->getOperand(2);
  if (!BitcastRes.isIdenticalTo(LowerUnmergeInstr->getOperand(2)))
    return false;

  // Verify that this inst is a G_BITCAST, and get the input register
  auto *BitcastInst = MRI.getUniqueVRegDef(BitcastRes.getReg());
  if (!BitcastInst || BitcastInst->getOpcode() != TargetOpcode::G_BITCAST)
    return false;

  BitcastInput = BitcastInst->getOperand(1).getReg();
  return true;
}

void PISAPostLegalizerCombinerImpl::applyV2i1ZextToBfeBfi(
    MachineInstr &BitcastMI, Register BitcastInput) const {

  auto *MRI = B.getMRI();
  auto Bfe1Res = MRI->createGenericVirtualRegister(I32);
  auto Bfe2Res = MRI->createGenericVirtualRegister(I32);
  auto BfiRes = MRI->createGenericVirtualRegister(I32);

  auto Zero = B.buildConstant(I32, 0);
  auto One = B.buildConstant(I32, 1);
  auto Eight = B.buildConstant(I32, 8);
  auto Sixteen = B.buildConstant(I32, 16);

  B.buildIntrinsic(Intrinsic::pisa_ubfe, {DstOp(Bfe1Res)})
      .addUse(BitcastInput)
      .addUse(One.getReg(0))   // Width
      .addUse(Zero.getReg(0)); // Offset

  B.buildIntrinsic(Intrinsic::pisa_ubfe, {DstOp(Bfe2Res)})
      .addUse(BitcastInput)
      .addUse(One.getReg(0))      // Width
      .addUse(Sixteen.getReg(0)); // Offset

  B.buildIntrinsic(Intrinsic::pisa_bfi, {DstOp(BfiRes)})
      .addUse(Bfe1Res)
      .addUse(Bfe2Res)
      .addUse(Eight.getReg(0))  // Width
      .addUse(Eight.getReg(0)); // Offset

  /*
    Peephole optimisation:
    The example input given above continues with a G_STORE instr
    as the only user of our newly created G_TRUNC:
      [...]
      %52:_(s8) = G_TRUNC %48:_(s16)
      %10:_(<2 x s8>) = G_BUILD_VECTOR %51:_(s8), %52:_(s8)
      G_STORE %10:_(s16), %4:_(p1)

    If we would just bitcast the G_TRUNC result back to
    <2 x s8> to match all expected types in that case, the resulting
    PISA code inserts an unnecessary mov into a different register

    To avoid that, check if the next user is only a G_STORE instruction.
    In that case, we can just modify the type of the stored register
    to s16 and change the MachineMemOperand's type to s16 as well.

    In all other cases, insert a G_BITCAST to <2 x s8>.
  */

  auto StoreInput = BitcastMI.getOperand(0).getReg();
  if (MRI->hasOneNonDBGUse(StoreInput) &&
      MRI->use_instr_nodbg_begin(StoreInput)->getOpcode() ==
          TargetOpcode::G_STORE) {
    // Our only use is a G_STORE
    B.buildInstr(TargetOpcode::G_TRUNC).addDef(StoreInput).addUse(BfiRes);

    MRI->setType(StoreInput, I16);
    for (auto *MemOp : MRI->use_instr_nodbg_begin(StoreInput)->memoperands())
      MemOp->setType(I16);
  } else {
    // We either have multiple users or the following user is not a G_STORE
    // Insert a bitcast to <2 x s8> so that nothing breaks
    auto TruncRes = MRI->createGenericVirtualRegister(I16);
    B.buildInstr(TargetOpcode::G_TRUNC).addDef(TruncRes).addUse(BfiRes);

    B.buildInstr(TargetOpcode::G_BITCAST).addDef(StoreInput).addUse(TruncRes);
  }
  BitcastMI.eraseFromParent();
}

// %374:_(s16) = G_CONSTANT i16 1
// %375:_(s16) = G_AND %321:_, %374:_
// %421:_(s16) = G_CONSTANT i16 0
// %422:_(s16) = G_CONSTANT i16 1
// %405:_(s16) = G_PISA_SELECT %375:_, %422:_, %421:_
// => %374:_(s16) = G_CONSTANT i16 1
// => %375:_(s16) = G_AND %321:_, %374:_
// => %405:_(s16) = COPY %375
bool PISAPostLegalizerCombinerImpl::matchAndSelect(
    MachineInstr &MI, Register &Replacement) const {
  auto &SelectMI = MI;

  auto Const0 =
      getIConstantVRegValWithLookThrough(SelectMI.getOperand(3).getReg(), MRI);
  auto Const1 =
      getIConstantVRegValWithLookThrough(SelectMI.getOperand(2).getReg(), MRI);
  if (!Const0.has_value() || !Const1.has_value())
    return false;
  if ((Const0->Value != 0) || (Const1->Value != 1))
    return false;

  auto &AndMI = *getDefIgnoringCopies(SelectMI.getOperand(1).getReg(), MRI);
  if (AndMI.getOpcode() != TargetOpcode::G_AND)
    return false;
  auto AndConst =
      getIConstantVRegValWithLookThrough(AndMI.getOperand(2).getReg(), MRI);
  if (!AndConst.has_value() || (AndConst->Value != 1))
    return false;

  if (MRI.getType(AndMI.getOperand(0).getReg()) !=
      MRI.getType(SelectMI.getOperand(0).getReg()))
    return false;

  Replacement = AndMI.getOperand(0).getReg();
  return true;
}
void PISAPostLegalizerCombinerImpl::applyAndSelect(MachineInstr &MI,
                                                   Register Replacement) const {

  B.buildCopy(MI.getOperand(0).getReg(), Replacement);
  MI.eraseFromParent();
}

// %23:_(s32) = G_CONSTANT i32 8
// %17:_(s8) = G_TRUNC %1:reg32b(s32)
// %24:_(s32) = G_LSHR %1:reg32b, %23:_(s32)
// %25:_(s8) = G_TRUNC %24:_(s32)
// %28:_(s8) = G_TRUNC %2:reg32b(s32)
// %31:_(s32) = G_LSHR %2:reg32b, %23:_(s32)
// %32:_(s8) = G_TRUNC %31:_(s32)
///%149:_(<4 x s8>) = G_BUILD_VECTOR %17:_(s8), %25:_(s8), %28:_(s8), %32:_(s8)
// %141:_(s32) = G_BITCAST %149:_(<4 x s8>)
// => %A = G_AND %1, 0xFFFF
// => %B = G_SHL %2, 16
// => %141 = G_OR %A, %B
bool PISAPostLegalizerCombinerImpl::matchBuildRegFrom2(
    MachineInstr &MI, Reg2MatchInfo &MatchInfo) const {
  auto &BitcastMI = MI;

  auto DstTy = MRI.getType(BitcastMI.getOperand(0).getReg());
  if (DstTy.isVector() || (DstTy.getScalarSizeInBits() != 32))
    return false;
  auto SrcReg = BitcastMI.getOperand(1).getReg();
  auto SrcTy = MRI.getType(SrcReg);
  if (!SrcTy.isVector() || (SrcTy.getScalarSizeInBits() != 8))
    return false;
  auto &BuildVecMI = *getDefIgnoringCopies(SrcReg, MRI);
  if (BuildVecMI.getOpcode() != TargetOpcode::G_BUILD_VECTOR)
    return false;

  for (auto I = 0; I < 2; I++) {
    auto LoReg = BuildVecMI.getOperand(1 + (I * 2)).getReg();
    auto HiReg = BuildVecMI.getOperand(2 + (I * 2)).getReg();
    auto &LoTruncMI = *getDefIgnoringCopies(LoReg, MRI);
    auto &HiTruncMI = *getDefIgnoringCopies(HiReg, MRI);
    if (LoTruncMI.getOpcode() != TargetOpcode::G_TRUNC)
      return false;
    if (HiTruncMI.getOpcode() != TargetOpcode::G_TRUNC)
      return false;
    auto &HiShiftMI =
        *getDefIgnoringCopies(HiTruncMI.getOperand(1).getReg(), MRI);
    if (HiShiftMI.getOpcode() != TargetOpcode::G_LSHR)
      return false;
    if (LoTruncMI.getOperand(1).getReg() != HiShiftMI.getOperand(1).getReg())
      return false;
    auto ShiftConst = getIConstantVRegValWithLookThrough(
        HiShiftMI.getOperand(2).getReg(), MRI);
    if (!ShiftConst.has_value() || (ShiftConst->Value != 8))
      return false;
    if (I == 0)
      MatchInfo.Reg0 = LoTruncMI.getOperand(1).getReg();
    else
      MatchInfo.Reg1 = LoTruncMI.getOperand(1).getReg();
  }
  return true;
}
void PISAPostLegalizerCombinerImpl::applyBuildRegFrom2(
    MachineInstr &MI, Reg2MatchInfo &MatchInfo) const {
  auto MaskReg = MRI.createGenericVirtualRegister(LLT::integer(32));
  auto Mask = B.buildConstant(MaskReg, 0xFFFF);
  auto ShiftReg = MRI.createGenericVirtualRegister(LLT::integer(32));
  auto Shift = B.buildConstant(ShiftReg, 16);

  auto AReg = MRI.createGenericVirtualRegister(LLT::integer(32));
  auto BReg = MRI.createGenericVirtualRegister(LLT::integer(32));

  auto RegSize = MRI.getType(MatchInfo.Reg0).getSizeInBits();
  assert((RegSize == 16 || RegSize == 32 || RegSize == 64) &&
         "only supporting 16/32/64bit registers");
  Register ASrcReg, BSrcReg;
  if (RegSize == 16) {
    ASrcReg = MRI.createGenericVirtualRegister(LLT::integer(32));
    BSrcReg = MRI.createGenericVirtualRegister(LLT::integer(32));
    B.buildZExt(ASrcReg, MatchInfo.Reg0);
    B.buildZExt(BSrcReg, MatchInfo.Reg1);
  } else if (RegSize == 64) {
    ASrcReg = MRI.createGenericVirtualRegister(LLT::integer(32));
    BSrcReg = MRI.createGenericVirtualRegister(LLT::integer(32));
    B.buildTrunc(ASrcReg, MatchInfo.Reg0);
    B.buildTrunc(BSrcReg, MatchInfo.Reg1);
  } else {
    ASrcReg = MatchInfo.Reg0;
    BSrcReg = MatchInfo.Reg1;
  }
  B.buildAnd(AReg, ASrcReg, Mask);
  B.buildShl(BReg, BSrcReg, Shift);
  B.buildOr(MI.getOperand(0).getReg(), AReg, BReg);
  MI.eraseFromParent();
}

// Match a G_BUILD_VECTOR whose two elements are consecutive lanes (base,
// base+1) of a single wider source vector, produced by G_UNMERGE_VALUES.
// Such a build is really a sub-vector slice of that source; recording
// (source, base) lets applyBuildVectorFromUnmergeLanes rewrite it to a
// G_EXTRACT_SUBVECTOR, which ISel lowers to a composite sub-register COPY.
bool PISAPostLegalizerCombinerImpl::matchBuildVectorFromUnmergeLanes(
    MachineInstr &MI, std::tuple<Register, int64_t> &MatchInfo) const {
  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);
  // Only nameable 2-element pairs (map to .xy / .zw) are handled.
  if (!DstTy.isVector() || DstTy.getNumElements() != 2)
    return false;
  const unsigned NumElts = 2;

  const MachineInstr *Unmerge = nullptr;
  Register SrcReg;
  int64_t BaseLane = -1;
  for (unsigned I = 0; I < NumElts; ++I) {
    auto Def = getDefSrcRegIgnoringCopies(MI.getOperand(1 + I).getReg(), MRI);
    if (!Def)
      return false;
    const MachineInstr *D = Def->MI;
    if (D->getOpcode() != TargetOpcode::G_UNMERGE_VALUES)
      return false;
    // The source operand of G_UNMERGE_VALUES is its last operand; the ones
    // before it are the per-lane results.
    const unsigned NumLanes = D->getNumOperands() - 1;
    int Lane = -1;
    for (unsigned J = 0; J < NumLanes; ++J)
      if (D->getOperand(J).getReg() == Def->Reg) {
        Lane = (int)J;
        break;
      }
    if (Lane < 0)
      return false;
    if (I == 0) {
      Unmerge = D;
      SrcReg = D->getOperand(NumLanes).getReg();
      BaseLane = Lane;
    } else if (D != Unmerge || Lane != BaseLane + (int)I) {
      return false;
    }
  }

  // Base must form a nameable composite sub-register (.xy at 0, .zw at 2).
  if (BaseLane != 0 && BaseLane != 2)
    return false;

  LLT SrcTy = MRI.getType(SrcReg);
  // The .xy/.zw composite sub-registers are only defined on 4-lane register
  // classes (Reg*bx4). On wider vectors (v5-v8, v16, ...) the elements have
  // per-lane sub-registers only; a .zw view there is not a real sub-register
  // and would be mis-lowered. Restrict to <=4-lane sources (the intended
  // 3-4-element scope) and let wider vectors keep the element-wise path.
  if (!SrcTy.isVector() || SrcTy.getElementType() != DstTy.getElementType() ||
      SrcTy.getNumElements() <= NumElts || SrcTy.getNumElements() > 4 ||
      (uint64_t)BaseLane + NumElts > SrcTy.getNumElements())
    return false;

  MatchInfo = std::make_tuple(SrcReg, BaseLane);
  return true;
}

void PISAPostLegalizerCombinerImpl::applyBuildVectorFromUnmergeLanes(
    MachineInstr &MI, std::tuple<Register, int64_t> &MatchInfo) const {
  auto [SrcReg, BaseLane] = MatchInfo;
  B.buildExtractSubvector(MI.getOperand(0).getReg(), SrcReg, BaseLane);
  MI.eraseFromParent();
}

// Match a 4-element G_BUILD_VECTOR that concatenates two whole 2-element
// sub-vectors: elements [0,1] are lanes 0,1 of source A and elements [2,3] are
// lanes 0,1 of source B (each produced by G_UNMERGE_VALUES). Such a build is a
// concatenation; recording (A, B) lets the apply rewrite it to
// G_CONCAT_VECTORS, which ISel writes into the .xy / .zw slices directly.
bool PISAPostLegalizerCombinerImpl::matchBuildVectorConcatSubvectors(
    MachineInstr &MI, SmallVector<Register, 4> &MatchInfo) const {
  Register DstReg = MI.getOperand(0).getReg();
  LLT DstTy = MRI.getType(DstReg);
  if (!DstTy.isVector() || DstTy.getNumElements() != 4)
    return false;
  const unsigned M = 2; // sub-vector width -> nameable .xy / .zw
  const unsigned K = DstTy.getNumElements() / M;
  MatchInfo.clear();
  for (unsigned G = 0; G < K; ++G) {
    const MachineInstr *Unmerge = nullptr;
    Register SrcReg;
    for (unsigned J = 0; J < M; ++J) {
      auto Def = getDefSrcRegIgnoringCopies(
          MI.getOperand(1 + G * M + J).getReg(), MRI);
      if (!Def)
        return false;
      const MachineInstr *D = Def->MI;
      if (D->getOpcode() != TargetOpcode::G_UNMERGE_VALUES)
        return false;
      const unsigned NumLanes = D->getNumOperands() - 1;
      int Lane = -1;
      for (unsigned L = 0; L < NumLanes; ++L)
        if (D->getOperand(L).getReg() == Def->Reg) {
          Lane = (int)L;
          break;
        }
      // Each source must be consumed whole and in order (lane J at position J).
      if (Lane != (int)J)
        return false;
      if (J == 0) {
        Unmerge = D;
        SrcReg = D->getOperand(NumLanes).getReg();
      } else if (D != Unmerge) {
        return false;
      }
    }
    LLT SrcTy = MRI.getType(SrcReg);
    if (!SrcTy.isVector() || SrcTy.getNumElements() != M ||
        SrcTy.getElementType() != DstTy.getElementType())
      return false;
    MatchInfo.push_back(SrcReg);
  }
  return MatchInfo.size() == K;
}

void PISAPostLegalizerCombinerImpl::applyBuildVectorConcatSubvectors(
    MachineInstr &MI, SmallVector<Register, 4> &MatchInfo) const {
  B.buildConcatVectors(MI.getOperand(0).getReg(), MatchInfo);
  MI.eraseFromParent();
}

// if NumSignBits == BitWidth:
//  %24:_(s32) = G_CONSTANT i32 0
//  %25:_(s32) = G_CONSTANT i32 1
//  %26:_(s32) = G_AND %18:_, %25:_
//  %9:_(s1) = G_ICMP intpred(ne), %26:_(s32), %24:_
// => %24:_(s32) = G_CONSTANT i32 0
//    %26:_(s32) = COPY %18:_(s32)
//    %9:_(s1) = G_ICMP intpred(ne), %26:_(s32), %24:_
bool PISAPostLegalizerCombinerImpl::matchCmpAndAllOnes(
    MachineInstr &MI, GISelValueTracking *VT, Register &MatchInfo) const {
  Register SrcReg;
  CmpInst::Predicate Pred;
  if (!mi_match(MI, MRI,
                m_GICmp(m_Pred(Pred), m_GAnd(m_Reg(SrcReg), m_SpecificICst(1)),
                        m_SpecificICst(0))) ||
      (Pred != CmpInst::ICMP_NE))
    return false;

  auto BitWidth = MRI.getType(SrcReg).getSizeInBits();
  if (BitWidth < 16)
    return false;

  if (VT->computeNumSignBits(SrcReg) < BitWidth)
    return false;

  MatchInfo = SrcReg;
  return true;
}
void PISAPostLegalizerCombinerImpl::applyCmpAndAllOnes(
    MachineInstr &MI, Register &MatchInfo) const {
  Observer.changingInstr(MI);
  MI.getOperand(2).setReg(MatchInfo);
  Observer.changedInstr(MI);
}

//  %11:_(s16) = G_TRUNC %8:_(s32)
//  %12:_(s16) = G_CONSTANT i16 1
//  %13:_(s16) = G_AND %11:_, %12:_
//=> %12:_(s32) = G_CONSTANT i32 1
//   %13:_(s32) = G_AND %8:_, %12:_
//   %11:_(s16) = G_TRUNC %13:_(s32)
bool PISAPostLegalizerCombinerImpl::matchFixIllegalShiftAmt(
    MachineInstr &MI) const {
  assert(MI.getOpcode() == TargetOpcode::G_SHL ||
         MI.getOpcode() == TargetOpcode::G_LSHR ||
         MI.getOpcode() == TargetOpcode::G_ASHR);
  return MRI.getType(MI.getOperand(2).getReg()) != I32;
}

void PISAPostLegalizerCombinerImpl::applyFixIllegalShiftAmt(
    MachineInstr &MI) const {
  Register ShAmtReg = MI.getOperand(2).getReg();
  LLT ShAmtTy = MRI.getType(ShAmtReg);
  MachineIRBuilder MIB(MI);
  Register NewShAmt = (ShAmtTy.getSizeInBits() > 32)
                          ? MIB.buildTrunc(I32, ShAmtReg).getReg(0)
                          : MIB.buildZExt(I32, ShAmtReg).getReg(0);
  Observer.changingInstr(MI);
  MI.getOperand(2).setReg(NewShAmt);
  Observer.changedInstr(MI);
}

bool PISAPostLegalizerCombinerImpl::matchSinkTrunc(
    MachineInstr &MI, std::tuple<Register, int64_t> &MatchInfo) const {
  const unsigned Opcode = MI.getOpcode();
  assert(Opcode == TargetOpcode::G_AND || Opcode == TargetOpcode::G_OR ||
         Opcode == TargetOpcode::G_XOR);

  int64_t Mask;
  Register SrcReg;
  if (!mi_match(MI, MRI,
                m_BinOp(Opcode, m_GTrunc(m_Reg(SrcReg)), m_ICst(Mask))))
    return false;

  // We don't want to create 64-bit boolean operations
  auto BitWidth = MRI.getType(SrcReg).getSizeInBits();
  if (BitWidth > 32)
    return false;

  MatchInfo = std::make_tuple(SrcReg, Mask);
  return true;
}
void PISAPostLegalizerCombinerImpl::applySinkTrunc(
    MachineInstr &MI, std::tuple<Register, int64_t> &MatchInfo) const {
  int64_t Mask;
  Register SrcReg;
  std::tie(SrcReg, Mask) = MatchInfo;
  auto SrcTy = MRI.getType(SrcReg);
  auto MaskReg = MRI.createGenericVirtualRegister(SrcTy);
  auto DstReg = MRI.createGenericVirtualRegister(SrcTy);
  B.buildConstant(MaskReg, Mask);
  B.buildInstr(MI.getOpcode()).addDef(DstReg).addUse(SrcReg).addUse(MaskReg);
  B.buildInstr(TargetOpcode::G_TRUNC)
      .addDef(MI.getOperand(0).getReg())
      .addUse(DstReg);
  MI.eraseFromParent();
}

//  %22:_(s16) = G_TRUNC %26:_(s32)
//  %2:_(s8) = G_TRUNC %22:_(s16)
//=> %2:_(s8) = G_TRUNC %26:_(s32)
bool PISAPostLegalizerCombinerImpl::matchTruncTrunc(MachineInstr &MI,
                                                    Register &MatchInfo) const {
  Register SrcReg;
  if (!mi_match(MI, MRI, m_GTrunc(m_GTrunc(m_Reg(SrcReg)))))
    return false;

  MatchInfo = SrcReg;
  return true;
}
void PISAPostLegalizerCombinerImpl::applyTruncTrunc(MachineInstr &MI,
                                                    Register &MatchInfo) const {
  Observer.changingInstr(MI);
  MI.getOperand(1).setReg(MatchInfo);
  Observer.changedInstr(MI);
}

// %1:registers(s32) = G_ABS %0:reg32b
// %4:registers(s32) = G_INTRINSIC_CONVERGENT intrinsic(@llvm.pisa.ired),
// IRedOp::UMAX, %1:registers(s32), %2:reg32b(s32), %3:reg32b(s32)
// => %1:registers(s32) = G_INTRINSIC_CONVERGENT intrinsic(@llvm.pisa.ired),
// IRedOp::ABSMAX, %0:registers(s32), %2:reg32b(s32), %3:reg32b(s32)
bool PISAPostLegalizerCombinerImpl::matchAbsRedMaxToRedAbsMax(
    MachineInstr &MI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {
  unsigned AbsOpcode;
  unsigned TargetInstOpCode = 0;
  unsigned TargetOpType = 0;
  unsigned IntrID = 0;
  switch (MI.getOpcode()) {
  default:
    return false;
  case TargetOpcode::G_INTRINSIC_CONVERGENT: {
    auto II = cast<GIntrinsic>(MI).getIntrinsicID();
    if (II == Intrinsic::pisa_ired) {
      if (MI.getOperand(2).getImm() != pisa::IRedOp::UMAX)
        return false;
      AbsOpcode = TargetOpcode::G_ABS;
      IntrID = Intrinsic::pisa_ired;
      TargetOpType = pisa::IRedOp::ABSMAX;
    } else if (II == Intrinsic::pisa_fred) {
      if (MI.getOperand(2).getImm() != pisa::FRedOp::MAX)
        return false;
      AbsOpcode = TargetOpcode::G_INTRINSIC;
      IntrID = Intrinsic::pisa_fred;
      TargetOpType = pisa::FRedOp::ABSMAX;
    } else
      return false;
    break;
  }
  }
  unsigned SrcOpIdx = IntrID ? 3 : 2;
  auto [SrcMI, SrcRegIdx] =
      PISA::getDefIgnoringBitcasts(MI.getOperand(SrcOpIdx).getReg(), MRI);
  MachineInstr *SrcMIPtr = SrcMI;
  if (SrcMIPtr->getOpcode() != AbsOpcode)
    return false;
  if (AbsOpcode == TargetOpcode::G_INTRINSIC &&
      cast<GIntrinsic>(*SrcMIPtr).getIntrinsicID() != Intrinsic::pisa_fabs)
    return false;
  // For G_INTRINSIC, source reg is operand 2; for generic opcodes, operand 1.
  unsigned AbsSrcIdx = AbsOpcode == TargetOpcode::G_INTRINSIC ? 2 : 1;
  if (IntrID)
    MatchInfo = [SrcMIPtr, AbsSrcIdx, &MI, IntrID,
                 TargetOpType](MachineIRBuilder &B) {
      auto MIB = B.buildIntrinsic(IntrID, {MI.getOperand(0)});
      MIB.addImm(TargetOpType);
      MIB.addUse(SrcMIPtr->getOperand(AbsSrcIdx).getReg());
      MIB.addUse(MI.getOperand(4).getReg());
      MIB.addUse(MI.getOperand(5).getReg());
      // pisa_fred carries a trailing i1 nanp operand; pisa_ired does not.
      if (IntrID == Intrinsic::pisa_fred)
        MIB.add(MI.getOperand(6)); // copy nanp from original
      MIB.setMIFlags(MI.getFlags());
    };
  else
    MatchInfo = [SrcMIPtr, &MI, TargetInstOpCode,
                 TargetOpType](MachineIRBuilder &B) {
      B.buildInstr(TargetInstOpCode)
          .addDef(MI.getOperand(0).getReg())
          .addImm(TargetOpType)
          .addUse(SrcMIPtr->getOperand(1).getReg())
          .addUse(MI.getOperand(3).getReg())
          .addUse(MI.getOperand(4).getReg())
          .add(MI.getOperand(5)) // copy nanp from original
          .setMIFlags(MI.getFlags());
    };
  return true;
}

// %45:_(s32) = G_CONSTANT i32 -1
// %60:_(s32) = G_CONSTANT i32 0
// %46:_(s32) = G_SELECT %10:_(s1), %45:_, %60:_
// %47:_(s32) = G_SELECT %35:_(s1), %45:_, %60:_
// %48:_(s32) = G_AND %46:_, %47:_
// %59:_(s32) = G_CONSTANT i32 31
// %58:_(s32) = G_SHL %48:_, %59:_(s32)
// %14:_(s32) = G_ASHR %58:_, %59:_(s32)
// => %14:_(s32) = G_AND %46:_, %47:_
bool PISAPostLegalizerCombinerImpl::matchShiftTrueFalse(
    MachineInstr &AShrMI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {

  // Result can be only all zeros or all ones. But if sources are also only
  // all zeros/ones, then shl/ashr is a redundant artifact from legalization.

  auto DstReg = AShrMI.getOperand(0).getReg();
  auto DstTy = MRI.getType(DstReg);
  auto ShiftAmount = DstTy.getScalarSizeInBits() - 1;

  Register SrcReg;
  if (!mi_match(AShrMI, MRI,
                m_GAShr(m_GShl(m_Reg(SrcReg), m_SpecificICst(ShiftAmount)),
                        m_SpecificICst(ShiftAmount))))
    return false;

  std::function<bool(Register)> Match = [&](Register Reg) {
    auto CValue = getIConstantVRegValWithLookThrough(Reg, MRI);
    if (CValue.has_value())
      return CValue->Value.isZero() || CValue->Value.isAllOnes();

    auto *MI = getDefIgnoringCopies(Reg, MRI);
    if (!MI)
      return false;

    switch (MI->getOpcode()) {
    case TargetOpcode::G_TRUNC:
      return Match(MI->getOperand(1).getReg());
    case TargetOpcode::G_AND:
    case TargetOpcode::G_OR:
    case TargetOpcode::G_XOR:
      return Match(MI->getOperand(1).getReg()) &&
             Match(MI->getOperand(2).getReg());
    case TargetOpcode::G_SELECT:
    case PISA::G_PISA_SELECT:
      return Match(MI->getOperand(2).getReg()) &&
             Match(MI->getOperand(3).getReg());
    default:
      return false;
    }
  };

  if (!Match(SrcReg))
    return false;

  MatchInfo = [DstReg, SrcReg, this](MachineIRBuilder &B) {
    if (MRI.hasOneNonDBGUse(SrcReg)) {
      auto *MI = getDefIgnoringCopies(SrcReg, MRI);
      if (MI) {
        MI->getOperand(0).setReg(DstReg);
        return;
      }
    }

    B.buildCopy(DstReg, SrcReg);
  };
  return true;
}

// %54:_(s8), %55:_(s8), %56:_(s8), %57:_(s8) = G_UNMERGE_VALUES %21:_(<4 x s8>)
// %39:_(s16) = G_ANYEXT %54:_(s8)
// %40:_(s16) = G_ANYEXT %55:_(s8)
// %41:_(s16) = G_ADD %39:_, %40:_
// %36:_(s16) = G_ANYEXT %56:_(s8)
// %37:_(s16) = G_ANYEXT %57:_(s8)
// %38:_(s16) = G_ADD %36:_, %37:_
// %35:_(s16) = G_ADD %41:_, %38:_
// => %40:_(s32) = G_INTRINSIC intrinsic(@llvm.pisa.dp4a.uu), 0, %21, 0x01010101
//    %35:_(s16) = G_TRUNC %40:_(s32)
bool PISAPostLegalizerCombinerImpl::matchAddInt8Reduction(
    MachineInstr &MI,
    std::function<void(MachineIRBuilder &)> &MatchInfo) const {

  auto Dst = MI.getOperand(0).getReg();
  auto DstTy = MRI.getType(Dst);
  if (!DstTy.isScalar() || DstTy.getSizeInBits() > 32)
    return false;

  std::array<Register, 4> Srcs;
  // Helper to match anyext or zext of a register
  auto MatchExt = [](Register &Reg) {
    return m_any_of(m_GAnyExt(m_Reg(Reg)), m_GZExt(m_Reg(Reg)));
  };
  if (!mi_match(MI, MRI,
                m_any_of(m_GAdd(m_OneNonDBGUse(m_GAdd(MatchExt(Srcs[0]),
                                                      MatchExt(Srcs[1]))),
                                m_OneNonDBGUse(m_GAdd(MatchExt(Srcs[2]),
                                                      MatchExt(Srcs[3])))),
                         m_GAdd(m_OneNonDBGUse(m_GAdd(
                                    m_OneNonDBGUse(m_GAdd(MatchExt(Srcs[0]),
                                                          MatchExt(Srcs[1]))),
                                    MatchExt(Srcs[2]))),
                                MatchExt(Srcs[3])))))
    return false;

  // All registers must be unique.
  SmallSet<Register, 4> Unique(Srcs.begin(), Srcs.end());
  if (Unique.size() < Srcs.size())
    return false;

  // All register must be i8 from the same unmerge instruction.
  auto GetUnmerge = [=](Register &Reg) {
    auto *UnmergeMI = MRI.getVRegDef(Reg);
    return UnmergeMI->getOpcode() == TargetOpcode::G_UNMERGE_VALUES ? UnmergeMI
                                                                    : nullptr;
  };

  MachineInstr *UnmergeMI = GetUnmerge(Srcs[0]);
  if (!UnmergeMI)
    return false;
  for (unsigned I = 1; I < Srcs.size(); ++I) {
    if (UnmergeMI != GetUnmerge(Srcs[I]))
      return false;
  }

  auto VectorReg =
      UnmergeMI->getOperand(UnmergeMI->getNumOperands() - 1).getReg();
  if (MRI.getType(VectorReg) !=
      LLT::vector(ElementCount::getFixed(4), LLT::integer(8)))
    return false;

  MatchInfo = [=](MachineIRBuilder &B) {
    auto Acc = MRI.createGenericVirtualRegister(I32);
    auto X = MRI.createGenericVirtualRegister(I32);
    auto Y = MRI.createGenericVirtualRegister(I32);

    B.buildConstant(Acc, 0);
    B.buildBitcast(X, VectorReg);
    B.buildConstant(Y, 0x01010101);

    // Check if dst can be directly used, or result must be truncated.
    if (DstTy == I32) {
      B.buildIntrinsic(Intrinsic::pisa_dp4a_uu, {DstOp(Dst)})
          .addUse(Acc)
          .addUse(X)
          .addUse(Y)
          .addImm(0);
    } else {
      auto Dst32 = MRI.createGenericVirtualRegister(I32);
      B.buildIntrinsic(Intrinsic::pisa_dp4a_uu, {DstOp(Dst32)})
          .addUse(Acc)
          .addUse(X)
          .addUse(Y)
          .addImm(0);
      B.buildTrunc(Dst, Dst32);
    }
  };
  return true;
}

// %370:_(i16), %371:_(i16) = G_UNMERGE_VALUES %366:_(<2 x i16>)
// %378:_(f16) = G_BITCAST %370:_(i16)
// %379:_(f16) = G_BITCAST %371:_(i16)
// %37:_(<2 x f16>) = G_BUILD_VECTOR %378:_(f16), %379:_(f16)
// => %37:_(<2 x f16>) = G_BITCAST %366:_(<2 x i16>)
bool PISAPostLegalizerCombinerImpl::matchUnmergeBitcastBuildVectorToBitcast(
    MachineInstr &MI, Register &MatchInfo) const {
  Register UnmergeSrc;
  for (unsigned I = 1, E = MI.getNumOperands(); I < E; I++) {
    auto [SrcMI, SrcRegIdx] =
        PISA::getDefIgnoringBitcasts(MI.getOperand(I).getReg(), MRI, true);
    if (!SrcMI || SrcMI->getOpcode() != TargetOpcode::G_UNMERGE_VALUES)
      return false;
    if (SrcMI->getNumOperands() != MI.getNumOperands())
      return false;
    auto SrcReg = SrcMI->getOperand(SrcMI->getNumOperands() - 1).getReg();
    if (SrcRegIdx != (I - 1))
      return false;
    if (I == 1)
      UnmergeSrc = SrcReg;
    else if (UnmergeSrc != SrcReg)
      return false;
  }
  MatchInfo = UnmergeSrc;
  return true;
}
void PISAPostLegalizerCombinerImpl::applyUnmergeBitcastBuildVectorToBitcast(
    MachineInstr &MI, Register &MatchInfo) const {
  auto DstReg = MI.getOperand(0).getReg();
  if (MRI.getType(DstReg) != MRI.getType(MatchInfo))
    B.buildBitcast(MI.getOperand(0).getReg(), MatchInfo);
  else
    B.buildCopy(MI.getOperand(0).getReg(), MatchInfo);
  MI.eraseFromParent();
}

// Return true if the G_FENCE corresponds to a subgroup or workitem scope.
// These are execution barriers, not memory ordering fences, and should
// be preserved.
static bool isSubgroupFence(const MachineInstr &MI) {
  auto ScopeID = static_cast<SyncScope::ID>(MI.getOperand(1).getImm());
  if (ScopeID == SyncScope::SingleThread)
    return true;
  auto &Ctx = MI.getMF()->getFunction().getContext();
  if (auto Name = Ctx.getSyncScopeName(ScopeID))
    return Name->starts_with("subgroup") || Name->starts_with("workitem");
  return false;
}

// Extract the base scope (e.g. "workgroup") from a sync scope name that may
// include an address space suffix (e.g. "workgroup-shared", "workgroup-global",
// "workgroup-generic", or just "workgroup").
static StringRef getBaseScopeName(const MachineInstr &MI) {
  auto ScopeID = static_cast<SyncScope::ID>(MI.getOperand(1).getImm());
  auto &Ctx = MI.getMF()->getFunction().getContext();
  auto ScopeName = Ctx.getSyncScopeName(ScopeID);
  if (!ScopeName)
    return StringRef();

  for (StringRef Suffix : {"-shared", "-global", "-generic"})
    if (ScopeName->ends_with(Suffix))
      return ScopeName->drop_back(Suffix.size());
  return *ScopeName;
}

bool PISAPostLegalizerCombinerImpl::matchRedundantFence(
    MachineInstr &MI, MachineInstr *&PrevFence) const {
  assert(MI.getOpcode() == TargetOpcode::G_FENCE);

  if (isSubgroupFence(MI))
    return false;

  unsigned Scope = MI.getOperand(1).getImm();

  auto It = MI.getIterator();
  const MachineBasicBlock &MBB = *MI.getParent();
  while (It != MBB.begin()) {
    --It;
    if (It->getOpcode() == TargetOpcode::G_FENCE) {
      if (It->getOperand(1).getImm() == Scope) {
        PrevFence = &*It;
        return true;
      }
      // TODO: A wider scope subsumes a narrower one on the same addrspace.
      // E.g. for `fence.global.gpu; fence.global.workgroup` we could drop
      // the workgroup fence.
      return false;
    }
    // TODO: Skip memory instructions with scope that do not interfere.
    if (It->mayLoadOrStore() || It->hasUnmodeledSideEffects())
      return false;
  }
  return false;
}

void PISAPostLegalizerCombinerImpl::applyRedundantFence(
    MachineInstr &MI, MachineInstr *&PrevFence) const {
  assert(PrevFence && "Expected a preceding compatible fence");

  auto CurOrd = static_cast<AtomicOrdering>(MI.getOperand(0).getImm());
  auto PrevOrd = static_cast<AtomicOrdering>(PrevFence->getOperand(0).getImm());

  // Pick the stronger ordering (e.g. choose seq_cst over acquire).
  // Acquire and release cannot be compared; use acq_rel for the
  // resulting fence.
  auto Merged = getMergedAtomicOrdering(PrevOrd, CurOrd);
  PrevFence->getOperand(0).setImm(static_cast<int64_t>(Merged));
  MI.eraseFromParent();
}

// %28:_(<8 x i32>) = G_BUILD_VECTOR %91:_(i32), %92:_(i32), ...
// %37:_(<4 x i32>) = G_EXTRACT_SUBVECTOR %28:_(<8 x i32>), 0
// => %37:_(<4 x i32>) = G_BUILD_VECTOR %91:_(i32), %92:_(i32), ...
bool PISAPostLegalizerCombinerImpl::matchExtractSubvectorBuildVector(
    MachineInstr &MI, SmallVector<Register, 8> &MatchInfo) const {
  assert(MI.getOpcode() == TargetOpcode::G_EXTRACT_SUBVECTOR);
  if (!MI.getOperand(2).isImm())
    return false;
  auto *BV = getDefIgnoringCopies(MI.getOperand(1).getReg(), MRI);
  if (!BV || BV->getOpcode() != TargetOpcode::G_BUILD_VECTOR)
    return false;
  uint64_t Offset = MI.getOperand(2).getImm();
  unsigned NumDstElts = MRI.getType(MI.getOperand(0).getReg()).getNumElements();
  if (NumDstElts != 4)
    return false;
  unsigned NumSrcElts = BV->getNumOperands() - 1; // operand 0 is the dst
  if (Offset + NumDstElts > NumSrcElts)
    return false;
  MatchInfo.clear();
  for (unsigned I = 0; I < NumDstElts; ++I)
    MatchInfo.push_back(BV->getOperand(1 + Offset + I).getReg());
  return true;
}

void PISAPostLegalizerCombinerImpl::applyExtractSubvectorBuildVector(
    MachineInstr &MI, SmallVector<Register, 8> &MatchInfo) const {
  Register DstReg = MI.getOperand(0).getReg();
  B.setInstrAndDebugLoc(MI);
  B.buildBuildVector(DstReg, MatchInfo);
  MI.eraseFromParent();
}

// Match adjacent G_FENCE instructions that have the same memory ordering and
// same base scope but differ only in address space. These can be merged into
// a single generic-address-space fence.
bool PISAPostLegalizerCombinerImpl::matchMergeAdjacentFences(
    MachineInstr &MI, MachineInstr *&PrevFence) const {
  assert(MI.getOpcode() == TargetOpcode::G_FENCE);

  if (isSubgroupFence(MI))
    return false;

  auto Order = static_cast<AtomicOrdering>(MI.getOperand(0).getImm());
  auto BaseScopeNameA = getBaseScopeName(MI);
  if (BaseScopeNameA.empty())
    return false;

  auto It = MI.getIterator();
  const MachineBasicBlock &MBB = *MI.getParent();
  while (It != MBB.begin()) {
    --It;
    if (It->getOpcode() == TargetOpcode::G_FENCE) {
      auto ItOrder = static_cast<AtomicOrdering>(It->getOperand(0).getImm());
      if (ItOrder != Order)
        return false;

      if (It->getOperand(1).getImm() == MI.getOperand(1).getImm())
        return false;

      auto BaseScopeNameIt = getBaseScopeName(*It);
      if (BaseScopeNameIt.empty() || BaseScopeNameIt != BaseScopeNameA)
        return false;

      PrevFence = &*It;
      return true;
    }
    if (It->mayLoadOrStore() || It->hasUnmodeledSideEffects())
      return false;
  }
  return false;
}

// Merge two fences with same ordering and same base scope but different
// address spaces into a single fence with generic address space.
void PISAPostLegalizerCombinerImpl::applyMergeAdjacentFences(
    MachineInstr &MI, MachineInstr *&PrevFence) const {
  assert(PrevFence && "Expected a preceding compatible fence");

  StringRef BaseScopeName = getBaseScopeName(*PrevFence);
  assert(!BaseScopeName.empty() && "Expected a named sync scope");
  std::string GenericName = (BaseScopeName + "-generic").str();
  auto &Ctx = MI.getMF()->getFunction().getContext();
  auto GenericID = Ctx.getOrInsertSyncScopeID(GenericName);
  PrevFence->getOperand(1).setImm(static_cast<int64_t>(GenericID));

  MI.eraseFromParent();
}

// %29:_(<64 x s32>) = IMPLICIT_DEF
// %30:_(<64 x s32>) = G_INSERT_SUBVECTOR %29:_, %7:_(<8 x s32>), 0
// %31:_(<64 x s32>) = G_INSERT_SUBVECTOR %30:_, %10:_(<8 x s32>), 8
// %32:_(<64 x s32>) = G_INSERT_SUBVECTOR %31:_, %13:_(<8 x s32>), 16
// %33:_(<64 x s32>) = G_INSERT_SUBVECTOR %32:_, %16:_(<8 x s32>), 24
// %34:_(<64 x s32>) = G_INSERT_SUBVECTOR %33:_, %19:_(<8 x s32>), 32
// %35:_(<64 x s32>) = G_INSERT_SUBVECTOR %34:_, %22:_(<8 x s32>), 40
// %36:_(<64 x s32>) = G_INSERT_SUBVECTOR %35:_, %25:_(<8 x s32>), 48
// %37:_(<64 x s32>) = G_INSERT_SUBVECTOR %36:_, %28:_(<8 x s32>), 56
// %4:_(<32 x s32>) = G_EXTRACT_SUBVECTOR %37:_(<64 x s32>), 32
// =>
// %IMP:_(<32 x s32>) = IMPLICIT_DEF
// %34:_(<32 x s32>) = G_INSERT_SUBVECTOR %IMP:_, %19:_(<8 x s32>), 0
// %35:_(<32 x s32>) = G_INSERT_SUBVECTOR %34:_, %22:_(<8 x s32>), 8
// %36:_(<32 x s32>) = G_INSERT_SUBVECTOR %35:_, %25:_(<8 x s32>), 16
// %37:_(<32 x s32>) = G_INSERT_SUBVECTOR %36:_, %28:_(<8 x s32>), 24
// %4:_(<32 x s32>) = COPY %37:_(<32 x s32>)
bool PISAPostLegalizerCombinerImpl::matchExtractSubvectorPartial(
    MachineInstr &MI, SmallVector<MachineInstr *, 8> &MatchInfo) const {
  assert(MI.getOpcode() == TargetOpcode::G_EXTRACT_SUBVECTOR);

  Register SrcReg = MI.getOperand(1).getReg();
  uint64_t ExtractIdx = MI.getOperand(2).getImm();
  LLT DstTy = MRI.getType(MI.getOperand(0).getReg());
  LLT SrcTy = MRI.getType(SrcReg);
  unsigned DstNumElts = DstTy.getNumElements();

  if (DstTy.getScalarSizeInBits() != 32)
    return false;
  if (DstNumElts < 8 || DstNumElts == SrcTy.getNumElements())
    return false;

  // Walk the chain of G_INSERT_SUBVECTOR instructions feeding the source.
  // Collect inserts that are fully contained within the extracted range.
  MatchInfo.clear();
  MachineInstr *Cur = MRI.getVRegDef(SrcReg);
  while (Cur && Cur->getOpcode() == TargetOpcode::G_INSERT_SUBVECTOR) {
    uint64_t InsIdx = Cur->getOperand(3).getImm();
    Register InsSubReg = Cur->getOperand(2).getReg();
    unsigned InsNumElts = MRI.getType(InsSubReg).getNumElements();

    // Reject if insert starts before/extend past the extract window.
    if (InsIdx < ExtractIdx && InsIdx + InsNumElts > ExtractIdx)
      return false;
    if (InsIdx >= ExtractIdx && InsIdx + InsNumElts > ExtractIdx + DstNumElts)
      return false;

    if (InsIdx >= ExtractIdx)
      MatchInfo.push_back(Cur);
    Cur = MRI.getVRegDef(Cur->getOperand(1).getReg());
  }

  // Base of the chain must be IMPLICIT_DEF and we need at least one insert.
  if (MatchInfo.empty() || !Cur ||
      Cur->getOpcode() != TargetOpcode::IMPLICIT_DEF)
    return false;
  return true;
}

void PISAPostLegalizerCombinerImpl::applyExtractSubvectorPartial(
    MachineInstr &MI, const SmallVector<MachineInstr *, 8> &MatchInfo) const {
  Register DstReg = MI.getOperand(0).getReg();
  uint64_t ExtractIdx = MI.getOperand(2).getImm();
  LLT DstTy = MRI.getType(DstReg);

  // Rebuild in reverse order (innermost/earliest insert first).
  B.setInstrAndDebugLoc(MI);
  Register AccReg = B.buildUndef(DstTy).getReg(0);

  for (const MachineInstr *Ins : llvm::reverse(MatchInfo)) {
    Register InsSubReg = Ins->getOperand(2).getReg();
    uint64_t InsIdx = Ins->getOperand(3).getImm();
    uint64_t NewIdx = InsIdx - ExtractIdx;
    AccReg = B.buildInsertSubvector(DstTy, AccReg, InsSubReg, NewIdx).getReg(0);
  }

  B.buildCopy(DstReg, AccReg);
  MI.eraseFromParent();
}

// umin(x, UINT_MAX) -> x,  umax(x, 0)       -> x
// smin(x, INT_MAX)  -> x,  smax(x, INT_MIN)  -> x
bool PISAPostLegalizerCombinerImpl::matchMinMaxIdentityFold(
    MachineInstr &MI, Register &MatchInfo) const {
  assert(MI.getOpcode() == TargetOpcode::G_UMIN ||
         MI.getOpcode() == TargetOpcode::G_UMAX ||
         MI.getOpcode() == TargetOpcode::G_SMIN ||
         MI.getOpcode() == TargetOpcode::G_SMAX);
  Register LHS = MI.getOperand(1).getReg();
  Register RHS = MI.getOperand(2).getReg();
  auto IsIdentity = [&](const APInt &Val) -> bool {
    switch (MI.getOpcode()) {
    case TargetOpcode::G_UMIN:
      return Val.isAllOnes();
    case TargetOpcode::G_UMAX:
      return Val.isZero();
    case TargetOpcode::G_SMIN:
      return Val.isMaxSignedValue();
    case TargetOpcode::G_SMAX:
      return Val.isMinSignedValue();
    default:
      return false;
    }
  };
  if (auto C = getIConstantVRegValWithLookThrough(RHS, MRI))
    if (IsIdentity(C->Value)) {
      MatchInfo = LHS;
      return true;
    }
  if (auto C = getIConstantVRegValWithLookThrough(LHS, MRI))
    if (IsIdentity(C->Value)) {
      MatchInfo = RHS;
      return true;
    }
  return false;
}

// Pass boilerplate
// ================

class PISAPostLegalizerCombiner : public MachineFunctionPass {
  PISAPostLegalizerCombinerImplRuleConfig RuleConfig;

public:
  static char ID;

  PISAPostLegalizerCombiner();

  StringRef getPassName() const override { return "PISAPostLegalizerCombiner"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // end anonymous namespace

void PISAPostLegalizerCombiner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  AU.addRequired<GISelValueTrackingAnalysisLegacy>();
  AU.addPreserved<GISelValueTrackingAnalysisLegacy>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

PISAPostLegalizerCombiner::PISAPostLegalizerCombiner()
    : MachineFunctionPass(ID) {
  initializePISAPostLegalizerCombinerPass(*PassRegistry::getPassRegistry());
  if (!RuleConfig.parseCommandLineOption())
    report_fatal_error("Invalid rule identifier");
}

bool PISAPostLegalizerCombiner::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;

  const Function &F = MF.getFunction();
  bool EnableOpt =
      MF.getTarget().getOptLevel() != CodeGenOptLevel::None && !skipFunction(F);

  GISelValueTracking *KB =
      &getAnalysis<GISelValueTrackingAnalysisLegacy>().get(MF);
  MachineDominatorTree *MDT =
      &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  CombinerInfo CInfo(
      /*AllowIllegalOps=*/true, /*ShouldLegalizeIllegal=*/false,
      /*LegalizerInfo=*/nullptr, EnableOpt, F.hasOptSize(), F.hasMinSize());

  const PISASubtarget &STI = MF.getSubtarget<PISASubtarget>();
  PISAPostLegalizerCombinerImpl Impl(MF, CInfo, *KB, /*CSEInfo=*/nullptr,
                                     RuleConfig, STI, MDT,
                                     STI.getLegalizerInfo());
  return Impl.combineMachineInstrs();
}

char PISAPostLegalizerCombiner::ID = 0;
INITIALIZE_PASS_BEGIN(PISAPostLegalizerCombiner, DEBUG_TYPE,
                      "Combine PISA machine instrs after legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(GISelValueTrackingAnalysisLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(PISAPostLegalizerCombiner, DEBUG_TYPE,
                    "Combine PISA machine instrs after legalization", false,
                    false)

namespace llvm {
FunctionPass *createPISAPostLegalizerCombiner() {
  return new PISAPostLegalizerCombiner();
}
} // end namespace llvm
