//===-- AArch64TargetTransformInfo.cpp - AArch64 specific TTI -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64TargetTransformInfo.h"
#include "AArch64ExpandImm.h"
#include "AArch64PerfectShuffle.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/CostTable.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"
#include "llvm/Transforms/Vectorize/LoopVectorizationLegality.h"
#include <algorithm>
using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "aarch64tti"

static cl::opt<bool> EnableFalkorHWPFUnrollFix("enable-falkor-hwpf-unroll-fix",
                                               cl::init(true), cl::Hidden);

static cl::opt<unsigned> SVEGatherOverhead("sve-gather-overhead", cl::init(10),
                                           cl::Hidden);

static cl::opt<unsigned> SVEScatterOverhead("sve-scatter-overhead",
                                            cl::init(10), cl::Hidden);

class TailFoldingKind {
private:
  uint8_t Bits = 0; // Currently defaults to disabled.

public:
  enum TailFoldingOpts {
    TFDisabled = 0x0,
    TFReductions = 0x01,
    TFRecurrences = 0x02,
    TFSimple = 0x80,
    TFAll = TFReductions | TFRecurrences | TFSimple
  };

  void operator=(const std::string &Val) {
    if (Val.empty())
      return;
    SmallVector<StringRef, 6> TailFoldTypes;
    StringRef(Val).split(TailFoldTypes, '+', -1, false);
    for (auto TailFoldType : TailFoldTypes) {
      if (TailFoldType == "disabled")
        Bits = 0;
      else if (TailFoldType == "all")
        Bits = TFAll;
      else if (TailFoldType == "default")
        Bits = 0; // Currently defaults to never tail-folding.
      else if (TailFoldType == "simple")
        add(TFSimple);
      else if (TailFoldType == "reductions")
        add(TFReductions);
      else if (TailFoldType == "recurrences")
        add(TFRecurrences);
      else if (TailFoldType == "noreductions")
        remove(TFReductions);
      else if (TailFoldType == "norecurrences")
        remove(TFRecurrences);
      else {
        errs()
            << "invalid argument " << TailFoldType.str()
            << " to -sve-tail-folding=; each element must be one of: disabled, "
               "all, default, simple, reductions, noreductions, recurrences, "
               "norecurrences\n";
      }
    }
  }

  operator uint8_t() const { return Bits; }

  void add(uint8_t Flag) { Bits |= Flag; }
  void remove(uint8_t Flag) { Bits &= ~Flag; }
};

TailFoldingKind TailFoldingKindLoc;

cl::opt<TailFoldingKind, true, cl::parser<std::string>> SVETailFolding(
    "sve-tail-folding",
    cl::desc(
        "Control the use of vectorisation using tail-folding for SVE:"
        "\ndisabled    No loop types will vectorize using tail-folding"
        "\ndefault     Uses the default tail-folding settings for the target "
        "CPU"
        "\nall         All legal loop types will vectorize using tail-folding"
        "\nsimple      Use tail-folding for simple loops (not reductions or "
        "recurrences)"
        "\nreductions  Use tail-folding for loops containing reductions"
        "\nrecurrences Use tail-folding for loops containing fixed order "
        "recurrences"),
    cl::location(TailFoldingKindLoc));

bool AArch64TTIImpl::areInlineCompatible(const Function *Caller,
                                         const Function *Callee) const {
  const TargetMachine &TM = getTLI()->getTargetMachine();

  const FeatureBitset &CallerBits =
      TM.getSubtargetImpl(*Caller)->getFeatureBits();
  const FeatureBitset &CalleeBits =
      TM.getSubtargetImpl(*Callee)->getFeatureBits();

  // Inline a callee if its target-features are a subset of the callers
  // target-features.
  return (CallerBits & CalleeBits) == CalleeBits;
}

bool AArch64TTIImpl::shouldMaximizeVectorBandwidth(
    TargetTransformInfo::RegisterKind K) const {
  assert(K != TargetTransformInfo::RGK_Scalar);
  return K == TargetTransformInfo::RGK_FixedWidthVector;
}

/// Calculate the cost of materializing a 64-bit value. This helper
/// method might only calculate a fraction of a larger immediate. Therefore it
/// is valid to return a cost of ZERO.
InstructionCost AArch64TTIImpl::getIntImmCost(int64_t Val) {
  // Check if the immediate can be encoded within an instruction.
  if (Val == 0 || AArch64_AM::isLogicalImmediate(Val, 64))
    return 0;

  if (Val < 0)
    Val = ~Val;

  // Calculate how many moves we will need to materialize this constant.
  SmallVector<AArch64_IMM::ImmInsnModel, 4> Insn;
  AArch64_IMM::expandMOVImm(Val, 64, Insn);
  return Insn.size();
}

/// Calculate the cost of materializing the given constant.
InstructionCost AArch64TTIImpl::getIntImmCost(const APInt &Imm, Type *Ty,
                                              TTI::TargetCostKind CostKind) {
  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  if (BitSize == 0)
    return ~0U;

  // Sign-extend all constants to a multiple of 64-bit.
  APInt ImmVal = Imm;
  if (BitSize & 0x3f)
    ImmVal = Imm.sext((BitSize + 63) & ~0x3fU);

  // Split the constant into 64-bit chunks and calculate the cost for each
  // chunk.
  InstructionCost Cost = 0;
  for (unsigned ShiftVal = 0; ShiftVal < BitSize; ShiftVal += 64) {
    APInt Tmp = ImmVal.ashr(ShiftVal).sextOrTrunc(64);
    int64_t Val = Tmp.getSExtValue();
    Cost += getIntImmCost(Val);
  }
  // We need at least one instruction to materialze the constant.
  return std::max<InstructionCost>(1, Cost);
}

InstructionCost AArch64TTIImpl::getIntImmCostInst(unsigned Opcode, unsigned Idx,
                                                  const APInt &Imm, Type *Ty,
                                                  TTI::TargetCostKind CostKind,
                                                  Instruction *Inst) {
  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  // There is no cost model for constants with a bit size of 0. Return TCC_Free
  // here, so that constant hoisting will ignore this constant.
  if (BitSize == 0)
    return TTI::TCC_Free;

  unsigned ImmIdx = ~0U;
  switch (Opcode) {
  default:
    return TTI::TCC_Free;
  case Instruction::GetElementPtr:
    // Always hoist the base address of a GetElementPtr.
    if (Idx == 0)
      return 2 * TTI::TCC_Basic;
    return TTI::TCC_Free;
  case Instruction::Store:
    ImmIdx = 0;
    break;
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::ICmp:
    ImmIdx = 1;
    break;
  // Always return TCC_Free for the shift value of a shift instruction.
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    if (Idx == 1)
      return TTI::TCC_Free;
    break;
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::IntToPtr:
  case Instruction::PtrToInt:
  case Instruction::BitCast:
  case Instruction::PHI:
  case Instruction::Call:
  case Instruction::Select:
  case Instruction::Ret:
  case Instruction::Load:
    break;
  }

  if (Idx == ImmIdx) {
    int NumConstants = (BitSize + 63) / 64;
    InstructionCost Cost = AArch64TTIImpl::getIntImmCost(Imm, Ty, CostKind);
    return (Cost <= NumConstants * TTI::TCC_Basic)
               ? static_cast<int>(TTI::TCC_Free)
               : Cost;
  }
  return AArch64TTIImpl::getIntImmCost(Imm, Ty, CostKind);
}

InstructionCost
AArch64TTIImpl::getIntImmCostIntrin(Intrinsic::ID IID, unsigned Idx,
                                    const APInt &Imm, Type *Ty,
                                    TTI::TargetCostKind CostKind) {
  assert(Ty->isIntegerTy());

  unsigned BitSize = Ty->getPrimitiveSizeInBits();
  // There is no cost model for constants with a bit size of 0. Return TCC_Free
  // here, so that constant hoisting will ignore this constant.
  if (BitSize == 0)
    return TTI::TCC_Free;

  // Most (all?) AArch64 intrinsics do not support folding immediates into the
  // selected instruction, so we compute the materialization cost for the
  // immediate directly.
  if (IID >= Intrinsic::aarch64_addg && IID <= Intrinsic::aarch64_udiv)
    return AArch64TTIImpl::getIntImmCost(Imm, Ty, CostKind);

  switch (IID) {
  default:
    return TTI::TCC_Free;
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::usub_with_overflow:
  case Intrinsic::smul_with_overflow:
  case Intrinsic::umul_with_overflow:
    if (Idx == 1) {
      int NumConstants = (BitSize + 63) / 64;
      InstructionCost Cost = AArch64TTIImpl::getIntImmCost(Imm, Ty, CostKind);
      return (Cost <= NumConstants * TTI::TCC_Basic)
                 ? static_cast<int>(TTI::TCC_Free)
                 : Cost;
    }
    break;
  case Intrinsic::experimental_stackmap:
    if ((Idx < 2) || (Imm.getBitWidth() <= 64 && isInt<64>(Imm.getSExtValue())))
      return TTI::TCC_Free;
    break;
  case Intrinsic::experimental_patchpoint_void:
  case Intrinsic::experimental_patchpoint_i64:
    if ((Idx < 4) || (Imm.getBitWidth() <= 64 && isInt<64>(Imm.getSExtValue())))
      return TTI::TCC_Free;
    break;
  case Intrinsic::experimental_gc_statepoint:
    if ((Idx < 5) || (Imm.getBitWidth() <= 64 && isInt<64>(Imm.getSExtValue())))
      return TTI::TCC_Free;
    break;
  }
  return AArch64TTIImpl::getIntImmCost(Imm, Ty, CostKind);
}

TargetTransformInfo::PopcntSupportKind
AArch64TTIImpl::getPopcntSupport(unsigned TyWidth) {
  assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
  if (TyWidth == 32 || TyWidth == 64)
    return TTI::PSK_FastHardware;
  // TODO: AArch64TargetLowering::LowerCTPOP() supports 128bit popcount.
  return TTI::PSK_Software;
}

InstructionCost
AArch64TTIImpl::getIntrinsicInstrCost(const IntrinsicCostAttributes &ICA,
                                      TTI::TargetCostKind CostKind) {
  auto *RetTy = ICA.getReturnType();
  switch (ICA.getID()) {
  case Intrinsic::umin:
  case Intrinsic::umax:
  case Intrinsic::smin:
  case Intrinsic::smax: {
    static const auto ValidMinMaxTys = {MVT::v8i8,  MVT::v16i8, MVT::v4i16,
                                        MVT::v8i16, MVT::v2i32, MVT::v4i32};
    auto LT = getTypeLegalizationCost(RetTy);
    // v2i64 types get converted to cmp+bif hence the cost of 2
    if (LT.second == MVT::v2i64)
      return LT.first * 2;
    if (any_of(ValidMinMaxTys, [&LT](MVT M) { return M == LT.second; }))
      return LT.first;
    break;
  }
  case Intrinsic::sadd_sat:
  case Intrinsic::ssub_sat:
  case Intrinsic::uadd_sat:
  case Intrinsic::usub_sat: {
    static const auto ValidSatTys = {MVT::v8i8,  MVT::v16i8, MVT::v4i16,
                                     MVT::v8i16, MVT::v2i32, MVT::v4i32,
                                     MVT::v2i64};
    auto LT = getTypeLegalizationCost(RetTy);
    // This is a base cost of 1 for the vadd, plus 3 extract shifts if we
    // need to extend the type, as it uses shr(qadd(shl, shl)).
    unsigned Instrs =
        LT.second.getScalarSizeInBits() == RetTy->getScalarSizeInBits() ? 1 : 4;
    if (any_of(ValidSatTys, [&LT](MVT M) { return M == LT.second; }))
      return LT.first * Instrs;
    break;
  }
  case Intrinsic::abs: {
    static const auto ValidAbsTys = {MVT::v8i8,  MVT::v16i8, MVT::v4i16,
                                     MVT::v8i16, MVT::v2i32, MVT::v4i32,
                                     MVT::v2i64};
    auto LT = getTypeLegalizationCost(RetTy);
    if (any_of(ValidAbsTys, [&LT](MVT M) { return M == LT.second; }))
      return LT.first;
    break;
  }
  case Intrinsic::experimental_stepvector: {
    InstructionCost Cost = 1; // Cost of the `index' instruction
    auto LT = getTypeLegalizationCost(RetTy);
    // Legalisation of illegal vectors involves an `index' instruction plus
    // (LT.first - 1) vector adds.
    if (LT.first > 1) {
      Type *LegalVTy = EVT(LT.second).getTypeForEVT(RetTy->getContext());
      InstructionCost AddCost =
          getArithmeticInstrCost(Instruction::Add, LegalVTy, CostKind);
      Cost += AddCost * (LT.first - 1);
    }
    return Cost;
  }
  case Intrinsic::bitreverse: {
    static const CostTblEntry BitreverseTbl[] = {
        {Intrinsic::bitreverse, MVT::i32, 1},
        {Intrinsic::bitreverse, MVT::i64, 1},
        {Intrinsic::bitreverse, MVT::v8i8, 1},
        {Intrinsic::bitreverse, MVT::v16i8, 1},
        {Intrinsic::bitreverse, MVT::v4i16, 2},
        {Intrinsic::bitreverse, MVT::v8i16, 2},
        {Intrinsic::bitreverse, MVT::v2i32, 2},
        {Intrinsic::bitreverse, MVT::v4i32, 2},
        {Intrinsic::bitreverse, MVT::v1i64, 2},
        {Intrinsic::bitreverse, MVT::v2i64, 2},
    };
    const auto LegalisationCost = getTypeLegalizationCost(RetTy);
    const auto *Entry =
        CostTableLookup(BitreverseTbl, ICA.getID(), LegalisationCost.second);
    if (Entry) {
      // Cost Model is using the legal type(i32) that i8 and i16 will be
      // converted to +1 so that we match the actual lowering cost
      if (TLI->getValueType(DL, RetTy, true) == MVT::i8 ||
          TLI->getValueType(DL, RetTy, true) == MVT::i16)
        return LegalisationCost.first * Entry->Cost + 1;

      return LegalisationCost.first * Entry->Cost;
    }
    break;
  }
  case Intrinsic::ctpop: {
    if (!ST->hasNEON()) {
      // 32-bit or 64-bit ctpop without NEON is 12 instructions.
      return getTypeLegalizationCost(RetTy).first * 12;
    }
    static const CostTblEntry CtpopCostTbl[] = {
        {ISD::CTPOP, MVT::v2i64, 4},
        {ISD::CTPOP, MVT::v4i32, 3},
        {ISD::CTPOP, MVT::v8i16, 2},
        {ISD::CTPOP, MVT::v16i8, 1},
        {ISD::CTPOP, MVT::i64,   4},
        {ISD::CTPOP, MVT::v2i32, 3},
        {ISD::CTPOP, MVT::v4i16, 2},
        {ISD::CTPOP, MVT::v8i8,  1},
        {ISD::CTPOP, MVT::i32,   5},
    };
    auto LT = getTypeLegalizationCost(RetTy);
    MVT MTy = LT.second;
    if (const auto *Entry = CostTableLookup(CtpopCostTbl, ISD::CTPOP, MTy)) {
      // Extra cost of +1 when illegal vector types are legalized by promoting
      // the integer type.
      int ExtraCost = MTy.isVector() && MTy.getScalarSizeInBits() !=
                                            RetTy->getScalarSizeInBits()
                          ? 1
                          : 0;
      return LT.first * Entry->Cost + ExtraCost;
    }
    break;
  }
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::usub_with_overflow:
  case Intrinsic::smul_with_overflow:
  case Intrinsic::umul_with_overflow: {
    static const CostTblEntry WithOverflowCostTbl[] = {
        {Intrinsic::sadd_with_overflow, MVT::i8, 3},
        {Intrinsic::uadd_with_overflow, MVT::i8, 3},
        {Intrinsic::sadd_with_overflow, MVT::i16, 3},
        {Intrinsic::uadd_with_overflow, MVT::i16, 3},
        {Intrinsic::sadd_with_overflow, MVT::i32, 1},
        {Intrinsic::uadd_with_overflow, MVT::i32, 1},
        {Intrinsic::sadd_with_overflow, MVT::i64, 1},
        {Intrinsic::uadd_with_overflow, MVT::i64, 1},
        {Intrinsic::ssub_with_overflow, MVT::i8, 3},
        {Intrinsic::usub_with_overflow, MVT::i8, 3},
        {Intrinsic::ssub_with_overflow, MVT::i16, 3},
        {Intrinsic::usub_with_overflow, MVT::i16, 3},
        {Intrinsic::ssub_with_overflow, MVT::i32, 1},
        {Intrinsic::usub_with_overflow, MVT::i32, 1},
        {Intrinsic::ssub_with_overflow, MVT::i64, 1},
        {Intrinsic::usub_with_overflow, MVT::i64, 1},
        {Intrinsic::smul_with_overflow, MVT::i8, 5},
        {Intrinsic::umul_with_overflow, MVT::i8, 4},
        {Intrinsic::smul_with_overflow, MVT::i16, 5},
        {Intrinsic::umul_with_overflow, MVT::i16, 4},
        {Intrinsic::smul_with_overflow, MVT::i32, 2}, // eg umull;tst
        {Intrinsic::umul_with_overflow, MVT::i32, 2}, // eg umull;cmp sxtw
        {Intrinsic::smul_with_overflow, MVT::i64, 3}, // eg mul;smulh;cmp
        {Intrinsic::umul_with_overflow, MVT::i64, 3}, // eg mul;umulh;cmp asr
    };
    EVT MTy = TLI->getValueType(DL, RetTy->getContainedType(0), true);
    if (MTy.isSimple())
      if (const auto *Entry = CostTableLookup(WithOverflowCostTbl, ICA.getID(),
                                              MTy.getSimpleVT()))
        return Entry->Cost;
    break;
  }
  case Intrinsic::fptosi_sat:
  case Intrinsic::fptoui_sat: {
    if (ICA.getArgTypes().empty())
      break;
    bool IsSigned = ICA.getID() == Intrinsic::fptosi_sat;
    auto LT = getTypeLegalizationCost(ICA.getArgTypes()[0]);
    EVT MTy = TLI->getValueType(DL, RetTy);
    // Check for the legal types, which are where the size of the input and the
    // output are the same, or we are using cvt f64->i32 or f32->i64.
    if ((LT.second == MVT::f32 || LT.second == MVT::f64 ||
         LT.second == MVT::v2f32 || LT.second == MVT::v4f32 ||
         LT.second == MVT::v2f64) &&
        (LT.second.getScalarSizeInBits() == MTy.getScalarSizeInBits() ||
         (LT.second == MVT::f64 && MTy == MVT::i32) ||
         (LT.second == MVT::f32 && MTy == MVT::i64)))
      return LT.first;
    // Similarly for fp16 sizes
    if (ST->hasFullFP16() &&
        ((LT.second == MVT::f16 && MTy == MVT::i32) ||
         ((LT.second == MVT::v4f16 || LT.second == MVT::v8f16) &&
          (LT.second.getScalarSizeInBits() == MTy.getScalarSizeInBits()))))
      return LT.first;

    // Otherwise we use a legal convert followed by a min+max
    if ((LT.second.getScalarType() == MVT::f32 ||
         LT.second.getScalarType() == MVT::f64 ||
         (ST->hasFullFP16() && LT.second.getScalarType() == MVT::f16)) &&
        LT.second.getScalarSizeInBits() >= MTy.getScalarSizeInBits()) {
      Type *LegalTy =
          Type::getIntNTy(RetTy->getContext(), LT.second.getScalarSizeInBits());
      if (LT.second.isVector())
        LegalTy = VectorType::get(LegalTy, LT.second.getVectorElementCount());
      InstructionCost Cost = 1;
      IntrinsicCostAttributes Attrs1(IsSigned ? Intrinsic::smin : Intrinsic::umin,
                                    LegalTy, {LegalTy, LegalTy});
      Cost += getIntrinsicInstrCost(Attrs1, CostKind);
      IntrinsicCostAttributes Attrs2(IsSigned ? Intrinsic::smax : Intrinsic::umax,
                                    LegalTy, {LegalTy, LegalTy});
      Cost += getIntrinsicInstrCost(Attrs2, CostKind);
      return LT.first * Cost;
    }
    break;
  }
  default:
    break;
  }
  return BaseT::getIntrinsicInstrCost(ICA, CostKind);
}

/// The function will remove redundant reinterprets casting in the presence
/// of the control flow
static Optional<Instruction *> processPhiNode(InstCombiner &IC,
                                              IntrinsicInst &II) {
  SmallVector<Instruction *, 32> Worklist;
  auto RequiredType = II.getType();

  auto *PN = dyn_cast<PHINode>(II.getArgOperand(0));
  assert(PN && "Expected Phi Node!");

  // Don't create a new Phi unless we can remove the old one.
  if (!PN->hasOneUse())
    return None;

  for (Value *IncValPhi : PN->incoming_values()) {
    auto *Reinterpret = dyn_cast<IntrinsicInst>(IncValPhi);
    if (!Reinterpret ||
        Reinterpret->getIntrinsicID() !=
            Intrinsic::aarch64_sve_convert_to_svbool ||
        RequiredType != Reinterpret->getArgOperand(0)->getType())
      return None;
  }

  // Create the new Phi
  LLVMContext &Ctx = PN->getContext();
  IRBuilder<> Builder(Ctx);
  Builder.SetInsertPoint(PN);
  PHINode *NPN = Builder.CreatePHI(RequiredType, PN->getNumIncomingValues());
  Worklist.push_back(PN);

  for (unsigned I = 0; I < PN->getNumIncomingValues(); I++) {
    auto *Reinterpret = cast<Instruction>(PN->getIncomingValue(I));
    NPN->addIncoming(Reinterpret->getOperand(0), PN->getIncomingBlock(I));
    Worklist.push_back(Reinterpret);
  }

  // Cleanup Phi Node and reinterprets
  return IC.replaceInstUsesWith(II, NPN);
}

// (from_svbool (binop (to_svbool pred) (svbool_t _) (svbool_t _))))
// => (binop (pred) (from_svbool _) (from_svbool _))
//
// The above transformation eliminates a `to_svbool` in the predicate
// operand of bitwise operation `binop` by narrowing the vector width of
// the operation. For example, it would convert a `<vscale x 16 x i1>
// and` into a `<vscale x 4 x i1> and`. This is profitable because
// to_svbool must zero the new lanes during widening, whereas
// from_svbool is free.
static Optional<Instruction *> tryCombineFromSVBoolBinOp(InstCombiner &IC,
                                                         IntrinsicInst &II) {
  auto BinOp = dyn_cast<IntrinsicInst>(II.getOperand(0));
  if (!BinOp)
    return None;

  auto IntrinsicID = BinOp->getIntrinsicID();
  switch (IntrinsicID) {
  case Intrinsic::aarch64_sve_and_z:
  case Intrinsic::aarch64_sve_bic_z:
  case Intrinsic::aarch64_sve_eor_z:
  case Intrinsic::aarch64_sve_nand_z:
  case Intrinsic::aarch64_sve_nor_z:
  case Intrinsic::aarch64_sve_orn_z:
  case Intrinsic::aarch64_sve_orr_z:
    break;
  default:
    return None;
  }

  auto BinOpPred = BinOp->getOperand(0);
  auto BinOpOp1 = BinOp->getOperand(1);
  auto BinOpOp2 = BinOp->getOperand(2);

  auto PredIntr = dyn_cast<IntrinsicInst>(BinOpPred);
  if (!PredIntr ||
      PredIntr->getIntrinsicID() != Intrinsic::aarch64_sve_convert_to_svbool)
    return None;

  auto PredOp = PredIntr->getOperand(0);
  auto PredOpTy = cast<VectorType>(PredOp->getType());
  if (PredOpTy != II.getType())
    return None;

  IRBuilder<> Builder(II.getContext());
  Builder.SetInsertPoint(&II);

  SmallVector<Value *> NarrowedBinOpArgs = {PredOp};
  auto NarrowBinOpOp1 = Builder.CreateIntrinsic(
      Intrinsic::aarch64_sve_convert_from_svbool, {PredOpTy}, {BinOpOp1});
  NarrowedBinOpArgs.push_back(NarrowBinOpOp1);
  if (BinOpOp1 == BinOpOp2)
    NarrowedBinOpArgs.push_back(NarrowBinOpOp1);
  else
    NarrowedBinOpArgs.push_back(Builder.CreateIntrinsic(
        Intrinsic::aarch64_sve_convert_from_svbool, {PredOpTy}, {BinOpOp2}));

  auto NarrowedBinOp =
      Builder.CreateIntrinsic(IntrinsicID, {PredOpTy}, NarrowedBinOpArgs);
  return IC.replaceInstUsesWith(II, NarrowedBinOp);
}

static Optional<Instruction *> instCombineConvertFromSVBool(InstCombiner &IC,
                                                            IntrinsicInst &II) {
  // If the reinterpret instruction operand is a PHI Node
  if (isa<PHINode>(II.getArgOperand(0)))
    return processPhiNode(IC, II);

  if (auto BinOpCombine = tryCombineFromSVBoolBinOp(IC, II))
    return BinOpCombine;

  SmallVector<Instruction *, 32> CandidatesForRemoval;
  Value *Cursor = II.getOperand(0), *EarliestReplacement = nullptr;

  const auto *IVTy = cast<VectorType>(II.getType());

  // Walk the chain of conversions.
  while (Cursor) {
    // If the type of the cursor has fewer lanes than the final result, zeroing
    // must take place, which breaks the equivalence chain.
    const auto *CursorVTy = cast<VectorType>(Cursor->getType());
    if (CursorVTy->getElementCount().getKnownMinValue() <
        IVTy->getElementCount().getKnownMinValue())
      break;

    // If the cursor has the same type as I, it is a viable replacement.
    if (Cursor->getType() == IVTy)
      EarliestReplacement = Cursor;

    auto *IntrinsicCursor = dyn_cast<IntrinsicInst>(Cursor);

    // If this is not an SVE conversion intrinsic, this is the end of the chain.
    if (!IntrinsicCursor || !(IntrinsicCursor->getIntrinsicID() ==
                                  Intrinsic::aarch64_sve_convert_to_svbool ||
                              IntrinsicCursor->getIntrinsicID() ==
                                  Intrinsic::aarch64_sve_convert_from_svbool))
      break;

    CandidatesForRemoval.insert(CandidatesForRemoval.begin(), IntrinsicCursor);
    Cursor = IntrinsicCursor->getOperand(0);
  }

  // If no viable replacement in the conversion chain was found, there is
  // nothing to do.
  if (!EarliestReplacement)
    return None;

  return IC.replaceInstUsesWith(II, EarliestReplacement);
}

static Optional<Instruction *> instCombineSVESel(InstCombiner &IC,
                                                 IntrinsicInst &II) {
  IRBuilder<> Builder(&II);
  auto Select = Builder.CreateSelect(II.getOperand(0), II.getOperand(1),
                                     II.getOperand(2));
  return IC.replaceInstUsesWith(II, Select);
}

static Optional<Instruction *> instCombineSVEDup(InstCombiner &IC,
                                                 IntrinsicInst &II) {
  IntrinsicInst *Pg = dyn_cast<IntrinsicInst>(II.getArgOperand(1));
  if (!Pg)
    return None;

  if (Pg->getIntrinsicID() != Intrinsic::aarch64_sve_ptrue)
    return None;

  const auto PTruePattern =
      cast<ConstantInt>(Pg->getOperand(0))->getZExtValue();
  if (PTruePattern != AArch64SVEPredPattern::vl1)
    return None;

  // The intrinsic is inserting into lane zero so use an insert instead.
  auto *IdxTy = Type::getInt64Ty(II.getContext());
  auto *Insert = InsertElementInst::Create(
      II.getArgOperand(0), II.getArgOperand(2), ConstantInt::get(IdxTy, 0));
  Insert->insertBefore(&II);
  Insert->takeName(&II);

  return IC.replaceInstUsesWith(II, Insert);
}

static Optional<Instruction *> instCombineSVEDupX(InstCombiner &IC,
                                                  IntrinsicInst &II) {
  // Replace DupX with a regular IR splat.
  IRBuilder<> Builder(II.getContext());
  Builder.SetInsertPoint(&II);
  auto *RetTy = cast<ScalableVectorType>(II.getType());
  Value *Splat =
      Builder.CreateVectorSplat(RetTy->getElementCount(), II.getArgOperand(0));
  Splat->takeName(&II);
  return IC.replaceInstUsesWith(II, Splat);
}

static Optional<Instruction *> instCombineSVECmpNE(InstCombiner &IC,
                                                   IntrinsicInst &II) {
  LLVMContext &Ctx = II.getContext();
  IRBuilder<> Builder(Ctx);
  Builder.SetInsertPoint(&II);

  // Check that the predicate is all active
  auto *Pg = dyn_cast<IntrinsicInst>(II.getArgOperand(0));
  if (!Pg || Pg->getIntrinsicID() != Intrinsic::aarch64_sve_ptrue)
    return None;

  const auto PTruePattern =
      cast<ConstantInt>(Pg->getOperand(0))->getZExtValue();
  if (PTruePattern != AArch64SVEPredPattern::all)
    return None;

  // Check that we have a compare of zero..
  auto *SplatValue =
      dyn_cast_or_null<ConstantInt>(getSplatValue(II.getArgOperand(2)));
  if (!SplatValue || !SplatValue->isZero())
    return None;

  // ..against a dupq
  auto *DupQLane = dyn_cast<IntrinsicInst>(II.getArgOperand(1));
  if (!DupQLane ||
      DupQLane->getIntrinsicID() != Intrinsic::aarch64_sve_dupq_lane)
    return None;

  // Where the dupq is a lane 0 replicate of a vector insert
  if (!cast<ConstantInt>(DupQLane->getArgOperand(1))->isZero())
    return None;

  auto *VecIns = dyn_cast<IntrinsicInst>(DupQLane->getArgOperand(0));
  if (!VecIns || VecIns->getIntrinsicID() != Intrinsic::vector_insert)
    return None;

  // Where the vector insert is a fixed constant vector insert into undef at
  // index zero
  if (!isa<UndefValue>(VecIns->getArgOperand(0)))
    return None;

  if (!cast<ConstantInt>(VecIns->getArgOperand(2))->isZero())
    return None;

  auto *ConstVec = dyn_cast<Constant>(VecIns->getArgOperand(1));
  if (!ConstVec)
    return None;

  auto *VecTy = dyn_cast<FixedVectorType>(ConstVec->getType());
  auto *OutTy = dyn_cast<ScalableVectorType>(II.getType());
  if (!VecTy || !OutTy || VecTy->getNumElements() != OutTy->getMinNumElements())
    return None;

  unsigned NumElts = VecTy->getNumElements();
  unsigned PredicateBits = 0;

  // Expand intrinsic operands to a 16-bit byte level predicate
  for (unsigned I = 0; I < NumElts; ++I) {
    auto *Arg = dyn_cast<ConstantInt>(ConstVec->getAggregateElement(I));
    if (!Arg)
      return None;
    if (!Arg->isZero())
      PredicateBits |= 1 << (I * (16 / NumElts));
  }

  // If all bits are zero bail early with an empty predicate
  if (PredicateBits == 0) {
    auto *PFalse = Constant::getNullValue(II.getType());
    PFalse->takeName(&II);
    return IC.replaceInstUsesWith(II, PFalse);
  }

  // Calculate largest predicate type used (where byte predicate is largest)
  unsigned Mask = 8;
  for (unsigned I = 0; I < 16; ++I)
    if ((PredicateBits & (1 << I)) != 0)
      Mask |= (I % 8);

  unsigned PredSize = Mask & -Mask;
  auto *PredType = ScalableVectorType::get(
      Type::getInt1Ty(Ctx), AArch64::SVEBitsPerBlock / (PredSize * 8));

  // Ensure all relevant bits are set
  for (unsigned I = 0; I < 16; I += PredSize)
    if ((PredicateBits & (1 << I)) == 0)
      return None;

  auto *PTruePat =
      ConstantInt::get(Type::getInt32Ty(Ctx), AArch64SVEPredPattern::all);
  auto *PTrue = Builder.CreateIntrinsic(Intrinsic::aarch64_sve_ptrue,
                                        {PredType}, {PTruePat});
  auto *ConvertToSVBool = Builder.CreateIntrinsic(
      Intrinsic::aarch64_sve_convert_to_svbool, {PredType}, {PTrue});
  auto *ConvertFromSVBool =
      Builder.CreateIntrinsic(Intrinsic::aarch64_sve_convert_from_svbool,
                              {II.getType()}, {ConvertToSVBool});

  ConvertFromSVBool->takeName(&II);
  return IC.replaceInstUsesWith(II, ConvertFromSVBool);
}

static Optional<Instruction *> instCombineSVELast(InstCombiner &IC,
                                                  IntrinsicInst &II) {
  IRBuilder<> Builder(II.getContext());
  Builder.SetInsertPoint(&II);
  Value *Pg = II.getArgOperand(0);
  Value *Vec = II.getArgOperand(1);
  auto IntrinsicID = II.getIntrinsicID();
  bool IsAfter = IntrinsicID == Intrinsic::aarch64_sve_lasta;

  // lastX(splat(X)) --> X
  if (auto *SplatVal = getSplatValue(Vec))
    return IC.replaceInstUsesWith(II, SplatVal);

  // If x and/or y is a splat value then:
  // lastX (binop (x, y)) --> binop(lastX(x), lastX(y))
  Value *LHS, *RHS;
  if (match(Vec, m_OneUse(m_BinOp(m_Value(LHS), m_Value(RHS))))) {
    if (isSplatValue(LHS) || isSplatValue(RHS)) {
      auto *OldBinOp = cast<BinaryOperator>(Vec);
      auto OpC = OldBinOp->getOpcode();
      auto *NewLHS =
          Builder.CreateIntrinsic(IntrinsicID, {Vec->getType()}, {Pg, LHS});
      auto *NewRHS =
          Builder.CreateIntrinsic(IntrinsicID, {Vec->getType()}, {Pg, RHS});
      auto *NewBinOp = BinaryOperator::CreateWithCopiedFlags(
          OpC, NewLHS, NewRHS, OldBinOp, OldBinOp->getName(), &II);
      return IC.replaceInstUsesWith(II, NewBinOp);
    }
  }

  auto *C = dyn_cast<Constant>(Pg);
  if (IsAfter && C && C->isNullValue()) {
    // The intrinsic is extracting lane 0 so use an extract instead.
    auto *IdxTy = Type::getInt64Ty(II.getContext());
    auto *Extract = ExtractElementInst::Create(Vec, ConstantInt::get(IdxTy, 0));
    Extract->insertBefore(&II);
    Extract->takeName(&II);
    return IC.replaceInstUsesWith(II, Extract);
  }

  auto *IntrPG = dyn_cast<IntrinsicInst>(Pg);
  if (!IntrPG)
    return None;

  if (IntrPG->getIntrinsicID() != Intrinsic::aarch64_sve_ptrue)
    return None;

  const auto PTruePattern =
      cast<ConstantInt>(IntrPG->getOperand(0))->getZExtValue();

  // Can the intrinsic's predicate be converted to a known constant index?
  unsigned MinNumElts = getNumElementsFromSVEPredPattern(PTruePattern);
  if (!MinNumElts)
    return None;

  unsigned Idx = MinNumElts - 1;
  // Increment the index if extracting the element after the last active
  // predicate element.
  if (IsAfter)
    ++Idx;

  // Ignore extracts whose index is larger than the known minimum vector
  // length. NOTE: This is an artificial constraint where we prefer to
  // maintain what the user asked for until an alternative is proven faster.
  auto *PgVTy = cast<ScalableVectorType>(Pg->getType());
  if (Idx >= PgVTy->getMinNumElements())
    return None;

  // The intrinsic is extracting a fixed lane so use an extract instead.
  auto *IdxTy = Type::getInt64Ty(II.getContext());
  auto *Extract = ExtractElementInst::Create(Vec, ConstantInt::get(IdxTy, Idx));
  Extract->insertBefore(&II);
  Extract->takeName(&II);
  return IC.replaceInstUsesWith(II, Extract);
}

static Optional<Instruction *> instCombineSVECondLast(InstCombiner &IC,
                                                      IntrinsicInst &II) {
  // The SIMD&FP variant of CLAST[AB] is significantly faster than the scalar
  // integer variant across a variety of micro-architectures. Replace scalar
  // integer CLAST[AB] intrinsic with optimal SIMD&FP variant. A simple
  // bitcast-to-fp + clast[ab] + bitcast-to-int will cost a cycle or two more
  // depending on the micro-architecture, but has been observed as generally
  // being faster, particularly when the CLAST[AB] op is a loop-carried
  // dependency.
  IRBuilder<> Builder(II.getContext());
  Builder.SetInsertPoint(&II);
  Value *Pg = II.getArgOperand(0);
  Value *Fallback = II.getArgOperand(1);
  Value *Vec = II.getArgOperand(2);
  Type *Ty = II.getType();

  if (!Ty->isIntegerTy())
    return None;

  Type *FPTy;
  switch (cast<IntegerType>(Ty)->getBitWidth()) {
  default:
    return None;
  case 16:
    FPTy = Builder.getHalfTy();
    break;
  case 32:
    FPTy = Builder.getFloatTy();
    break;
  case 64:
    FPTy = Builder.getDoubleTy();
    break;
  }

  Value *FPFallBack = Builder.CreateBitCast(Fallback, FPTy);
  auto *FPVTy = VectorType::get(
      FPTy, cast<VectorType>(Vec->getType())->getElementCount());
  Value *FPVec = Builder.CreateBitCast(Vec, FPVTy);
  auto *FPII = Builder.CreateIntrinsic(II.getIntrinsicID(), {FPVec->getType()},
                                       {Pg, FPFallBack, FPVec});
  Value *FPIItoInt = Builder.CreateBitCast(FPII, II.getType());
  return IC.replaceInstUsesWith(II, FPIItoInt);
}

static Optional<Instruction *> instCombineRDFFR(InstCombiner &IC,
                                                IntrinsicInst &II) {
  LLVMContext &Ctx = II.getContext();
  IRBuilder<> Builder(Ctx);
  Builder.SetInsertPoint(&II);
  // Replace rdffr with predicated rdffr.z intrinsic, so that optimizePTestInstr
  // can work with RDFFR_PP for ptest elimination.
  auto *AllPat =
      ConstantInt::get(Type::getInt32Ty(Ctx), AArch64SVEPredPattern::all);
  auto *PTrue = Builder.CreateIntrinsic(Intrinsic::aarch64_sve_ptrue,
                                        {II.getType()}, {AllPat});
  auto *RDFFR =
      Builder.CreateIntrinsic(Intrinsic::aarch64_sve_rdffr_z, {}, {PTrue});
  RDFFR->takeName(&II);
  return IC.replaceInstUsesWith(II, RDFFR);
}

static Optional<Instruction *>
instCombineSVECntElts(InstCombiner &IC, IntrinsicInst &II, unsigned NumElts) {
  const auto Pattern = cast<ConstantInt>(II.getArgOperand(0))->getZExtValue();

  if (Pattern == AArch64SVEPredPattern::all) {
    LLVMContext &Ctx = II.getContext();
    IRBuilder<> Builder(Ctx);
    Builder.SetInsertPoint(&II);

    Constant *StepVal = ConstantInt::get(II.getType(), NumElts);
    auto *VScale = Builder.CreateVScale(StepVal);
    VScale->takeName(&II);
    return IC.replaceInstUsesWith(II, VScale);
  }

  unsigned MinNumElts = getNumElementsFromSVEPredPattern(Pattern);

  return MinNumElts && NumElts >= MinNumElts
             ? Optional<Instruction *>(IC.replaceInstUsesWith(
                   II, ConstantInt::get(II.getType(), MinNumElts)))
             : None;
}

static Optional<Instruction *> instCombineSVEPTest(InstCombiner &IC,
                                                   IntrinsicInst &II) {
  IntrinsicInst *Op1 = dyn_cast<IntrinsicInst>(II.getArgOperand(0));
  IntrinsicInst *Op2 = dyn_cast<IntrinsicInst>(II.getArgOperand(1));

  if (Op1 && Op2 &&
      Op1->getIntrinsicID() == Intrinsic::aarch64_sve_convert_to_svbool &&
      Op2->getIntrinsicID() == Intrinsic::aarch64_sve_convert_to_svbool &&
      Op1->getArgOperand(0)->getType() == Op2->getArgOperand(0)->getType()) {

    IRBuilder<> Builder(II.getContext());
    Builder.SetInsertPoint(&II);

    Value *Ops[] = {Op1->getArgOperand(0), Op2->getArgOperand(0)};
    Type *Tys[] = {Op1->getArgOperand(0)->getType()};

    auto *PTest = Builder.CreateIntrinsic(II.getIntrinsicID(), Tys, Ops);

    PTest->takeName(&II);
    return IC.replaceInstUsesWith(II, PTest);
  }

  return None;
}

static Optional<Instruction *> instCombineSVEVectorFMLA(InstCombiner &IC,
                                                        IntrinsicInst &II) {
  // fold (fadd p a (fmul p b c)) -> (fma p a b c)
  Value *P = II.getOperand(0);
  Value *A = II.getOperand(1);
  auto FMul = II.getOperand(2);
  Value *B, *C;
  if (!match(FMul, m_Intrinsic<Intrinsic::aarch64_sve_fmul>(
                       m_Specific(P), m_Value(B), m_Value(C))))
    return None;

  if (!FMul->hasOneUse())
    return None;

  llvm::FastMathFlags FAddFlags = II.getFastMathFlags();
  // Stop the combine when the flags on the inputs differ in case dropping flags
  // would lead to us missing out on more beneficial optimizations.
  if (FAddFlags != cast<CallInst>(FMul)->getFastMathFlags())
    return None;
  if (!FAddFlags.allowContract())
    return None;

  IRBuilder<> Builder(II.getContext());
  Builder.SetInsertPoint(&II);
  auto FMLA = Builder.CreateIntrinsic(Intrinsic::aarch64_sve_fmla,
                                      {II.getType()}, {P, A, B, C}, &II);
  FMLA->setFastMathFlags(FAddFlags);
  return IC.replaceInstUsesWith(II, FMLA);
}

static bool isAllActivePredicate(Value *Pred) {
  // Look through convert.from.svbool(convert.to.svbool(...) chain.
  Value *UncastedPred;
  if (match(Pred, m_Intrinsic<Intrinsic::aarch64_sve_convert_from_svbool>(
                      m_Intrinsic<Intrinsic::aarch64_sve_convert_to_svbool>(
                          m_Value(UncastedPred)))))
    // If the predicate has the same or less lanes than the uncasted
    // predicate then we know the casting has no effect.
    if (cast<ScalableVectorType>(Pred->getType())->getMinNumElements() <=
        cast<ScalableVectorType>(UncastedPred->getType())->getMinNumElements())
      Pred = UncastedPred;

  return match(Pred, m_Intrinsic<Intrinsic::aarch64_sve_ptrue>(
                         m_ConstantInt<AArch64SVEPredPattern::all>()));
}

static Optional<Instruction *>
instCombineSVELD1(InstCombiner &IC, IntrinsicInst &II, const DataLayout &DL) {
  IRBuilder<> Builder(II.getContext());
  Builder.SetInsertPoint(&II);

  Value *Pred = II.getOperand(0);
  Value *PtrOp = II.getOperand(1);
  Type *VecTy = II.getType();
  Value *VecPtr = Builder.CreateBitCast(PtrOp, VecTy->getPointerTo());

  if (isAllActivePredicate(Pred)) {
    LoadInst *Load = Builder.CreateLoad(VecTy, VecPtr);
    Load->copyMetadata(II);
    return IC.replaceInstUsesWith(II, Load);
  }

  CallInst *MaskedLoad =
      Builder.CreateMaskedLoad(VecTy, VecPtr, PtrOp->getPointerAlignment(DL),
                               Pred, ConstantAggregateZero::get(VecTy));
  MaskedLoad->copyMetadata(II);
  return IC.replaceInstUsesWith(II, MaskedLoad);
}

static Optional<Instruction *>
instCombineSVEST1(InstCombiner &IC, IntrinsicInst &II, const DataLayout &DL) {
  IRBuilder<> Builder(II.getContext());
  Builder.SetInsertPoint(&II);

  Value *VecOp = II.getOperand(0);
  Value *Pred = II.getOperand(1);
  Value *PtrOp = II.getOperand(2);
  Value *VecPtr =
      Builder.CreateBitCast(PtrOp, VecOp->getType()->getPointerTo());

  if (isAllActivePredicate(Pred)) {
    StoreInst *Store = Builder.CreateStore(VecOp, VecPtr);
    Store->copyMetadata(II);
    return IC.eraseInstFromFunction(II);
  }

  CallInst *MaskedStore = Builder.CreateMaskedStore(
      VecOp, VecPtr, PtrOp->getPointerAlignment(DL), Pred);
  MaskedStore->copyMetadata(II);
  return IC.eraseInstFromFunction(II);
}

static Instruction::BinaryOps intrinsicIDToBinOpCode(unsigned Intrinsic) {
  switch (Intrinsic) {
  case Intrinsic::aarch64_sve_fmul:
    return Instruction::BinaryOps::FMul;
  case Intrinsic::aarch64_sve_fadd:
    return Instruction::BinaryOps::FAdd;
  case Intrinsic::aarch64_sve_fsub:
    return Instruction::BinaryOps::FSub;
  default:
    return Instruction::BinaryOpsEnd;
  }
}

static Optional<Instruction *> instCombineSVEVectorBinOp(InstCombiner &IC,
                                                         IntrinsicInst &II) {
  auto *OpPredicate = II.getOperand(0);
  auto BinOpCode = intrinsicIDToBinOpCode(II.getIntrinsicID());
  if (BinOpCode == Instruction::BinaryOpsEnd ||
      !match(OpPredicate, m_Intrinsic<Intrinsic::aarch64_sve_ptrue>(
                              m_ConstantInt<AArch64SVEPredPattern::all>())))
    return None;
  IRBuilder<> Builder(II.getContext());
  Builder.SetInsertPoint(&II);
  Builder.setFastMathFlags(II.getFastMathFlags());
  auto BinOp =
      Builder.CreateBinOp(BinOpCode, II.getOperand(1), II.getOperand(2));
  return IC.replaceInstUsesWith(II, BinOp);
}

static Optional<Instruction *> instCombineSVEVectorFAdd(InstCombiner &IC,
                                                        IntrinsicInst &II) {
  if (auto FMLA = instCombineSVEVectorFMLA(IC, II))
    return FMLA;
  return instCombineSVEVectorBinOp(IC, II);
}

static Optional<Instruction *> instCombineSVEVectorMul(InstCombiner &IC,
                                                       IntrinsicInst &II) {
  auto *OpPredicate = II.getOperand(0);
  auto *OpMultiplicand = II.getOperand(1);
  auto *OpMultiplier = II.getOperand(2);

  IRBuilder<> Builder(II.getContext());
  Builder.SetInsertPoint(&II);

  // Return true if a given instruction is a unit splat value, false otherwise.
  auto IsUnitSplat = [](auto *I) {
    auto *SplatValue = getSplatValue(I);
    if (!SplatValue)
      return false;
    return match(SplatValue, m_FPOne()) || match(SplatValue, m_One());
  };

  // Return true if a given instruction is an aarch64_sve_dup intrinsic call
  // with a unit splat value, false otherwise.
  auto IsUnitDup = [](auto *I) {
    auto *IntrI = dyn_cast<IntrinsicInst>(I);
    if (!IntrI || IntrI->getIntrinsicID() != Intrinsic::aarch64_sve_dup)
      return false;

    auto *SplatValue = IntrI->getOperand(2);
    return match(SplatValue, m_FPOne()) || match(SplatValue, m_One());
  };

  if (IsUnitSplat(OpMultiplier)) {
    // [f]mul pg %n, (dupx 1) => %n
    OpMultiplicand->takeName(&II);
    return IC.replaceInstUsesWith(II, OpMultiplicand);
  } else if (IsUnitDup(OpMultiplier)) {
    // [f]mul pg %n, (dup pg 1) => %n
    auto *DupInst = cast<IntrinsicInst>(OpMultiplier);
    auto *DupPg = DupInst->getOperand(1);
    // TODO: this is naive. The optimization is still valid if DupPg
    // 'encompasses' OpPredicate, not only if they're the same predicate.
    if (OpPredicate == DupPg) {
      OpMultiplicand->takeName(&II);
      return IC.replaceInstUsesWith(II, OpMultiplicand);
    }
  }

  return instCombineSVEVectorBinOp(IC, II);
}

static Optional<Instruction *> instCombineSVEUnpack(InstCombiner &IC,
                                                    IntrinsicInst &II) {
  IRBuilder<> Builder(II.getContext());
  Builder.SetInsertPoint(&II);
  Value *UnpackArg = II.getArgOperand(0);
  auto *RetTy = cast<ScalableVectorType>(II.getType());
  bool IsSigned = II.getIntrinsicID() == Intrinsic::aarch64_sve_sunpkhi ||
                  II.getIntrinsicID() == Intrinsic::aarch64_sve_sunpklo;

  // Hi = uunpkhi(splat(X)) --> Hi = splat(extend(X))
  // Lo = uunpklo(splat(X)) --> Lo = splat(extend(X))
  if (auto *ScalarArg = getSplatValue(UnpackArg)) {
    ScalarArg =
        Builder.CreateIntCast(ScalarArg, RetTy->getScalarType(), IsSigned);
    Value *NewVal =
        Builder.CreateVectorSplat(RetTy->getElementCount(), ScalarArg);
    NewVal->takeName(&II);
    return IC.replaceInstUsesWith(II, NewVal);
  }

  return None;
}
static Optional<Instruction *> instCombineSVETBL(InstCombiner &IC,
                                                 IntrinsicInst &II) {
  auto *OpVal = II.getOperand(0);
  auto *OpIndices = II.getOperand(1);
  VectorType *VTy = cast<VectorType>(II.getType());

  // Check whether OpIndices is a constant splat value < minimal element count
  // of result.
  auto *SplatValue = dyn_cast_or_null<ConstantInt>(getSplatValue(OpIndices));
  if (!SplatValue ||
      SplatValue->getValue().uge(VTy->getElementCount().getKnownMinValue()))
    return None;

  // Convert sve_tbl(OpVal sve_dup_x(SplatValue)) to
  // splat_vector(extractelement(OpVal, SplatValue)) for further optimization.
  IRBuilder<> Builder(II.getContext());
  Builder.SetInsertPoint(&II);
  auto *Extract = Builder.CreateExtractElement(OpVal, SplatValue);
  auto *VectorSplat =
      Builder.CreateVectorSplat(VTy->getElementCount(), Extract);

  VectorSplat->takeName(&II);
  return IC.replaceInstUsesWith(II, VectorSplat);
}

static Optional<Instruction *> instCombineSVETupleGet(InstCombiner &IC,
                                                      IntrinsicInst &II) {
  // Try to remove sequences of tuple get/set.
  Value *SetTuple, *SetIndex, *SetValue;
  auto *GetTuple = II.getArgOperand(0);
  auto *GetIndex = II.getArgOperand(1);
  // Check that we have tuple_get(GetTuple, GetIndex) where GetTuple is a
  // call to tuple_set i.e. tuple_set(SetTuple, SetIndex, SetValue).
  // Make sure that the types of the current intrinsic and SetValue match
  // in order to safely remove the sequence.
  if (!match(GetTuple,
             m_Intrinsic<Intrinsic::aarch64_sve_tuple_set>(
                 m_Value(SetTuple), m_Value(SetIndex), m_Value(SetValue))) ||
      SetValue->getType() != II.getType())
    return None;
  // Case where we get the same index right after setting it.
  // tuple_get(tuple_set(SetTuple, SetIndex, SetValue), GetIndex) --> SetValue
  if (GetIndex == SetIndex)
    return IC.replaceInstUsesWith(II, SetValue);
  // If we are getting a different index than what was set in the tuple_set
  // intrinsic. We can just set the input tuple to the one up in the chain.
  // tuple_get(tuple_set(SetTuple, SetIndex, SetValue), GetIndex)
  // --> tuple_get(SetTuple, GetIndex)
  return IC.replaceOperand(II, 0, SetTuple);
}

static Optional<Instruction *> instCombineSVEZip(InstCombiner &IC,
                                                 IntrinsicInst &II) {
  // zip1(uzp1(A, B), uzp2(A, B)) --> A
  // zip2(uzp1(A, B), uzp2(A, B)) --> B
  Value *A, *B;
  if (match(II.getArgOperand(0),
            m_Intrinsic<Intrinsic::aarch64_sve_uzp1>(m_Value(A), m_Value(B))) &&
      match(II.getArgOperand(1), m_Intrinsic<Intrinsic::aarch64_sve_uzp2>(
                                     m_Specific(A), m_Specific(B))))
    return IC.replaceInstUsesWith(
        II, (II.getIntrinsicID() == Intrinsic::aarch64_sve_zip1 ? A : B));

  return None;
}

static Optional<Instruction *> instCombineLD1GatherIndex(InstCombiner &IC,
                                                         IntrinsicInst &II) {
  Value *Mask = II.getOperand(0);
  Value *BasePtr = II.getOperand(1);
  Value *Index = II.getOperand(2);
  Type *Ty = II.getType();
  Value *PassThru = ConstantAggregateZero::get(Ty);

  // Contiguous gather => masked load.
  // (sve.ld1.gather.index Mask BasePtr (sve.index IndexBase 1))
  // => (masked.load (gep BasePtr IndexBase) Align Mask zeroinitializer)
  Value *IndexBase;
  if (match(Index, m_Intrinsic<Intrinsic::aarch64_sve_index>(
                       m_Value(IndexBase), m_SpecificInt(1)))) {
    IRBuilder<> Builder(II.getContext());
    Builder.SetInsertPoint(&II);

    Align Alignment =
        BasePtr->getPointerAlignment(II.getModule()->getDataLayout());

    Type *VecPtrTy = PointerType::getUnqual(Ty);
    Value *Ptr = Builder.CreateGEP(
        cast<VectorType>(Ty)->getElementType(), BasePtr, IndexBase);
    Ptr = Builder.CreateBitCast(Ptr, VecPtrTy);
    CallInst *MaskedLoad =
        Builder.CreateMaskedLoad(Ty, Ptr, Alignment, Mask, PassThru);
    MaskedLoad->takeName(&II);
    return IC.replaceInstUsesWith(II, MaskedLoad);
  }

  return None;
}

static Optional<Instruction *> instCombineST1ScatterIndex(InstCombiner &IC,
                                                          IntrinsicInst &II) {
  Value *Val = II.getOperand(0);
  Value *Mask = II.getOperand(1);
  Value *BasePtr = II.getOperand(2);
  Value *Index = II.getOperand(3);
  Type *Ty = Val->getType();

  // Contiguous scatter => masked store.
  // (sve.st1.scatter.index Value Mask BasePtr (sve.index IndexBase 1))
  // => (masked.store Value (gep BasePtr IndexBase) Align Mask)
  Value *IndexBase;
  if (match(Index, m_Intrinsic<Intrinsic::aarch64_sve_index>(
                       m_Value(IndexBase), m_SpecificInt(1)))) {
    IRBuilder<> Builder(II.getContext());
    Builder.SetInsertPoint(&II);

    Align Alignment =
        BasePtr->getPointerAlignment(II.getModule()->getDataLayout());

    Value *Ptr = Builder.CreateGEP(
        cast<VectorType>(Ty)->getElementType(), BasePtr, IndexBase);
    Type *VecPtrTy = PointerType::getUnqual(Ty);
    Ptr = Builder.CreateBitCast(Ptr, VecPtrTy);

    (void)Builder.CreateMaskedStore(Val, Ptr, Alignment, Mask);

    return IC.eraseInstFromFunction(II);
  }

  return None;
}

static Optional<Instruction *> instCombineSVESDIV(InstCombiner &IC,
                                                  IntrinsicInst &II) {
  IRBuilder<> Builder(II.getContext());
  Builder.SetInsertPoint(&II);
  Type *Int32Ty = Builder.getInt32Ty();
  Value *Pred = II.getOperand(0);
  Value *Vec = II.getOperand(1);
  Value *DivVec = II.getOperand(2);

  Value *SplatValue = getSplatValue(DivVec);
  ConstantInt *SplatConstantInt = dyn_cast_or_null<ConstantInt>(SplatValue);
  if (!SplatConstantInt)
    return None;
  APInt Divisor = SplatConstantInt->getValue();

  if (Divisor.isPowerOf2()) {
    Constant *DivisorLog2 = ConstantInt::get(Int32Ty, Divisor.logBase2());
    auto ASRD = Builder.CreateIntrinsic(
        Intrinsic::aarch64_sve_asrd, {II.getType()}, {Pred, Vec, DivisorLog2});
    return IC.replaceInstUsesWith(II, ASRD);
  }
  if (Divisor.isNegatedPowerOf2()) {
    Divisor.negate();
    Constant *DivisorLog2 = ConstantInt::get(Int32Ty, Divisor.logBase2());
    auto ASRD = Builder.CreateIntrinsic(
        Intrinsic::aarch64_sve_asrd, {II.getType()}, {Pred, Vec, DivisorLog2});
    auto NEG = Builder.CreateIntrinsic(Intrinsic::aarch64_sve_neg,
                                       {ASRD->getType()}, {ASRD, Pred, ASRD});
    return IC.replaceInstUsesWith(II, NEG);
  }

  return None;
}

static Optional<Instruction *> instCombineMaxMinNM(InstCombiner &IC,
                                                   IntrinsicInst &II) {
  Value *A = II.getArgOperand(0);
  Value *B = II.getArgOperand(1);
  if (A == B)
    return IC.replaceInstUsesWith(II, A);

  return None;
}

static Optional<Instruction *> instCombineSVESrshl(InstCombiner &IC,
                                                   IntrinsicInst &II) {
  IRBuilder<> Builder(&II);
  Value *Pred = II.getOperand(0);
  Value *Vec = II.getOperand(1);
  Value *Shift = II.getOperand(2);

  // Convert SRSHL into the simpler LSL intrinsic when fed by an ABS intrinsic.
  Value *AbsPred, *MergedValue;
  if (!match(Vec, m_Intrinsic<Intrinsic::aarch64_sve_sqabs>(
                      m_Value(MergedValue), m_Value(AbsPred), m_Value())) &&
      !match(Vec, m_Intrinsic<Intrinsic::aarch64_sve_abs>(
                      m_Value(MergedValue), m_Value(AbsPred), m_Value())))

    return None;

  // Transform is valid if any of the following are true:
  // * The ABS merge value is an undef or non-negative
  // * The ABS predicate is all active
  // * The ABS predicate and the SRSHL predicates are the same
  if (!isa<UndefValue>(MergedValue) &&
      !match(MergedValue, m_NonNegative()) &&
      AbsPred != Pred && !isAllActivePredicate(AbsPred))
    return None;

  // Only valid when the shift amount is non-negative, otherwise the rounding
  // behaviour of SRSHL cannot be ignored.
  if (!match(Shift, m_NonNegative()))
    return None;

  auto LSL = Builder.CreateIntrinsic(Intrinsic::aarch64_sve_lsl, {II.getType()},
                                     {Pred, Vec, Shift});

  return IC.replaceInstUsesWith(II, LSL);
}

Optional<Instruction *>
AArch64TTIImpl::instCombineIntrinsic(InstCombiner &IC,
                                     IntrinsicInst &II) const {
  Intrinsic::ID IID = II.getIntrinsicID();
  switch (IID) {
  default:
    break;
  case Intrinsic::aarch64_neon_fmaxnm:
  case Intrinsic::aarch64_neon_fminnm:
    return instCombineMaxMinNM(IC, II);
  case Intrinsic::aarch64_sve_convert_from_svbool:
    return instCombineConvertFromSVBool(IC, II);
  case Intrinsic::aarch64_sve_dup:
    return instCombineSVEDup(IC, II);
  case Intrinsic::aarch64_sve_dup_x:
    return instCombineSVEDupX(IC, II);
  case Intrinsic::aarch64_sve_cmpne:
  case Intrinsic::aarch64_sve_cmpne_wide:
    return instCombineSVECmpNE(IC, II);
  case Intrinsic::aarch64_sve_rdffr:
    return instCombineRDFFR(IC, II);
  case Intrinsic::aarch64_sve_lasta:
  case Intrinsic::aarch64_sve_lastb:
    return instCombineSVELast(IC, II);
  case Intrinsic::aarch64_sve_clasta_n:
  case Intrinsic::aarch64_sve_clastb_n:
    return instCombineSVECondLast(IC, II);
  case Intrinsic::aarch64_sve_cntd:
    return instCombineSVECntElts(IC, II, 2);
  case Intrinsic::aarch64_sve_cntw:
    return instCombineSVECntElts(IC, II, 4);
  case Intrinsic::aarch64_sve_cnth:
    return instCombineSVECntElts(IC, II, 8);
  case Intrinsic::aarch64_sve_cntb:
    return instCombineSVECntElts(IC, II, 16);
  case Intrinsic::aarch64_sve_ptest_any:
  case Intrinsic::aarch64_sve_ptest_first:
  case Intrinsic::aarch64_sve_ptest_last:
    return instCombineSVEPTest(IC, II);
  case Intrinsic::aarch64_sve_mul:
  case Intrinsic::aarch64_sve_fmul:
    return instCombineSVEVectorMul(IC, II);
  case Intrinsic::aarch64_sve_fadd:
    return instCombineSVEVectorFAdd(IC, II);
  case Intrinsic::aarch64_sve_fsub:
    return instCombineSVEVectorBinOp(IC, II);
  case Intrinsic::aarch64_sve_tbl:
    return instCombineSVETBL(IC, II);
  case Intrinsic::aarch64_sve_uunpkhi:
  case Intrinsic::aarch64_sve_uunpklo:
  case Intrinsic::aarch64_sve_sunpkhi:
  case Intrinsic::aarch64_sve_sunpklo:
    return instCombineSVEUnpack(IC, II);
  case Intrinsic::aarch64_sve_tuple_get:
    return instCombineSVETupleGet(IC, II);
  case Intrinsic::aarch64_sve_zip1:
  case Intrinsic::aarch64_sve_zip2:
    return instCombineSVEZip(IC, II);
  case Intrinsic::aarch64_sve_ld1_gather_index:
    return instCombineLD1GatherIndex(IC, II);
  case Intrinsic::aarch64_sve_st1_scatter_index:
    return instCombineST1ScatterIndex(IC, II);
  case Intrinsic::aarch64_sve_ld1:
    return instCombineSVELD1(IC, II, DL);
  case Intrinsic::aarch64_sve_st1:
    return instCombineSVEST1(IC, II, DL);
  case Intrinsic::aarch64_sve_sdiv:
    return instCombineSVESDIV(IC, II);
  case Intrinsic::aarch64_sve_sel:
    return instCombineSVESel(IC, II);
  case Intrinsic::aarch64_sve_srshl:
    return instCombineSVESrshl(IC, II);
  }

  return None;
}

Optional<Value *> AArch64TTIImpl::simplifyDemandedVectorEltsIntrinsic(
    InstCombiner &IC, IntrinsicInst &II, APInt OrigDemandedElts,
    APInt &UndefElts, APInt &UndefElts2, APInt &UndefElts3,
    std::function<void(Instruction *, unsigned, APInt, APInt &)>
        SimplifyAndSetOp) const {
  switch (II.getIntrinsicID()) {
  default:
    break;
  case Intrinsic::aarch64_neon_fcvtxn:
  case Intrinsic::aarch64_neon_rshrn:
  case Intrinsic::aarch64_neon_sqrshrn:
  case Intrinsic::aarch64_neon_sqrshrun:
  case Intrinsic::aarch64_neon_sqshrn:
  case Intrinsic::aarch64_neon_sqshrun:
  case Intrinsic::aarch64_neon_sqxtn:
  case Intrinsic::aarch64_neon_sqxtun:
  case Intrinsic::aarch64_neon_uqrshrn:
  case Intrinsic::aarch64_neon_uqshrn:
  case Intrinsic::aarch64_neon_uqxtn:
    SimplifyAndSetOp(&II, 0, OrigDemandedElts, UndefElts);
    break;
  }

  return None;
}

bool AArch64TTIImpl::isWideningInstruction(Type *DstTy, unsigned Opcode,
                                           ArrayRef<const Value *> Args) {

  // A helper that returns a vector type from the given type. The number of
  // elements in type Ty determines the vector width.
  auto toVectorTy = [&](Type *ArgTy) {
    return VectorType::get(ArgTy->getScalarType(),
                           cast<VectorType>(DstTy)->getElementCount());
  };

  // Exit early if DstTy is not a vector type whose elements are at least
  // 16-bits wide.
  if (!DstTy->isVectorTy() || DstTy->getScalarSizeInBits() < 16)
    return false;

  // Determine if the operation has a widening variant. We consider both the
  // "long" (e.g., usubl) and "wide" (e.g., usubw) versions of the
  // instructions.
  //
  // TODO: Add additional widening operations (e.g., shl, etc.) once we
  //       verify that their extending operands are eliminated during code
  //       generation.
  switch (Opcode) {
  case Instruction::Add: // UADDL(2), SADDL(2), UADDW(2), SADDW(2).
  case Instruction::Sub: // USUBL(2), SSUBL(2), USUBW(2), SSUBW(2).
  case Instruction::Mul: // SMULL(2), UMULL(2)
    break;
  default:
    return false;
  }

  // To be a widening instruction (either the "wide" or "long" versions), the
  // second operand must be a sign- or zero extend.
  if (Args.size() != 2 ||
      (!isa<SExtInst>(Args[1]) && !isa<ZExtInst>(Args[1])))
    return false;
  auto *Extend = cast<CastInst>(Args[1]);
  auto *Arg0 = dyn_cast<CastInst>(Args[0]);

  // A mul only has a mull version (not like addw). Both operands need to be
  // extending and the same type.
  if (Opcode == Instruction::Mul &&
      (!Arg0 || Arg0->getOpcode() != Extend->getOpcode() ||
       Arg0->getOperand(0)->getType() != Extend->getOperand(0)->getType()))
    return false;

  // Legalize the destination type and ensure it can be used in a widening
  // operation.
  auto DstTyL = getTypeLegalizationCost(DstTy);
  unsigned DstElTySize = DstTyL.second.getScalarSizeInBits();
  if (!DstTyL.second.isVector() || DstElTySize != DstTy->getScalarSizeInBits())
    return false;

  // Legalize the source type and ensure it can be used in a widening
  // operation.
  auto *SrcTy = toVectorTy(Extend->getSrcTy());
  auto SrcTyL = getTypeLegalizationCost(SrcTy);
  unsigned SrcElTySize = SrcTyL.second.getScalarSizeInBits();
  if (!SrcTyL.second.isVector() || SrcElTySize != SrcTy->getScalarSizeInBits())
    return false;

  // Get the total number of vector elements in the legalized types.
  InstructionCost NumDstEls =
      DstTyL.first * DstTyL.second.getVectorMinNumElements();
  InstructionCost NumSrcEls =
      SrcTyL.first * SrcTyL.second.getVectorMinNumElements();

  // Return true if the legalized types have the same number of vector elements
  // and the destination element type size is twice that of the source type.
  return NumDstEls == NumSrcEls && 2 * SrcElTySize == DstElTySize;
}

InstructionCost AArch64TTIImpl::getCastInstrCost(unsigned Opcode, Type *Dst,
                                                 Type *Src,
                                                 TTI::CastContextHint CCH,
                                                 TTI::TargetCostKind CostKind,
                                                 const Instruction *I) {
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  // If the cast is observable, and it is used by a widening instruction (e.g.,
  // uaddl, saddw, etc.), it may be free.
  if (I && I->hasOneUser()) {
    auto *SingleUser = cast<Instruction>(*I->user_begin());
    SmallVector<const Value *, 4> Operands(SingleUser->operand_values());
    if (isWideningInstruction(Dst, SingleUser->getOpcode(), Operands)) {
      // If the cast is the second operand, it is free. We will generate either
      // a "wide" or "long" version of the widening instruction.
      if (I == SingleUser->getOperand(1))
        return 0;
      // If the cast is not the second operand, it will be free if it looks the
      // same as the second operand. In this case, we will generate a "long"
      // version of the widening instruction.
      if (auto *Cast = dyn_cast<CastInst>(SingleUser->getOperand(1)))
        if (I->getOpcode() == unsigned(Cast->getOpcode()) &&
            cast<CastInst>(I)->getSrcTy() == Cast->getSrcTy())
          return 0;
    }
  }

  // TODO: Allow non-throughput costs that aren't binary.
  auto AdjustCost = [&CostKind](InstructionCost Cost) -> InstructionCost {
    if (CostKind != TTI::TCK_RecipThroughput)
      return Cost == 0 ? 0 : 1;
    return Cost;
  };

  EVT SrcTy = TLI->getValueType(DL, Src);
  EVT DstTy = TLI->getValueType(DL, Dst);

  if (!SrcTy.isSimple() || !DstTy.isSimple())
    return AdjustCost(
        BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I));

  static const TypeConversionCostTblEntry
  ConversionTbl[] = {
    { ISD::TRUNCATE, MVT::v2i8,   MVT::v2i64,  1},  // xtn
    { ISD::TRUNCATE, MVT::v2i16,  MVT::v2i64,  1},  // xtn
    { ISD::TRUNCATE, MVT::v2i32,  MVT::v2i64,  1},  // xtn
    { ISD::TRUNCATE, MVT::v4i8,   MVT::v4i32,  1},  // xtn
    { ISD::TRUNCATE, MVT::v4i8,   MVT::v4i64,  3},  // 2 xtn + 1 uzp1
    { ISD::TRUNCATE, MVT::v4i16,  MVT::v4i32,  1},  // xtn
    { ISD::TRUNCATE, MVT::v4i16,  MVT::v4i64,  2},  // 1 uzp1 + 1 xtn
    { ISD::TRUNCATE, MVT::v4i32,  MVT::v4i64,  1},  // 1 uzp1
    { ISD::TRUNCATE, MVT::v8i8,   MVT::v8i16,  1},  // 1 xtn
    { ISD::TRUNCATE, MVT::v8i8,   MVT::v8i32,  2},  // 1 uzp1 + 1 xtn
    { ISD::TRUNCATE, MVT::v8i8,   MVT::v8i64,  4},  // 3 x uzp1 + xtn
    { ISD::TRUNCATE, MVT::v8i16,  MVT::v8i32,  1},  // 1 uzp1
    { ISD::TRUNCATE, MVT::v8i16,  MVT::v8i64,  3},  // 3 x uzp1
    { ISD::TRUNCATE, MVT::v8i32,  MVT::v8i64,  2},  // 2 x uzp1
    { ISD::TRUNCATE, MVT::v16i8,  MVT::v16i16, 1},  // uzp1
    { ISD::TRUNCATE, MVT::v16i8,  MVT::v16i32, 3},  // (2 + 1) x uzp1
    { ISD::TRUNCATE, MVT::v16i8,  MVT::v16i64, 7},  // (4 + 2 + 1) x uzp1
    { ISD::TRUNCATE, MVT::v16i16, MVT::v16i32, 2},  // 2 x uzp1
    { ISD::TRUNCATE, MVT::v16i16, MVT::v16i64, 6},  // (4 + 2) x uzp1
    { ISD::TRUNCATE, MVT::v16i32, MVT::v16i64, 4},  // 4 x uzp1

    // Truncations on nxvmiN
    { ISD::TRUNCATE, MVT::nxv2i1, MVT::nxv2i16, 1 },
    { ISD::TRUNCATE, MVT::nxv2i1, MVT::nxv2i32, 1 },
    { ISD::TRUNCATE, MVT::nxv2i1, MVT::nxv2i64, 1 },
    { ISD::TRUNCATE, MVT::nxv4i1, MVT::nxv4i16, 1 },
    { ISD::TRUNCATE, MVT::nxv4i1, MVT::nxv4i32, 1 },
    { ISD::TRUNCATE, MVT::nxv4i1, MVT::nxv4i64, 2 },
    { ISD::TRUNCATE, MVT::nxv8i1, MVT::nxv8i16, 1 },
    { ISD::TRUNCATE, MVT::nxv8i1, MVT::nxv8i32, 3 },
    { ISD::TRUNCATE, MVT::nxv8i1, MVT::nxv8i64, 5 },
    { ISD::TRUNCATE, MVT::nxv16i1, MVT::nxv16i8, 1 },
    { ISD::TRUNCATE, MVT::nxv2i16, MVT::nxv2i32, 1 },
    { ISD::TRUNCATE, MVT::nxv2i32, MVT::nxv2i64, 1 },
    { ISD::TRUNCATE, MVT::nxv4i16, MVT::nxv4i32, 1 },
    { ISD::TRUNCATE, MVT::nxv4i32, MVT::nxv4i64, 2 },
    { ISD::TRUNCATE, MVT::nxv8i16, MVT::nxv8i32, 3 },
    { ISD::TRUNCATE, MVT::nxv8i32, MVT::nxv8i64, 6 },

    // The number of shll instructions for the extension.
    { ISD::SIGN_EXTEND, MVT::v4i64,  MVT::v4i16, 3 },
    { ISD::ZERO_EXTEND, MVT::v4i64,  MVT::v4i16, 3 },
    { ISD::SIGN_EXTEND, MVT::v4i64,  MVT::v4i32, 2 },
    { ISD::ZERO_EXTEND, MVT::v4i64,  MVT::v4i32, 2 },
    { ISD::SIGN_EXTEND, MVT::v8i32,  MVT::v8i8,  3 },
    { ISD::ZERO_EXTEND, MVT::v8i32,  MVT::v8i8,  3 },
    { ISD::SIGN_EXTEND, MVT::v8i32,  MVT::v8i16, 2 },
    { ISD::ZERO_EXTEND, MVT::v8i32,  MVT::v8i16, 2 },
    { ISD::SIGN_EXTEND, MVT::v8i64,  MVT::v8i8,  7 },
    { ISD::ZERO_EXTEND, MVT::v8i64,  MVT::v8i8,  7 },
    { ISD::SIGN_EXTEND, MVT::v8i64,  MVT::v8i16, 6 },
    { ISD::ZERO_EXTEND, MVT::v8i64,  MVT::v8i16, 6 },
    { ISD::SIGN_EXTEND, MVT::v16i16, MVT::v16i8, 2 },
    { ISD::ZERO_EXTEND, MVT::v16i16, MVT::v16i8, 2 },
    { ISD::SIGN_EXTEND, MVT::v16i32, MVT::v16i8, 6 },
    { ISD::ZERO_EXTEND, MVT::v16i32, MVT::v16i8, 6 },

    // LowerVectorINT_TO_FP:
    { ISD::SINT_TO_FP, MVT::v2f32, MVT::v2i32, 1 },
    { ISD::SINT_TO_FP, MVT::v4f32, MVT::v4i32, 1 },
    { ISD::SINT_TO_FP, MVT::v2f64, MVT::v2i64, 1 },
    { ISD::UINT_TO_FP, MVT::v2f32, MVT::v2i32, 1 },
    { ISD::UINT_TO_FP, MVT::v4f32, MVT::v4i32, 1 },
    { ISD::UINT_TO_FP, MVT::v2f64, MVT::v2i64, 1 },

    // Complex: to v2f32
    { ISD::SINT_TO_FP, MVT::v2f32, MVT::v2i8,  3 },
    { ISD::SINT_TO_FP, MVT::v2f32, MVT::v2i16, 3 },
    { ISD::SINT_TO_FP, MVT::v2f32, MVT::v2i64, 2 },
    { ISD::UINT_TO_FP, MVT::v2f32, MVT::v2i8,  3 },
    { ISD::UINT_TO_FP, MVT::v2f32, MVT::v2i16, 3 },
    { ISD::UINT_TO_FP, MVT::v2f32, MVT::v2i64, 2 },

    // Complex: to v4f32
    { ISD::SINT_TO_FP, MVT::v4f32, MVT::v4i8,  4 },
    { ISD::SINT_TO_FP, MVT::v4f32, MVT::v4i16, 2 },
    { ISD::UINT_TO_FP, MVT::v4f32, MVT::v4i8,  3 },
    { ISD::UINT_TO_FP, MVT::v4f32, MVT::v4i16, 2 },

    // Complex: to v8f32
    { ISD::SINT_TO_FP, MVT::v8f32, MVT::v8i8,  10 },
    { ISD::SINT_TO_FP, MVT::v8f32, MVT::v8i16, 4 },
    { ISD::UINT_TO_FP, MVT::v8f32, MVT::v8i8,  10 },
    { ISD::UINT_TO_FP, MVT::v8f32, MVT::v8i16, 4 },

    // Complex: to v16f32
    { ISD::SINT_TO_FP, MVT::v16f32, MVT::v16i8, 21 },
    { ISD::UINT_TO_FP, MVT::v16f32, MVT::v16i8, 21 },

    // Complex: to v2f64
    { ISD::SINT_TO_FP, MVT::v2f64, MVT::v2i8,  4 },
    { ISD::SINT_TO_FP, MVT::v2f64, MVT::v2i16, 4 },
    { ISD::SINT_TO_FP, MVT::v2f64, MVT::v2i32, 2 },
    { ISD::UINT_TO_FP, MVT::v2f64, MVT::v2i8,  4 },
    { ISD::UINT_TO_FP, MVT::v2f64, MVT::v2i16, 4 },
    { ISD::UINT_TO_FP, MVT::v2f64, MVT::v2i32, 2 },

    // Complex: to v4f64
    { ISD::SINT_TO_FP, MVT::v4f64, MVT::v4i32,  4 },
    { ISD::UINT_TO_FP, MVT::v4f64, MVT::v4i32,  4 },

    // LowerVectorFP_TO_INT
    { ISD::FP_TO_SINT, MVT::v2i32, MVT::v2f32, 1 },
    { ISD::FP_TO_SINT, MVT::v4i32, MVT::v4f32, 1 },
    { ISD::FP_TO_SINT, MVT::v2i64, MVT::v2f64, 1 },
    { ISD::FP_TO_UINT, MVT::v2i32, MVT::v2f32, 1 },
    { ISD::FP_TO_UINT, MVT::v4i32, MVT::v4f32, 1 },
    { ISD::FP_TO_UINT, MVT::v2i64, MVT::v2f64, 1 },

    // Complex, from v2f32: legal type is v2i32 (no cost) or v2i64 (1 ext).
    { ISD::FP_TO_SINT, MVT::v2i64, MVT::v2f32, 2 },
    { ISD::FP_TO_SINT, MVT::v2i16, MVT::v2f32, 1 },
    { ISD::FP_TO_SINT, MVT::v2i8,  MVT::v2f32, 1 },
    { ISD::FP_TO_UINT, MVT::v2i64, MVT::v2f32, 2 },
    { ISD::FP_TO_UINT, MVT::v2i16, MVT::v2f32, 1 },
    { ISD::FP_TO_UINT, MVT::v2i8,  MVT::v2f32, 1 },

    // Complex, from v4f32: legal type is v4i16, 1 narrowing => ~2
    { ISD::FP_TO_SINT, MVT::v4i16, MVT::v4f32, 2 },
    { ISD::FP_TO_SINT, MVT::v4i8,  MVT::v4f32, 2 },
    { ISD::FP_TO_UINT, MVT::v4i16, MVT::v4f32, 2 },
    { ISD::FP_TO_UINT, MVT::v4i8,  MVT::v4f32, 2 },

    // Complex, from nxv2f32.
    { ISD::FP_TO_SINT, MVT::nxv2i64, MVT::nxv2f32, 1 },
    { ISD::FP_TO_SINT, MVT::nxv2i32, MVT::nxv2f32, 1 },
    { ISD::FP_TO_SINT, MVT::nxv2i16, MVT::nxv2f32, 1 },
    { ISD::FP_TO_SINT, MVT::nxv2i8,  MVT::nxv2f32, 1 },
    { ISD::FP_TO_UINT, MVT::nxv2i64, MVT::nxv2f32, 1 },
    { ISD::FP_TO_UINT, MVT::nxv2i32, MVT::nxv2f32, 1 },
    { ISD::FP_TO_UINT, MVT::nxv2i16, MVT::nxv2f32, 1 },
    { ISD::FP_TO_UINT, MVT::nxv2i8,  MVT::nxv2f32, 1 },

    // Complex, from v2f64: legal type is v2i32, 1 narrowing => ~2.
    { ISD::FP_TO_SINT, MVT::v2i32, MVT::v2f64, 2 },
    { ISD::FP_TO_SINT, MVT::v2i16, MVT::v2f64, 2 },
    { ISD::FP_TO_SINT, MVT::v2i8,  MVT::v2f64, 2 },
    { ISD::FP_TO_UINT, MVT::v2i32, MVT::v2f64, 2 },
    { ISD::FP_TO_UINT, MVT::v2i16, MVT::v2f64, 2 },
    { ISD::FP_TO_UINT, MVT::v2i8,  MVT::v2f64, 2 },

    // Complex, from nxv2f64.
    { ISD::FP_TO_SINT, MVT::nxv2i64, MVT::nxv2f64, 1 },
    { ISD::FP_TO_SINT, MVT::nxv2i32, MVT::nxv2f64, 1 },
    { ISD::FP_TO_SINT, MVT::nxv2i16, MVT::nxv2f64, 1 },
    { ISD::FP_TO_SINT, MVT::nxv2i8,  MVT::nxv2f64, 1 },
    { ISD::FP_TO_UINT, MVT::nxv2i64, MVT::nxv2f64, 1 },
    { ISD::FP_TO_UINT, MVT::nxv2i32, MVT::nxv2f64, 1 },
    { ISD::FP_TO_UINT, MVT::nxv2i16, MVT::nxv2f64, 1 },
    { ISD::FP_TO_UINT, MVT::nxv2i8,  MVT::nxv2f64, 1 },

    // Complex, from nxv4f32.
    { ISD::FP_TO_SINT, MVT::nxv4i64, MVT::nxv4f32, 4 },
    { ISD::FP_TO_SINT, MVT::nxv4i32, MVT::nxv4f32, 1 },
    { ISD::FP_TO_SINT, MVT::nxv4i16, MVT::nxv4f32, 1 },
    { ISD::FP_TO_SINT, MVT::nxv4i8,  MVT::nxv4f32, 1 },
    { ISD::FP_TO_UINT, MVT::nxv4i64, MVT::nxv4f32, 4 },
    { ISD::FP_TO_UINT, MVT::nxv4i32, MVT::nxv4f32, 1 },
    { ISD::FP_TO_UINT, MVT::nxv4i16, MVT::nxv4f32, 1 },
    { ISD::FP_TO_UINT, MVT::nxv4i8,  MVT::nxv4f32, 1 },

    // Complex, from nxv8f64. Illegal -> illegal conversions not required.
    { ISD::FP_TO_SINT, MVT::nxv8i16, MVT::nxv8f64, 7 },
    { ISD::FP_TO_SINT, MVT::nxv8i8,  MVT::nxv8f64, 7 },
    { ISD::FP_TO_UINT, MVT::nxv8i16, MVT::nxv8f64, 7 },
    { ISD::FP_TO_UINT, MVT::nxv8i8,  MVT::nxv8f64, 7 },

    // Complex, from nxv4f64. Illegal -> illegal conversions not required.
    { ISD::FP_TO_SINT, MVT::nxv4i32, MVT::nxv4f64, 3 },
    { ISD::FP_TO_SINT, MVT::nxv4i16, MVT::nxv4f64, 3 },
    { ISD::FP_TO_SINT, MVT::nxv4i8,  MVT::nxv4f64, 3 },
    { ISD::FP_TO_UINT, MVT::nxv4i32, MVT::nxv4f64, 3 },
    { ISD::FP_TO_UINT, MVT::nxv4i16, MVT::nxv4f64, 3 },
    { ISD::FP_TO_UINT, MVT::nxv4i8,  MVT::nxv4f64, 3 },

    // Complex, from nxv8f32. Illegal -> illegal conversions not required.
    { ISD::FP_TO_SINT, MVT::nxv8i16, MVT::nxv8f32, 3 },
    { ISD::FP_TO_SINT, MVT::nxv8i8,  MVT::nxv8f32, 3 },
    { ISD::FP_TO_UINT, MVT::nxv8i16, MVT::nxv8f32, 3 },
    { ISD::FP_TO_UINT, MVT::nxv8i8,  MVT::nxv8f32, 3 },

    // Complex, from nxv8f16.
    { ISD::FP_TO_SINT, MVT::nxv8i64, MVT::nxv8f16, 10 },
    { ISD::FP_TO_SINT, MVT::nxv8i32, MVT::nxv8f16, 4 },
    { ISD::FP_TO_SINT, MVT::nxv8i16, MVT::nxv8f16, 1 },
    { ISD::FP_TO_SINT, MVT::nxv8i8,  MVT::nxv8f16, 1 },
    { ISD::FP_TO_UINT, MVT::nxv8i64, MVT::nxv8f16, 10 },
    { ISD::FP_TO_UINT, MVT::nxv8i32, MVT::nxv8f16, 4 },
    { ISD::FP_TO_UINT, MVT::nxv8i16, MVT::nxv8f16, 1 },
    { ISD::FP_TO_UINT, MVT::nxv8i8,  MVT::nxv8f16, 1 },

    // Complex, from nxv4f16.
    { ISD::FP_TO_SINT, MVT::nxv4i64, MVT::nxv4f16, 4 },
    { ISD::FP_TO_SINT, MVT::nxv4i32, MVT::nxv4f16, 1 },
    { ISD::FP_TO_SINT, MVT::nxv4i16, MVT::nxv4f16, 1 },
    { ISD::FP_TO_SINT, MVT::nxv4i8,  MVT::nxv4f16, 1 },
    { ISD::FP_TO_UINT, MVT::nxv4i64, MVT::nxv4f16, 4 },
    { ISD::FP_TO_UINT, MVT::nxv4i32, MVT::nxv4f16, 1 },
    { ISD::FP_TO_UINT, MVT::nxv4i16, MVT::nxv4f16, 1 },
    { ISD::FP_TO_UINT, MVT::nxv4i8,  MVT::nxv4f16, 1 },

    // Complex, from nxv2f16.
    { ISD::FP_TO_SINT, MVT::nxv2i64, MVT::nxv2f16, 1 },
    { ISD::FP_TO_SINT, MVT::nxv2i32, MVT::nxv2f16, 1 },
    { ISD::FP_TO_SINT, MVT::nxv2i16, MVT::nxv2f16, 1 },
    { ISD::FP_TO_SINT, MVT::nxv2i8,  MVT::nxv2f16, 1 },
    { ISD::FP_TO_UINT, MVT::nxv2i64, MVT::nxv2f16, 1 },
    { ISD::FP_TO_UINT, MVT::nxv2i32, MVT::nxv2f16, 1 },
    { ISD::FP_TO_UINT, MVT::nxv2i16, MVT::nxv2f16, 1 },
    { ISD::FP_TO_UINT, MVT::nxv2i8,  MVT::nxv2f16, 1 },

    // Truncate from nxvmf32 to nxvmf16.
    { ISD::FP_ROUND, MVT::nxv2f16, MVT::nxv2f32, 1 },
    { ISD::FP_ROUND, MVT::nxv4f16, MVT::nxv4f32, 1 },
    { ISD::FP_ROUND, MVT::nxv8f16, MVT::nxv8f32, 3 },

    // Truncate from nxvmf64 to nxvmf16.
    { ISD::FP_ROUND, MVT::nxv2f16, MVT::nxv2f64, 1 },
    { ISD::FP_ROUND, MVT::nxv4f16, MVT::nxv4f64, 3 },
    { ISD::FP_ROUND, MVT::nxv8f16, MVT::nxv8f64, 7 },

    // Truncate from nxvmf64 to nxvmf32.
    { ISD::FP_ROUND, MVT::nxv2f32, MVT::nxv2f64, 1 },
    { ISD::FP_ROUND, MVT::nxv4f32, MVT::nxv4f64, 3 },
    { ISD::FP_ROUND, MVT::nxv8f32, MVT::nxv8f64, 6 },

    // Extend from nxvmf16 to nxvmf32.
    { ISD::FP_EXTEND, MVT::nxv2f32, MVT::nxv2f16, 1},
    { ISD::FP_EXTEND, MVT::nxv4f32, MVT::nxv4f16, 1},
    { ISD::FP_EXTEND, MVT::nxv8f32, MVT::nxv8f16, 2},

    // Extend from nxvmf16 to nxvmf64.
    { ISD::FP_EXTEND, MVT::nxv2f64, MVT::nxv2f16, 1},
    { ISD::FP_EXTEND, MVT::nxv4f64, MVT::nxv4f16, 2},
    { ISD::FP_EXTEND, MVT::nxv8f64, MVT::nxv8f16, 4},

    // Extend from nxvmf32 to nxvmf64.
    { ISD::FP_EXTEND, MVT::nxv2f64, MVT::nxv2f32, 1},
    { ISD::FP_EXTEND, MVT::nxv4f64, MVT::nxv4f32, 2},
    { ISD::FP_EXTEND, MVT::nxv8f64, MVT::nxv8f32, 6},

    // Bitcasts from float to integer
    { ISD::BITCAST, MVT::nxv2f16, MVT::nxv2i16, 0 },
    { ISD::BITCAST, MVT::nxv4f16, MVT::nxv4i16, 0 },
    { ISD::BITCAST, MVT::nxv2f32, MVT::nxv2i32, 0 },

    // Bitcasts from integer to float
    { ISD::BITCAST, MVT::nxv2i16, MVT::nxv2f16, 0 },
    { ISD::BITCAST, MVT::nxv4i16, MVT::nxv4f16, 0 },
    { ISD::BITCAST, MVT::nxv2i32, MVT::nxv2f32, 0 },
  };

  if (const auto *Entry = ConvertCostTableLookup(ConversionTbl, ISD,
                                                 DstTy.getSimpleVT(),
                                                 SrcTy.getSimpleVT()))
    return AdjustCost(Entry->Cost);

  static const TypeConversionCostTblEntry FP16Tbl[] = {
      {ISD::FP_TO_SINT, MVT::v4i8, MVT::v4f16, 1}, // fcvtzs
      {ISD::FP_TO_UINT, MVT::v4i8, MVT::v4f16, 1},
      {ISD::FP_TO_SINT, MVT::v4i16, MVT::v4f16, 1}, // fcvtzs
      {ISD::FP_TO_UINT, MVT::v4i16, MVT::v4f16, 1},
      {ISD::FP_TO_SINT, MVT::v4i32, MVT::v4f16, 2}, // fcvtl+fcvtzs
      {ISD::FP_TO_UINT, MVT::v4i32, MVT::v4f16, 2},
      {ISD::FP_TO_SINT, MVT::v8i8, MVT::v8f16, 2}, // fcvtzs+xtn
      {ISD::FP_TO_UINT, MVT::v8i8, MVT::v8f16, 2},
      {ISD::FP_TO_SINT, MVT::v8i16, MVT::v8f16, 1}, // fcvtzs
      {ISD::FP_TO_UINT, MVT::v8i16, MVT::v8f16, 1},
      {ISD::FP_TO_SINT, MVT::v8i32, MVT::v8f16, 4}, // 2*fcvtl+2*fcvtzs
      {ISD::FP_TO_UINT, MVT::v8i32, MVT::v8f16, 4},
      {ISD::FP_TO_SINT, MVT::v16i8, MVT::v16f16, 3}, // 2*fcvtzs+xtn
      {ISD::FP_TO_UINT, MVT::v16i8, MVT::v16f16, 3},
      {ISD::FP_TO_SINT, MVT::v16i16, MVT::v16f16, 2}, // 2*fcvtzs
      {ISD::FP_TO_UINT, MVT::v16i16, MVT::v16f16, 2},
      {ISD::FP_TO_SINT, MVT::v16i32, MVT::v16f16, 8}, // 4*fcvtl+4*fcvtzs
      {ISD::FP_TO_UINT, MVT::v16i32, MVT::v16f16, 8},
      {ISD::UINT_TO_FP, MVT::v8f16, MVT::v8i8, 2},   // ushll + ucvtf
      {ISD::SINT_TO_FP, MVT::v8f16, MVT::v8i8, 2},   // sshll + scvtf
      {ISD::UINT_TO_FP, MVT::v16f16, MVT::v16i8, 4}, // 2 * ushl(2) + 2 * ucvtf
      {ISD::SINT_TO_FP, MVT::v16f16, MVT::v16i8, 4}, // 2 * sshl(2) + 2 * scvtf
  };

  if (ST->hasFullFP16())
    if (const auto *Entry = ConvertCostTableLookup(
            FP16Tbl, ISD, DstTy.getSimpleVT(), SrcTy.getSimpleVT()))
      return AdjustCost(Entry->Cost);

  return AdjustCost(
      BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I));
}

InstructionCost AArch64TTIImpl::getExtractWithExtendCost(unsigned Opcode,
                                                         Type *Dst,
                                                         VectorType *VecTy,
                                                         unsigned Index) {

  // Make sure we were given a valid extend opcode.
  assert((Opcode == Instruction::SExt || Opcode == Instruction::ZExt) &&
         "Invalid opcode");

  // We are extending an element we extract from a vector, so the source type
  // of the extend is the element type of the vector.
  auto *Src = VecTy->getElementType();

  // Sign- and zero-extends are for integer types only.
  assert(isa<IntegerType>(Dst) && isa<IntegerType>(Src) && "Invalid type");

  // Get the cost for the extract. We compute the cost (if any) for the extend
  // below.
  InstructionCost Cost =
      getVectorInstrCost(Instruction::ExtractElement, VecTy, Index);

  // Legalize the types.
  auto VecLT = getTypeLegalizationCost(VecTy);
  auto DstVT = TLI->getValueType(DL, Dst);
  auto SrcVT = TLI->getValueType(DL, Src);
  TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;

  // If the resulting type is still a vector and the destination type is legal,
  // we may get the extension for free. If not, get the default cost for the
  // extend.
  if (!VecLT.second.isVector() || !TLI->isTypeLegal(DstVT))
    return Cost + getCastInstrCost(Opcode, Dst, Src, TTI::CastContextHint::None,
                                   CostKind);

  // The destination type should be larger than the element type. If not, get
  // the default cost for the extend.
  if (DstVT.getFixedSizeInBits() < SrcVT.getFixedSizeInBits())
    return Cost + getCastInstrCost(Opcode, Dst, Src, TTI::CastContextHint::None,
                                   CostKind);

  switch (Opcode) {
  default:
    llvm_unreachable("Opcode should be either SExt or ZExt");

  // For sign-extends, we only need a smov, which performs the extension
  // automatically.
  case Instruction::SExt:
    return Cost;

  // For zero-extends, the extend is performed automatically by a umov unless
  // the destination type is i64 and the element type is i8 or i16.
  case Instruction::ZExt:
    if (DstVT.getSizeInBits() != 64u || SrcVT.getSizeInBits() == 32u)
      return Cost;
  }

  // If we are unable to perform the extend for free, get the default cost.
  return Cost + getCastInstrCost(Opcode, Dst, Src, TTI::CastContextHint::None,
                                 CostKind);
}

InstructionCost AArch64TTIImpl::getCFInstrCost(unsigned Opcode,
                                               TTI::TargetCostKind CostKind,
                                               const Instruction *I) {
  if (CostKind != TTI::TCK_RecipThroughput)
    return Opcode == Instruction::PHI ? 0 : 1;
  assert(CostKind == TTI::TCK_RecipThroughput && "unexpected CostKind");
  // Branches are assumed to be predicted.
  return 0;
}

InstructionCost AArch64TTIImpl::getVectorInstrCostHelper(Type *Val,
                                                         unsigned Index,
                                                         bool HasRealUse) {
  assert(Val->isVectorTy() && "This must be a vector type");

  if (Index != -1U) {
    // Legalize the type.
    std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Val);

    // This type is legalized to a scalar type.
    if (!LT.second.isVector())
      return 0;

    // The type may be split. For fixed-width vectors we can normalize the
    // index to the new type.
    if (LT.second.isFixedLengthVector()) {
      unsigned Width = LT.second.getVectorNumElements();
      Index = Index % Width;
    }

    // The element at index zero is already inside the vector.
    // - For a physical (HasRealUse==true) insert-element or extract-element
    // instruction that extracts integers, an explicit FPR -> GPR move is
    // needed. So it has non-zero cost.
    // - For the rest of cases (virtual instruction or element type is float),
    // consider the instruction free.
    //
    // FIXME:
    // If the extract-element and insert-element instructions could be
    // simplified away (e.g., could be combined into users by looking at use-def
    // context), they have no cost. This is not done in the first place for
    // compile-time considerations.
    if (Index == 0 && (!HasRealUse || !Val->getScalarType()->isIntegerTy()))
      return 0;
  }

  // All other insert/extracts cost this much.
  return ST->getVectorInsertExtractBaseCost();
}

InstructionCost AArch64TTIImpl::getVectorInstrCost(unsigned Opcode, Type *Val,
                                                   unsigned Index) {
  return getVectorInstrCostHelper(Val, Index, false /* HasRealUse */);
}

InstructionCost AArch64TTIImpl::getVectorInstrCost(const Instruction &I,
                                                   Type *Val, unsigned Index) {
  return getVectorInstrCostHelper(Val, Index, true /* HasRealUse */);
}

InstructionCost AArch64TTIImpl::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
    TTI::OperandValueInfo Op1Info, TTI::OperandValueInfo Op2Info,
    ArrayRef<const Value *> Args,
    const Instruction *CxtI) {

  // TODO: Handle more cost kinds.
  if (CostKind != TTI::TCK_RecipThroughput)
    return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info,
                                         Op2Info, Args, CxtI);

  // Legalize the type.
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Ty);
  int ISD = TLI->InstructionOpcodeToISD(Opcode);

  switch (ISD) {
  default:
    return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info,
                                         Op2Info);
  case ISD::SDIV:
    if (Op2Info.isConstant() && Op2Info.isUniform() && Op2Info.isPowerOf2()) {
      // On AArch64, scalar signed division by constants power-of-two are
      // normally expanded to the sequence ADD + CMP + SELECT + SRA.
      // The OperandValue properties many not be same as that of previous
      // operation; conservatively assume OP_None.
      InstructionCost Cost = getArithmeticInstrCost(
          Instruction::Add, Ty, CostKind,
          Op1Info.getNoProps(), Op2Info.getNoProps());
      Cost += getArithmeticInstrCost(Instruction::Sub, Ty, CostKind,
                                     Op1Info.getNoProps(), Op2Info.getNoProps());
      Cost += getArithmeticInstrCost(
          Instruction::Select, Ty, CostKind,
          Op1Info.getNoProps(), Op2Info.getNoProps());
      Cost += getArithmeticInstrCost(Instruction::AShr, Ty, CostKind,
                                     Op1Info.getNoProps(), Op2Info.getNoProps());
      return Cost;
    }
    [[fallthrough]];
  case ISD::UDIV: {
    if (Op2Info.isConstant() && Op2Info.isUniform()) {
      auto VT = TLI->getValueType(DL, Ty);
      if (TLI->isOperationLegalOrCustom(ISD::MULHU, VT)) {
        // Vector signed division by constant are expanded to the
        // sequence MULHS + ADD/SUB + SRA + SRL + ADD, and unsigned division
        // to MULHS + SUB + SRL + ADD + SRL.
        InstructionCost MulCost = getArithmeticInstrCost(
            Instruction::Mul, Ty, CostKind, Op1Info.getNoProps(), Op2Info.getNoProps());
        InstructionCost AddCost = getArithmeticInstrCost(
            Instruction::Add, Ty, CostKind, Op1Info.getNoProps(), Op2Info.getNoProps());
        InstructionCost ShrCost = getArithmeticInstrCost(
            Instruction::AShr, Ty, CostKind, Op1Info.getNoProps(), Op2Info.getNoProps());
        return MulCost * 2 + AddCost * 2 + ShrCost * 2 + 1;
      }
    }

    InstructionCost Cost = BaseT::getArithmeticInstrCost(
        Opcode, Ty, CostKind, Op1Info, Op2Info);
    if (Ty->isVectorTy()) {
      // On AArch64, vector divisions are not supported natively and are
      // expanded into scalar divisions of each pair of elements.
      Cost += getArithmeticInstrCost(Instruction::ExtractElement, Ty, CostKind,
                                     Op1Info, Op2Info);
      Cost += getArithmeticInstrCost(Instruction::InsertElement, Ty, CostKind,
                                     Op1Info, Op2Info);
      // TODO: if one of the arguments is scalar, then it's not necessary to
      // double the cost of handling the vector elements.
      Cost += Cost;
    }
    return Cost;
  }
  case ISD::MUL:
    // Since we do not have a MUL.2d instruction, a mul <2 x i64> is expensive
    // as elements are extracted from the vectors and the muls scalarized.
    // As getScalarizationOverhead is a bit too pessimistic, we estimate the
    // cost for a i64 vector directly here, which is:
    // - four 2-cost i64 extracts,
    // - two 2-cost i64 inserts, and
    // - two 1-cost muls.
    // So, for a v2i64 with LT.First = 1 the cost is 14, and for a v4i64 with
    // LT.first = 2 the cost is 28. If both operands are extensions it will not
    // need to scalarize so the cost can be cheaper (smull or umull).
    if (LT.second != MVT::v2i64 || isWideningInstruction(Ty, Opcode, Args))
      return LT.first;
    return LT.first * 14;
  case ISD::ADD:
  case ISD::XOR:
  case ISD::OR:
  case ISD::AND:
  case ISD::SRL:
  case ISD::SRA:
  case ISD::SHL:
    // These nodes are marked as 'custom' for combining purposes only.
    // We know that they are legal. See LowerAdd in ISelLowering.
    return LT.first;

  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
  case ISD::FDIV:
  case ISD::FNEG:
    // These nodes are marked as 'custom' just to lower them to SVE.
    // We know said lowering will incur no additional cost.
    if (!Ty->getScalarType()->isFP128Ty())
      return 2 * LT.first;

    return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info,
                                         Op2Info);
  }
}

InstructionCost AArch64TTIImpl::getAddressComputationCost(Type *Ty,
                                                          ScalarEvolution *SE,
                                                          const SCEV *Ptr) {
  // Address computations in vectorized code with non-consecutive addresses will
  // likely result in more instructions compared to scalar code where the
  // computation can more often be merged into the index mode. The resulting
  // extra micro-ops can significantly decrease throughput.
  unsigned NumVectorInstToHideOverhead = 10;
  int MaxMergeDistance = 64;

  if (Ty->isVectorTy() && SE &&
      !BaseT::isConstantStridedAccessLessThan(SE, Ptr, MaxMergeDistance + 1))
    return NumVectorInstToHideOverhead;

  // In many cases the address computation is not merged into the instruction
  // addressing mode.
  return 1;
}

InstructionCost AArch64TTIImpl::getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                                   Type *CondTy,
                                                   CmpInst::Predicate VecPred,
                                                   TTI::TargetCostKind CostKind,
                                                   const Instruction *I) {
  // TODO: Handle other cost kinds.
  if (CostKind != TTI::TCK_RecipThroughput)
    return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind,
                                     I);

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  // We don't lower some vector selects well that are wider than the register
  // width.
  if (isa<FixedVectorType>(ValTy) && ISD == ISD::SELECT) {
    // We would need this many instructions to hide the scalarization happening.
    const int AmortizationCost = 20;

    // If VecPred is not set, check if we can get a predicate from the context
    // instruction, if its type matches the requested ValTy.
    if (VecPred == CmpInst::BAD_ICMP_PREDICATE && I && I->getType() == ValTy) {
      CmpInst::Predicate CurrentPred;
      if (match(I, m_Select(m_Cmp(CurrentPred, m_Value(), m_Value()), m_Value(),
                            m_Value())))
        VecPred = CurrentPred;
    }
    // Check if we have a compare/select chain that can be lowered using
    // a (F)CMxx & BFI pair.
    if (CmpInst::isIntPredicate(VecPred) || VecPred == CmpInst::FCMP_OLE ||
        VecPred == CmpInst::FCMP_OLT || VecPred == CmpInst::FCMP_OGT ||
        VecPred == CmpInst::FCMP_OGE || VecPred == CmpInst::FCMP_OEQ ||
        VecPred == CmpInst::FCMP_UNE) {
      static const auto ValidMinMaxTys = {
          MVT::v8i8,  MVT::v16i8, MVT::v4i16, MVT::v8i16, MVT::v2i32,
          MVT::v4i32, MVT::v2i64, MVT::v2f32, MVT::v4f32, MVT::v2f64};
      static const auto ValidFP16MinMaxTys = {MVT::v4f16, MVT::v8f16};

      auto LT = getTypeLegalizationCost(ValTy);
      if (any_of(ValidMinMaxTys, [&LT](MVT M) { return M == LT.second; }) ||
          (ST->hasFullFP16() &&
           any_of(ValidFP16MinMaxTys, [&LT](MVT M) { return M == LT.second; })))
        return LT.first;
    }

    static const TypeConversionCostTblEntry
    VectorSelectTbl[] = {
      { ISD::SELECT, MVT::v16i1, MVT::v16i16, 16 },
      { ISD::SELECT, MVT::v8i1, MVT::v8i32, 8 },
      { ISD::SELECT, MVT::v16i1, MVT::v16i32, 16 },
      { ISD::SELECT, MVT::v4i1, MVT::v4i64, 4 * AmortizationCost },
      { ISD::SELECT, MVT::v8i1, MVT::v8i64, 8 * AmortizationCost },
      { ISD::SELECT, MVT::v16i1, MVT::v16i64, 16 * AmortizationCost }
    };

    EVT SelCondTy = TLI->getValueType(DL, CondTy);
    EVT SelValTy = TLI->getValueType(DL, ValTy);
    if (SelCondTy.isSimple() && SelValTy.isSimple()) {
      if (const auto *Entry = ConvertCostTableLookup(VectorSelectTbl, ISD,
                                                     SelCondTy.getSimpleVT(),
                                                     SelValTy.getSimpleVT()))
        return Entry->Cost;
    }
  }
  // The base case handles scalable vectors fine for now, since it treats the
  // cost as 1 * legalization cost.
  return BaseT::getCmpSelInstrCost(Opcode, ValTy, CondTy, VecPred, CostKind, I);
}

AArch64TTIImpl::TTI::MemCmpExpansionOptions
AArch64TTIImpl::enableMemCmpExpansion(bool OptSize, bool IsZeroCmp) const {
  TTI::MemCmpExpansionOptions Options;
  if (ST->requiresStrictAlign()) {
    // TODO: Add cost modeling for strict align. Misaligned loads expand to
    // a bunch of instructions when strict align is enabled.
    return Options;
  }
  Options.AllowOverlappingLoads = true;
  Options.MaxNumLoads = TLI->getMaxExpandSizeMemcmp(OptSize);
  Options.NumLoadsPerBlock = Options.MaxNumLoads;
  // TODO: Though vector loads usually perform well on AArch64, in some targets
  // they may wake up the FP unit, which raises the power consumption.  Perhaps
  // they could be used with no holds barred (-O3).
  Options.LoadSizes = {8, 4, 2, 1};
  return Options;
}

bool AArch64TTIImpl::prefersVectorizedAddressing() const {
  return ST->hasSVE();
}

InstructionCost
AArch64TTIImpl::getMaskedMemoryOpCost(unsigned Opcode, Type *Src,
                                      Align Alignment, unsigned AddressSpace,
                                      TTI::TargetCostKind CostKind) {
  if (useNeonVector(Src))
    return BaseT::getMaskedMemoryOpCost(Opcode, Src, Alignment, AddressSpace,
                                        CostKind);
  auto LT = getTypeLegalizationCost(Src);
  if (!LT.first.isValid())
    return InstructionCost::getInvalid();

  // The code-generator is currently not able to handle scalable vectors
  // of <vscale x 1 x eltty> yet, so return an invalid cost to avoid selecting
  // it. This change will be removed when code-generation for these types is
  // sufficiently reliable.
  if (cast<VectorType>(Src)->getElementCount() == ElementCount::getScalable(1))
    return InstructionCost::getInvalid();

  return LT.first * 2;
}

static unsigned getSVEGatherScatterOverhead(unsigned Opcode) {
  return Opcode == Instruction::Load ? SVEGatherOverhead : SVEScatterOverhead;
}

InstructionCost AArch64TTIImpl::getGatherScatterOpCost(
    unsigned Opcode, Type *DataTy, const Value *Ptr, bool VariableMask,
    Align Alignment, TTI::TargetCostKind CostKind, const Instruction *I) {
  if (useNeonVector(DataTy))
    return BaseT::getGatherScatterOpCost(Opcode, DataTy, Ptr, VariableMask,
                                         Alignment, CostKind, I);
  auto *VT = cast<VectorType>(DataTy);
  auto LT = getTypeLegalizationCost(DataTy);
  if (!LT.first.isValid())
    return InstructionCost::getInvalid();

  // The code-generator is currently not able to handle scalable vectors
  // of <vscale x 1 x eltty> yet, so return an invalid cost to avoid selecting
  // it. This change will be removed when code-generation for these types is
  // sufficiently reliable.
  if (cast<VectorType>(DataTy)->getElementCount() ==
      ElementCount::getScalable(1))
    return InstructionCost::getInvalid();

  ElementCount LegalVF = LT.second.getVectorElementCount();
  InstructionCost MemOpCost =
      getMemoryOpCost(Opcode, VT->getElementType(), Alignment, 0, CostKind,
                      {TTI::OK_AnyValue, TTI::OP_None}, I);
  // Add on an overhead cost for using gathers/scatters.
  // TODO: At the moment this is applied unilaterally for all CPUs, but at some
  // point we may want a per-CPU overhead.
  MemOpCost *= getSVEGatherScatterOverhead(Opcode);
  return LT.first * MemOpCost * getMaxNumElements(LegalVF);
}

bool AArch64TTIImpl::useNeonVector(const Type *Ty) const {
  return isa<FixedVectorType>(Ty) && !ST->useSVEForFixedLengthVectors();
}

InstructionCost AArch64TTIImpl::getMemoryOpCost(unsigned Opcode, Type *Ty,
                                                MaybeAlign Alignment,
                                                unsigned AddressSpace,
                                                TTI::TargetCostKind CostKind,
                                                TTI::OperandValueInfo OpInfo,
                                                const Instruction *I) {
  EVT VT = TLI->getValueType(DL, Ty, true);
  // Type legalization can't handle structs
  if (VT == MVT::Other)
    return BaseT::getMemoryOpCost(Opcode, Ty, Alignment, AddressSpace,
                                  CostKind);

  auto LT = getTypeLegalizationCost(Ty);
  if (!LT.first.isValid())
    return InstructionCost::getInvalid();

  // The code-generator is currently not able to handle scalable vectors
  // of <vscale x 1 x eltty> yet, so return an invalid cost to avoid selecting
  // it. This change will be removed when code-generation for these types is
  // sufficiently reliable.
  if (auto *VTy = dyn_cast<ScalableVectorType>(Ty))
    if (VTy->getElementCount() == ElementCount::getScalable(1))
      return InstructionCost::getInvalid();

  // TODO: consider latency as well for TCK_SizeAndLatency.
  if (CostKind == TTI::TCK_CodeSize || CostKind == TTI::TCK_SizeAndLatency)
    return LT.first;

  if (CostKind != TTI::TCK_RecipThroughput)
    return 1;

  if (ST->isMisaligned128StoreSlow() && Opcode == Instruction::Store &&
      LT.second.is128BitVector() && (!Alignment || *Alignment < Align(16))) {
    // Unaligned stores are extremely inefficient. We don't split all
    // unaligned 128-bit stores because the negative impact that has shown in
    // practice on inlined block copy code.
    // We make such stores expensive so that we will only vectorize if there
    // are 6 other instructions getting vectorized.
    const int AmortizationCost = 6;

    return LT.first * 2 * AmortizationCost;
  }

  // Check truncating stores and extending loads.
  if (useNeonVector(Ty) &&
      Ty->getScalarSizeInBits() != LT.second.getScalarSizeInBits()) {
    // v4i8 types are lowered to scalar a load/store and sshll/xtn.
    if (VT == MVT::v4i8)
      return 2;
    // Otherwise we need to scalarize.
    return cast<FixedVectorType>(Ty)->getNumElements() * 2;
  }

  return LT.first;
}

InstructionCost AArch64TTIImpl::getInterleavedMemoryOpCost(
    unsigned Opcode, Type *VecTy, unsigned Factor, ArrayRef<unsigned> Indices,
    Align Alignment, unsigned AddressSpace, TTI::TargetCostKind CostKind,
    bool UseMaskForCond, bool UseMaskForGaps) {
  assert(Factor >= 2 && "Invalid interleave factor");
  auto *VecVTy = cast<FixedVectorType>(VecTy);

  if (!UseMaskForCond && !UseMaskForGaps &&
      Factor <= TLI->getMaxSupportedInterleaveFactor()) {
    unsigned NumElts = VecVTy->getNumElements();
    auto *SubVecTy =
        FixedVectorType::get(VecTy->getScalarType(), NumElts / Factor);

    // ldN/stN only support legal vector types of size 64 or 128 in bits.
    // Accesses having vector types that are a multiple of 128 bits can be
    // matched to more than one ldN/stN instruction.
    bool UseScalable;
    if (NumElts % Factor == 0 &&
        TLI->isLegalInterleavedAccessType(SubVecTy, DL, UseScalable))
      return Factor * TLI->getNumInterleavedAccesses(SubVecTy, DL, UseScalable);
  }

  return BaseT::getInterleavedMemoryOpCost(Opcode, VecTy, Factor, Indices,
                                           Alignment, AddressSpace, CostKind,
                                           UseMaskForCond, UseMaskForGaps);
}

InstructionCost
AArch64TTIImpl::getCostOfKeepingLiveOverCall(ArrayRef<Type *> Tys) {
  InstructionCost Cost = 0;
  TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
  for (auto *I : Tys) {
    if (!I->isVectorTy())
      continue;
    if (I->getScalarSizeInBits() * cast<FixedVectorType>(I)->getNumElements() ==
        128)
      Cost += getMemoryOpCost(Instruction::Store, I, Align(128), 0, CostKind) +
              getMemoryOpCost(Instruction::Load, I, Align(128), 0, CostKind);
  }
  return Cost;
}

unsigned AArch64TTIImpl::getMaxInterleaveFactor(unsigned VF) {
  return ST->getMaxInterleaveFactor();
}

// For Falkor, we want to avoid having too many strided loads in a loop since
// that can exhaust the HW prefetcher resources.  We adjust the unroller
// MaxCount preference below to attempt to ensure unrolling doesn't create too
// many strided loads.
static void
getFalkorUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                              TargetTransformInfo::UnrollingPreferences &UP) {
  enum { MaxStridedLoads = 7 };
  auto countStridedLoads = [](Loop *L, ScalarEvolution &SE) {
    int StridedLoads = 0;
    // FIXME? We could make this more precise by looking at the CFG and
    // e.g. not counting loads in each side of an if-then-else diamond.
    for (const auto BB : L->blocks()) {
      for (auto &I : *BB) {
        LoadInst *LMemI = dyn_cast<LoadInst>(&I);
        if (!LMemI)
          continue;

        Value *PtrValue = LMemI->getPointerOperand();
        if (L->isLoopInvariant(PtrValue))
          continue;

        const SCEV *LSCEV = SE.getSCEV(PtrValue);
        const SCEVAddRecExpr *LSCEVAddRec = dyn_cast<SCEVAddRecExpr>(LSCEV);
        if (!LSCEVAddRec || !LSCEVAddRec->isAffine())
          continue;

        // FIXME? We could take pairing of unrolled load copies into account
        // by looking at the AddRec, but we would probably have to limit this
        // to loops with no stores or other memory optimization barriers.
        ++StridedLoads;
        // We've seen enough strided loads that seeing more won't make a
        // difference.
        if (StridedLoads > MaxStridedLoads / 2)
          return StridedLoads;
      }
    }
    return StridedLoads;
  };

  int StridedLoads = countStridedLoads(L, SE);
  LLVM_DEBUG(dbgs() << "falkor-hwpf: detected " << StridedLoads
                    << " strided loads\n");
  // Pick the largest power of 2 unroll count that won't result in too many
  // strided loads.
  if (StridedLoads) {
    UP.MaxCount = 1 << Log2_32(MaxStridedLoads / StridedLoads);
    LLVM_DEBUG(dbgs() << "falkor-hwpf: setting unroll MaxCount to "
                      << UP.MaxCount << '\n');
  }
}

void AArch64TTIImpl::getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                                             TTI::UnrollingPreferences &UP,
                                             OptimizationRemarkEmitter *ORE) {
  // Enable partial unrolling and runtime unrolling.
  BaseT::getUnrollingPreferences(L, SE, UP, ORE);

  UP.UpperBound = true;

  // For inner loop, it is more likely to be a hot one, and the runtime check
  // can be promoted out from LICM pass, so the overhead is less, let's try
  // a larger threshold to unroll more loops.
  if (L->getLoopDepth() > 1)
    UP.PartialThreshold *= 2;

  // Disable partial & runtime unrolling on -Os.
  UP.PartialOptSizeThreshold = 0;

  if (ST->getProcFamily() == AArch64Subtarget::Falkor &&
      EnableFalkorHWPFUnrollFix)
    getFalkorUnrollingPreferences(L, SE, UP);

  // Scan the loop: don't unroll loops with calls as this could prevent
  // inlining. Don't unroll vector loops either, as they don't benefit much from
  // unrolling.
  for (auto *BB : L->getBlocks()) {
    for (auto &I : *BB) {
      // Don't unroll vectorised loop.
      if (I.getType()->isVectorTy())
        return;

      if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
        if (const Function *F = cast<CallBase>(I).getCalledFunction()) {
          if (!isLoweredToCall(F))
            continue;
        }
        return;
      }
    }
  }

  // Enable runtime unrolling for in-order models
  // If mcpu is omitted, getProcFamily() returns AArch64Subtarget::Others, so by
  // checking for that case, we can ensure that the default behaviour is
  // unchanged
  if (ST->getProcFamily() != AArch64Subtarget::Others &&
      !ST->getSchedModel().isOutOfOrder()) {
    UP.Runtime = true;
    UP.Partial = true;
    UP.UnrollRemainder = true;
    UP.DefaultUnrollRuntimeCount = 4;

    UP.UnrollAndJam = true;
    UP.UnrollAndJamInnerLoopThreshold = 60;
  }
}

void AArch64TTIImpl::getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                                           TTI::PeelingPreferences &PP) {
  BaseT::getPeelingPreferences(L, SE, PP);
}

Value *AArch64TTIImpl::getOrCreateResultFromMemIntrinsic(IntrinsicInst *Inst,
                                                         Type *ExpectedType) {
  switch (Inst->getIntrinsicID()) {
  default:
    return nullptr;
  case Intrinsic::aarch64_neon_st2:
  case Intrinsic::aarch64_neon_st3:
  case Intrinsic::aarch64_neon_st4: {
    // Create a struct type
    StructType *ST = dyn_cast<StructType>(ExpectedType);
    if (!ST)
      return nullptr;
    unsigned NumElts = Inst->arg_size() - 1;
    if (ST->getNumElements() != NumElts)
      return nullptr;
    for (unsigned i = 0, e = NumElts; i != e; ++i) {
      if (Inst->getArgOperand(i)->getType() != ST->getElementType(i))
        return nullptr;
    }
    Value *Res = UndefValue::get(ExpectedType);
    IRBuilder<> Builder(Inst);
    for (unsigned i = 0, e = NumElts; i != e; ++i) {
      Value *L = Inst->getArgOperand(i);
      Res = Builder.CreateInsertValue(Res, L, i);
    }
    return Res;
  }
  case Intrinsic::aarch64_neon_ld2:
  case Intrinsic::aarch64_neon_ld3:
  case Intrinsic::aarch64_neon_ld4:
    if (Inst->getType() == ExpectedType)
      return Inst;
    return nullptr;
  }
}

bool AArch64TTIImpl::getTgtMemIntrinsic(IntrinsicInst *Inst,
                                        MemIntrinsicInfo &Info) {
  switch (Inst->getIntrinsicID()) {
  default:
    break;
  case Intrinsic::aarch64_neon_ld2:
  case Intrinsic::aarch64_neon_ld3:
  case Intrinsic::aarch64_neon_ld4:
    Info.ReadMem = true;
    Info.WriteMem = false;
    Info.PtrVal = Inst->getArgOperand(0);
    break;
  case Intrinsic::aarch64_neon_st2:
  case Intrinsic::aarch64_neon_st3:
  case Intrinsic::aarch64_neon_st4:
    Info.ReadMem = false;
    Info.WriteMem = true;
    Info.PtrVal = Inst->getArgOperand(Inst->arg_size() - 1);
    break;
  }

  switch (Inst->getIntrinsicID()) {
  default:
    return false;
  case Intrinsic::aarch64_neon_ld2:
  case Intrinsic::aarch64_neon_st2:
    Info.MatchingId = VECTOR_LDST_TWO_ELEMENTS;
    break;
  case Intrinsic::aarch64_neon_ld3:
  case Intrinsic::aarch64_neon_st3:
    Info.MatchingId = VECTOR_LDST_THREE_ELEMENTS;
    break;
  case Intrinsic::aarch64_neon_ld4:
  case Intrinsic::aarch64_neon_st4:
    Info.MatchingId = VECTOR_LDST_FOUR_ELEMENTS;
    break;
  }
  return true;
}

/// See if \p I should be considered for address type promotion. We check if \p
/// I is a sext with right type and used in memory accesses. If it used in a
/// "complex" getelementptr, we allow it to be promoted without finding other
/// sext instructions that sign extended the same initial value. A getelementptr
/// is considered as "complex" if it has more than 2 operands.
bool AArch64TTIImpl::shouldConsiderAddressTypePromotion(
    const Instruction &I, bool &AllowPromotionWithoutCommonHeader) {
  bool Considerable = false;
  AllowPromotionWithoutCommonHeader = false;
  if (!isa<SExtInst>(&I))
    return false;
  Type *ConsideredSExtType =
      Type::getInt64Ty(I.getParent()->getParent()->getContext());
  if (I.getType() != ConsideredSExtType)
    return false;
  // See if the sext is the one with the right type and used in at least one
  // GetElementPtrInst.
  for (const User *U : I.users()) {
    if (const GetElementPtrInst *GEPInst = dyn_cast<GetElementPtrInst>(U)) {
      Considerable = true;
      // A getelementptr is considered as "complex" if it has more than 2
      // operands. We will promote a SExt used in such complex GEP as we
      // expect some computation to be merged if they are done on 64 bits.
      if (GEPInst->getNumOperands() > 2) {
        AllowPromotionWithoutCommonHeader = true;
        break;
      }
    }
  }
  return Considerable;
}

bool AArch64TTIImpl::isLegalToVectorizeReduction(
    const RecurrenceDescriptor &RdxDesc, ElementCount VF) const {
  if (!VF.isScalable())
    return true;

  Type *Ty = RdxDesc.getRecurrenceType();
  if (Ty->isBFloatTy() || !isElementTypeLegalForScalableVector(Ty))
    return false;

  switch (RdxDesc.getRecurrenceKind()) {
  case RecurKind::Add:
  case RecurKind::FAdd:
  case RecurKind::And:
  case RecurKind::Or:
  case RecurKind::Xor:
  case RecurKind::SMin:
  case RecurKind::SMax:
  case RecurKind::UMin:
  case RecurKind::UMax:
  case RecurKind::FMin:
  case RecurKind::FMax:
  case RecurKind::SelectICmp:
  case RecurKind::SelectFCmp:
  case RecurKind::FMulAdd:
    return true;
  default:
    return false;
  }
}

InstructionCost
AArch64TTIImpl::getMinMaxReductionCost(VectorType *Ty, VectorType *CondTy,
                                       bool IsUnsigned,
                                       TTI::TargetCostKind CostKind) {
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Ty);

  if (LT.second.getScalarType() == MVT::f16 && !ST->hasFullFP16())
    return BaseT::getMinMaxReductionCost(Ty, CondTy, IsUnsigned, CostKind);

  assert((isa<ScalableVectorType>(Ty) == isa<ScalableVectorType>(CondTy)) &&
         "Both vector needs to be equally scalable");

  InstructionCost LegalizationCost = 0;
  if (LT.first > 1) {
    Type *LegalVTy = EVT(LT.second).getTypeForEVT(Ty->getContext());
    unsigned MinMaxOpcode =
        Ty->isFPOrFPVectorTy()
            ? Intrinsic::maxnum
            : (IsUnsigned ? Intrinsic::umin : Intrinsic::smin);
    IntrinsicCostAttributes Attrs(MinMaxOpcode, LegalVTy, {LegalVTy, LegalVTy});
    LegalizationCost = getIntrinsicInstrCost(Attrs, CostKind) * (LT.first - 1);
  }

  return LegalizationCost + /*Cost of horizontal reduction*/ 2;
}

InstructionCost AArch64TTIImpl::getArithmeticReductionCostSVE(
    unsigned Opcode, VectorType *ValTy, TTI::TargetCostKind CostKind) {
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(ValTy);
  InstructionCost LegalizationCost = 0;
  if (LT.first > 1) {
    Type *LegalVTy = EVT(LT.second).getTypeForEVT(ValTy->getContext());
    LegalizationCost = getArithmeticInstrCost(Opcode, LegalVTy, CostKind);
    LegalizationCost *= LT.first - 1;
  }

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");
  // Add the final reduction cost for the legal horizontal reduction
  switch (ISD) {
  case ISD::ADD:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::FADD:
    return LegalizationCost + 2;
  default:
    return InstructionCost::getInvalid();
  }
}

InstructionCost
AArch64TTIImpl::getArithmeticReductionCost(unsigned Opcode, VectorType *ValTy,
                                           Optional<FastMathFlags> FMF,
                                           TTI::TargetCostKind CostKind) {
  if (TTI::requiresOrderedReduction(FMF)) {
    if (auto *FixedVTy = dyn_cast<FixedVectorType>(ValTy)) {
      InstructionCost BaseCost =
          BaseT::getArithmeticReductionCost(Opcode, ValTy, FMF, CostKind);
      // Add on extra cost to reflect the extra overhead on some CPUs. We still
      // end up vectorizing for more computationally intensive loops.
      return BaseCost + FixedVTy->getNumElements();
    }

    if (Opcode != Instruction::FAdd)
      return InstructionCost::getInvalid();

    auto *VTy = cast<ScalableVectorType>(ValTy);
    InstructionCost Cost =
        getArithmeticInstrCost(Opcode, VTy->getScalarType(), CostKind);
    Cost *= getMaxNumElements(VTy->getElementCount());
    return Cost;
  }

  if (isa<ScalableVectorType>(ValTy))
    return getArithmeticReductionCostSVE(Opcode, ValTy, CostKind);

  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(ValTy);
  MVT MTy = LT.second;
  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  assert(ISD && "Invalid opcode");

  // Horizontal adds can use the 'addv' instruction. We model the cost of these
  // instructions as twice a normal vector add, plus 1 for each legalization
  // step (LT.first). This is the only arithmetic vector reduction operation for
  // which we have an instruction.
  // OR, XOR and AND costs should match the codegen from:
  // OR: llvm/test/CodeGen/AArch64/reduce-or.ll
  // XOR: llvm/test/CodeGen/AArch64/reduce-xor.ll
  // AND: llvm/test/CodeGen/AArch64/reduce-and.ll
  static const CostTblEntry CostTblNoPairwise[]{
      {ISD::ADD, MVT::v8i8,   2},
      {ISD::ADD, MVT::v16i8,  2},
      {ISD::ADD, MVT::v4i16,  2},
      {ISD::ADD, MVT::v8i16,  2},
      {ISD::ADD, MVT::v4i32,  2},
      {ISD::ADD, MVT::v2i64,  2},
      {ISD::OR,  MVT::v8i8,  15},
      {ISD::OR,  MVT::v16i8, 17},
      {ISD::OR,  MVT::v4i16,  7},
      {ISD::OR,  MVT::v8i16,  9},
      {ISD::OR,  MVT::v2i32,  3},
      {ISD::OR,  MVT::v4i32,  5},
      {ISD::OR,  MVT::v2i64,  3},
      {ISD::XOR, MVT::v8i8,  15},
      {ISD::XOR, MVT::v16i8, 17},
      {ISD::XOR, MVT::v4i16,  7},
      {ISD::XOR, MVT::v8i16,  9},
      {ISD::XOR, MVT::v2i32,  3},
      {ISD::XOR, MVT::v4i32,  5},
      {ISD::XOR, MVT::v2i64,  3},
      {ISD::AND, MVT::v8i8,  15},
      {ISD::AND, MVT::v16i8, 17},
      {ISD::AND, MVT::v4i16,  7},
      {ISD::AND, MVT::v8i16,  9},
      {ISD::AND, MVT::v2i32,  3},
      {ISD::AND, MVT::v4i32,  5},
      {ISD::AND, MVT::v2i64,  3},
  };
  switch (ISD) {
  default:
    break;
  case ISD::ADD:
    if (const auto *Entry = CostTableLookup(CostTblNoPairwise, ISD, MTy))
      return (LT.first - 1) + Entry->Cost;
    break;
  case ISD::XOR:
  case ISD::AND:
  case ISD::OR:
    const auto *Entry = CostTableLookup(CostTblNoPairwise, ISD, MTy);
    if (!Entry)
      break;
    auto *ValVTy = cast<FixedVectorType>(ValTy);
    if (!ValVTy->getElementType()->isIntegerTy(1) &&
        MTy.getVectorNumElements() <= ValVTy->getNumElements() &&
        isPowerOf2_32(ValVTy->getNumElements())) {
      InstructionCost ExtraCost = 0;
      if (LT.first != 1) {
        // Type needs to be split, so there is an extra cost of LT.first - 1
        // arithmetic ops.
        auto *Ty = FixedVectorType::get(ValTy->getElementType(),
                                        MTy.getVectorNumElements());
        ExtraCost = getArithmeticInstrCost(Opcode, Ty, CostKind);
        ExtraCost *= LT.first - 1;
      }
      return Entry->Cost + ExtraCost;
    }
    break;
  }
  return BaseT::getArithmeticReductionCost(Opcode, ValTy, FMF, CostKind);
}

InstructionCost AArch64TTIImpl::getSpliceCost(VectorType *Tp, int Index) {
  static const CostTblEntry ShuffleTbl[] = {
      { TTI::SK_Splice, MVT::nxv16i8,  1 },
      { TTI::SK_Splice, MVT::nxv8i16,  1 },
      { TTI::SK_Splice, MVT::nxv4i32,  1 },
      { TTI::SK_Splice, MVT::nxv2i64,  1 },
      { TTI::SK_Splice, MVT::nxv2f16,  1 },
      { TTI::SK_Splice, MVT::nxv4f16,  1 },
      { TTI::SK_Splice, MVT::nxv8f16,  1 },
      { TTI::SK_Splice, MVT::nxv2bf16, 1 },
      { TTI::SK_Splice, MVT::nxv4bf16, 1 },
      { TTI::SK_Splice, MVT::nxv8bf16, 1 },
      { TTI::SK_Splice, MVT::nxv2f32,  1 },
      { TTI::SK_Splice, MVT::nxv4f32,  1 },
      { TTI::SK_Splice, MVT::nxv2f64,  1 },
  };

  // The code-generator is currently not able to handle scalable vectors
  // of <vscale x 1 x eltty> yet, so return an invalid cost to avoid selecting
  // it. This change will be removed when code-generation for these types is
  // sufficiently reliable.
  if (Tp->getElementCount() == ElementCount::getScalable(1))
    return InstructionCost::getInvalid();

  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Tp);
  Type *LegalVTy = EVT(LT.second).getTypeForEVT(Tp->getContext());
  TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
  EVT PromotedVT = LT.second.getScalarType() == MVT::i1
                       ? TLI->getPromotedVTForPredicate(EVT(LT.second))
                       : LT.second;
  Type *PromotedVTy = EVT(PromotedVT).getTypeForEVT(Tp->getContext());
  InstructionCost LegalizationCost = 0;
  if (Index < 0) {
    LegalizationCost =
        getCmpSelInstrCost(Instruction::ICmp, PromotedVTy, PromotedVTy,
                           CmpInst::BAD_ICMP_PREDICATE, CostKind) +
        getCmpSelInstrCost(Instruction::Select, PromotedVTy, LegalVTy,
                           CmpInst::BAD_ICMP_PREDICATE, CostKind);
  }

  // Predicated splice are promoted when lowering. See AArch64ISelLowering.cpp
  // Cost performed on a promoted type.
  if (LT.second.getScalarType() == MVT::i1) {
    LegalizationCost +=
        getCastInstrCost(Instruction::ZExt, PromotedVTy, LegalVTy,
                         TTI::CastContextHint::None, CostKind) +
        getCastInstrCost(Instruction::Trunc, LegalVTy, PromotedVTy,
                         TTI::CastContextHint::None, CostKind);
  }
  const auto *Entry =
      CostTableLookup(ShuffleTbl, TTI::SK_Splice, PromotedVT.getSimpleVT());
  assert(Entry && "Illegal Type for Splice");
  LegalizationCost += Entry->Cost;
  return LegalizationCost * LT.first;
}

InstructionCost AArch64TTIImpl::getShuffleCost(TTI::ShuffleKind Kind,
                                               VectorType *Tp,
                                               ArrayRef<int> Mask,
                                               TTI::TargetCostKind CostKind,
                                               int Index, VectorType *SubTp,
                                               ArrayRef<const Value *> Args) {
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Tp);
  // If we have a Mask, and the LT is being legalized somehow, split the Mask
  // into smaller vectors and sum the cost of each shuffle.
  if (!Mask.empty() && isa<FixedVectorType>(Tp) && LT.second.isVector() &&
      Tp->getScalarSizeInBits() == LT.second.getScalarSizeInBits() &&
      cast<FixedVectorType>(Tp)->getNumElements() >
          LT.second.getVectorNumElements() &&
      !Index && !SubTp) {
    unsigned TpNumElts = cast<FixedVectorType>(Tp)->getNumElements();
    assert(Mask.size() == TpNumElts && "Expected Mask and Tp size to match!");
    unsigned LTNumElts = LT.second.getVectorNumElements();
    unsigned NumVecs = (TpNumElts + LTNumElts - 1) / LTNumElts;
    VectorType *NTp =
        VectorType::get(Tp->getScalarType(), LT.second.getVectorElementCount());
    InstructionCost Cost;
    for (unsigned N = 0; N < NumVecs; N++) {
      SmallVector<int> NMask;
      // Split the existing mask into chunks of size LTNumElts. Track the source
      // sub-vectors to ensure the result has at most 2 inputs.
      unsigned Source1, Source2;
      unsigned NumSources = 0;
      for (unsigned E = 0; E < LTNumElts; E++) {
        int MaskElt = (N * LTNumElts + E < TpNumElts) ? Mask[N * LTNumElts + E]
                                                      : UndefMaskElem;
        if (MaskElt < 0) {
          NMask.push_back(UndefMaskElem);
          continue;
        }

        // Calculate which source from the input this comes from and whether it
        // is new to us.
        unsigned Source = MaskElt / LTNumElts;
        if (NumSources == 0) {
          Source1 = Source;
          NumSources = 1;
        } else if (NumSources == 1 && Source != Source1) {
          Source2 = Source;
          NumSources = 2;
        } else if (NumSources >= 2 && Source != Source1 && Source != Source2) {
          NumSources++;
        }

        // Add to the new mask. For the NumSources>2 case these are not correct,
        // but are only used for the modular lane number.
        if (Source == Source1)
          NMask.push_back(MaskElt % LTNumElts);
        else if (Source == Source2)
          NMask.push_back(MaskElt % LTNumElts + LTNumElts);
        else
          NMask.push_back(MaskElt % LTNumElts);
      }
      // If the sub-mask has at most 2 input sub-vectors then re-cost it using
      // getShuffleCost. If not then cost it using the worst case.
      if (NumSources <= 2)
        Cost += getShuffleCost(NumSources <= 1 ? TTI::SK_PermuteSingleSrc
                                               : TTI::SK_PermuteTwoSrc,
                               NTp, NMask, CostKind, 0, nullptr, Args);
      else if (any_of(enumerate(NMask), [&](const auto &ME) {
                 return ME.value() % LTNumElts == ME.index();
               }))
        Cost += LTNumElts - 1;
      else
        Cost += LTNumElts;
    }
    return Cost;
  }

  Kind = improveShuffleKindFromMask(Kind, Mask);

  // Check for broadcast loads.
  if (Kind == TTI::SK_Broadcast) {
    bool IsLoad = !Args.empty() && isa<LoadInst>(Args[0]);
    if (IsLoad && LT.second.isVector() &&
        isLegalBroadcastLoad(Tp->getElementType(),
                             LT.second.getVectorElementCount()))
      return 0; // broadcast is handled by ld1r
  }

  // If we have 4 elements for the shuffle and a Mask, get the cost straight
  // from the perfect shuffle tables.
  if (Mask.size() == 4 && Tp->getElementCount() == ElementCount::getFixed(4) &&
      (Tp->getScalarSizeInBits() == 16 || Tp->getScalarSizeInBits() == 32) &&
      all_of(Mask, [](int E) { return E < 8; }))
    return getPerfectShuffleCost(Mask);

  if (Kind == TTI::SK_Broadcast || Kind == TTI::SK_Transpose ||
      Kind == TTI::SK_Select || Kind == TTI::SK_PermuteSingleSrc ||
      Kind == TTI::SK_Reverse || Kind == TTI::SK_Splice) {
    static const CostTblEntry ShuffleTbl[] = {
        // Broadcast shuffle kinds can be performed with 'dup'.
        {TTI::SK_Broadcast, MVT::v8i8, 1},
        {TTI::SK_Broadcast, MVT::v16i8, 1},
        {TTI::SK_Broadcast, MVT::v4i16, 1},
        {TTI::SK_Broadcast, MVT::v8i16, 1},
        {TTI::SK_Broadcast, MVT::v2i32, 1},
        {TTI::SK_Broadcast, MVT::v4i32, 1},
        {TTI::SK_Broadcast, MVT::v2i64, 1},
        {TTI::SK_Broadcast, MVT::v2f32, 1},
        {TTI::SK_Broadcast, MVT::v4f32, 1},
        {TTI::SK_Broadcast, MVT::v2f64, 1},
        // Transpose shuffle kinds can be performed with 'trn1/trn2' and
        // 'zip1/zip2' instructions.
        {TTI::SK_Transpose, MVT::v8i8, 1},
        {TTI::SK_Transpose, MVT::v16i8, 1},
        {TTI::SK_Transpose, MVT::v4i16, 1},
        {TTI::SK_Transpose, MVT::v8i16, 1},
        {TTI::SK_Transpose, MVT::v2i32, 1},
        {TTI::SK_Transpose, MVT::v4i32, 1},
        {TTI::SK_Transpose, MVT::v2i64, 1},
        {TTI::SK_Transpose, MVT::v2f32, 1},
        {TTI::SK_Transpose, MVT::v4f32, 1},
        {TTI::SK_Transpose, MVT::v2f64, 1},
        // Select shuffle kinds.
        // TODO: handle vXi8/vXi16.
        {TTI::SK_Select, MVT::v2i32, 1}, // mov.
        {TTI::SK_Select, MVT::v4i32, 2}, // rev+trn (or similar).
        {TTI::SK_Select, MVT::v2i64, 1}, // mov.
        {TTI::SK_Select, MVT::v2f32, 1}, // mov.
        {TTI::SK_Select, MVT::v4f32, 2}, // rev+trn (or similar).
        {TTI::SK_Select, MVT::v2f64, 1}, // mov.
        // PermuteSingleSrc shuffle kinds.
        {TTI::SK_PermuteSingleSrc, MVT::v2i32, 1}, // mov.
        {TTI::SK_PermuteSingleSrc, MVT::v4i32, 3}, // perfectshuffle worst case.
        {TTI::SK_PermuteSingleSrc, MVT::v2i64, 1}, // mov.
        {TTI::SK_PermuteSingleSrc, MVT::v2f32, 1}, // mov.
        {TTI::SK_PermuteSingleSrc, MVT::v4f32, 3}, // perfectshuffle worst case.
        {TTI::SK_PermuteSingleSrc, MVT::v2f64, 1}, // mov.
        {TTI::SK_PermuteSingleSrc, MVT::v4i16, 3}, // perfectshuffle worst case.
        {TTI::SK_PermuteSingleSrc, MVT::v4f16, 3}, // perfectshuffle worst case.
        {TTI::SK_PermuteSingleSrc, MVT::v4bf16, 3}, // same
        {TTI::SK_PermuteSingleSrc, MVT::v8i16, 8},  // constpool + load + tbl
        {TTI::SK_PermuteSingleSrc, MVT::v8f16, 8},  // constpool + load + tbl
        {TTI::SK_PermuteSingleSrc, MVT::v8bf16, 8}, // constpool + load + tbl
        {TTI::SK_PermuteSingleSrc, MVT::v8i8, 8},   // constpool + load + tbl
        {TTI::SK_PermuteSingleSrc, MVT::v16i8, 8},  // constpool + load + tbl
        // Reverse can be lowered with `rev`.
        {TTI::SK_Reverse, MVT::v2i32, 1}, // REV64
        {TTI::SK_Reverse, MVT::v4i32, 2}, // REV64; EXT
        {TTI::SK_Reverse, MVT::v2i64, 1}, // EXT
        {TTI::SK_Reverse, MVT::v2f32, 1}, // REV64
        {TTI::SK_Reverse, MVT::v4f32, 2}, // REV64; EXT
        {TTI::SK_Reverse, MVT::v2f64, 1}, // EXT
        {TTI::SK_Reverse, MVT::v8f16, 2}, // REV64; EXT
        {TTI::SK_Reverse, MVT::v8i16, 2}, // REV64; EXT
        {TTI::SK_Reverse, MVT::v16i8, 2}, // REV64; EXT
        {TTI::SK_Reverse, MVT::v4f16, 1}, // REV64
        {TTI::SK_Reverse, MVT::v4i16, 1}, // REV64
        {TTI::SK_Reverse, MVT::v8i8, 1},  // REV64
        // Splice can all be lowered as `ext`.
        {TTI::SK_Splice, MVT::v2i32, 1},
        {TTI::SK_Splice, MVT::v4i32, 1},
        {TTI::SK_Splice, MVT::v2i64, 1},
        {TTI::SK_Splice, MVT::v2f32, 1},
        {TTI::SK_Splice, MVT::v4f32, 1},
        {TTI::SK_Splice, MVT::v2f64, 1},
        {TTI::SK_Splice, MVT::v8f16, 1},
        {TTI::SK_Splice, MVT::v8bf16, 1},
        {TTI::SK_Splice, MVT::v8i16, 1},
        {TTI::SK_Splice, MVT::v16i8, 1},
        {TTI::SK_Splice, MVT::v4bf16, 1},
        {TTI::SK_Splice, MVT::v4f16, 1},
        {TTI::SK_Splice, MVT::v4i16, 1},
        {TTI::SK_Splice, MVT::v8i8, 1},
        // Broadcast shuffle kinds for scalable vectors
        {TTI::SK_Broadcast, MVT::nxv16i8, 1},
        {TTI::SK_Broadcast, MVT::nxv8i16, 1},
        {TTI::SK_Broadcast, MVT::nxv4i32, 1},
        {TTI::SK_Broadcast, MVT::nxv2i64, 1},
        {TTI::SK_Broadcast, MVT::nxv2f16, 1},
        {TTI::SK_Broadcast, MVT::nxv4f16, 1},
        {TTI::SK_Broadcast, MVT::nxv8f16, 1},
        {TTI::SK_Broadcast, MVT::nxv2bf16, 1},
        {TTI::SK_Broadcast, MVT::nxv4bf16, 1},
        {TTI::SK_Broadcast, MVT::nxv8bf16, 1},
        {TTI::SK_Broadcast, MVT::nxv2f32, 1},
        {TTI::SK_Broadcast, MVT::nxv4f32, 1},
        {TTI::SK_Broadcast, MVT::nxv2f64, 1},
        {TTI::SK_Broadcast, MVT::nxv16i1, 1},
        {TTI::SK_Broadcast, MVT::nxv8i1, 1},
        {TTI::SK_Broadcast, MVT::nxv4i1, 1},
        {TTI::SK_Broadcast, MVT::nxv2i1, 1},
        // Handle the cases for vector.reverse with scalable vectors
        {TTI::SK_Reverse, MVT::nxv16i8, 1},
        {TTI::SK_Reverse, MVT::nxv8i16, 1},
        {TTI::SK_Reverse, MVT::nxv4i32, 1},
        {TTI::SK_Reverse, MVT::nxv2i64, 1},
        {TTI::SK_Reverse, MVT::nxv2f16, 1},
        {TTI::SK_Reverse, MVT::nxv4f16, 1},
        {TTI::SK_Reverse, MVT::nxv8f16, 1},
        {TTI::SK_Reverse, MVT::nxv2bf16, 1},
        {TTI::SK_Reverse, MVT::nxv4bf16, 1},
        {TTI::SK_Reverse, MVT::nxv8bf16, 1},
        {TTI::SK_Reverse, MVT::nxv2f32, 1},
        {TTI::SK_Reverse, MVT::nxv4f32, 1},
        {TTI::SK_Reverse, MVT::nxv2f64, 1},
        {TTI::SK_Reverse, MVT::nxv16i1, 1},
        {TTI::SK_Reverse, MVT::nxv8i1, 1},
        {TTI::SK_Reverse, MVT::nxv4i1, 1},
        {TTI::SK_Reverse, MVT::nxv2i1, 1},
    };
    if (const auto *Entry = CostTableLookup(ShuffleTbl, Kind, LT.second))
      return LT.first * Entry->Cost;
  }

  if (Kind == TTI::SK_Splice && isa<ScalableVectorType>(Tp))
    return getSpliceCost(Tp, Index);

  // Inserting a subvector can often be done with either a D, S or H register
  // move, so long as the inserted vector is "aligned".
  if (Kind == TTI::SK_InsertSubvector && LT.second.isFixedLengthVector() &&
      LT.second.getSizeInBits() <= 128 && SubTp) {
    std::pair<InstructionCost, MVT> SubLT = getTypeLegalizationCost(SubTp);
    if (SubLT.second.isVector()) {
      int NumElts = LT.second.getVectorNumElements();
      int NumSubElts = SubLT.second.getVectorNumElements();
      if ((Index % NumSubElts) == 0 && (NumElts % NumSubElts) == 0)
        return SubLT.first;
    }
  }

  return BaseT::getShuffleCost(Kind, Tp, Mask, CostKind, Index, SubTp);
}

bool AArch64TTIImpl::preferPredicateOverEpilogue(
    Loop *L, LoopInfo *LI, ScalarEvolution &SE, AssumptionCache &AC,
    TargetLibraryInfo *TLI, DominatorTree *DT, LoopVectorizationLegality *LVL,
    InterleavedAccessInfo *IAI) {
  if (!ST->hasSVE() || TailFoldingKindLoc == TailFoldingKind::TFDisabled)
    return false;

  // We don't currently support vectorisation with interleaving for SVE - with
  // such loops we're better off not using tail-folding. This gives us a chance
  // to fall back on fixed-width vectorisation using NEON's ld2/st2/etc.
  if (IAI->hasGroups())
    return false;

  TailFoldingKind Required; // Defaults to 0.
  if (LVL->getReductionVars().size())
    Required.add(TailFoldingKind::TFReductions);
  if (LVL->getFixedOrderRecurrences().size())
    Required.add(TailFoldingKind::TFRecurrences);
  if (!Required)
    Required.add(TailFoldingKind::TFSimple);

  return (TailFoldingKindLoc & Required) == Required;
}

InstructionCost
AArch64TTIImpl::getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                                     int64_t BaseOffset, bool HasBaseReg,
                                     int64_t Scale, unsigned AddrSpace) const {
  // Scaling factors are not free at all.
  // Operands                     | Rt Latency
  // -------------------------------------------
  // Rt, [Xn, Xm]                 | 4
  // -------------------------------------------
  // Rt, [Xn, Xm, lsl #imm]       | Rn: 4 Rm: 5
  // Rt, [Xn, Wm, <extend> #imm]  |
  TargetLoweringBase::AddrMode AM;
  AM.BaseGV = BaseGV;
  AM.BaseOffs = BaseOffset;
  AM.HasBaseReg = HasBaseReg;
  AM.Scale = Scale;
  if (getTLI()->isLegalAddressingMode(DL, AM, Ty, AddrSpace))
    // Scale represents reg2 * scale, thus account for 1 if
    // it is not equal to 0 or 1.
    return AM.Scale != 0 && AM.Scale != 1;
  return -1;
}
