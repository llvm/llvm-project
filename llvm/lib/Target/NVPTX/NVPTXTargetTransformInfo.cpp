//===-- NVPTXTargetTransformInfo.cpp - NVPTX specific TTI -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NVPTXTargetTransformInfo.h"
#include "NVVMProperties.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/SimplifyQuery.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/NVPTXAddrSpace.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"
#include <optional>
using namespace llvm;

#define DEBUG_TYPE "NVPTXtti"

// Convert NVVM intrinsics to target-generic LLVM code where possible.
static Instruction *convertNvvmIntrinsicToLlvm(InstCombiner &IC,
                                               IntrinsicInst *II) {
  // Each NVVM intrinsic we can simplify can be replaced with one of:
  //
  //  * an LLVM intrinsic,
  //  * an LLVM cast operation,
  //  * an LLVM binary operation, or
  //  * ad-hoc LLVM IR for the particular operation.

  // Some transformations are only valid when the module's
  // flush-denormals-to-zero (ftz) setting is true/false, whereas other
  // transformations are valid regardless of the module's ftz setting.
  enum FtzRequirementTy {
    FTZ_Any,       // Any ftz setting is ok.
    FTZ_MustBeOn,  // Transformation is valid only if ftz is on.
    FTZ_MustBeOff, // Transformation is valid only if ftz is off.
  };
  // Classes of NVVM intrinsics that can't be replaced one-to-one with a
  // target-generic intrinsic, cast op, or binary op but that we can nonetheless
  // simplify.
  enum SpecialCase {
    SPC_Reciprocal,
    SCP_FunnelShiftClamp,
  };

  // SimplifyAction is a poor-man's variant (plus an additional flag) that
  // represents how to replace an NVVM intrinsic with target-generic LLVM IR.
  struct SimplifyAction {
    // Invariant: At most one of these Optionals has a value.
    std::optional<Intrinsic::ID> IID;
    std::optional<Instruction::CastOps> CastOp;
    std::optional<Instruction::BinaryOps> BinaryOp;
    std::optional<SpecialCase> Special;

    FtzRequirementTy FtzRequirement = FTZ_Any;
    // Denormal handling is guarded by different attributes depending on the
    // type (denormal-fp-math vs denormal-fp-math-f32), take note of halfs.
    bool IsHalfTy = false;

    SimplifyAction() = default;

    SimplifyAction(Intrinsic::ID IID, FtzRequirementTy FtzReq,
                   bool IsHalfTy = false)
        : IID(IID), FtzRequirement(FtzReq), IsHalfTy(IsHalfTy) {}

    // Cast operations don't have anything to do with FTZ, so we skip that
    // argument.
    SimplifyAction(Instruction::CastOps CastOp) : CastOp(CastOp) {}

    SimplifyAction(Instruction::BinaryOps BinaryOp, FtzRequirementTy FtzReq)
        : BinaryOp(BinaryOp), FtzRequirement(FtzReq) {}

    SimplifyAction(SpecialCase Special, FtzRequirementTy FtzReq)
        : Special(Special), FtzRequirement(FtzReq) {}
  };

  // Try to generate a SimplifyAction describing how to replace our
  // IntrinsicInstr with target-generic LLVM IR.
  const SimplifyAction Action = [II]() -> SimplifyAction {
    switch (II->getIntrinsicID()) {
    // NVVM intrinsics that map directly to LLVM intrinsics.
    case Intrinsic::nvvm_ceil_d:
      return {Intrinsic::ceil, FTZ_Any};
    case Intrinsic::nvvm_ceil_f:
      return {Intrinsic::ceil, FTZ_MustBeOff};
    case Intrinsic::nvvm_ceil_ftz_f:
      return {Intrinsic::ceil, FTZ_MustBeOn};
    case Intrinsic::nvvm_floor_d:
      return {Intrinsic::floor, FTZ_Any};
    case Intrinsic::nvvm_floor_f:
      return {Intrinsic::floor, FTZ_MustBeOff};
    case Intrinsic::nvvm_floor_ftz_f:
      return {Intrinsic::floor, FTZ_MustBeOn};
    case Intrinsic::nvvm_fma_rn_d:
      return {Intrinsic::fma, FTZ_Any};
    case Intrinsic::nvvm_fma_rn_f:
      return {Intrinsic::fma, FTZ_MustBeOff};
    case Intrinsic::nvvm_fma_rn_ftz_f:
      return {Intrinsic::fma, FTZ_MustBeOn};
    case Intrinsic::nvvm_fma_rn_f16:
      return {Intrinsic::fma, FTZ_MustBeOff, true};
    case Intrinsic::nvvm_fma_rn_ftz_f16:
      return {Intrinsic::fma, FTZ_MustBeOn, true};
    case Intrinsic::nvvm_fma_rn_f16x2:
      return {Intrinsic::fma, FTZ_MustBeOff, true};
    case Intrinsic::nvvm_fma_rn_ftz_f16x2:
      return {Intrinsic::fma, FTZ_MustBeOn, true};
    case Intrinsic::nvvm_fma_rn_bf16:
      return {Intrinsic::fma, FTZ_MustBeOff, true};
    case Intrinsic::nvvm_fma_rn_bf16x2:
      return {Intrinsic::fma, FTZ_MustBeOff, true};
    case Intrinsic::nvvm_fmax_d:
      return {Intrinsic::maximumnum, FTZ_Any};
    case Intrinsic::nvvm_fmax_f:
      return {Intrinsic::maximumnum, FTZ_MustBeOff};
    case Intrinsic::nvvm_fmax_ftz_f:
      return {Intrinsic::maximumnum, FTZ_MustBeOn};
    case Intrinsic::nvvm_fmax_nan_f:
      return {Intrinsic::maximum, FTZ_MustBeOff};
    case Intrinsic::nvvm_fmax_ftz_nan_f:
      return {Intrinsic::maximum, FTZ_MustBeOn};
    case Intrinsic::nvvm_fmax_f16:
      return {Intrinsic::maximumnum, FTZ_MustBeOff, true};
    case Intrinsic::nvvm_fmax_ftz_f16:
      return {Intrinsic::maximumnum, FTZ_MustBeOn, true};
    case Intrinsic::nvvm_fmax_f16x2:
      return {Intrinsic::maximumnum, FTZ_MustBeOff, true};
    case Intrinsic::nvvm_fmax_ftz_f16x2:
      return {Intrinsic::maximumnum, FTZ_MustBeOn, true};
    case Intrinsic::nvvm_fmax_nan_f16:
      return {Intrinsic::maximum, FTZ_MustBeOff, true};
    case Intrinsic::nvvm_fmax_ftz_nan_f16:
      return {Intrinsic::maximum, FTZ_MustBeOn, true};
    case Intrinsic::nvvm_fmax_nan_f16x2:
      return {Intrinsic::maximum, FTZ_MustBeOff, true};
    case Intrinsic::nvvm_fmax_ftz_nan_f16x2:
      return {Intrinsic::maximum, FTZ_MustBeOn, true};
    case Intrinsic::nvvm_fmin_d:
      return {Intrinsic::minimumnum, FTZ_Any};
    case Intrinsic::nvvm_fmin_f:
      return {Intrinsic::minimumnum, FTZ_MustBeOff};
    case Intrinsic::nvvm_fmin_ftz_f:
      return {Intrinsic::minimumnum, FTZ_MustBeOn};
    case Intrinsic::nvvm_fmin_nan_f:
      return {Intrinsic::minimum, FTZ_MustBeOff};
    case Intrinsic::nvvm_fmin_ftz_nan_f:
      return {Intrinsic::minimum, FTZ_MustBeOn};
    case Intrinsic::nvvm_fmin_f16:
      return {Intrinsic::minimumnum, FTZ_MustBeOff, true};
    case Intrinsic::nvvm_fmin_ftz_f16:
      return {Intrinsic::minimumnum, FTZ_MustBeOn, true};
    case Intrinsic::nvvm_fmin_f16x2:
      return {Intrinsic::minimumnum, FTZ_MustBeOff, true};
    case Intrinsic::nvvm_fmin_ftz_f16x2:
      return {Intrinsic::minimumnum, FTZ_MustBeOn, true};
    case Intrinsic::nvvm_fmin_nan_f16:
      return {Intrinsic::minimum, FTZ_MustBeOff, true};
    case Intrinsic::nvvm_fmin_ftz_nan_f16:
      return {Intrinsic::minimum, FTZ_MustBeOn, true};
    case Intrinsic::nvvm_fmin_nan_f16x2:
      return {Intrinsic::minimum, FTZ_MustBeOff, true};
    case Intrinsic::nvvm_fmin_ftz_nan_f16x2:
      return {Intrinsic::minimum, FTZ_MustBeOn, true};
    case Intrinsic::nvvm_sqrt_rn_d:
      return {Intrinsic::sqrt, FTZ_Any};
    case Intrinsic::nvvm_sqrt_f:
      // nvvm_sqrt_f is a special case.  For  most intrinsics, foo_ftz_f is the
      // ftz version, and foo_f is the non-ftz version.  But nvvm_sqrt_f adopts
      // the ftz-ness of the surrounding code.  sqrt_rn_f and sqrt_rn_ftz_f are
      // the versions with explicit ftz-ness.
      return {Intrinsic::sqrt, FTZ_Any};
    case Intrinsic::nvvm_trunc_d:
      return {Intrinsic::trunc, FTZ_Any};
    case Intrinsic::nvvm_trunc_f:
      return {Intrinsic::trunc, FTZ_MustBeOff};
    case Intrinsic::nvvm_trunc_ftz_f:
      return {Intrinsic::trunc, FTZ_MustBeOn};

    // NVVM intrinsics that map to LLVM cast operations.
    // Note - we cannot map intrinsics like nvvm_d2ll_rz to LLVM's
    // FPToSI, as NaN to int conversion with FPToSI is considered UB and is
    // eliminated. NVVM conversion intrinsics are translated to PTX cvt
    // instructions which define the outcome for NaN rather than leaving as UB.
    // Therefore, translate NVVM intrinsics to sitofp/uitofp, but not to
    // fptosi/fptoui.
    case Intrinsic::nvvm_i2d_rn:
    case Intrinsic::nvvm_i2f_rn:
    case Intrinsic::nvvm_ll2d_rn:
    case Intrinsic::nvvm_ll2f_rn:
      return {Instruction::SIToFP};
    case Intrinsic::nvvm_ui2d_rn:
    case Intrinsic::nvvm_ui2f_rn:
    case Intrinsic::nvvm_ull2d_rn:
    case Intrinsic::nvvm_ull2f_rn:
      return {Instruction::UIToFP};

    // NVVM intrinsics that map to LLVM binary ops.
    case Intrinsic::nvvm_div_rn_d:
      return {Instruction::FDiv, FTZ_Any};

    // The remainder of cases are NVVM intrinsics that map to LLVM idioms, but
    // need special handling.
    //
    // We seem to be missing intrinsics for rcp.approx.{ftz.}f32, which is just
    // as well.
    case Intrinsic::nvvm_rcp_rn_d:
      return {SPC_Reciprocal, FTZ_Any};

    case Intrinsic::nvvm_fshl_clamp:
    case Intrinsic::nvvm_fshr_clamp:
      return {SCP_FunnelShiftClamp, FTZ_Any};

      // We do not currently simplify intrinsics that give an approximate
      // answer. These include:
      //
      //   - nvvm_cos_approx_{f,ftz_f}
      //   - nvvm_ex2_approx(_ftz)
      //   - nvvm_lg2_approx_{d,f,ftz_f}
      //   - nvvm_sin_approx_{f,ftz_f}
      //   - nvvm_sqrt_approx_{f,ftz_f}
      //   - nvvm_rsqrt_approx_{d,f,ftz_f}
      //   - nvvm_div_approx_{ftz_d,ftz_f,f}
      //   - nvvm_rcp_approx_ftz_d
      //
      // Ideally we'd encode them as e.g. "fast call @llvm.cos", where "fast"
      // means that fastmath is enabled in the intrinsic.  Unfortunately only
      // binary operators (currently) have a fastmath bit in SelectionDAG, so
      // this information gets lost and we can't select on it.
      //
      // TODO: div and rcp are lowered to a binary op, so these we could in
      // theory lower them to "fast fdiv".

    default:
      return {};
    }
  }();

  // If Action.FtzRequirementTy is not satisfied by the module's ftz state, we
  // can bail out now.  (Notice that in the case that IID is not an NVVM
  // intrinsic, we don't have to look up any module metadata, as
  // FtzRequirementTy will be FTZ_Any.)
  if (Action.FtzRequirement != FTZ_Any) {
    // FIXME: Broken for f64
    DenormalMode Mode = II->getFunction()->getDenormalMode(
        Action.IsHalfTy ? APFloat::IEEEhalf() : APFloat::IEEEsingle());
    bool FtzEnabled = Mode.Output == DenormalMode::PreserveSign;

    if (FtzEnabled != (Action.FtzRequirement == FTZ_MustBeOn))
      return nullptr;
  }

  // Simplify to target-generic intrinsic.
  if (Action.IID) {
    SmallVector<Value *, 4> Args(II->args());
    // All the target-generic intrinsics currently of interest to us have one
    // type argument, equal to that of the nvvm intrinsic's argument.
    Type *Tys[] = {II->getArgOperand(0)->getType()};
    return CallInst::Create(
        Intrinsic::getOrInsertDeclaration(II->getModule(), *Action.IID, Tys),
        Args);
  }

  // Simplify to target-generic binary op.
  if (Action.BinaryOp)
    return BinaryOperator::Create(*Action.BinaryOp, II->getArgOperand(0),
                                  II->getArgOperand(1), II->getName());

  // Simplify to target-generic cast op.
  if (Action.CastOp)
    return CastInst::Create(*Action.CastOp, II->getArgOperand(0), II->getType(),
                            II->getName());

  // All that's left are the special cases.
  if (!Action.Special)
    return nullptr;

  switch (*Action.Special) {
  case SPC_Reciprocal:
    // Simplify reciprocal.
    return BinaryOperator::Create(
        Instruction::FDiv, ConstantFP::get(II->getArgOperand(0)->getType(), 1),
        II->getArgOperand(0), II->getName());

  case SCP_FunnelShiftClamp: {
    // Canonicalize a clamping funnel shift to the generic llvm funnel shift
    // when possible, as this is easier for llvm to optimize further.
    if (const auto *ShiftConst = dyn_cast<ConstantInt>(II->getArgOperand(2))) {
      const bool IsLeft = II->getIntrinsicID() == Intrinsic::nvvm_fshl_clamp;
      if (ShiftConst->getZExtValue() >= II->getType()->getIntegerBitWidth())
        return IC.replaceInstUsesWith(*II, II->getArgOperand(IsLeft ? 1 : 0));

      const unsigned FshIID = IsLeft ? Intrinsic::fshl : Intrinsic::fshr;
      return CallInst::Create(Intrinsic::getOrInsertDeclaration(
                                  II->getModule(), FshIID, II->getType()),
                              SmallVector<Value *, 3>(II->args()));
    }
    return nullptr;
  }
  }
  llvm_unreachable("All SpecialCase enumerators should be handled in switch.");
}

// Returns whether a pointer in AS is in the given specific address space, or
// nullopt when this cannot be determined at compile time.
static std::optional<bool> isInAddressSpace(unsigned AS, unsigned SpecificAS) {
  if (AS == NVPTXAS::ADDRESS_SPACE_GENERIC ||
      AS == NVPTXAS::ADDRESS_SPACE_ENTRY_PARAM)
    return std::nullopt; // Got to check at run-time.
  if (AS == NVPTXAS::ADDRESS_SPACE_SHARED_CLUSTER &&
      SpecificAS == NVPTXAS::ADDRESS_SPACE_SHARED)
    return std::nullopt;
  return AS == SpecificAS ||
         (AS == NVPTXAS::ADDRESS_SPACE_SHARED &&
          SpecificAS == NVPTXAS::ADDRESS_SPACE_SHARED_CLUSTER);
}

// Returns true/false when we know the answer, nullopt otherwise.
static std::optional<bool> evaluateIsSpace(Intrinsic::ID IID, unsigned AS) {
  switch (IID) {
  case Intrinsic::nvvm_isspacep_global:
    return isInAddressSpace(AS, NVPTXAS::ADDRESS_SPACE_GLOBAL);
  case Intrinsic::nvvm_isspacep_local:
    return isInAddressSpace(AS, NVPTXAS::ADDRESS_SPACE_LOCAL);
  case Intrinsic::nvvm_isspacep_shared:
    return isInAddressSpace(AS, NVPTXAS::ADDRESS_SPACE_SHARED);
  case Intrinsic::nvvm_isspacep_shared_cluster:
    return isInAddressSpace(AS, NVPTXAS::ADDRESS_SPACE_SHARED_CLUSTER);
  case Intrinsic::nvvm_isspacep_const:
    return isInAddressSpace(AS, NVPTXAS::ADDRESS_SPACE_CONST);
  default:
    llvm_unreachable("Unexpected intrinsic");
  }
}

// Returns an instruction pointer (may be nullptr if we do not know the answer).
// Returns nullopt if `II` is not one of the `isspacep` intrinsics.
//
// TODO: If InferAddressSpaces were run early enough in the pipeline this could
// be removed in favor of the constant folding that occurs there through
// rewriteIntrinsicWithAddressSpace
static std::optional<Instruction *>
handleSpaceCheckIntrinsics(InstCombiner &IC, IntrinsicInst &II) {

  switch (auto IID = II.getIntrinsicID()) {
  case Intrinsic::nvvm_isspacep_global:
  case Intrinsic::nvvm_isspacep_local:
  case Intrinsic::nvvm_isspacep_shared:
  case Intrinsic::nvvm_isspacep_shared_cluster:
  case Intrinsic::nvvm_isspacep_const: {
    Value *Op0 = II.getArgOperand(0);
    unsigned AS = Op0->getType()->getPointerAddressSpace();
    // Peek through ASC to generic AS.
    // TODO: we could dig deeper through both ASCs and GEPs.
    if (AS == NVPTXAS::ADDRESS_SPACE_GENERIC)
      if (auto *ASCO = dyn_cast<AddrSpaceCastOperator>(Op0))
        AS = ASCO->getOperand(0)->getType()->getPointerAddressSpace();

    if (std::optional<bool> Answer = evaluateIsSpace(IID, AS))
      return IC.replaceInstUsesWith(II,
                                    ConstantInt::get(II.getType(), *Answer));
    return nullptr; // Don't know the answer, got to check at run time.
  }
  default:
    return std::nullopt;
  }
}

static Instruction *foldAbsIntoRedux(InstCombiner &IC, IntrinsicInst &II) {
  Intrinsic::ID AbsIID;
  switch (II.getIntrinsicID()) {
  case Intrinsic::nvvm_redux_sync_fmin:
  case Intrinsic::nvvm_redux_sync_fmin_abs:
    AbsIID = Intrinsic::nvvm_redux_sync_fmin_abs;
    break;
  case Intrinsic::nvvm_redux_sync_fmax:
  case Intrinsic::nvvm_redux_sync_fmax_abs:
    AbsIID = Intrinsic::nvvm_redux_sync_fmax_abs;
    break;
  case Intrinsic::nvvm_redux_sync_fmin_NaN:
  case Intrinsic::nvvm_redux_sync_fmin_abs_NaN:
    AbsIID = Intrinsic::nvvm_redux_sync_fmin_abs_NaN;
    break;
  case Intrinsic::nvvm_redux_sync_fmax_NaN:
  case Intrinsic::nvvm_redux_sync_fmax_abs_NaN:
    AbsIID = Intrinsic::nvvm_redux_sync_fmax_abs_NaN;
    break;
  default:
    return nullptr;
  }

  auto *Abs = dyn_cast<IntrinsicInst>(II.getArgOperand(0));
  if (!Abs || Abs->getIntrinsicID() != Intrinsic::fabs)
    return nullptr;

  II.setCalledFunction(
      Intrinsic::getOrInsertDeclaration(II.getModule(), AbsIID));
  // These attributes described the absolute value, not its source.
  II.removeParamAttr(0, Attribute::NoFPClass);
  II.removeParamAttr(0, Attribute::Returned);
  return IC.replaceOperand(II, 0, Abs->getArgOperand(0));
}

std::optional<Instruction *>
NVPTXTTIImpl::instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const {
  if (std::optional<Instruction *> I = handleSpaceCheckIntrinsics(IC, II))
    return *I;
  if (Instruction *I = convertNvvmIntrinsicToLlvm(IC, &II))
    return I;
  if (Instruction *I = foldAbsIntoRedux(IC, II))
    return I;
  return std::nullopt;
}

InstructionCost
NVPTXTTIImpl::getInstructionCost(const User *U,
                                 ArrayRef<const Value *> Operands,
                                 TTI::TargetCostKind CostKind) const {
  if (const auto *CI = dyn_cast<CallInst>(U))
    if (const auto *IA = dyn_cast<InlineAsm>(CI->getCalledOperand())) {
      // Without this implementation getCallCost() would return the number
      // of arguments+1 as the cost. Because the cost-model assumes it is a call
      // since it is classified as a call in the IR. A better cost model would
      // be to return the number of asm instructions embedded in the asm
      // string.
      StringRef AsmStr = IA->getAsmString();
      const unsigned InstCount =
          count_if(split(AsmStr, ';'), [](StringRef AsmInst) {
            // Trim off scopes denoted by '{' and '}' as these can be ignored
            AsmInst = AsmInst.trim().ltrim("{} \t\n\v\f\r");
            // This is pretty coarse but does a reasonably good job of
            // identifying things that look like instructions, possibly with a
            // predicate ("@").
            return !AsmInst.empty() &&
                   (AsmInst[0] == '@' || isAlpha(AsmInst[0]) ||
                    AsmInst.contains(".pragma"));
          });
      return InstCount * TargetTransformInfo::TCC_Basic;
    }

  return BaseT::getInstructionCost(U, Operands, CostKind);
}

InstructionCost NVPTXTTIImpl::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
    TTI::OperandValueInfo Op1Info, TTI::OperandValueInfo Op2Info,
    ArrayRef<const Value *> Args, const Instruction *CtxI) const {
  // Legalize the type.
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Ty);

  int ISD = TLI->InstructionOpcodeToISD(Opcode);

  switch (ISD) {
  default:
    return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info,
                                         Op2Info);
  case ISD::ADD:
  case ISD::MUL:
  case ISD::XOR:
  case ISD::OR:
  case ISD::AND:
    // The machine code (SASS) simulates an i64 with two i32. Therefore, we
    // estimate that arithmetic operations on i64 are twice as expensive as
    // those on types that can fit into one machine register.
    if (LT.second.SimpleTy == MVT::i64)
      return 2 * LT.first;
    // Delegate other cases to the basic TTI.
    return BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info,
                                         Op2Info);
  }
}

void NVPTXTTIImpl::getUnrollingPreferences(
    Loop *L, ScalarEvolution &SE, TTI::UnrollingPreferences &UP,
    OptimizationRemarkEmitter *ORE) const {
  BaseT::getUnrollingPreferences(L, SE, UP, ORE);

  // Enable partial unrolling and runtime unrolling, but reduce the
  // threshold.  This partially unrolls small loops which are often
  // unrolled by the PTX to SASS compiler and unrolling earlier can be
  // beneficial.
  UP.Partial = UP.Runtime = true;
  UP.PartialThreshold = UP.Threshold / 4;
}

void NVPTXTTIImpl::getPeelingPreferences(Loop *L, ScalarEvolution &SE,
                                         TTI::PeelingPreferences &PP) const {
  BaseT::getPeelingPreferences(L, SE, PP);
}

bool NVPTXTTIImpl::collectFlatAddressOperands(SmallVectorImpl<int> &OpIndexes,
                                              Intrinsic::ID IID) const {
  switch (IID) {
  case Intrinsic::nvvm_isspacep_const:
  case Intrinsic::nvvm_isspacep_global:
  case Intrinsic::nvvm_isspacep_local:
  case Intrinsic::nvvm_isspacep_shared:
  case Intrinsic::nvvm_isspacep_shared_cluster:
  case Intrinsic::nvvm_prefetch_tensormap: {
    OpIndexes.push_back(0);
    return true;
  }
  }
  return false;
}

Value *NVPTXTTIImpl::rewriteIntrinsicWithAddressSpace(IntrinsicInst *II,
                                                      Value *OldV,
                                                      Value *NewV) const {
  const Intrinsic::ID IID = II->getIntrinsicID();
  switch (IID) {
  case Intrinsic::nvvm_isspacep_const:
  case Intrinsic::nvvm_isspacep_global:
  case Intrinsic::nvvm_isspacep_local:
  case Intrinsic::nvvm_isspacep_shared:
  case Intrinsic::nvvm_isspacep_shared_cluster: {
    const unsigned NewAS = NewV->getType()->getPointerAddressSpace();
    if (const auto R = evaluateIsSpace(IID, NewAS))
      return ConstantInt::get(II->getType(), *R);
    return nullptr;
  }
  case Intrinsic::nvvm_prefetch_tensormap: {
    IRBuilder<> Builder(II);
    const unsigned NewAS = NewV->getType()->getPointerAddressSpace();
    if (NewAS == NVPTXAS::ADDRESS_SPACE_CONST ||
        NewAS == NVPTXAS::ADDRESS_SPACE_ENTRY_PARAM)
      return Builder.CreateUnaryIntrinsic(Intrinsic::nvvm_prefetch_tensormap,
                                          NewV);
    return nullptr;
  }
  }
  return nullptr;
}

bool NVPTXTTIImpl::isLegalMaskedStore(Type *DataTy, Align Alignment,
                                      unsigned AddrSpace,
                                      TTI::MaskKind MaskKind) const {
  if (MaskKind != TTI::MaskKind::ConstantMask)
    return false;

  //  We currently only support this feature for 256-bit vectors, so the
  //  alignment must be at least 32
  if (Alignment < 32)
    return false;

  if (!ST->has256BitVectorLoadStore(AddrSpace))
    return false;

  auto *VTy = dyn_cast<FixedVectorType>(DataTy);
  if (!VTy)
    return false;

  auto *ElemTy = VTy->getScalarType();
  return (ElemTy->getScalarSizeInBits() == 32 && VTy->getNumElements() == 8) ||
         (ElemTy->getScalarSizeInBits() == 64 && VTy->getNumElements() == 4);
}

bool NVPTXTTIImpl::isLegalMaskedLoad(Type *DataTy, Align Alignment,
                                     unsigned /*AddrSpace*/,
                                     TTI::MaskKind MaskKind) const {
  if (MaskKind != TTI::MaskKind::ConstantMask)
    return false;

  if (Alignment < DL.getTypeStoreSize(DataTy))
    return false;

  // We do not support sub-byte element type masked loads.
  auto *VTy = dyn_cast<FixedVectorType>(DataTy);
  if (!VTy)
    return false;
  return VTy->getElementType()->getScalarSizeInBits() >= 8;
}

unsigned NVPTXTTIImpl::getLoadStoreVecRegBitWidth(unsigned AddrSpace) const {
  // 256 bit loads/stores are currently only supported for global address space
  if (ST->has256BitVectorLoadStore(AddrSpace))
    return 256;
  return 128;
}

unsigned NVPTXTTIImpl::getAssumedAddrSpace(const Value *V) const {
  if (isa<AllocaInst>(V))
    return ADDRESS_SPACE_LOCAL;

  if (const Argument *Arg = dyn_cast<Argument>(V)) {
    if (isKernelFunction(*Arg->getParent())) {
      const NVPTXTargetMachine &TM =
          static_cast<const NVPTXTargetMachine &>(getTLI()->getTargetMachine());
      if (TM.getDrvInterface() == NVPTX::CUDA && !Arg->hasByValAttr())
        return ADDRESS_SPACE_GLOBAL;
    } else {
      // We assume that all device parameters that are passed byval will be
      // placed in the local AS. Very simple cases will be updated after ISel to
      // use the device param space where possible.
      if (Arg->hasByValAttr())
        return ADDRESS_SPACE_LOCAL;
    }
  }

  return -1;
}

void NVPTXTTIImpl::collectKernelLaunchBounds(
    const Function &F,
    SmallVectorImpl<std::pair<StringRef, int64_t>> &LB) const {
  if (const auto Val = getMaxClusterRank(F))
    LB.push_back({"maxclusterrank", *Val});

  const auto MaxNTID = getMaxNTID(F);
  if (MaxNTID.size() > 0)
    LB.push_back({"maxntidx", MaxNTID[0]});
  if (MaxNTID.size() > 1)
    LB.push_back({"maxntidy", MaxNTID[1]});
  if (MaxNTID.size() > 2)
    LB.push_back({"maxntidz", MaxNTID[2]});
}

// Global addresses can only be materialized if they are in generic global or
// constant space.
bool NVPTXTTIImpl::shouldBuildLookupTablesForConstant(Constant *C) const {
  if (const auto *GV = dyn_cast<GlobalValue>(C)) {
    const unsigned AS = GV->getAddressSpace();
    return AS == NVPTXAS::ADDRESS_SPACE_GENERIC ||
           AS == NVPTXAS::ADDRESS_SPACE_GLOBAL ||
           AS == NVPTXAS::ADDRESS_SPACE_CONST;
  }
  return true;
}

// Operands whose defined, uniform values make the collective result uniform.
// Other data operands may vary between lanes.
static SmallVector<unsigned, 2> getUniformityOperands(const IntrinsicInst &II) {
  using namespace PatternMatch;
  uint64_t Clamp, Index;
  switch (II.getIntrinsicID()) {
  case Intrinsic::nvvm_shfl_sync_idx_f32:
  case Intrinsic::nvvm_shfl_sync_idx_f32p:
  case Intrinsic::nvvm_shfl_sync_idx_i32:
  case Intrinsic::nvvm_shfl_sync_idx_i32p:
    // Require no segmentation or per-lane fallback to the input value.
    if (match(II.getArgOperand(3), m_ConstantInt(Clamp)) &&
        (Clamp & 0x1f00) == 0 &&
        ((Clamp & 31) == 31 ||
         (match(II.getArgOperand(2), m_ConstantInt(Index)) &&
          (Index & 31) <= (Clamp & 31))))
      return {0, 2};
    return {};
  case Intrinsic::nvvm_barrier_cta_red_and_aligned_all:
  case Intrinsic::nvvm_barrier_cta_red_and_all:
  case Intrinsic::nvvm_barrier_cta_red_or_aligned_all:
  case Intrinsic::nvvm_barrier_cta_red_or_all:
  case Intrinsic::nvvm_barrier_cta_red_popc_aligned_all:
  case Intrinsic::nvvm_barrier_cta_red_popc_all:
  case Intrinsic::nvvm_match_all_sync_i32p:
  case Intrinsic::nvvm_match_all_sync_i64p:
  case Intrinsic::nvvm_vote_all_sync:
  case Intrinsic::nvvm_vote_any_sync:
  case Intrinsic::nvvm_vote_ballot_sync:
  case Intrinsic::nvvm_vote_uni_sync:
    return {0};
  case Intrinsic::nvvm_match_any_sync_i32:
  case Intrinsic::nvvm_match_any_sync_i64:
  case Intrinsic::nvvm_barrier_cta_red_and_aligned_count:
  case Intrinsic::nvvm_barrier_cta_red_and_count:
  case Intrinsic::nvvm_barrier_cta_red_or_aligned_count:
  case Intrinsic::nvvm_barrier_cta_red_or_count:
  case Intrinsic::nvvm_barrier_cta_red_popc_aligned_count:
  case Intrinsic::nvvm_barrier_cta_red_popc_count:
    return {0, 1};
  case Intrinsic::nvvm_redux_sync_add:
  case Intrinsic::nvvm_redux_sync_and:
  case Intrinsic::nvvm_redux_sync_fmax:
  case Intrinsic::nvvm_redux_sync_fmax_NaN:
  case Intrinsic::nvvm_redux_sync_fmax_abs:
  case Intrinsic::nvvm_redux_sync_fmax_abs_NaN:
  case Intrinsic::nvvm_redux_sync_fmin:
  case Intrinsic::nvvm_redux_sync_fmin_NaN:
  case Intrinsic::nvvm_redux_sync_fmin_abs:
  case Intrinsic::nvvm_redux_sync_fmin_abs_NaN:
  case Intrinsic::nvvm_redux_sync_max:
  case Intrinsic::nvvm_redux_sync_min:
  case Intrinsic::nvvm_redux_sync_or:
  case Intrinsic::nvvm_redux_sync_umax:
  case Intrinsic::nvvm_redux_sync_umin:
  case Intrinsic::nvvm_redux_sync_xor:
    return {1};
  default:
    return {};
  }
}

ValueUniformity NVPTXTTIImpl::getValueUniformity(const Value *V) const {
  using namespace PatternMatch;

  if (const auto *Arg = dyn_cast<Argument>(V))
    return isKernelFunction(*Arg->getParent()) ? ValueUniformity::Default
                                               : ValueUniformity::NeverUniform;

  const auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return ValueUniformity::Default;

  // Even a uniform address can observe different writes between lanes.
  if (const auto *LI = dyn_cast<LoadInst>(I))
    return LI->isSimple() && LI->getPointerAddressSpace() == ADDRESS_SPACE_CONST
               ? ValueUniformity::Default
               : ValueUniformity::NeverUniform;
  if (I->isAtomic() || isa<AllocaInst>(I))
    return ValueUniformity::NeverUniform;
  if (const auto *Cast = dyn_cast<AddrSpaceCastInst>(I))
    if (Cast->getSrcAddressSpace() == ADDRESS_SPACE_LOCAL &&
        Cast->getDestAddressSpace() == ADDRESS_SPACE_GENERIC)
      return ValueUniformity::NeverUniform;
  if (const auto *Freeze = dyn_cast<FreezeInst>(I))
    if (!isGuaranteedNotToBeUndefOrPoison(Freeze->getOperand(0)))
      return ValueUniformity::NeverUniform;

  if (const auto *II = dyn_cast<IntrinsicInst>(I)) {
    auto UniformityOperands = getUniformityOperands(*II);
    if (!UniformityOperands.empty()) {
      // Undef controls can select different, individually valid groups/lanes.
      return llvm::all_of(UniformityOperands,
                          [&](unsigned Op) {
                            return isGuaranteedNotToBeUndefOrPoison(
                                II->getArgOperand(Op));
                          })
                 ? ValueUniformity::Custom
                 : ValueUniformity::NeverUniform;
    }

    Intrinsic::ID IID = II->getIntrinsicID();
    if (const auto *CFP = dyn_cast<ConstrainedFPIntrinsic>(II)) {
      switch (IID) {
      case Intrinsic::experimental_constrained_lround:
      case Intrinsic::experimental_constrained_llround:
      case Intrinsic::experimental_constrained_lrint:
      case Intrinsic::experimental_constrained_llrint:
        // An out-of-range result can be an arbitrary defined integer.
        return ValueUniformity::NeverUniform;
      default:
        return CFP->getExceptionBehavior() == fp::ebIgnore &&
                       CFP->getRoundingMode() != RoundingMode::Dynamic
                   ? ValueUniformity::Default
                   : ValueUniformity::NeverUniform;
      }
    }

    switch (IID) {
    case Intrinsic::frexp:
      // The exponent is unspecified for NaN and infinity.
      return isKnownNeverInfOrNaN(II->getArgOperand(0),
                                  SimplifyQuery(I->getDataLayout()))
                 ? ValueUniformity::Default
                 : ValueUniformity::NeverUniform;
    case Intrinsic::nvvm_read_ptx_sreg_tid_x:
    case Intrinsic::nvvm_read_ptx_sreg_tid_y:
    case Intrinsic::nvvm_read_ptx_sreg_tid_z: {
      auto ReqNTID = getReqNTID(*I->getFunction());
      if (!isKernelFunction(*I->getFunction()) || ReqNTID.empty())
        return ValueUniformity::NeverUniform;
      ReqNTID.resize(3, 1);
      unsigned Dim = IID == Intrinsic::nvvm_read_ptx_sreg_tid_x   ? 0
                     : IID == Intrinsic::nvvm_read_ptx_sreg_tid_y ? 1
                                                                  : 2;
      unsigned Stride = llvm::product_of(ArrayRef(ReqNTID).take_front(Dim));
      return ReqNTID[Dim] == 1 || Stride % 32 == 0
                 ? ValueUniformity::AlwaysUniform
                 : ValueUniformity::NeverUniform;
    }
    case Intrinsic::nvvm_is_explicit_cluster:
    case Intrinsic::nvvm_read_ptx_sreg_aggr_smem_size:
    case Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_w:
    case Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_x:
    case Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_y:
    case Intrinsic::nvvm_read_ptx_sreg_cluster_ctaid_z:
    case Intrinsic::nvvm_read_ptx_sreg_cluster_ctarank:
    case Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_w:
    case Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_x:
    case Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_y:
    case Intrinsic::nvvm_read_ptx_sreg_cluster_nctaid_z:
    case Intrinsic::nvvm_read_ptx_sreg_cluster_nctarank:
    case Intrinsic::nvvm_read_ptx_sreg_clusterid_w:
    case Intrinsic::nvvm_read_ptx_sreg_clusterid_x:
    case Intrinsic::nvvm_read_ptx_sreg_clusterid_y:
    case Intrinsic::nvvm_read_ptx_sreg_clusterid_z:
    case Intrinsic::nvvm_read_ptx_sreg_ctaid_w:
    case Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
    case Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
    case Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
    case Intrinsic::nvvm_read_ptx_sreg_dynamic_smem_size:
    case Intrinsic::nvvm_read_ptx_sreg_gridid:
    case Intrinsic::nvvm_read_ptx_sreg_nclusterid_w:
    case Intrinsic::nvvm_read_ptx_sreg_nclusterid_x:
    case Intrinsic::nvvm_read_ptx_sreg_nclusterid_y:
    case Intrinsic::nvvm_read_ptx_sreg_nclusterid_z:
    case Intrinsic::nvvm_read_ptx_sreg_nctaid_w:
    case Intrinsic::nvvm_read_ptx_sreg_nctaid_x:
    case Intrinsic::nvvm_read_ptx_sreg_nctaid_y:
    case Intrinsic::nvvm_read_ptx_sreg_nctaid_z:
    case Intrinsic::nvvm_read_ptx_sreg_nsmid:
    case Intrinsic::nvvm_read_ptx_sreg_ntid_w:
    case Intrinsic::nvvm_read_ptx_sreg_ntid_x:
    case Intrinsic::nvvm_read_ptx_sreg_ntid_y:
    case Intrinsic::nvvm_read_ptx_sreg_ntid_z:
    case Intrinsic::nvvm_read_ptx_sreg_nwarpid:
    case Intrinsic::nvvm_read_ptx_sreg_reserved_smem_offset_begin:
    case Intrinsic::nvvm_read_ptx_sreg_reserved_smem_offset_cap:
    case Intrinsic::nvvm_read_ptx_sreg_reserved_smem_offset_end:
    case Intrinsic::nvvm_read_ptx_sreg_tid_w:
    case Intrinsic::nvvm_read_ptx_sreg_total_smem_size:
    case Intrinsic::nvvm_read_ptx_sreg_warpsize:
      return ValueUniformity::AlwaysUniform;
    default:
      return ValueUniformity::NeverUniform;
    }
  }
  if (isa<CallBase>(I))
    return ValueUniformity::NeverUniform;

  uint64_t C;
  if ((match(V, m_Shr(m_Intrinsic<Intrinsic::nvvm_read_ptx_sreg_tid_x>(),
                      m_ConstantInt(C))) &&
       C >= 5) ||
      (match(V, m_c_And(m_Intrinsic<Intrinsic::nvvm_read_ptx_sreg_tid_x>(),
                        m_ConstantInt(C))) &&
       (C & 31) == 0)) {
    // maxntid limits only the total thread count, not individual dimensions.
    const auto ReqNTID = getReqNTID(*I->getFunction());
    if (isKernelFunction(*I->getFunction()) && !ReqNTID.empty() &&
        (ReqNTID[0] <= 32 || ReqNTID[0] % 32 == 0 ||
         llvm::all_of(drop_begin(ReqNTID), [](unsigned N) { return N == 1; })))
      return ValueUniformity::AlwaysUniform;
  }
  return ValueUniformity::Default;
}

bool NVPTXTTIImpl::isUniform(const Instruction *I,
                             const SmallBitVector &UniformArgs) const {
  return llvm::all_of(getUniformityOperands(*cast<IntrinsicInst>(I)),
                      [&](unsigned Op) { return UniformArgs[Op]; });
}
