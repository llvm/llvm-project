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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/NVPTXAddrSpace.h"
#include "llvm/Transforms/InstCombine/InstCombiner.h"
#include <optional>
using namespace llvm;

#define DEBUG_TYPE "NVPTXtti"

// Whether the given intrinsic reads threadIdx.x/y/z.
static bool readsThreadIndex(const IntrinsicInst *II) {
  switch (II->getIntrinsicID()) {
    default: return false;
    case Intrinsic::nvvm_read_ptx_sreg_tid_x:
    case Intrinsic::nvvm_read_ptx_sreg_tid_y:
    case Intrinsic::nvvm_read_ptx_sreg_tid_z:
      return true;
  }
}

static bool readsLaneId(const IntrinsicInst *II) {
  return II->getIntrinsicID() == Intrinsic::nvvm_read_ptx_sreg_laneid;
}

bool NVPTXTTIImpl::isSourceOfDivergence(const Value *V) const {
  // Without inter-procedural analysis, we conservatively assume that arguments
  // to __device__ functions are divergent.
  if (const Argument *Arg = dyn_cast<Argument>(V))
    return !isKernelFunction(*Arg->getParent());

  if (const Instruction *I = dyn_cast<Instruction>(V)) {
    // Without pointer analysis, we conservatively assume values loaded from
    // generic or local address space are divergent.
    if (const LoadInst *LI = dyn_cast<LoadInst>(I)) {
      unsigned AS = LI->getPointerAddressSpace();
      return AS == ADDRESS_SPACE_GENERIC || AS == ADDRESS_SPACE_LOCAL;
    }
    // Atomic instructions may cause divergence. Atomic instructions are
    // executed sequentially across all threads in a warp. Therefore, an earlier
    // executed thread may see different memory inputs than a later executed
    // thread. For example, suppose *a = 0 initially.
    //
    //   atom.global.add.s32 d, [a], 1
    //
    // returns 0 for the first thread that enters the critical region, and 1 for
    // the second thread.
    if (I->isAtomic())
      return true;
    if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
      // Instructions that read threadIdx are obviously divergent.
      if (readsThreadIndex(II) || readsLaneId(II))
        return true;
    }
    // Conservatively consider the return value of function calls as divergent.
    // We could analyze callees with bodies more precisely using
    // inter-procedural analysis.
    if (isa<CallInst>(I))
      return true;
  }

  return false;
}

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

// Returns true/false when we know the answer, nullopt otherwise.
static std::optional<bool> evaluateIsSpace(Intrinsic::ID IID, unsigned AS) {
  if (AS == NVPTXAS::ADDRESS_SPACE_GENERIC ||
      AS == NVPTXAS::ADDRESS_SPACE_ENTRY_PARAM)
    return std::nullopt; // Got to check at run-time.
  switch (IID) {
  case Intrinsic::nvvm_isspacep_global:
    return AS == NVPTXAS::ADDRESS_SPACE_GLOBAL;
  case Intrinsic::nvvm_isspacep_local:
    return AS == NVPTXAS::ADDRESS_SPACE_LOCAL;
  case Intrinsic::nvvm_isspacep_shared:
    // If shared cluster this can't be evaluated at compile time.
    if (AS == NVPTXAS::ADDRESS_SPACE_SHARED_CLUSTER)
      return std::nullopt;
    return AS == NVPTXAS::ADDRESS_SPACE_SHARED;
  case Intrinsic::nvvm_isspacep_shared_cluster:
    return AS == NVPTXAS::ADDRESS_SPACE_SHARED_CLUSTER ||
           AS == NVPTXAS::ADDRESS_SPACE_SHARED;
  case Intrinsic::nvvm_isspacep_const:
    return AS == NVPTXAS::ADDRESS_SPACE_CONST;
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

std::optional<Instruction *>
NVPTXTTIImpl::instCombineIntrinsic(InstCombiner &IC, IntrinsicInst &II) const {
  if (std::optional<Instruction *> I = handleSpaceCheckIntrinsics(IC, II))
    return *I;
  if (Instruction *I = convertNvvmIntrinsicToLlvm(IC, &II))
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

static bool isV2F32Ty(Type *Ty) {
  // PTX has native packed arithmetic for pairs of f32 values.
  auto *VTy = dyn_cast<FixedVectorType>(Ty);
  return VTy && VTy->getNumElements() == 2 &&
         VTy->getElementType()->isFloatTy();
}

static bool isF32VectorArithmeticOpcode(unsigned Opcode) {
  // These opcodes map to packed f32x2 PTX arithmetic when the vector is v2f32.
  switch (Opcode) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
    return true;
  default:
    return false;
  }
}

static bool isDemandedSplat(ArrayRef<Value *> VL, const APInt &DemandedElts) {
  // Packed f32x2 arithmetic can consume scalar f32 operands as broadcasts, so
  // splat buildvectors are free when the demanded lanes are identical.
  if (VL.empty() || DemandedElts.getBitWidth() != VL.size())
    return false;

  Value *Splat = nullptr;
  for (unsigned Idx = 0, E = VL.size(); Idx != E; ++Idx) {
    if (!DemandedElts[Idx] || isa<UndefValue>(VL[Idx]))
      continue;
    if (!Splat) {
      Splat = VL[Idx];
      continue;
    }
    if (VL[Idx] != Splat)
      return false;
  }
  return Splat != nullptr;
}

static bool isCheapPTXVectorInsertExtract(Type *Ty, const DataLayout &DL,
                                          const TargetLoweringBase *TLI) {
  assert(TLI && "Expected NVPTX TargetLowering");
  auto *VTy = dyn_cast<FixedVectorType>(Ty);
  return !VTy || NVPTX::isPackedVectorTy(TLI->getValueType(DL, Ty));
}

static bool hasNativeNVPTXVectorArithmetic(unsigned Opcode, Type *Ty,
                                           const DataLayout &DL,
                                           const TargetLoweringBase *TLI) {
  assert(TLI && "Expected NVPTX TargetLowering");
  if (!isa<FixedVectorType>(Ty))
    return false;

  if (isF32VectorArithmeticOpcode(Opcode) && isV2F32Ty(Ty))
    return true;

  int ISD = TLI->InstructionOpcodeToISD(Opcode);
  if (ISD == 0)
    return false;

  EVT VT = TLI->getValueType(DL, Ty);
  return TLI->isOperationLegal(ISD, VT);
}

static InstructionCost getVectorRegisterPieceCost(Type *Ty,
                                                  const DataLayout &DL) {
  auto *VTy = dyn_cast<FixedVectorType>(Ty);
  if (!VTy)
    return 0;

  constexpr unsigned NVPTXRegBits = 32;
  unsigned TyBits = DL.getTypeSizeInBits(Ty).getFixedValue();
  unsigned NumPieces = divideCeil(TyBits, NVPTXRegBits);
  return NumPieces;
}

static InstructionCost getNonNativeVectorOpPenalty(
    Type *Ty, const DataLayout &DL, const TargetLoweringBase *TLI) {
  auto *VTy = dyn_cast<FixedVectorType>(Ty);
  if (!VTy || isCheapPTXVectorInsertExtract(Ty, DL, TLI))
    return 0;

  // Model a non-native vector operation by the scalar lane work plus the
  // register-sized pieces that have to be assembled/disassembled around it.
  return VTy->getNumElements() + getVectorRegisterPieceCost(Ty, DL);
}

static InstructionCost getNonNativeVectorShufflePenalty(
    VectorType *DstTy, VectorType *SrcTy, VectorType *SubTp,
    const DataLayout &DL, const TargetLoweringBase *TLI) {
  InstructionCost Cost = 0;
  for (VectorType *Ty : {DstTy, SrcTy, SubTp}) {
    if (!Ty)
      continue;
    Cost += getNonNativeVectorOpPenalty(Ty, DL, TLI);
  }
  return Cost;
}

InstructionCost NVPTXTTIImpl::getShuffleCost(
    TTI::ShuffleKind Kind, VectorType *DstTy, VectorType *SrcTy,
    ArrayRef<int> Mask, TTI::TargetCostKind CostKind, int Index,
    VectorType *SubTp, ArrayRef<const Value *> Args,
    const Instruction *CxtI) const {
  InstructionCost Cost =
      BaseT::getShuffleCost(Kind, DstTy, SrcTy, Mask, CostKind, Index, SubTp,
                            Args, CxtI);

  if (CostKind != TTI::TCK_RecipThroughput)
    return Cost;

  // A scalar f32 broadcast feeding f32x2 arithmetic can use the scalar operand
  // form of the packed PTX instruction, so do not charge a shuffle for it.
  if (isV2F32Ty(DstTy) &&
      (Kind == TTI::SK_Broadcast ||
       (Kind == TTI::SK_PermuteSingleSrc &&
        ShuffleVectorInst::isZeroEltSplatMask(Mask, Mask.size())))) {
    return TTI::TCC_Free;
  }

  return Cost + getNonNativeVectorShufflePenalty(DstTy, SrcTy, SubTp, DL, TLI);
}

InstructionCost NVPTXTTIImpl::getVectorInstrCost(
    unsigned Opcode, Type *Val, TTI::TargetCostKind CostKind, unsigned Index,
    const Value *Op0, const Value *Op1, TTI::VectorInstrContext VIC) const {
  InstructionCost Cost =
      BaseT::getVectorInstrCost(Opcode, Val, CostKind, Index, Op0, Op1, VIC);

  if (CostKind != TTI::TCK_RecipThroughput ||
      (Opcode != Instruction::InsertElement &&
       Opcode != Instruction::ExtractElement) ||
      VIC != TTI::VectorInstrContext::None ||
      isCheapPTXVectorInsertExtract(Val, DL, TLI))
    return Cost;

  return Cost + getNonNativeVectorOpPenalty(Val, DL, TLI);
}

InstructionCost NVPTXTTIImpl::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, TTI::TargetCostKind CostKind,
    TTI::OperandValueInfo Op1Info, TTI::OperandValueInfo Op2Info,
    ArrayRef<const Value *> Args, const Instruction *CxtI) const {
  InstructionCost Cost =
      BaseT::getArithmeticInstrCost(Opcode, Ty, CostKind, Op1Info, Op2Info);
  if (CostKind == TTI::TCK_RecipThroughput) {
    if (auto *VTy = dyn_cast<FixedVectorType>(Ty);
        VTy && !hasNativeNVPTXVectorArithmetic(Opcode, Ty, DL, TLI)) {
      InstructionCost ScalarCost =
          VTy->getNumElements() *
          BaseT::getArithmeticInstrCost(Opcode, VTy->getElementType(), CostKind,
                                        Op1Info, Op2Info);
      InstructionCost LegalizedCost =
          ScalarCost + getNonNativeVectorOpPenalty(Ty, DL, TLI);
      if (LegalizedCost > Cost)
        Cost = LegalizedCost;
    }
  }

  // Legalize the type.
  std::pair<InstructionCost, MVT> LT = getTypeLegalizationCost(Ty);

  int ISD = TLI->InstructionOpcodeToISD(Opcode);

  switch (ISD) {
  default:
    return Cost;
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
    return Cost;
  }
}

InstructionCost NVPTXTTIImpl::getScalarizationOverhead(
    VectorType *InTy, const APInt &DemandedElts, bool Insert, bool Extract,
    TTI::TargetCostKind CostKind, bool ForPoisonSrc, ArrayRef<Value *> VL,
    TTI::VectorInstrContext VIC) const {
  if (!InTy->getElementCount().isFixed())
    return InstructionCost::getInvalid();

  auto VT = getTLI()->getValueType(DL, InTy);
  auto NumElements = InTy->getElementCount().getFixedValue();
  InstructionCost Cost = 0;
  if (Insert && !VL.empty()) {
    bool AllConstant = all_of(seq(NumElements), [&](int Idx) {
      return !DemandedElts[Idx] || isa<Constant>(VL[Idx]);
    });
    if (AllConstant) {
      Cost += TTI::TCC_Free;
      Insert = false;
    } else if (isV2F32Ty(InTy) && isDemandedSplat(VL, DemandedElts)) {
      // A splat buildvector can be represented by a scalar broadcast operand of
      // the packed f32 instruction.
      Cost += TTI::TCC_Free;
      Insert = false;
    }
  }
  if (Insert && NVPTX::isPackedVectorTy(VT) && VT.is32BitVector()) {
    // Can be built in a single 32-bit mov (64-bit regs are emulated in SASS
    // with 2x 32-bit regs)
    Cost += 1;
    Insert = false;
  }
  if (Insert && VT == MVT::v4i8) {
    Cost += 3; // 3 x PRMT
    for (auto Idx : seq(NumElements))
      if (DemandedElts[Idx])
        Cost += 1; // zext operand to i32
    Insert = false;
  }
  InstructionCost BaseCost = BaseT::getScalarizationOverhead(
      InTy, DemandedElts, Insert, Extract, CostKind, ForPoisonSrc, VL, VIC);

  if (Extract && CostKind == TTI::TCK_RecipThroughput &&
      !NVPTX::isPackedVectorTy(VT) && !VT.is32BitVector() &&
      !VT.is16BitVector()) {
    InstructionCost ExtractCost = DemandedElts.popcount();
    if (!Insert)
      return Cost + ExtractCost;
  }

  return Cost + BaseCost;
}

InstructionCost NVPTXTTIImpl::getCastInstrCost(
    unsigned Opcode, Type *Dst, Type *Src, TTI::CastContextHint CCH,
    TTI::TargetCostKind CostKind, const Instruction *I) const {
  InstructionCost Cost =
      BaseT::getCastInstrCost(Opcode, Dst, Src, CCH, CostKind, I);
  if (CostKind != TTI::TCK_RecipThroughput)
    return Cost;

  auto *DstVTy = dyn_cast<FixedVectorType>(Dst);
  auto *SrcVTy = dyn_cast<FixedVectorType>(Src);
  if (!DstVTy || !SrcVTy)
    return Cost;

  if (DstVTy->getNumElements() == SrcVTy->getNumElements()) {
    InstructionCost ScalarCost =
        DstVTy->getNumElements() *
        BaseT::getCastInstrCost(Opcode, DstVTy->getElementType(),
                                SrcVTy->getElementType(), CCH, CostKind, I);
    InstructionCost LegalizedCost =
        ScalarCost + getNonNativeVectorOpPenalty(Dst, DL, TLI) +
        getNonNativeVectorOpPenalty(Src, DL, TLI);
    if (LegalizedCost > Cost)
      Cost = LegalizedCost;
  } else if (Opcode == Instruction::BitCast) {
    int ISD = TLI->InstructionOpcodeToISD(Opcode);
    if (ISD == 0)
      return Cost + getNonNativeVectorOpPenalty(Dst, DL, TLI) +
             getNonNativeVectorOpPenalty(Src, DL, TLI);

    EVT DstVT = TLI->getValueType(DL, Dst);
    if (!TLI->isOperationLegal(ISD, DstVT))
      Cost += getNonNativeVectorOpPenalty(Dst, DL, TLI) +
              getNonNativeVectorOpPenalty(Src, DL, TLI);
  }
  return Cost;
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

ValueUniformity NVPTXTTIImpl::getValueUniformity(const Value *V) const {
  if (isSourceOfDivergence(V))
    return ValueUniformity::NeverUniform;

  return ValueUniformity::Default;
}
