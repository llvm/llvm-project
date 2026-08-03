//===-- PISAEmitIntrinsics.cpp - emit PISA intrinsics ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISA.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicsPISA.h"

#define DEBUG_TYPE "pisa-emit-intrinsics"
#define DEBUG_NAME "PISA emit intrinsics"

using namespace llvm;

namespace llvm {
void initializePISAEmitIntrinsicsPass(PassRegistry &);
} // namespace llvm

// Helper that answers two related questions about the SNaN/QNaN payload of
// a value, both used to decide when @llvm.fabs is safe to lower to the
// PISA hardware fabs (which quiets SNaN inputs):
//
//   * cannotBeSNaN(V) - def-side walk: is V's bit pattern provably never a
//     signaling NaN? True for IEEE arithmetic results (always produce QNaN),
//     integer-to-FP conversions, nnan-flagged producers, non-signaling
//     constants, nofpclass(snan) arguments / call results, and sign-only
//     recursion through fabs / fneg / copysign.
//
//   * allUsesAreSNaNInsensitive(V) - use-side walk: do every transitive
//     consumer of V treat an SNaN payload the same as a QNaN payload?
//     True when all uses reach fcmp (boolean result), maxnum / minnum /
//     maximumnum / minimumnum (which quiet SNaN internally per IEEE
//     754-2019), or further sign-only / lane-shuffle ops feeding the same
//     set.
//
// Either condition alone is sufficient to rewrite @llvm.fabs to
// @llvm.pisa.fabs. The two queries are conceptually dual (def-side vs
// use-side) and share the same recursion infrastructure, depth limit and
// cache, so they live in a single class. Per-Value caching is important:
// fabs calls in straight-line code often share common subexpressions, and
// extractelement / lane-shuffle chains naturally revisit the same value.
//
// Other operations (arithmetic-as-a-consumer, bitcast-to-int, store, ret,
// opaque calls, llvm.maximum / llvm.minimum, etc.) are conservatively
// rejected. IEEE arithmetic does in fact quiet SNaN at the output, but
// keeping the use-side rule narrow protects the optimisation under
// constrained FP environments and integer-bit observers.
class SNaNPayloadAnalysis {
  static constexpr unsigned MaxDepth = 8;

  // One cache entry per Value, holding the result of either or both queries
  // once they've been computed. std::optional makes the "not yet computed"
  // state explicit, so a Value asked only the def-side question never
  // populates the use-side answer (and vice versa).
  struct CachedResult {
    std::optional<bool> CannotBeSNaN;
    std::optional<bool> AllUsesInsensitive;
  };
  DenseMap<const Value *, CachedResult> Cache;

  // --- Def-side computation -------------------------------------------------
  bool computeCannotBeSNaN(const Value *V, unsigned Depth) {
    if (Depth > MaxDepth)
      return false;

    // Constants: check the bit pattern directly.
    if (const auto *CFP = dyn_cast<ConstantFP>(V))
      return !CFP->getValueAPF().isSignaling();

    if (const auto *C = dyn_cast<Constant>(V)) {
      if (isa<UndefValue>(C) || isa<PoisonValue>(C))
        return true;
      if (auto *VTy = dyn_cast<FixedVectorType>(C->getType())) {
        for (unsigned I = 0, E = VTy->getNumElements(); I != E; ++I) {
          const Constant *Elt = C->getAggregateElement(I);
          // getAggregateElement may return null for constants it can't split
          // (e.g. some ConstantExpr forms). Be conservative.
          if (!Elt || !computeCannotBeSNaN(Elt, Depth + 1))
            return false;
        }
        return true;
      }
      // Other constants (ConstantExpr, etc.); be conservative.
      return false;
    }

    // Function arguments / call results with a `nofpclass(snan)` attribute.
    if (const auto *Arg = dyn_cast<Argument>(V))
      return (Arg->getNoFPClass() & fcSNan) == fcSNan;
    if (const auto *CB = dyn_cast<CallBase>(V))
      if ((CB->getRetNoFPClass() & fcSNan) == fcSNan)
        return true;

    // nnan implies no NaN at all, hence no signaling NaN.
    if (const auto *FPOp = dyn_cast<FPMathOperator>(V))
      if (FPOp->hasNoNaNs())
        return true;

    const auto *I = dyn_cast<Instruction>(V);
    if (!I)
      return false;

    switch (I->getOpcode()) {
    // IEEE 754 arithmetic ops produce QNaN, never SNaN.
    case Instruction::FAdd:
    case Instruction::FSub:
    case Instruction::FMul:
    case Instruction::FDiv:
    case Instruction::FRem:
      // PISA frem is legalized to fdiv+fma+selects, all IEEE arithmetic that
      // produces QNaN whenever it produces a NaN.
      return true;
    // Integer-to-FP conversions can never produce a NaN.
    case Instruction::SIToFP:
    case Instruction::UIToFP:
      return true;
    // FNeg only flips the sign bit; SNaN-ness is preserved.
    case Instruction::FNeg:
      return cannotBeSNaN(I->getOperand(0), Depth + 1);
    case Instruction::Select:
      return cannotBeSNaN(I->getOperand(1), Depth + 1) &&
             cannotBeSNaN(I->getOperand(2), Depth + 1);
    case Instruction::ExtractElement:
      return cannotBeSNaN(I->getOperand(0), Depth + 1);
    case Instruction::InsertElement:
      return cannotBeSNaN(I->getOperand(0), Depth + 1) &&
             cannotBeSNaN(I->getOperand(1), Depth + 1);
    case Instruction::Call: {
      const auto *II = dyn_cast<IntrinsicInst>(I);
      if (!II)
        return false;
      switch (II->getIntrinsicID()) {
      // Generic LLVM math intrinsics produce QNaN per IEEE 754.
      case Intrinsic::fma:
      case Intrinsic::fmuladd:
      case Intrinsic::sqrt:
      case Intrinsic::sin:
      case Intrinsic::cos:
      case Intrinsic::tan:
      case Intrinsic::exp:
      case Intrinsic::exp2:
      case Intrinsic::log:
      case Intrinsic::log2:
      case Intrinsic::log10:
      case Intrinsic::pow:
      case Intrinsic::powi:
      case Intrinsic::minnum:
      case Intrinsic::maxnum:
      case Intrinsic::minimum:
      case Intrinsic::maximum:
      case Intrinsic::canonicalize:
      case Intrinsic::pisa_fabs:
      case Intrinsic::pisa_fadd:
      case Intrinsic::pisa_fsub:
      case Intrinsic::pisa_fmul:
      case Intrinsic::pisa_fma:
      case Intrinsic::pisa_fdiv_rnd:
      case Intrinsic::pisa_pow_rnd:
      case Intrinsic::pisa_fsqrt_rnd:
      case Intrinsic::pisa_frnd_rnd:
      case Intrinsic::pisa_frcp:
      case Intrinsic::pisa_frcp_rnd:
      case Intrinsic::pisa_frsqrt:
      case Intrinsic::pisa_sin_rnd:
      case Intrinsic::pisa_cos_rnd:
      case Intrinsic::pisa_tanh_rnd:
      case Intrinsic::pisa_exp_rnd:
      case Intrinsic::pisa_exp2_rnd:
      case Intrinsic::pisa_log_rnd:
      case Intrinsic::pisa_log2_rnd:
      case Intrinsic::pisa_log10_rnd:
      case Intrinsic::pisa_fmin_sat:
      case Intrinsic::pisa_fmax_sat:
      case Intrinsic::pisa_ftrunc:
      case Intrinsic::pisa_sitofp:
      case Intrinsic::pisa_uitofp:
        return true;
      // IR fabs / copysign preserve the magnitude operand's NaN payload, so
      // SNaN-ness is preserved; recurse into the source.
      case Intrinsic::fabs:
      case Intrinsic::copysign:
        return cannotBeSNaN(II->getArgOperand(0), Depth + 1);
      default:
        return false;
      }
    }
    default:
      return false;
    }
  }

  // --- Use-side computation -------------------------------------------------
  bool computeAllUsesInsensitive(const Value *V, unsigned Depth) {
    if (Depth > MaxDepth)
      return false;
    if (V->use_empty())
      // No observable users at all - vacuously insensitive.
      return true;
    for (const User *U : V->users()) {
      const auto *I = dyn_cast<Instruction>(U);
      if (!I || !isUserInsensitive(*I, Depth + 1))
        return false;
    }
    return true;
  }

  bool isUserInsensitive(const Instruction &I, unsigned Depth) {
    switch (I.getOpcode()) {
    case Instruction::FCmp:
      return true;
    case Instruction::ExtractElement:
    case Instruction::InsertElement:
    case Instruction::ShuffleVector:
      return allUsesAreSNaNInsensitive(&I, Depth);
    case Instruction::Call: {
      const auto *II = dyn_cast<IntrinsicInst>(&I);
      if (!II)
        return false;
      switch (II->getIntrinsicID()) {
      // Per IEEE 754-2019 maxnum/minnum treat SNaN as missing data, quieting
      // it internally. The distinction in the result is gone.
      case Intrinsic::maxnum:
      case Intrinsic::minnum:
      case Intrinsic::maximumnum:
      case Intrinsic::minimumnum:
        return true;
      // Constrained maxnum/minnum and non-signaling fcmp: the data result
      // is insensitive to SNaN vs QNaN, but under fpexcept.strict /
      // fpexcept.maytrap the exception flags differ (IEEE 754 mandates
      // FE_INVALID for SNaN operands). Only safe when exceptions are
      // ignored.
      case Intrinsic::experimental_constrained_maxnum:
      case Intrinsic::experimental_constrained_minnum:
      case Intrinsic::experimental_constrained_fcmp: {
        const auto *CFP = cast<ConstrainedFPIntrinsic>(II);
        auto EB = CFP->getExceptionBehavior();
        return EB && *EB == fp::ebIgnore;
      }
      // Sign-only operations are bit-preserving on the payload, so the
      // distinction propagates through; recurse to the consumers.
      case Intrinsic::fabs:
      case Intrinsic::pisa_fabs:
      case Intrinsic::copysign:
        return allUsesAreSNaNInsensitive(II, Depth);
      // experimental.constrained.fcmps - the signaling comparison signals
      // FE_INVALID on *any* NaN (SNaN or QNaN), so quieting makes no
      // difference to exception behavior.
      case Intrinsic::experimental_constrained_fcmps:
        return true;
      default:
        return false;
      }
    }
    default:
      return false;
    }
  }

public:
  // Query whether V is provably never a signaling NaN. Result is cached.
  bool cannotBeSNaN(const Value *V, unsigned Depth = 0) {
    auto &Slot = Cache[V];
    if (Slot.CannotBeSNaN)
      return *Slot.CannotBeSNaN;
    bool Result = computeCannotBeSNaN(V, Depth);
    // Re-fetch: recursion may have inserted intermediate entries and
    // invalidated the earlier reference.
    Cache[V].CannotBeSNaN = Result;
    return Result;
  }

  // Query whether all transitive users of V are SNaN-payload-insensitive.
  // Result is cached.
  bool allUsesAreSNaNInsensitive(const Value *V, unsigned Depth = 0) {
    auto &Slot = Cache[V];
    if (Slot.AllUsesInsensitive)
      return *Slot.AllUsesInsensitive;
    bool Result = computeAllUsesInsensitive(V, Depth);
    // Re-fetch: recursion may have inserted intermediate entries and
    // invalidated the earlier reference.
    Cache[V].AllUsesInsensitive = Result;
    return Result;
  }
};

namespace {
class PISAEmitIntrinsics : public FunctionPass,
                           public InstVisitor<PISAEmitIntrinsics> {

  IRBuilder<> *IRB = nullptr;
  bool Changed = false;
  SNaNPayloadAnalysis SNaNAnalysis;

public:
  static char ID;
  PISAEmitIntrinsics() : FunctionPass(ID) {
    initializePISAEmitIntrinsicsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  void visitIntrinsicInst(IntrinsicInst &I);
};
} // namespace

char PISAEmitIntrinsics::ID = 0;
INITIALIZE_PASS(PISAEmitIntrinsics, DEBUG_TYPE, DEBUG_NAME, false, false)

// given a metadata node, return CmpInst::Predicate representation
static CmpInst::Predicate getPredicateFromName(StringRef Name) {
  CmpInst::Predicate Pred;
  Pred = StringSwitch<CmpInst::Predicate>(Name)
             .Case("false", CmpInst::FCMP_FALSE)
             .Case("oeq", CmpInst::FCMP_OEQ)
             .Case("ogt", CmpInst::FCMP_OGT)
             .Case("oge", CmpInst::FCMP_OGE)
             .Case("olt", CmpInst::FCMP_OLT)
             .Case("ole", CmpInst::FCMP_OLE)
             .Case("one", CmpInst::FCMP_ONE)
             .Case("ord", CmpInst::FCMP_ORD)
             .Case("uno", CmpInst::FCMP_UNO)
             .Case("ueq", CmpInst::FCMP_UEQ)
             .Case("ugt", CmpInst::FCMP_UGT)
             .Case("uge", CmpInst::FCMP_UGE)
             .Case("ult", CmpInst::FCMP_ULT)
             .Case("ule", CmpInst::FCMP_ULE)
             .Case("une", CmpInst::FCMP_UNE)
             .Case("true", CmpInst::FCMP_TRUE)
             .Default(CmpInst::BAD_FCMP_PREDICATE);
  return Pred;
}

// replace one intrinsic with another, stripping some args
static void replaceIntrinsicWith(IntrinsicInst *II, Intrinsic::ID IID,
                                 unsigned IgnoreLast) {
  IRBuilder<> IRB(II);

  SmallVector<Value *, 4> Args;
  // NOLINTNEXTLINE(llvm-qualified-auto)
  for (auto It = II->arg_begin(); It != (II->arg_end() - IgnoreLast); ++It)
    Args.push_back(*It);
  auto *NewI = IRB.CreateIntrinsic(II->getType(), IID, Args,
                                   isa<FPMathOperator>(II) ? II : nullptr);
  II->replaceAllUsesWith(NewI);
  II->eraseFromParent();
}

// replace metadata with equivalent numerical representation
static void replaceRoundingModeMD(IntrinsicInst *II, Intrinsic::ID IID,
                                  bool IsConstrained) {
  IRBuilder<> IRB(II);
  auto *ImmTy = Type::getInt8Ty(II->getContext());

  // constrained have roundmode, exception on the end
  // non-constrained only have roundmode on the end
  auto IgnoreLast = IsConstrained ? 2 : 1;
  auto *MD =
      cast<MetadataAsValue>(II->getArgOperand(II->arg_size() - IgnoreLast))
          ->getMetadata();
  auto RoundMode = convertStrToRoundingMode(cast<MDString>(MD)->getString());
  // PISA has no runtime rounding-mode control. Map round.dynamic to the
  // IEEE 754 default (round-to-nearest-even).
  if (!RoundMode || *RoundMode == RoundingMode::Dynamic)
    RoundMode = RoundingMode::NearestTiesToEven;
  Constant *RoundVal = ConstantInt::get(ImmTy, (unsigned)*RoundMode);

  SmallVector<Value *, 4> Args;
  // NOLINTNEXTLINE(llvm-qualified-auto)
  for (auto It = II->arg_begin(); It != (II->arg_end() - IgnoreLast); ++It)
    Args.push_back(*It);
  Args.push_back(RoundVal);
  switch (IID) {
  default:
    break;
  case Intrinsic::pisa_fadd:
  case Intrinsic::pisa_fsub:
  case Intrinsic::pisa_fmul:
  case Intrinsic::pisa_fma:
  case Intrinsic::pisa_ftrunc:
  case Intrinsic::pisa_uitofp:
  case Intrinsic::pisa_sitofp:
    Args.push_back(ConstantInt::getFalse(II->getContext())); // saturation
    break;
  }
  auto *NewI = IRB.CreateIntrinsic(II->getType(), IID, Args,
                                   isa<FPMathOperator>(II) ? II : nullptr);
  II->replaceAllUsesWith(NewI);
  II->eraseFromParent();
}

void PISAEmitIntrinsics::visitIntrinsicInst(IntrinsicInst &I) {
  IRB->SetInsertPoint(&I);
  auto II = I.getIntrinsicID();
  switch (II) {
// intrinsics with rounding mode (e.g. llvm.experimental.constrained*) are
// mapped into PISA equivalents, with rounding mode MD being mapped to an
// equivalent immediate values. ISel maps these to proper instructions.
#define REPLACE_ROUNDMODE(from, to, constrained)                               \
  case from: {                                                                 \
    replaceRoundingModeMD(&I, to, constrained);                                \
    Changed = true;                                                            \
    break;                                                                     \
  }
    REPLACE_ROUNDMODE(Intrinsic::pisa_fptoui_md, Intrinsic::pisa_fptoui_rnd,
                      false)
    REPLACE_ROUNDMODE(Intrinsic::pisa_fptosi_md, Intrinsic::pisa_fptosi_rnd,
                      false)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_fadd,
                      Intrinsic::pisa_fadd, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_fsub,
                      Intrinsic::pisa_fsub, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_fmul,
                      Intrinsic::pisa_fmul, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_fdiv,
                      Intrinsic::pisa_fdiv_rnd, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_uitofp,
                      Intrinsic::pisa_uitofp, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_sitofp,
                      Intrinsic::pisa_sitofp, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_sqrt,
                      Intrinsic::pisa_fsqrt_rnd, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_fma,
                      Intrinsic::pisa_fma, true)
    // TODO: check if we need to split into mul+add
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_fmuladd,
                      Intrinsic::pisa_fma, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_fptrunc,
                      Intrinsic::pisa_ftrunc, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_rint,
                      Intrinsic::pisa_frnd_rnd, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_log2,
                      Intrinsic::pisa_log2_rnd, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_sin,
                      Intrinsic::pisa_sin_rnd, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_cos,
                      Intrinsic::pisa_cos_rnd, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_tanh,
                      Intrinsic::pisa_tanh_rnd, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_exp,
                      Intrinsic::pisa_exp_rnd, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_exp2,
                      Intrinsic::pisa_exp2_rnd, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_log,
                      Intrinsic::pisa_log_rnd, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_log10,
                      Intrinsic::pisa_log10_rnd, true)
    REPLACE_ROUNDMODE(Intrinsic::experimental_constrained_pow,
                      Intrinsic::pisa_pow_rnd, true)
#undef REPLACE_ROUNDMODE
#define REPLACE_INTRINSIC(from, to, strip)                                     \
  case from: {                                                                 \
    replaceIntrinsicWith(&I, to, strip);                                       \
    Changed = true;                                                            \
    break;                                                                     \
  }
    REPLACE_INTRINSIC(Intrinsic::experimental_constrained_minnum,
                      Intrinsic::minnum, 1)
    REPLACE_INTRINSIC(Intrinsic::experimental_constrained_maxnum,
                      Intrinsic::maxnum, 1)
    REPLACE_INTRINSIC(Intrinsic::experimental_constrained_floor,
                      Intrinsic::floor, 1)
    REPLACE_INTRINSIC(Intrinsic::experimental_constrained_ceil, Intrinsic::ceil,
                      1)
    REPLACE_INTRINSIC(Intrinsic::experimental_constrained_round,
                      Intrinsic::round, 1)
    REPLACE_INTRINSIC(Intrinsic::experimental_constrained_trunc,
                      Intrinsic::trunc, 1)
#undef REPLACE_INTRINSIC
  case Intrinsic::experimental_constrained_fptosi: {
    // exceptions are not supported
    auto *NewI = IRB->CreateFPToSI(I.getOperand(0), I.getType());
    I.replaceAllUsesWith(NewI);
    I.eraseFromParent();
    Changed = true;
  } break;
  case Intrinsic::experimental_constrained_fptoui: {
    // exceptions are not supported
    auto *NewI = IRB->CreateFPToUI(I.getOperand(0), I.getType());
    I.replaceAllUsesWith(NewI);
    I.eraseFromParent();
    Changed = true;
  } break;
  case Intrinsic::experimental_constrained_fpext: {
    // exceptions are not supported
    auto *NewI = IRB->CreateFPExt(I.getOperand(0), I.getType());
    I.replaceAllUsesWith(NewI);
    I.eraseFromParent();
    Changed = true;
  } break;
  case Intrinsic::experimental_constrained_fcmp: {
    // exceptions are not supported
    auto *MD = cast<MetadataAsValue>(I.getOperand(2))->getMetadata();
    auto Predicate = getPredicateFromName(cast<MDString>(MD)->getString());
    auto *NewI = IRB->CreateFCmp(Predicate, I.getOperand(0), I.getOperand(1));
    I.replaceAllUsesWith(NewI);
    I.eraseFromParent();
    Changed = true;
  } break;
  case Intrinsic::experimental_constrained_fcmps: {
    // exceptions are not supported
    auto *MD = cast<MetadataAsValue>(I.getOperand(2))->getMetadata();
    auto Predicate = getPredicateFromName(cast<MDString>(MD)->getString());
    auto *NewI = IRB->CreateFCmpS(Predicate, I.getOperand(0), I.getOperand(1));
    I.replaceAllUsesWith(NewI);
    I.eraseFromParent();
    Changed = true;
  } break;
  case Intrinsic::experimental_constrained_frem: {
    // https://llvm.org/docs/LangRef.html#llvm-experimental-constrained-frem-intrinsic
    // ... rounding mode argument has no effect ...
    auto *NewI = IRB->CreateFRem(I.getOperand(0), I.getOperand(1));
    if (auto *CastedNewI = dyn_cast<Instruction>(NewI))
      CastedNewI->setFastMathFlags(I.getFastMathFlags());
    I.replaceAllUsesWith(NewI);
    I.eraseFromParent();
    Changed = true;
  } break;
  case Intrinsic::fabs: {
    // PISA fabs and IEEE fabs differ only on signaling NaN inputs: PISA fabs
    // quiets the NaN, while IEEE fabs preserves the payload. The cheaper
    // PISA fabs is safe whenever the SNaN-vs-QNaN distinction cannot affect
    // any observable result at this call site. Two sufficient conditions:
    //
    //   * cannotBeSNaN(&I) - the fabs result is provably never an SNaN.
    //     For a fabs call this folds in the nnan flag on the call itself
    //     (FPMathOperator::hasNoNaNs) and recurses into the operand's
    //     def chain (IEEE arithmetic, sitofp, nnan producers, ...).
    //   * allUsesAreSNaNInsensitive(&I) - no consumer of the fabs result
    //     observes the SNaN payload (uses bottom out in maxnum/minnum,
    //     fcmp, or lane-shuffle/sign-only chains into the same).
    if (SNaNAnalysis.cannotBeSNaN(&I) ||
        SNaNAnalysis.allUsesAreSNaNInsensitive(&I)) {
      replaceIntrinsicWith(&I, Intrinsic::pisa_fabs, 0);
      Changed = true;
    }
  } break;
  default:
    break;
  }
}

bool PISAEmitIntrinsics::runOnFunction(Function &Func) {
  SNaNAnalysis = SNaNPayloadAnalysis();
  IRBuilder<> LocalIRB(Func.getContext());

  IRB = &LocalIRB;
  Changed = false;

  visit(Func);

  return Changed;
}

FunctionPass *llvm::createPISAEmitIntrinsicsPass() {
  return new PISAEmitIntrinsics();
}
