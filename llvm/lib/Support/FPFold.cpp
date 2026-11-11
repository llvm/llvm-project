//===- FPFold.cpp - Floating-point constant-folding helpers --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FPFold.h"
#include "DebugOptions.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include <cassert>
#include <utility>

using namespace llvm;

namespace {
enum class DenormalFoldingMode {
  TargetDefault,
  DisableFlushing,
  MatchDenormalFPEnv,
  DisableFolding
};

struct CreateDenormalFoldingMode {
  static void *call() {
    return new cl::opt<DenormalFoldingMode>(
        "const-folding-denormal-mode", cl::Hidden,
        cl::desc("Denormal handling during floating-point constant folding"),
        cl::init(DenormalFoldingMode::TargetDefault),
        cl::values(
            clEnumValN(DenormalFoldingMode::TargetDefault, "target-default",
                       "Use the target's preferred folding mode (default)"),
            clEnumValN(DenormalFoldingMode::DisableFlushing, "disable-flushing",
                       "Fold using IEEE denormal handling"),
            clEnumValN(DenormalFoldingMode::MatchDenormalFPEnv,
                       "match-denormal-fpenv",
                       "Fold using the supplied denormal mode"),
            clEnumValN(DenormalFoldingMode::DisableFolding, "disable-folding",
                       "Decline folds that depend on denormal handling")));
  }
};
} // namespace

static ManagedStatic<cl::opt<DenormalFoldingMode>, CreateDenormalFoldingMode>
    ConstFoldingDenormalMode;

void llvm::initFPFoldOptions() { *ConstFoldingDenormalMode; }

static DenormalMode getFoldingDenormalMode(FPOp Opcode,
                                           const fltSemantics &Semantics,
                                           DenormalMode Denorms,
                                           TargetFPInfo TFPI,
                                           bool OverrideDefaultFlushingMode) {
  if (OverrideDefaultFlushingMode)
    return Denorms;
  switch (*ConstFoldingDenormalMode) {
  case DenormalFoldingMode::TargetDefault:
    return TFPI.getTargetConstFoldDenormMode(Opcode, Semantics, Denorms);
  case DenormalFoldingMode::DisableFlushing:
    return DenormalMode::getIEEE();
  case DenormalFoldingMode::MatchDenormalFPEnv:
    return Denorms;
  case DenormalFoldingMode::DisableFolding:
    return DenormalMode::getDynamic();
  }
  llvm_unreachable("unknown denormal folding mode");
}

std::optional<APFloat> llvm::getFPValue(std::optional<FPFoldResult> Res) {
  return llvm::transformOptional(std::move(Res), &FPFoldResult::Value);
}

static APFloat flush(APFloat Value, DenormalMode::DenormalModeKind Mode) {
  if (!Value.isDenormal())
    return Value;

  if (Mode == DenormalMode::PreserveSign)
    return APFloat::getZero(Value.getSemantics(), Value.isNegative());
  if (Mode == DenormalMode::PositiveZero)
    return APFloat::getZero(Value.getSemantics());
  return Value;
}

/// Return true if gradual-underflow rounding may have promoted a result from
/// the subnormal range to the smallest normal value. When output denormals are
/// flushed, folding such a result as an ordinary IEEE operation is unsafe: an
/// FTZ operation may produce zero before gradual-underflow rounding can perform
/// that promotion.
static bool mayHaveBeenRoundedFromSubnormal(const APFloat &Result,
                                            APFloat::opStatus Status,
                                            RoundingMode RM) {
  if (!(Status & APFloat::opInexact) || !Result.isSmallestNormalized())
    return false;

  switch (RM) {
  default:
    llvm_unreachable("unknown rounding mode");
  case RoundingMode::NearestTiesToEven:
  case RoundingMode::NearestTiesToAway:
  case RoundingMode::Dynamic:
    return true;
  case RoundingMode::TowardPositive:
    return !Result.isNegative();
  case RoundingMode::TowardNegative:
    return Result.isNegative();
  case RoundingMode::TowardZero:
    return false;
  }
}

static APFloat::opStatus getSNaNStatus(ArrayRef<APFloat> Args) {
  for (const APFloat &Arg : Args)
    if (Arg.isSignaling())
      return APFloat::opInvalidOp;
  return APFloat::opOK;
}

/// The internal helper that dispatches to APFloat, and always produces
/// a status and result (so RM must be valid and non-dynamic if used, and
/// input args must already be flushed if necessary).
/// Result must be initialized with a copy of Args[0].
static APFloat::opStatus executeFold(FPOp Opcode, APFloat &Result,
                                     ArrayRef<APFloat> Args, RoundingMode RM) {
  assert(!Args.empty() && Result.bitwiseIsEqual(Args[0]) &&
         "Result must start as a copy of Args[0]");
  if (Args.size() == 2) {
    switch (Opcode) {
    default:
      llvm_unreachable("unsupported 2-arg FP folding op");
    case FPOp::Add:
      return Result.add(Args[1], RM);
    case FPOp::Sub:
      return Result.subtract(Args[1], RM);
    case FPOp::Mul:
      return Result.multiply(Args[1], RM);
    case FPOp::Div:
      return Result.divide(Args[1], RM);
    case FPOp::FRem:
      return Result.mod(Args[1]);
    case FPOp::MinNum:
      Result = minnum(Result, Args[1]);
      return getSNaNStatus(Args);
    case FPOp::MaxNum:
      Result = maxnum(Result, Args[1]);
      return getSNaNStatus(Args);
    case FPOp::Minimum:
      Result = minimum(Result, Args[1]);
      return getSNaNStatus(Args);
    case FPOp::Maximum:
      Result = maximum(Result, Args[1]);
      return getSNaNStatus(Args);
    case FPOp::MinimumNum:
      Result = minimumnum(Result, Args[1]);
      return getSNaNStatus(Args);
    case FPOp::MaximumNum:
      Result = maximumnum(Result, Args[1]);
      return getSNaNStatus(Args);
    }
  } else if (Args.size() == 1) {
    switch (Opcode) {
    default:
      llvm_unreachable("unsupported single-arg FP folding op");
    case FPOp::Ceil:
      return Result.roundToIntegral(RoundingMode::TowardPositive);
    case FPOp::Floor:
      return Result.roundToIntegral(RoundingMode::TowardNegative);
    case FPOp::Trunc:
      return Result.roundToIntegral(RoundingMode::TowardZero);
    case FPOp::Round:
      return Result.roundToIntegral(RoundingMode::NearestTiesToAway);
    case FPOp::RoundEven:
      return Result.roundToIntegral(RoundingMode::NearestTiesToEven);
    }
  } else if (Args.size() == 3 && Opcode == FPOp::FMA) {
    return Result.fusedMultiplyAdd(Args[1], Args[2], RM);
  }
  llvm_unreachable("unsupported FP folding op for given number of inputs");
}

/// Internal helper to fold or reject if we cannot give correct results.
/// If RM is necessary for the operation, it must be valid and non-dynamic.
/// Dynamic denormal modes reject the fold if subnormal inputs are present or
/// if the output is subnormal and may need flushing.
static std::optional<FPFoldResult> tryFoldFPOpImpl(FPOp Opcode,
                                                   ArrayRef<APFloat> Args,
                                                   RoundingMode RM,
                                                   DenormalMode Denorms) {
  assert(!Args.empty() && Args.size() <= 3);
  assert(Denorms.isValid() && "invalid denormal mode");

  SmallVector<APFloat, 3> FlushedArgs;
  for (unsigned I = 0; I < Args.size(); ++I) {
    assert(&Args[I].getSemantics() == &Args[0].getSemantics());
    if (Denorms.Input == DenormalMode::Dynamic && Args[I].isDenormal())
      return std::nullopt;
    FlushedArgs.push_back(flush(Args[I], Denorms.Input));
  }

  APFloat Result = FlushedArgs[0];
  APFloat::opStatus Status = executeFold(Opcode, Result, FlushedArgs, RM);

  if (Denorms.Output == DenormalMode::Dynamic && Result.isDenormal())
    return std::nullopt;
  if (Denorms.Output != DenormalMode::IEEE &&
      mayHaveBeenRoundedFromSubnormal(Result, Status, RM))
    return std::nullopt;
  return FPFoldResult{flush(std::move(Result), Denorms.Output), Status};
}

// Fold FPOps that do not require a rounding mode (or use the default RN)
std::optional<FPFoldResult> llvm::tryFoldFP(FPOp Opcode, ArrayRef<APFloat> Args,
                                            DenormalMode Denorms,
                                            TargetFPInfo TFPI,
                                            bool OverrideDefaultFlushingMode) {
  assert(!Args.empty());
  Denorms = getFoldingDenormalMode(Opcode, Args[0].getSemantics(), Denorms,
                                   TFPI, OverrideDefaultFlushingMode);
  return tryFoldFPOpImpl(Opcode, Args, RoundingMode::NearestTiesToEven,
                         Denorms);
}

// Fold FPOps that do require a rounding mode. Handle the possibility
// that the RM is dynamic (fold with RN and allow the fold
// only if no rounding is required).
std::optional<FPFoldResult>
llvm::tryFoldFPWithRM(FPOp Opcode, ArrayRef<APFloat> Args, RoundingMode RM,
                      DenormalMode Denorms, TargetFPInfo TFPI,
                      bool OverrideDefaultFlushingMode) {
  assert(RM != RoundingMode::Invalid && "invalid rounding mode");
  assert(!Args.empty());
  Denorms = getFoldingDenormalMode(Opcode, Args[0].getSemantics(), Denorms,
                                   TFPI, OverrideDefaultFlushingMode);
  RoundingMode EvalRM =
      RM == RoundingMode::Dynamic ? RoundingMode::NearestTiesToEven : RM;
  auto Res = tryFoldFPOpImpl(Opcode, Args, EvalRM, Denorms);
  if (RM == RoundingMode::Dynamic && Res && (Res->Status & APFloat::opInexact))
    return std::nullopt;
  return Res;
}

std::optional<APFloat::cmpResult>
llvm::tryFoldFCmp(const APFloat &LHS, const APFloat &RHS, DenormalMode Denorms,
                  TargetFPInfo TFPI, bool OverrideDefaultFlushingMode) {
  Denorms = getFoldingDenormalMode(FPOp::FCmp, LHS.getSemantics(), Denorms,
                                   TFPI, OverrideDefaultFlushingMode);
  assert(Denorms.isValid() && "invalid denormal mode");
  if (Denorms.Input == DenormalMode::Dynamic &&
      (LHS.isDenormal() || RHS.isDenormal()))
    return std::nullopt;

  APFloat LHSTmp = flush(LHS, Denorms.Input);
  APFloat RHSTmp = flush(RHS, Denorms.Input);
  return LHSTmp.compare(RHSTmp);
}
