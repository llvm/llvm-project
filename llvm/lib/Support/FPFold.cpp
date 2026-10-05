//===- FPFold.cpp - Floating-point constant-folding helpers --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FPFold.h"
#include "DebugOptions.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include <cassert>

using namespace llvm;

std::optional<APFloat> llvm::getFPValue(std::optional<FPFoldResult> Res) {
  return Res ? std::optional<APFloat>(std::move(Res->Value)) : std::nullopt;
}

namespace {
struct CreateAvoidFoldingToNaN {
  static void *call() {
    return new cl::opt<bool>(
        "avoid-folding-to-nan", cl::Hidden,
        cl::desc("Avoid constant-folding floating-point operations to NaN"),
        cl::init(false));
  }
};
} // namespace

static ManagedStatic<cl::opt<bool>, CreateAvoidFoldingToNaN> AvoidFoldingToNaN;

void llvm::initFPFoldOptions() { *AvoidFoldingToNaN; }

bool llvm::shouldAvoidFoldingToNaN() { return *AvoidFoldingToNaN; }

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
  case RoundingMode::NearestTiesToEven:
  case RoundingMode::NearestTiesToAway:
  case RoundingMode::Dynamic:
  case RoundingMode::Invalid:
    return true;
  case RoundingMode::TowardPositive:
    return !Result.isNegative();
  case RoundingMode::TowardNegative:
    return Result.isNegative();
  case RoundingMode::TowardZero:
    return false;
  }
  llvm_unreachable("unknown rounding mode");
}

static bool containsSNaN(ArrayRef<APFloat> Args) {
  for (const APFloat &Arg : Args)
    if (Arg.isSignaling())
      return true;
  return false;
}

/// The internal helper that dispatches to APFloat, and always produces
/// a status and result (so RM must be valid and non-dynamic if used, and
/// input args must already be flushed if necessary).
static APFloat::opStatus executeFold(FPOp Opcode, APFloat &Result,
                                     ArrayRef<APFloat> Args, RoundingMode RM) {
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
      return containsSNaN(Args) ? APFloat::opInvalidOp : APFloat::opOK;
    case FPOp::MaxNum:
      Result = maxnum(Result, Args[1]);
      return containsSNaN(Args) ? APFloat::opInvalidOp : APFloat::opOK;
    case FPOp::Minimum:
      Result = minimum(Result, Args[1]);
      return containsSNaN(Args) ? APFloat::opInvalidOp : APFloat::opOK;
    case FPOp::Maximum:
      Result = maximum(Result, Args[1]);
      return containsSNaN(Args) ? APFloat::opInvalidOp : APFloat::opOK;
    case FPOp::MinimumNum:
      Result = minimumnum(Result, Args[1]);
      return containsSNaN(Args) ? APFloat::opInvalidOp : APFloat::opOK;
    case FPOp::MaximumNum:
      Result = maximumnum(Result, Args[1]);
      return containsSNaN(Args) ? APFloat::opInvalidOp : APFloat::opOK;
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
/// Dynamic/Invalid denormal modes are handled - they reject the fold if
/// subnormal inputs are present, or if the output is subnormal and needs
/// flushed. Also optionally reject folds with NaN results (useful on
/// targets where specific NaN payloads matter).
static std::optional<FPFoldResult>
tryFoldFPOpImpl(FPOp Opcode, ArrayRef<APFloat> Args, RoundingMode RM,
                DenormalMode Denorms, bool AvoidFoldingToNaN) {
  assert(!Args.empty() && Args.size() <= 3);

  SmallVector<APFloat, 3> FlushedArgs;
  bool DenormInputModeUnknown = Denorms.Input == DenormalMode::Dynamic ||
                                Denorms.Input == DenormalMode::Invalid;
  for (unsigned I = 0; I < Args.size(); ++I) {
    if (DenormInputModeUnknown && Args[I].isDenormal())
      return std::nullopt;
    FlushedArgs.push_back(flush(Args[I], Denorms.Input));
  }

  APFloat Result = FlushedArgs[0];
  APFloat::opStatus Status = executeFold(Opcode, Result, FlushedArgs, RM);

  if (AvoidFoldingToNaN && Result.isNaN())
    return std::nullopt;
  if ((Denorms.Output == DenormalMode::Dynamic ||
       Denorms.Output == DenormalMode::Invalid) &&
      Result.isDenormal())
    return std::nullopt;
  if (Denorms.Output != DenormalMode::IEEE &&
      mayHaveBeenRoundedFromSubnormal(Result, Status, RM))
    return std::nullopt;
  return FPFoldResult{flush(std::move(Result), Denorms.Output), Status};
}

// Fold FPOps that do not require a rounding mode (or use the default RN)
std::optional<FPFoldResult> llvm::tryFoldFP(FPOp Opcode, ArrayRef<APFloat> Args,
                                            DenormalMode Denorms,
                                            bool AvoidFoldingToNaN) {
  return tryFoldFPOpImpl(Opcode, Args, RoundingMode::NearestTiesToEven, Denorms,
                         AvoidFoldingToNaN);
}

// Fold FPOps that do require a rounding mode. Handle the possibility
// that the RM is dynamic/invalid (fold with RN and allow the fold
// only if no rounding is required).
std::optional<FPFoldResult>
llvm::tryFoldFPWithRM(FPOp Opcode, ArrayRef<APFloat> Args, RoundingMode RM,
                      DenormalMode Denorms, bool AvoidFoldingToNaN) {
  bool RMUnknown = RM == RoundingMode::Invalid || RM == RoundingMode::Dynamic;
  RoundingMode EvalRM = RMUnknown ? RoundingMode::NearestTiesToEven : RM;
  auto Res = tryFoldFPOpImpl(Opcode, Args, EvalRM, Denorms, AvoidFoldingToNaN);
  if (RMUnknown && Res && (Res->Status & APFloat::opInexact))
    return std::nullopt;
  return Res;
}

std::optional<APFloat::cmpResult> llvm::tryFoldFCmp(const APFloat &LHS,
                                                    const APFloat &RHS,
                                                    DenormalMode Denorms) {
  if ((Denorms.Input == DenormalMode::Dynamic ||
       Denorms.Input == DenormalMode::Invalid) &&
      (LHS.isDenormal() || RHS.isDenormal()))
    return std::nullopt;

  APFloat LHSTmp = flush(LHS, Denorms.Input);
  APFloat RHSTmp = flush(RHS, Denorms.Input);
  return LHSTmp.compare(RHSTmp);
}
