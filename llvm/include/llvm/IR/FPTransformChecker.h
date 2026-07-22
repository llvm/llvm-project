//===-- llvm/FPTransformChecker.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file defines a class that helps checking conditions for floating-point
/// related IR transformations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_FPTRANSFORMCHECKER_H
#define LLVM_IR_FPTRANSFORMCHECKER_H

#include "llvm/IR/FMF.h"
#include "llvm/IR/FPEnv.h"
#include "llvm/IR/Operator.h"

namespace llvm {

struct SimplifyQuery;

/// The class keeps properties that can affect applicability of a floating-point
/// related transformation to the selected instruction(s).
class FPTransformChecker {
  static constexpr unsigned FastMathMask = FastMathFlags::AllFlagsMask;
  static constexpr unsigned FastMathBits = 8;
  static_assert((1U << FastMathBits) > FastMathMask, "Too few fast math bits");

  union {
    unsigned Flags;
    struct Fields {
      unsigned FastMath : FastMathBits;
      unsigned StrictFP : 1;
      unsigned Rounding : 3;
      unsigned Exceptions : 2;
    } F;
  } U;

public:
  static constexpr unsigned AllowReassoc = FastMathFlags::AllowReassoc;
  static constexpr unsigned NoNaNs = FastMathFlags::NoNaNs;
  static constexpr unsigned NoInfs = FastMathFlags::NoInfs;
  static constexpr unsigned NoSignedZeros = FastMathFlags::NoSignedZeros;
  static constexpr unsigned AllowReciprocal = FastMathFlags::AllowReciprocal;
  static constexpr unsigned AllowContract = FastMathFlags::AllowContract;
  static constexpr unsigned ApproxFunc = FastMathFlags::ApproxFunc;

  bool allowReassoc() const { return 0 != (U.F.FastMath & AllowReassoc); }
  bool noNaNs() const { return 0 != (U.F.FastMath & NoNaNs); }
  bool noInfs() const { return 0 != (U.F.FastMath & NoInfs); }
  bool noSignedZeros() const { return 0 != (U.F.FastMath & NoSignedZeros); }
  bool allowReciprocal() const { return 0 != (U.F.FastMath & AllowReciprocal); }
  bool allowContract() const { return 0 != (U.F.FastMath & AllowContract); }
  bool approxFunc() const { return 0 != (U.F.FastMath & ApproxFunc); }

  FastMathFlags getFMF() const { return FastMathFlags(U.F.FastMath); };

  /// Return true if the enclosing function has attribute StrictFP.
  bool isStrictFP() const { return U.F.StrictFP; }

  RoundingMode getRoundingMode() const {
    return static_cast<RoundingMode>(U.F.Rounding);
  }

  fp::ExceptionBehavior getExceptionBehavior() const {
    return static_cast<fp::ExceptionBehavior>(U.F.Exceptions);
  }

  FPTransformChecker() { U.Flags = 0; }
  FPTransformChecker(const Function *F) {
    U.Flags = 0;
    init(F);
  }
  LLVM_ABI FPTransformChecker(const Instruction *I);
  LLVM_ABI FPTransformChecker(const SimplifyQuery &Q);
  FPTransformChecker(FastMathFlags FMF) {
    U.Flags = 0;
    U.F.FastMath = FMF.getAsOpaqueInt();
  }

  FPTransformChecker withFastMath(FastMathFlags FMF) const {
    unsigned NewFlags =
        (U.Flags & ~FastMathMask) | (FMF.getAsOpaqueInt() & FastMathMask);
    return FPTransformChecker(NewFlags);
  }

  /// Check if the FP properties allow combining the function call with other
  /// call, like `exp(log(x))`.
  bool mayCombineCalls() const { return allowReassoc() && !isStrictFP(); }

  /// Returns true if the possibility of a signaling NaN can be safely
  /// ignored.
  bool canIgnoreSNaN() const {
    return (U.F.Exceptions == fp::ebIgnore || noNaNs());
  }

  /// Returns true if the rounding mode can be QRM at compile time or at
  /// run time.
  inline bool canRoundingModeBe(RoundingMode QRM) const {
    auto RM = getRoundingMode();
    return RM == QRM || RM == RoundingMode::Dynamic;
  }

  /// Returns true if the exception handling behavior and rounding mode
  /// match what is used in the default floating point environment.
  inline bool isDefaultFPEnvironment() {
    return U.F.Exceptions == fp::ebIgnore &&
           getRoundingMode() == RoundingMode::NearestTiesToEven;
  }

  /// Returns true, if the expression (x - x) can be a negative zero.
  bool canDiffWithItselfBeNegative() const {
    return canRoundingModeBe(RoundingMode::TowardNegative) && !noSignedZeros();
  }

private:
  FPTransformChecker(unsigned Flags) { U.Flags = 0; }
  LLVM_ABI void init(const Function *F);
  void with(const Instruction *I);
};

} // namespace llvm

#endif
