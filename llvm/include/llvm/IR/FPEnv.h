//===- FPEnv.h ---- FP Environment ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// @file
/// This file contains the declarations of entities that describe floating
/// point environment and related functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_FPENV_H
#define LLVM_IR_FPENV_H

#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/IR/FMF.h"
#include "llvm/Support/Compiler.h"
#include <optional>

namespace llvm {
class StringRef;

namespace Intrinsic {
typedef unsigned ID;
}

class Instruction;

namespace fp {

/// Exception behavior used for floating point operations.
///
/// Each of these values correspond to some metadata argument value of a
/// constrained floating point intrinsic. See the LLVM Language Reference Manual
/// for details.
enum ExceptionBehavior : uint8_t {
  ebIgnore,  ///< This corresponds to "fpexcept.ignore".
  ebMayTrap, ///< This corresponds to "fpexcept.maytrap".
  ebStrict   ///< This corresponds to "fpexcept.strict".
};

}

/// Keeps information about rounding mode used in a floating-point operation.
///
/// In addition to rounding mode used for the execution (the effective mode),
/// this class stores the method how the mode is specified. It may be a "static"
/// rounding node, where the rounding is encoded directly in the operation, or
/// an "assumed" mode, which assumes that the relevant mode is loaded into the
/// FP control register.
class RoundingSpec {
  RoundingMode Mode;
  bool IsAssumed;

public:
  RoundingSpec(RoundingMode RM, bool D) : Mode(RM), IsAssumed(D) {}

  static RoundingSpec makeStatic(RoundingMode RM) { return {RM, false}; }
  static RoundingSpec makeAssumed(RoundingMode RM) { return {RM, true}; }
  static RoundingSpec makeDynamic() { return {RoundingMode::Dynamic, true}; }

  RoundingMode getEffective() const { return Mode; }
  void setEffective(RoundingMode RM) { Mode = RM; }
  bool isAssumed() const { return IsAssumed; }
  void setAssumed(bool D) { IsAssumed = D; }
  bool isDynamic() const { return IsAssumed || Mode == RoundingMode::Dynamic; }
  bool isStatic() const { return !isDynamic(); }
  bool isDefault() const {
    return Mode == RoundingMode::NearestTiesToEven && IsAssumed;
  }
};

LLVM_ABI std::optional<RoundingSpec> readRoundingSpec(StringRef);
LLVM_ABI std::string printRoundingSpec(RoundingSpec R);

/// Returns a valid RoundingMode enumerator given a string that represents
/// rounding mode in operand bundles.
LLVM_ABI RoundingMode readRoundingMode(StringRef);

/// Returns a valid RoundingMode enumerator when given a string
/// that is valid as input in constrained intrinsic rounding mode
/// metadata.
LLVM_ABI std::optional<RoundingMode> convertStrToRoundingMode(StringRef);

/// For any RoundingMode enumerator, returns a string valid as input in
/// constrained intrinsic rounding mode metadata.
LLVM_ABI std::optional<StringRef> convertRoundingModeToStr(RoundingMode);

/// For any RoundingMode enumerator, returns a string to be used in operand
/// bundles.
LLVM_ABI std::optional<StringRef> convertRoundingModeToBundle(RoundingMode);

/// Returns a valid ExceptionBehavior enumerator when given a string
/// valid as input in constrained intrinsic exception behavior metadata.
LLVM_ABI std::optional<fp::ExceptionBehavior>
    convertStrToExceptionBehavior(StringRef);

/// Returns a valid ExceptionBehavior enumerator given a string from the operand
/// bundle argument.
LLVM_ABI std::optional<fp::ExceptionBehavior>
    convertBundleToExceptionBehavior(StringRef);

/// For any ExceptionBehavior enumerator, returns a string valid as
/// input in constrained intrinsic exception behavior metadata.
LLVM_ABI std::optional<StringRef>
    convertExceptionBehaviorToStr(fp::ExceptionBehavior);

/// Return string representing the given exception behavior for use in operand
/// bundles
LLVM_ABI std::optional<StringRef>
    convertExceptionBehaviorToBundle(fp::ExceptionBehavior);

inline raw_ostream &operator<<(raw_ostream &OS, fp::ExceptionBehavior EB) {
  OS << convertExceptionBehaviorToBundle(EB).value_or("invalid");
  return OS;
}

/// Returns true if the exception handling behavior and rounding mode
/// match what is used in the default floating point environment.
inline bool isDefaultFPEnvironment(fp::ExceptionBehavior EB, RoundingMode RM) {
  return EB == fp::ebIgnore && RM == RoundingMode::NearestTiesToEven;
}

/// Returns constrained intrinsic id to represent the given instruction in
/// strictfp function. If the instruction is already a constrained intrinsic or
/// does not have a constrained intrinsic counterpart, the function returns
/// zero.
LLVM_ABI Intrinsic::ID getConstrainedIntrinsicID(const Instruction &Instr);

/// Returns true if the rounding mode RM may be QRM at compile time or
/// at run time.
inline bool canRoundingModeBe(RoundingMode RM, RoundingMode QRM) {
  return RM == QRM || RM == RoundingMode::Dynamic;
}

/// Returns true if the possibility of a signaling NaN can be safely
/// ignored.
inline bool canIgnoreSNaN(fp::ExceptionBehavior EB, FastMathFlags FMF) {
  return (EB == fp::ebIgnore || FMF.noNaNs());
}
}
#endif
