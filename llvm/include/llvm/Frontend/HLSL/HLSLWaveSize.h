//===- HLSLResource.h - HLSL Resource helper objects ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects for working with HLSL WaveSize.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_HLSLWAVESIZE_H
#define LLVM_FRONTEND_HLSL_HLSLWAVESIZE_H

namespace llvm {
namespace hlsl {

// SM 6.6 allows WaveSize specification for only a single required size.
// SM 6.8+ allows specification of WaveSize as a min, max and preferred value.
struct WaveSize {
  unsigned Min = 0;
  unsigned Max = 0;
  unsigned Preferred = 0;

  WaveSize() = default;
  WaveSize(unsigned Min, unsigned Max = 0, unsigned Preferred = 0)
      : Min(Min), Max(Max), Preferred(Preferred) {}
  WaveSize(const WaveSize &) = default;
  WaveSize &operator=(const WaveSize &) = default;
  bool operator==(const WaveSize &Other) const {
    return Min == Other.Min && Max == Other.Max && Preferred == Other.Preferred;
  };

  // Valid non-zero values are powers of 2 between 4 and 128, inclusive.
  static bool isValidValue(unsigned Value) {
    return (Value >= 4 && Value <= 128 && ((Value & (Value - 1)) == 0));
  }
  // Valid representations:
  //    (not to be confused with encodings in metadata, PSV0, or RDAT)
  //  0, 0, 0: Not defined
  //  Min, 0, 0: single WaveSize (SM 6.6/6.7)
  //    (single WaveSize is represented in metadata with the single Min value)
  //  Min, Max (> Min), 0 or Preferred (>= Min and <= Max): Range (SM 6.8+)
  //    (WaveSizeRange represenation in metadata is the same)
  enum class ValidationResult {
    Success,
    InvalidMin,
    InvalidMax,
    InvalidPreferred,
    MaxOrPreferredWhenUndefined,
    PreferredWhenNoRange,
    MaxEqualsMin,
    MaxLessThanMin,
    PreferredOutOfRange,
    NoRangeOrMin,
  };
  ValidationResult validate() const {
    if (Min == 0) { // Not defined
      if (Max != 0 || Preferred != 0)
        return ValidationResult::MaxOrPreferredWhenUndefined;
      else
        // all 3 parameters are 0
        return ValidationResult::NoRangeOrMin;
    } else if (!isValidValue(Min)) {
      return ValidationResult::InvalidMin;
    } else if (Max == 0) { // single WaveSize (SM 6.6/6.7)
      if (Preferred != 0)
        return ValidationResult::PreferredWhenNoRange;
    } else if (!isValidValue(Max)) {
      return ValidationResult::InvalidMax;
    } else if (Min == Max) {
      return ValidationResult::MaxEqualsMin;
    } else if (Max < Min) {
      return ValidationResult::MaxLessThanMin;
    } else if (Preferred != 0) {
      if (!isValidValue(Preferred))
        return ValidationResult::InvalidPreferred;
      if (Preferred < Min || Preferred > Max)
        return ValidationResult::PreferredOutOfRange;
    }
    return ValidationResult::Success;
  }
  bool isValid() const { return validate() == ValidationResult::Success; }

  bool isDefined() const { return Min != 0; }
  bool isRange() const { return Max != 0; }
  bool hasPreferred() const { return Preferred != 0; }
};

} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLWAVESIZE_H
