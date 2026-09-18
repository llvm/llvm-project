//===- HLSLInterpolationTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/SemanticSignatures.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::hlsl;
using Mod = InterpolationModifier;
using InterpMode = dxbc::PSV::InterpolationMode;

TEST(HLSLInterpolationTest, AllModifierSets) {
  // Cover every subset, including redundant linear/center modifiers and all
  // combinations of competing sampling locations.
  for (unsigned Mask = 0; Mask != 64; ++Mask) {
    SCOPED_TRACE(Mask);
    auto Modifiers = static_cast<Mod>(Mask);
    InterpMode Expected;
    if (Modifiers == Mod::None)
      Expected = InterpMode::Undefined;
    else if (Modifiers == Mod::NoInterpolation)
      Expected = InterpMode::Constant;
    else if (any(Modifiers & Mod::NoInterpolation))
      Expected = InterpMode::Invalid;
    else {
      static constexpr InterpMode Modes[2][3] = {
          {InterpMode::Linear, InterpMode::LinearCentroid,
           InterpMode::LinearSample},
          {InterpMode::LinearNoperspective,
           InterpMode::LinearNoperspectiveCentroid,
           InterpMode::LinearNoperspectiveSample}};
      unsigned Location = any(Modifiers & Mod::Sample)     ? 2
                          : any(Modifiers & Mod::Centroid) ? 1
                                                           : 0;
      Expected = Modes[any(Modifiers & Mod::NoPerspective)][Location];
    }
    EXPECT_EQ(getInterpolationMode(Modifiers), Expected);
  }
}

TEST(HLSLInterpolationTest, SamplingLocations) {
  const struct {
    Mod Modifiers;
    Mod Expected;
  } Cases[] = {
      {Mod::None, Mod::None},
      {Mod::Center, Mod::Center},
      {Mod::Centroid, Mod::Centroid},
      {Mod::Center | Mod::Centroid, Mod::Centroid},
      {Mod::Sample, Mod::Sample},
      {Mod::Center | Mod::Sample, Mod::Sample},
      {Mod::Centroid | Mod::Sample, Mod::Sample},
      {Mod::Center | Mod::Centroid | Mod::Sample, Mod::Sample},
  };
  // Non-location keywords must not imply an explicit center modifier or change
  // sampling-location precedence, even in invalid interpolation combinations.
  for (const auto &Case : Cases)
    for (Mod Other :
         {Mod::None, Mod::NoInterpolation, Mod::Linear, Mod::NoPerspective,
          Mod::Linear | Mod::NoPerspective, Mod::NoInterpolation | Mod::Linear,
          Mod::NoInterpolation | Mod::NoPerspective,
          Mod::NoInterpolation | Mod::Linear | Mod::NoPerspective}) {
      Mod Modifiers = Case.Modifiers | Other;
      SCOPED_TRACE(static_cast<unsigned>(Modifiers));
      EXPECT_EQ(getInterpolationSamplingLocation(Modifiers), Case.Expected);
    }
}
