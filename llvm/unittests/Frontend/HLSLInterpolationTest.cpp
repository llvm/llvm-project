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
using SemanticKind = dxbc::PSV::SemanticKind;
using CompType = dxil::ElementType;

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
  for (const auto &[CaseModifiers, Expected] : Cases)
    for (Mod Other :
         {Mod::None, Mod::NoInterpolation, Mod::Linear, Mod::NoPerspective,
          Mod::Linear | Mod::NoPerspective, Mod::NoInterpolation | Mod::Linear,
          Mod::NoInterpolation | Mod::NoPerspective,
          Mod::NoInterpolation | Mod::Linear | Mod::NoPerspective}) {
      Mod Modifiers = CaseModifiers | Other;
      SCOPED_TRACE(static_cast<unsigned>(Modifiers));
      EXPECT_EQ(getInterpolationSamplingLocation(Modifiers), Expected);
    }
}

TEST(HLSLInterpolationTest, ComponentDefaults) {
  for (auto Stage : {Triple::Pixel, Triple::Vertex}) {
    SCOPED_TRACE(Triple::getEnvironmentTypeName(Stage).str());
    IOType IO = Stage == Triple::Pixel ? IOType::In : IOType::Out;
    for (CompType Type :
         {CompType::F16, CompType::F32, CompType::SNormF16, CompType::UNormF16,
          CompType::SNormF32, CompType::UNormF32})
      EXPECT_EQ(normalizeInterpolationMode(InterpMode::Undefined, Type,
                                           SemanticKind::Arbitrary, Stage, IO),
                InterpMode::Linear);
    for (CompType Type : {CompType::I1, CompType::I16, CompType::U16,
                          CompType::I32, CompType::U32})
      EXPECT_EQ(normalizeInterpolationMode(InterpMode::Undefined, Type,
                                           SemanticKind::Arbitrary, Stage, IO),
                InterpMode::Constant);
  }
}

TEST(HLSLInterpolationTest, PositionAndExplicitModes) {
  constexpr InterpMode PositionModes[] = {
      InterpMode::LinearNoperspective,
      InterpMode::Constant,
      InterpMode::LinearNoperspective,
      InterpMode::LinearNoperspectiveCentroid,
      InterpMode::LinearNoperspective,
      InterpMode::LinearNoperspectiveCentroid,
      InterpMode::LinearNoperspectiveSample,
      InterpMode::LinearNoperspectiveSample,
      InterpMode::Invalid};
  for (auto Stage : {Triple::Pixel, Triple::Vertex}) {
    SCOPED_TRACE(Triple::getEnvironmentTypeName(Stage).str());
    IOType IO = Stage == Triple::Pixel ? IOType::In : IOType::Out;
    for (unsigned I = 0; I != 9; ++I) {
      SCOPED_TRACE(I);
      auto Mode = static_cast<InterpMode>(I);
      EXPECT_EQ(normalizeInterpolationMode(Mode, CompType::F32,
                                           SemanticKind::Position, Stage, IO),
                PositionModes[I]);
      if (Mode != InterpMode::Undefined)
        EXPECT_EQ(normalizeInterpolationMode(
                      Mode, CompType::F32, SemanticKind::Arbitrary, Stage, IO),
                  Mode);
    }
  }
}

TEST(HLSLInterpolationTest, NonInterpolatedSignatures) {
  for (auto Stage : {Triple::Pixel, Triple::Vertex, Triple::Geometry,
                     Triple::Hull, Triple::Domain, Triple::Mesh,
                     Triple::Compute, Triple::Amplification, Triple::Library}) {
    SCOPED_TRACE(Triple::getEnvironmentTypeName(Stage).str());
    for (auto IO : {IOType::In, IOType::Out}) {
      SCOPED_TRACE(static_cast<unsigned>(IO));
      if ((Stage == Triple::Pixel && IO == IOType::In) ||
          (Stage == Triple::Vertex && IO == IOType::Out))
        continue;
      for (unsigned I = 0; I != 9; ++I)
        for (auto Kind : {SemanticKind::Arbitrary, SemanticKind::Position})
          EXPECT_EQ(normalizeInterpolationMode(static_cast<InterpMode>(I),
                                               CompType::F32, Kind, Stage, IO),
                    InterpMode::Undefined);
    }
  }
}
