//===- HLSLSemanticStagesTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/SemanticSignatures.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <ostream>

using namespace llvm;
using namespace llvm::hlsl;

namespace llvm::hlsl {

static StringRef getInterpretationName(SemanticInterpretation Interpretation) {
  switch (Interpretation) {
  case SemanticInterpretation::Invalid:
    return "Invalid";
  case SemanticInterpretation::NotAllocated:
    return "NotAllocated";
  case SemanticInterpretation::NotPacked:
    return "NotPacked";
  case SemanticInterpretation::Arbitrary:
    return "Arbitrary";
  case SemanticInterpretation::SV:
    return "SV";
  case SemanticInterpretation::SGV:
    return "SGV";
  case SemanticInterpretation::ClipCull:
    return "ClipCull";
  case SemanticInterpretation::TessFactor:
    return "TessFactor";
  case SemanticInterpretation::Target:
    return "Target";
  }
  llvm_unreachable("unhandled interpretation");
}

// Print the enum by name to keep the test failures readable.
static void PrintTo(SemanticInterpretation Interpretation, std::ostream *OS) {
  *OS << getInterpretationName(Interpretation).str();
}

} // namespace llvm::hlsl

namespace {

using SemanticKind = dxbc::PSV::SemanticKind;

// Every shader stage a semantic could possibly be used in, plus a stage that
// never holds a signature (Library) to make sure it is always rejected.
constexpr Triple::EnvironmentType AllStages[] = {
    Triple::Vertex,        Triple::Hull,  Triple::Domain,
    Triple::Geometry,      Triple::Pixel, Triple::Compute,
    Triple::Amplification, Triple::Mesh,  Triple::Library};

// The semantic kinds getAvailableStages knows about.
constexpr SemanticKind SupportedKinds[] = {SemanticKind::Arbitrary,
                                           SemanticKind::Position,
                                           SemanticKind::VertexID,
                                           SemanticKind::Target,
                                           SemanticKind::IsFrontFace,
                                           SemanticKind::ClipDistance,
                                           SemanticKind::CullDistance,
                                           SemanticKind::TessFactor,
                                           SemanticKind::InsideTessFactor,
                                           SemanticKind::DispatchThreadID,
                                           SemanticKind::GroupID,
                                           SemanticKind::GroupIndex,
                                           SemanticKind::GroupThreadID,
                                           SemanticKind::InstanceID,
                                           SemanticKind::ShadingRate,
                                           SemanticKind::Barycentrics,
                                           SemanticKind::StartVertexLocation,
                                           SemanticKind::StartInstanceLocation,
                                           SemanticKind::SampleIndex};

// Ensure all stages return a valid SemanticInterpretation
TEST(HLSLSemanticStagesTest, StagesAreValid) {
  for (SemanticKind Kind : SupportedKinds) {
    ArrayRef<SemanticStageInfo> Stages = getAvailableStages(Kind);
    EXPECT_FALSE(Stages.empty());
    for (const SemanticStageInfo &Info : Stages) {
      EXPECT_TRUE(any(Info.AllowedIOTypesMask))
          << "stage " << Triple::getEnvironmentTypeName(Info.Stage).str()
          << " allows no IOType";
      EXPECT_NE(Info.Interpretation, SemanticInterpretation::Invalid);
      EXPECT_THAT(AllStages, testing::Contains(Info.Stage));
    }
  }
}

// Ensure a stage is not listed twice with overlapping IOTypes
TEST(HLSLSemanticStagesTest, StageIOTypesAreDisjoint) {
  for (SemanticKind Kind : SupportedKinds) {
    ArrayRef<SemanticStageInfo> Stages = getAvailableStages(Kind);
    for (Triple::EnvironmentType Stage : AllStages) {
      IOType Seen = static_cast<IOType>(0);
      for (const SemanticStageInfo &Info : Stages) {
        if (Info.Stage != Stage)
          continue;
        EXPECT_FALSE(any(Seen & Info.AllowedIOTypesMask))
            << "stage " << Triple::getEnvironmentTypeName(Stage).str()
            << " is listed twice for the same IOType";
        Seen |= Info.AllowedIOTypesMask;
      }
    }
  }
}

// Per-semantic signature point tests.
//
// To cover a semantic, add a test named `<Semantic>SignaturePoints` that lists
// every point the semantic is valid at and passes the list to
// expectOnlySignaturePoints. List valid points only: every other
// stage/IOType combination is checked to be Invalid, so a point that becomes
// available by accident fails the test.

// A stage and IOType at which a semantic is valid, and how it is interpreted.
struct SignaturePoint {
  Triple::EnvironmentType Stage;
  IOType IO;
  SemanticInterpretation Interpretation;
};

// Checks that Kind is valid at exactly the points in Valid, with the listed
// interpretations, across every stage and IOType.
static void expectOnlySignaturePoints(SemanticKind Kind,
                                      ArrayRef<SignaturePoint> Valid) {
  constexpr IOType AllIOTypes[] = {IOType::In, IOType::Out,
                                   IOType::PatchConstantOrPrimitive};
  for (Triple::EnvironmentType Stage : AllStages) {
    for (IOType IO : AllIOTypes) {
      SemanticInterpretation Want = SemanticInterpretation::Invalid;
      for (const SignaturePoint &P : Valid)
        if (P.Stage == Stage && P.IO == IO)
          Want = P.Interpretation;
      EXPECT_EQ(getInterpretationKind(Kind, Stage, IO), Want)
          << "stage " << Triple::getEnvironmentTypeName(Stage).str()
          << ", IOType " << static_cast<unsigned>(IO);
    }
  }
}

// SV_ShadingRate is written per vertex/primitive by the geometry-producing
// stages and read by the pixel shader. Valid only on vertex, geometry, and mesh
// outputs, on hull, domain and geometry per-vertex data, and on pixel inputs.
TEST(HLSLSemanticStagesTest, ShadingRateSignaturePoints) {
  constexpr SemanticInterpretation SV = SemanticInterpretation::SV;
  const SignaturePoint Valid[] = {
      {Triple::Vertex, IOType::Out, SV},
      {Triple::Hull, IOType::In, SV},
      {Triple::Hull, IOType::Out, SV},
      {Triple::Domain, IOType::In, SV},
      {Triple::Domain, IOType::Out, SV},
      {Triple::Geometry, IOType::In, SV},
      {Triple::Geometry, IOType::Out, SV},
      {Triple::Pixel, IOType::In, SV},
      {Triple::Mesh, IOType::PatchConstantOrPrimitive, SV},
  };
  expectOnlySignaturePoints(SemanticKind::ShadingRate, Valid);
}

// SV_Barycentrics is generated by the rasterizer and read by the pixel shader.
// Valid only on pixel inputs, where it is in the signature but not packed.
TEST(HLSLSemanticStagesTest, BarycentricsSignaturePoints) {
  const SignaturePoint Valid[] = {
      {Triple::Pixel, IOType::In, SemanticInterpretation::NotPacked},
  };
  expectOnlySignaturePoints(SemanticKind::Barycentrics, Valid);
}

// SV_StartVertexLocation is read with a dedicated operation and is never placed
// in a signature. Valid only on vertex inputs.
TEST(HLSLSemanticStagesTest, StartVertexLocationSignaturePoints) {
  const SignaturePoint Valid[] = {
      {Triple::Vertex, IOType::In, SemanticInterpretation::NotAllocated},
  };
  expectOnlySignaturePoints(SemanticKind::StartVertexLocation, Valid);
}

// SV_StartInstanceLocation is read with a dedicated operation and is never
// placed in a signature. Valid only on vertex inputs.
TEST(HLSLSemanticStagesTest, StartInstanceLocationSignaturePoints) {
  const SignaturePoint Valid[] = {
      {Triple::Vertex, IOType::In, SemanticInterpretation::NotAllocated},
  };
  expectOnlySignaturePoints(SemanticKind::StartInstanceLocation, Valid);
}

// SV_SampleIndex is read with a dedicated operation, but is still added to the
// pixel input signature for compatibility. Valid only on pixel inputs, where it
// is in the signature but not packed.
TEST(HLSLSemanticStagesTest, SampleIndexSignaturePoints) {
  const SignaturePoint Valid[] = {
      {Triple::Pixel, IOType::In, SemanticInterpretation::NotPacked},
  };
  expectOnlySignaturePoints(SemanticKind::SampleIndex, Valid);
}

} // namespace
