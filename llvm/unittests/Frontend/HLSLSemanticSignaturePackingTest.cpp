//===- HLSLSemanticSignaturePackingTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/HLSL/SemanticSignaturePacking.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <initializer_list>

using namespace llvm;
using namespace llvm::hlsl;

namespace {

class HLSLSemanticSignaturePackingTest : public testing::Test {
protected:
  struct ElementConfig {
    dxbc::PSV::SemanticKind SemanticKind;
    uint32_t Rows;
    uint8_t Cols;
    dxil::ElementType CompType;
    dxbc::PSV::InterpolationMode InterpMode;
    uint32_t SemanticIndex = 0;
  };

  struct ExpectedLocation {
    uint32_t Row;
    uint8_t Col;
  };

  static constexpr ExpectedLocation Unallocated = {UnallocatedRow,
                                                   UnallocatedCol};

  struct TestConfig {
    Triple::EnvironmentType ShaderStage;
    IOType IOTy;
    SmallVector<ElementConfig> Elements;

    TestConfig(Triple::EnvironmentType ShaderStage, IOType IOTy,
               std::initializer_list<ElementConfig> Elements)
        : ShaderStage(ShaderStage), IOTy(IOTy), Elements(Elements) {}
  };

  SmallVector<SemanticSignatureElement>
  makeSignature(const TestConfig &Config) {
    SmallVector<SemanticSignatureElement> Elements;
    for (const ElementConfig &Element : Config.Elements) {
      SmallVector<uint32_t> SemanticIndices;
      for (uint32_t Row = 0; Row != Element.Rows; ++Row)
        SemanticIndices.push_back(Element.SemanticIndex + Row);

      Elements.emplace_back(
          /*SigId=*/static_cast<uint32_t>(Elements.size()),
          /*SemanticName=*/"TEST",
          /*CompType=*/Element.CompType,
          /*SemanticKind=*/Element.SemanticKind,
          /*SemanticIndices=*/SemanticIndices,
          /*Cols=*/Element.Cols);
      Elements.back().InterpMode = Element.InterpMode;
    }
    return Elements;
  }

  Error packStacked(SmallVectorImpl<SemanticSignatureElement> &Elements,
                    const TestConfig &Config) {
    return packSignatureStacked(Elements, Config.ShaderStage, Config.IOTy);
  }

  Error packIndexed(SmallVectorImpl<SemanticSignatureElement> &Elements,
                    const TestConfig &Config) {
    return packSignatureIndexed(Elements, Config.ShaderStage, Config.IOTy);
  }

  void expectPackingImpl(const TestConfig &Config, unsigned ExpectedRows,
                         std::initializer_list<ExpectedLocation> Locations,
                         bool Indexed) {
    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    ASSERT_EQ(Elements.size(), Locations.size());

    Error E =
        Indexed ? packIndexed(Elements, Config) : packStacked(Elements, Config);
    ASSERT_THAT_ERROR(std::move(E), Succeeded());

    unsigned Rows = 0;
    for (const SemanticSignatureElement &Element : Elements)
      if (Element.isAllocated())
        Rows = std::max(Rows, Element.StartRow + Element.Rows);
    EXPECT_EQ(Rows, ExpectedRows);

    unsigned Index = 0;
    for (ExpectedLocation Location : Locations) {
      EXPECT_EQ(Elements[Index].StartRow, Location.Row) << "element " << Index;
      EXPECT_EQ(Elements[Index].StartCol, Location.Col) << "element " << Index;
      ++Index;
    }
  }

  void expectPacking(const TestConfig &Config, unsigned ExpectedRows,
                     std::initializer_list<ExpectedLocation> Locations) {
    expectPackingImpl(Config, ExpectedRows, Locations, /*Indexed=*/false);
  }

  void expectIndexedPacking(const TestConfig &Config, unsigned ExpectedRows,
                            std::initializer_list<ExpectedLocation> Locations) {
    expectPackingImpl(Config, ExpectedRows, Locations, /*Indexed=*/true);
  }

  void expectPackingErrorImpl(const TestConfig &Config,
                              SignaturePackingError::ErrorKind ExpectedKind,
                              unsigned ExpectedElementIndex, bool Indexed) {
    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    Error E =
        Indexed ? packIndexed(Elements, Config) : packStacked(Elements, Config);
    if (!E) {
      ADD_FAILURE() << "expected a SignaturePackingError";
      return;
    }
    ASSERT_TRUE(E.isA<SignaturePackingError>());
    handleAllErrors(std::move(E), [&](const SignaturePackingError &PackingErr) {
      EXPECT_EQ(PackingErr.getErrorKind(), ExpectedKind);
      EXPECT_EQ(PackingErr.getElementIndex(), ExpectedElementIndex);
    });
  }

  void expectPackingError(const TestConfig &Config,
                          SignaturePackingError::ErrorKind ExpectedKind,
                          unsigned ExpectedElementIndex) {
    expectPackingErrorImpl(Config, ExpectedKind, ExpectedElementIndex,
                           /*Indexed=*/false);
  }

  void expectIndexedPackingError(const TestConfig &Config,
                                 SignaturePackingError::ErrorKind ExpectedKind,
                                 unsigned ExpectedElementIndex) {
    expectPackingErrorImpl(Config, ExpectedKind, ExpectedElementIndex,
                           /*Indexed=*/true);
  }
};

TEST_F(HLSLSemanticSignaturePackingTest, CreatesSignatureFromConfig) {
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Position, /*Rows=*/2, /*Cols=*/3,
        dxil::ElementType::F16, dxbc::PSV::InterpolationMode::Constant}});

  EXPECT_EQ(Config.ShaderStage, Triple::EnvironmentType::Vertex);
  EXPECT_EQ(Config.IOTy, IOType::Out);

  SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
  ASSERT_EQ(Elements.size(), 2u);

  EXPECT_EQ(Elements[0].SigId, 0u);
  EXPECT_EQ(Elements[0].SemanticName, "TEST");
  EXPECT_EQ(Elements[0].CompType, dxil::ElementType::F32);
  EXPECT_EQ(Elements[0].SemanticKind, dxbc::PSV::SemanticKind::Arbitrary);
  EXPECT_EQ(Elements[0].SemanticIndices, SmallVector<uint32_t>({0}));
  EXPECT_EQ(Elements[0].InterpMode, dxbc::PSV::InterpolationMode::Linear);
  EXPECT_EQ(Elements[0].Rows, 1u);
  EXPECT_EQ(Elements[0].Cols, 2u);
  EXPECT_EQ(Elements[0].StartRow, UnallocatedRow);
  EXPECT_EQ(Elements[0].StartCol, UnallocatedCol);
  EXPECT_EQ(Elements[0].UsageMask, 0u);
  EXPECT_EQ(Elements[0].DynIndexMask, 0u);
  EXPECT_EQ(Elements[0].GSStream, 0u);

  EXPECT_EQ(Elements[1].SigId, 1u);
  EXPECT_EQ(Elements[1].SemanticKind, dxbc::PSV::SemanticKind::Position);
  EXPECT_EQ(Elements[1].CompType, dxil::ElementType::F16);
  EXPECT_EQ(Elements[1].InterpMode, dxbc::PSV::InterpolationMode::Constant);
  EXPECT_EQ(Elements[1].SemanticIndices, SmallVector<uint32_t>({0, 1}));
  EXPECT_EQ(Elements[1].Rows, 2u);
  EXPECT_EQ(Elements[1].Cols, 3u);
}

//===----------------------------------------------------------------------===//
// Valid packing tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, SkipsNotAllocatedElements) {
  // Semantics accessed through dedicated intrinsics do not consume signature
  // rows and remain unallocated.

  // struct CSIn {
  //   uint3 DispatchThreadID : SV_DispatchThreadID;
  //   uint3 GroupID          : SV_GroupID;
  //   uint GroupIndex        : SV_GroupIndex;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Compute, IOType::In,
      {{dxbc::PSV::SemanticKind::DispatchThreadID, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::U32, dxbc::PSV::InterpolationMode::Undefined},
       {dxbc::PSV::SemanticKind::GroupID, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::U32, dxbc::PSV::InterpolationMode::Undefined},
       {dxbc::PSV::SemanticKind::GroupIndex, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::U32, dxbc::PSV::InterpolationMode::Undefined}});

  // Expected layout: no registers are used.
  expectPacking(Config, /*ExpectedRows=*/0,
                {Unallocated, Unallocated, Unallocated});
}

TEST_F(HLSLSemanticSignaturePackingTest, StacksInDeclarationOrder) {
  // Elements are assigned whole rows in declaration order, regardless of their
  // semantic interpretation.

  // struct VSIn {
  //   uint VertexID       : SV_VertexID;
  //   float2 Data         : DATA;
  //   float3 ClipDistance : SV_ClipDistance;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::In,
      {{dxbc::PSV::SemanticKind::VertexID, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::U32, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: VertexID.x       | unused.yzw
  // reg1: Data.xy          | unused.zw
  // reg2: ClipDistance.xyz | unused.w
  expectPacking(
      Config, /*ExpectedRows=*/3,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/2, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, DoesNotCoPackElements) {
  // Elements are never co-packed even when they would fit in one row.

  // struct VSIn {
  //   float A : A;
  //   float B : B;
  //   float C : C;
  //   float D : D;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::In,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: A.x | unused.yzw
  // reg1: B.x | unused.yzw
  // reg2: C.x | unused.yzw
  // reg3: D.x | unused.yzw
  expectPacking(Config, /*ExpectedRows=*/4,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/2, /*Col=*/0},
                 {/*Row=*/3, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, StacksMultiRowElements) {
  // A multi-row element occupies consecutive whole rows.

  // struct VSIn {
  //   float A[3]  : A;
  //   float3 B[2] : B;
  //   float4 C    : C;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::In,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/3, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/2, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/4,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: A[0].x   | unused.yzw
  // reg1: A[1].x   | unused.yzw
  // reg2: A[2].x   | unused.yzw
  // reg3: B[0].xyz | unused.w
  // reg4: B[1].xyz | unused.w
  // reg5: C.xyzw
  expectPacking(
      Config, /*ExpectedRows=*/6,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/3, /*Col=*/0}, {/*Row=*/5, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, ExactlyFillsSignature) {
  // An element may occupy all available signature rows.

  // struct VSIn {
  //   float4 A[32] : A;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::In,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/MaxSignatureRows,
        /*Cols=*/MaxSignatureCols, dxil::ElementType::F32,
        dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0-31: A[0-31].xyzw
  expectPacking(Config, /*ExpectedRows=*/MaxSignatureRows,
                {{/*Row=*/0, /*Col=*/0}});
}

//===----------------------------------------------------------------------===//
// Packing error tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, RejectsSignatureOverflow) {
  // A signature that requires more than 32 rows cannot be packed.

  // struct VSIn {
  //   float4 A0  : A0;
  //   ...
  //   float4 A32 : A32;
  // };
  TestConfig Config(Triple::EnvironmentType::Vertex, IOType::In, {});
  for (unsigned I = 0; I != MaxSignatureRows + 1; ++I)
    Config.Elements.push_back({dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1,
                               /*Cols=*/MaxSignatureCols,
                               dxil::ElementType::F32,
                               dxbc::PSV::InterpolationMode::Linear});

  // The last element is the one that no longer fits.
  expectPackingError(Config, SignaturePackingError::SignatureOverflow,
                     /*ExpectedElementIndex=*/MaxSignatureRows);
}

TEST_F(HLSLSemanticSignaturePackingTest, RejectsSingleElementOverflow) {
  // A single element may also require more rows than the signature provides.

  // struct VSIn {
  //   float4 A[33] : A;
  // };
  TestConfig Config(Triple::EnvironmentType::Vertex, IOType::In,
                    {{dxbc::PSV::SemanticKind::Arbitrary,
                      /*Rows=*/MaxSignatureRows + 1,
                      /*Cols=*/MaxSignatureCols, dxil::ElementType::F32,
                      dxbc::PSV::InterpolationMode::Linear}});

  expectPackingError(Config, SignaturePackingError::SignatureOverflow,
                     /*ExpectedElementIndex=*/0);
}

TEST_F(HLSLSemanticSignaturePackingTest, RejectsMultiRowSignatureOverflow) {
  // Each element is valid on its own, but together they require 33 rows.

  // struct VSIn {
  //   float4 A[31] : A;
  //   float4 B[2]  : B;
  // };
  TestConfig Config(Triple::EnvironmentType::Vertex, IOType::In,
                    {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/31,
                      /*Cols=*/MaxSignatureCols, dxil::ElementType::F32,
                      dxbc::PSV::InterpolationMode::Linear},
                     {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/2,
                      /*Cols=*/MaxSignatureCols, dxil::ElementType::F32,
                      dxbc::PSV::InterpolationMode::Linear}});

  expectPackingError(Config, SignaturePackingError::SignatureOverflow,
                     /*ExpectedElementIndex=*/1);
}

//===----------------------------------------------------------------------===//
// Indexed packing tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, IndexedUsesSemanticIndices) {
  // Target elements are assigned the row denoted by their semantic index, not
  // their declaration order. Every target starts at column zero.

  // struct PSOut {
  //   float4 Color3 : SV_Target3;
  //   float Color0  : SV_Target0;
  //   float2 Color2 : SV_Target2;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Pixel, IOType::Out,
      {{dxbc::PSV::SemanticKind::Target, /*Rows=*/1, /*Cols=*/4,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined,
        /*SemanticIndex=*/3},
       {dxbc::PSV::SemanticKind::Target, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined,
        /*SemanticIndex=*/0},
       {dxbc::PSV::SemanticKind::Target, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined,
        /*SemanticIndex=*/2}});

  // Expected layout:
  // reg0: Color0.x    | unused.yzw
  // reg1: unused.xyzw
  // reg2: Color2.xy   | unused.zw
  // reg3: Color3.xyzw
  expectIndexedPacking(
      Config, /*ExpectedRows=*/4,
      {{/*Row=*/3, /*Col=*/0}, {/*Row=*/0, /*Col=*/0}, {/*Row=*/2, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, IndexedLeavesSemanticIndexGaps) {
  // Rows without a corresponding target semantic remain unused.

  // struct PSOut {
  //   float4 Color1 : SV_Target1;
  //   float4 Color7 : SV_Target7;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Pixel, IOType::Out,
      {{dxbc::PSV::SemanticKind::Target, /*Rows=*/1, /*Cols=*/4,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined,
        /*SemanticIndex=*/1},
       {dxbc::PSV::SemanticKind::Target, /*Rows=*/1, /*Cols=*/4,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined,
        /*SemanticIndex=*/7}});

  // Expected layout:
  // reg0: unused.xyzw
  // reg1: Color1.xyzw
  // reg2-6: unused.xyzw
  // reg7: Color7.xyzw
  expectIndexedPacking(Config, /*ExpectedRows=*/8,
                       {{/*Row=*/1, /*Col=*/0}, {/*Row=*/7, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, IndexedRejectsSemanticIndexOverflow) {
  // A semantic index outside the 32-row signature cannot be allocated.

  // struct PSOut {
  //   float4 Color32 : SV_Target32;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Pixel, IOType::Out,
      {{dxbc::PSV::SemanticKind::Target, /*Rows=*/1, /*Cols=*/4,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined,
        /*SemanticIndex=*/MaxSignatureRows}});

  expectIndexedPackingError(Config, SignaturePackingError::SignatureOverflow,
                            /*ExpectedElementIndex=*/0);
}

} // namespace
