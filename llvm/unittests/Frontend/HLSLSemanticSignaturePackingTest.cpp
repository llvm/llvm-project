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
    uint32_t GSStream = 0;
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
    bool UseNative16BitTypes;
    SmallVector<ElementConfig> Elements;

    TestConfig(Triple::EnvironmentType ShaderStage, IOType IOTy,
               std::initializer_list<ElementConfig> Elements)
        : ShaderStage(ShaderStage), IOTy(IOTy), UseNative16BitTypes(false),
          Elements(Elements) {}

    TestConfig(Triple::EnvironmentType ShaderStage, IOType IOTy,
               bool UseNative16BitTypes,
               std::initializer_list<ElementConfig> Elements)
        : ShaderStage(ShaderStage), IOTy(IOTy),
          UseNative16BitTypes(UseNative16BitTypes), Elements(Elements) {}
  };

  enum class PackingMethod {
    Stacked,
    Indexed,
    PrefixStable,
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
      Elements.back().GSStream = Element.GSStream;
    }
    return Elements;
  }

  Error pack(PackingMethod Method,
             SmallVectorImpl<SemanticSignatureElement> &Elements,
             const TestConfig &Config) {
    switch (Method) {
    case PackingMethod::Stacked:
      return packSignatureStacked(Elements, Config.ShaderStage, Config.IOTy);
    case PackingMethod::Indexed:
      return packSignatureIndexed(Elements, Config.ShaderStage, Config.IOTy);
    case PackingMethod::PrefixStable:
      return packSignaturePrefixStable(Elements, Config.ShaderStage,
                                       Config.IOTy, Config.UseNative16BitTypes);
    }
    llvm_unreachable("invalid packing method");
  }

  void expectPacking(PackingMethod Method, const TestConfig &Config,
                     unsigned ExpectedRows,
                     std::initializer_list<ExpectedLocation> Locations) {
    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    ASSERT_EQ(Elements.size(), Locations.size());

    ASSERT_THAT_ERROR(pack(Method, Elements, Config), Succeeded());

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

  void expectPackingError(PackingMethod Method, const TestConfig &Config,
                          SignaturePackingError::ErrorKind ExpectedKind,
                          unsigned ExpectedElementIndex) {
    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    Error E = pack(Method, Elements, Config);
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

TEST_F(HLSLSemanticSignaturePackingTest, EmptySignature) {
  TestConfig Config(Triple::EnvironmentType::Vertex, IOType::Out, {});

  for (PackingMethod Method : {PackingMethod::Stacked, PackingMethod::Indexed,
                               PackingMethod::PrefixStable})
    expectPacking(Method, Config, /*ExpectedRows=*/0, {});
}

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
  for (PackingMethod Method : {PackingMethod::Stacked, PackingMethod::Indexed,
                               PackingMethod::PrefixStable})
    expectPacking(Method, Config, /*ExpectedRows=*/0,
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
      PackingMethod::Stacked, Config, /*ExpectedRows=*/3,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/2, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, CoPackingDependsOnMethod) {
  // Stacked packing gives each element a whole row, while prefix-stable packing
  // co-packs the same elements into the components of one row.

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

  // Stacked layout:
  // reg0: A.x | unused.yzw
  // reg1: B.x | unused.yzw
  // reg2: C.x | unused.yzw
  // reg3: D.x | unused.yzw
  expectPacking(PackingMethod::Stacked, Config, /*ExpectedRows=*/4,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/2, /*Col=*/0},
                 {/*Row=*/3, /*Col=*/0}});

  // Prefix-stable layout:
  // reg0: A.x | B.y | C.z | D.w
  expectPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/1,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/0, /*Col=*/1},
                 {/*Row=*/0, /*Col=*/2},
                 {/*Row=*/0, /*Col=*/3}});
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
      PackingMethod::Stacked, Config, /*ExpectedRows=*/6,
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
  for (PackingMethod Method :
       {PackingMethod::Stacked, PackingMethod::PrefixStable})
    expectPacking(Method, Config, /*ExpectedRows=*/MaxSignatureRows,
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
  for (PackingMethod Method :
       {PackingMethod::Stacked, PackingMethod::PrefixStable})
    expectPackingError(Method, Config, SignaturePackingError::SignatureOverflow,
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

  for (PackingMethod Method :
       {PackingMethod::Stacked, PackingMethod::PrefixStable})
    expectPackingError(Method, Config, SignaturePackingError::SignatureOverflow,
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

  for (PackingMethod Method :
       {PackingMethod::Stacked, PackingMethod::PrefixStable})
    expectPackingError(Method, Config, SignaturePackingError::SignatureOverflow,
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
  expectPacking(
      PackingMethod::Indexed, Config, /*ExpectedRows=*/4,
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
  expectPacking(PackingMethod::Indexed, Config, /*ExpectedRows=*/8,
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

  expectPackingError(PackingMethod::Indexed, Config,
                     SignaturePackingError::SignatureOverflow,
                     /*ExpectedElementIndex=*/0);
}

//===----------------------------------------------------------------------===//
// Basic prefix-stable packing tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableWhenAppended) {
  // Appending an element to a signature never moves the elements declared
  // before it; the appended element is only packed into the space they left.

  // struct Prefix {
  //   float3 A : A;
  //   float2 B : B;
  // };
  TestConfig PrefixConfig(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: A.xyz | unused.w
  // reg1: B.xy  | unused.zw
  expectPacking(PackingMethod::PrefixStable, PrefixConfig, /*ExpectedRows=*/2,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}});

  // struct Extended {
  //   float3 A : A;
  //   float2 B : B;
  //   float C  : C;
  // };
  TestConfig ExtendedConfig = PrefixConfig;
  ExtendedConfig.Elements.push_back(
      {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
       dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear});

  // Expected layout:
  // reg0: A.xyz | C.w
  // reg1: B.xy  | unused.zw
  //
  // C is packed into the gap A left behind, and A and B keep the locations
  // they were given in Prefix.
  expectPacking(
      PackingMethod::PrefixStable, ExtendedConfig, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/0, /*Col=*/3}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableFillsAllRows) {
  // A signature may use all 32 rows.

  // struct VSOut {
  //   float4 A0  : A0;
  //   ...
  //   float4 A31 : A31;
  // };
  TestConfig Config(Triple::EnvironmentType::Vertex, IOType::Out,
                    /*UseNative16BitTypes=*/false, {});
  for (unsigned I = 0; I != MaxSignatureRows; ++I)
    Config.Elements.push_back({dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1,
                               /*Cols=*/MaxSignatureCols,
                               dxil::ElementType::F32,
                               dxbc::PSV::InterpolationMode::Linear});

  SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
  ASSERT_THAT_ERROR(pack(PackingMethod::PrefixStable, Elements, Config),
                    Succeeded());

  for (unsigned I = 0; I != MaxSignatureRows; ++I) {
    EXPECT_EQ(Elements[I].StartRow, I) << "element " << I;
    EXPECT_EQ(Elements[I].StartCol, 0u) << "element " << I;
  }
}

//===----------------------------------------------------------------------===//
// Prefix-stable row compatibility tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableGeneralPacking) {
  // Native 16-bit types are enabled.
  // struct PSIn {
  //   float16_t2 A : A;
  //   float2 B     : B;
  //   float16_t3 C : C;
  //   float2 D     : D;
  //   int E        : E;
  //   float16_t2 F : F;
  //   float16_t G  : G;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Pixel, IOType::In,
      /*UseNative16BitTypes=*/true,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F16, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F16, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::I32, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F16, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F16, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: A.xy | F.zw
  // reg1: B.xy | D.zw
  // reg2: C.xyz | G.w
  // reg3: E.x | unused.yzw
  expectPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/4,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/2, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/2},
                 {/*Row=*/3, /*Col=*/0},
                 {/*Row=*/0, /*Col=*/2},
                 {/*Row=*/2, /*Col=*/3}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableNative16BitWidth) {
  // struct VSOut {
  //   float16_t2 A : A;
  //   float2 B     : B;
  //   float16_t2 C : C;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/true,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F16, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F16, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: A.xy | C.zw
  // reg1: B.xy | unused.zw
  expectPacking(
      PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/0, /*Col=*/2}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableInterpolationMode) {
  // struct VSOut {
  //   float2 A                : A;
  //   nointerpolation float2 B : B;
  //   float2 C                : C;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: A.xy | C.zw
  // reg1: B.xy | unused.zw
  expectPacking(
      PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/0, /*Col=*/2}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableCompatible16BitTypes) {
  // struct VSOut {
  //   nointerpolation int16_t A    : A;
  //   nointerpolation float16_t3 B : B;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/true,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::I16, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F16, dxbc::PSV::InterpolationMode::Constant}});

  // Expected layout:
  // reg0: A.x | B.yzw
  expectPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/1,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/0, /*Col=*/1}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableNormalized16BitTypes) {
  // A normalized 16-bit type has the same component width as any other 16-bit
  // type, so it co-packs with them but not with a 32-bit type.

  // struct VSOut {
  //   nointerpolation snorm half A : A;
  //   nointerpolation float16_t B  : B;
  //   nointerpolation float C      : C;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/true,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::SNormF16, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F16, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Constant}});

  // Expected layout:
  // reg0: A.x | B.y | unused.zw
  // reg1: C.x | unused.yzw
  expectPacking(
      PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/0, /*Col=*/1}, {/*Row=*/1, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableMinPrecisionWidth) {
  // Without native 16-bit types, 16-bit types are min-precision types which
  // occupy a full 32-bit component, so they co-pack with 32-bit types. This is
  // the same signature as PrefixStableNative16BitWidth, which packs
  // differently.

  // struct VSOut {
  //   min16float2 A : A;
  //   float2 B      : B;
  //   min16float2 C : C;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F16, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F16, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: A.xy | B.zw
  // reg1: C.xy | unused.zw
  expectPacking(
      PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/0, /*Col=*/2}, {/*Row=*/1, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableUndefinedInterpMode) {
  // An undefined interpolation mode does not constrain a register, but the
  // first defined mode packed into it does.

  // struct VSOut {
  //   float2 A                : A; // undefined interpolation mode
  //   float B                 : B; // linear
  //   nointerpolation float C : C; // nointerpolation
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Constant}});

  // Expected layout:
  // reg0: A.xy | B.z | unused.w
  // reg1: C.x  | unused.yzw
  expectPacking(
      PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/0, /*Col=*/2}, {/*Row=*/1, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest,
       PrefixStableUndefinedInterpModeAfterDefined) {
  // Once a register has a defined interpolation mode, an element with an
  // undefined mode cannot be packed into it.

  // struct VSOut {
  //   float2 A : A; // linear
  //   float2 B : B; // undefined interpolation mode
  //   float2 C : C; // linear
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: A.xy | C.zw
  // reg1: B.xy | unused.zw
  expectPacking(
      PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/0, /*Col=*/2}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableDistinctInterpModes) {
  // Every distinct interpolation mode requires its own register, including
  // modes that only differ by their centroid or noperspective qualifier.

  // struct VSOut {
  //   float2 A               : A;
  //   centroid float2 B      : B;
  //   noperspective float2 C : C;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::LinearCentroid},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32,
        dxbc::PSV::InterpolationMode::LinearNoperspective}});

  // Expected layout:
  // reg0: A.xy | unused.zw
  // reg1: B.xy | unused.zw
  // reg2: C.xy | unused.zw
  expectPacking(
      PackingMethod::PrefixStable, Config, /*ExpectedRows=*/3,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/2, /*Col=*/0}});
}

//===----------------------------------------------------------------------===//
// Prefix-stable component ordering tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableSystemValueOrdering) {
  // System values may be co-packed to the right of arbitrary values, and a
  // system generated value may be co-packed to the right of both.

  // struct PSIn {
  //   uint A             : A;
  //   float Position      : SV_Position;
  //   bool IsFrontFace   : SV_IsFrontFace;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Pixel, IOType::In,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::U32, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::Position, /*Rows=*/1,
        /*Cols=*/1, dxil::ElementType::F32,
        dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::IsFrontFace, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::I1, dxbc::PSV::InterpolationMode::Constant}});

  // Expected layout:
  // reg0: A.x | Position.y | IsFrontFace.z | unused.w
  expectPacking(
      PackingMethod::PrefixStable, Config, /*ExpectedRows=*/1,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/0, /*Col=*/1}, {/*Row=*/0, /*Col=*/2}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableArbitraryNotRightOfSV) {
  // Arbitrary values may never be placed to the right of a system value in the
  // same register, so B cannot co-pack with Position even though there is
  // space for it.

  // struct VSOut {
  //   float2 Position : SV_Position;
  //   float2 A        : A;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Position, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: Position.xy | unused.zw
  // reg1: A.xy        | unused.zw
  expectPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableSGVIsRightmost) {
  // Nothing may be placed to the right of a system generated value, so both A
  // and Position are pushed into the next register.

  // struct PSIn {
  //   bool IsFrontFace : SV_IsFrontFace;
  //   uint A           : A;
  //   float Position    : SV_Position;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Pixel, IOType::In,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::IsFrontFace, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::I1, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::U32, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::Position, /*Rows=*/1,
        /*Cols=*/1, dxil::ElementType::F32,
        dxbc::PSV::InterpolationMode::Constant}});

  // Expected layout:
  // reg0: IsFrontFace.x | unused.yzw
  // reg1: A.x | Position.y | unused.zw
  expectPacking(
      PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/1, /*Col=*/1}});
}

TEST_F(HLSLSemanticSignaturePackingTest,
       PrefixStableRejectsOverflowFromComponentOrdering) {
  // Nothing may be placed to the right of a system generated value, so
  // declaring one first leaves the rest of its register unusable by the
  // arbitrary values that follow, and they no longer fit in 32 rows. Every
  // element here shares an interpolation mode and a data width, so component
  // ordering is the only reason the signature overflows.
  //
  // Note that the optimal algorithm packs the arbitrary values into reg0 to
  // reg31 first and backfills IsFrontFace into reg0.w, so the very same
  // signature does fit when it is packed optimally.

  // struct PSIn {
  //   nointerpolation bool IsFrontFace : SV_IsFrontFace;
  //   nointerpolation int3 A0          : A0;
  //   ...
  //   nointerpolation int3 A31         : A31;
  // };
  TestConfig Config(Triple::EnvironmentType::Pixel, IOType::In,
                    /*UseNative16BitTypes=*/false,
                    {{dxbc::PSV::SemanticKind::IsFrontFace, /*Rows=*/1,
                      /*Cols=*/1, dxil::ElementType::I1,
                      dxbc::PSV::InterpolationMode::Constant}});
  for (unsigned I = 0; I != MaxSignatureRows; ++I)
    Config.Elements.push_back({dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1,
                               /*Cols=*/3, dxil::ElementType::I32,
                               dxbc::PSV::InterpolationMode::Constant});
  // The last element is the one that no longer fits.
  expectPackingError(PackingMethod::PrefixStable, Config,
                     SignaturePackingError::SignatureOverflow,
                     /*ExpectedElementIndex=*/MaxSignatureRows);
}

//===----------------------------------------------------------------------===//
// Prefix-stable dynamic indexing tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableIndexedRanges) {
  // An element with multiple rows occupies the same columns of a contiguous
  // range of rows, and other elements may be co-packed into the columns those
  // rows have left. A system value cannot be placed in a dynamically indexable
  // row, so Position starts a new register.

  // struct VSOut {
  //   float2 A[2]    : A;
  //   float B[2]     : B;
  //   float C        : C;
  //   float Position : SV_Position;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/2, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/2, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Position, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: A[0].xy | B[0].z | C.w
  // reg1: A[1].xy | B[1].z | unused.w
  // reg2: Position.x | unused.yzw
  expectPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/3,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/0, /*Col=*/2},
                 {/*Row=*/0, /*Col=*/3},
                 {/*Row=*/2, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableIndexedAfterSystemValue) {
  // A multi-row element is dynamically indexable, so it may not share any of
  // its rows with a system value, and it requires contiguous rows.

  // struct VSOut {
  //   float Position : SV_Position;
  //   float3 A[2]    : A;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Position, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/2, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: Position.x | unused.yzw
  // reg1: A[0].xyz   | unused.w
  // reg2: A[1].xyz   | unused.w
  expectPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/3,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}});
}

//===----------------------------------------------------------------------===//
// Prefix-stable tessellation factor tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableTessFactors) {
  // Indexed tess factors are reserved in the last column of their rows so that
  // arbitrary values can still be co-packed into the same rows.

  // struct PatchConstants {
  //   float TessFactor[2] : SV_TessFactor;
  //   float3 Data[2]      : DATA;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Hull, IOType::PatchConstantOrPrimitive,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::TessFactor, /*Rows=*/2, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/2, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined}});

  // Expected layout:
  // reg0: Data[0].xyz | TessFactor[0].w
  // reg1: Data[1].xyz | TessFactor[1].w
  expectPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
                {{/*Row=*/0, /*Col=*/3}, {/*Row=*/0, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableSingleRowTessFactor) {
  // A single row tess factor is not dynamically indexable and is packed like
  // any other system value, rather than being reserved in the last column.

  // struct PatchConstants {
  //   float TessFactor : SV_TessFactor;
  //   float3 Data      : DATA;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Hull, IOType::PatchConstantOrPrimitive,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::TessFactor, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined}});

  // Expected layout:
  // reg0: TessFactor.x | unused.yzw
  // reg1: Data.xyz     | unused.w
  expectPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest,
       PrefixStableIndexedTessFactorAfterIndexedElement) {
  // An indexed tess factor may only be placed in rows whose indexed range is
  // contained by its own, so it cannot be packed into the rows of the wider
  // indexed range of Data.

  // struct PatchConstants {
  //   float3 Data[3]      : DATA;
  //   float TessFactor[2] : SV_TessFactor;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Hull, IOType::PatchConstantOrPrimitive,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/3, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined},
       {dxbc::PSV::SemanticKind::TessFactor, /*Rows=*/2, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined}});

  // Expected layout:
  // reg0: Data[0].xyz | unused.w
  // reg1: Data[1].xyz | unused.w
  // reg2: Data[2].xyz | unused.w
  // reg3: unused.xyz  | TessFactor[0].w
  // reg4: unused.xyz  | TessFactor[1].w
  expectPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/5,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/3, /*Col=*/3}});
}

//===----------------------------------------------------------------------===//
// Prefix-stable clip/cull tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableClipCull) {
  // struct VSOut {
  //   float3 First         : First;
  //   float  Clip0         : SV_ClipDistance0;
  //   float3 Cull1         : SV_CullDistance1;
  //   float  Cull0         : SV_CullDistance0;
  //   float2 Clip1         : SV_ClipDistance1;
  //   float  WithFirst     : WithFirst;
  //   float  AfterClipCull : AfterClipCull;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: First.xyz       | WithFirst.w
  // reg1: Clip0.x         | Cull1.yzw
  // reg2: Cull0.x         | Clip1.yz | unused.w
  // reg3: AfterClipCull.x | unused.yzw
  expectPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/4,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/1},
                 {/*Row=*/2, /*Col=*/0},
                 {/*Row=*/2, /*Col=*/1},
                 {/*Row=*/0, /*Col=*/3},
                 {/*Row=*/3, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableIndexedClipCull) {
  // struct VSOut {
  //   float3 First         : First;
  //   float  Clip0         : SV_ClipDistance0;
  //   float2 Cull1[2]      : SV_CullDistance1;
  //   float  Clip1         : SV_ClipDistance1;
  //   float  WithFirst     : WithFirst;
  //   float  AfterClipCull : AfterClipCull;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/2, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: First.xyz       | WithFirst.w
  // reg1: Clip0.x         | Cull1[0].yz | Clip1.w
  // reg2: unused.x        | Cull1[1].yz | unused.w
  // reg3: AfterClipCull.x | unused.yzw
  expectPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/4,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/1},
                 {/*Row=*/1, /*Col=*/3},
                 {/*Row=*/0, /*Col=*/3},
                 {/*Row=*/3, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableMultipleIndexedClipCull) {
  // struct VSOut {
  //   float3 First         : First;
  //   float  Clip0         : SV_ClipDistance0;
  //   float2 Cull1[2]      : SV_CullDistance1;
  //   float  Clip1[2]      : SV_ClipDistance1;
  //   float  WithFirst     : WithFirst;
  //   float  AfterClipCull : AfterClipCull;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/2, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/2, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: First.xyz       | WithFirst.w
  // reg1: Clip0.x         | Cull1[0].yz | Clip1[0].w
  // reg2: unused.x        | Cull1[1].yz | Clip1[1].w
  // reg3: AfterClipCull.x | unused.yzw
  expectPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/4,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/1},
                 {/*Row=*/1, /*Col=*/3},
                 {/*Row=*/0, /*Col=*/3},
                 {/*Row=*/3, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableClipCullFillsTwoRows) {
  // Clip and cull distances may use a combined maximum of eight components
  // spread over two registers.

  // struct VSOut {
  //   float4 Clip0 : SV_ClipDistance0;
  //   float4 Cull0 : SV_CullDistance0;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/4,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/4,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: Clip0.xyzw
  // reg1: Cull0.xyzw
  expectPacking(PackingMethod::PrefixStable, Config,
                /*ExpectedRows=*/MaxClipCullRows,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableSeparatesClipCullRows) {
  // Non-indexed clip and cull distance registers do not need to be adjacent.
  // Elements declared between them may separate their reserved registers while
  // the clip/cull elements continue to count toward the common two-register
  // limit.

  // struct VSOut {
  //   float3 Clip0 : SV_ClipDistance0;
  //   float4 A[30] : A;
  //   float3 Cull0 : SV_CullDistance0;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/30, /*Cols=*/4,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0:     Clip0.xyz | unused.w
  // reg1-30:  A[0-29].xyzw
  // reg31:    Cull0.xyz | unused.w
  expectPacking(PackingMethod::PrefixStable, Config,
                /*ExpectedRows=*/MaxSignatureRows,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/31, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableClipCullAsArbitrary) {
  // Clip and cull distances are arbitrary values in a patch constant
  // signature, so the two register limit does not apply to them.

  // struct PatchConstants {
  //   float3 Clip0 : SV_ClipDistance0;
  //   float3 Clip1 : SV_ClipDistance1;
  //   float3 Cull0 : SV_CullDistance0;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Hull, IOType::PatchConstantOrPrimitive,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined}});

  // Expected layout:
  // reg0: Clip0.xyz | unused.w
  // reg1: Clip1.xyz | unused.w
  // reg2: Cull0.xyz | unused.w
  expectPacking(
      PackingMethod::PrefixStable, Config, /*ExpectedRows=*/3,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/2, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableClipCullWhenAppended) {
  // Clip and cull distances reserve whole registers ahead of the elements that
  // follow them, so appending elements does not move them either, not even
  // when the appended elements are clip/cull values themselves.

  // struct Prefix {
  //   float3 First    : First;
  //   float  Clip0    : SV_ClipDistance0;
  //   float2 Cull1[2] : SV_CullDistance1;
  // };
  TestConfig PrefixConfig(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/2, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: First.xyz | unused.w
  // reg1: Clip0.x   | Cull1[0].yz | unused.w
  // reg2: unused.x  | Cull1[1].yz | unused.w
  expectPacking(
      PackingMethod::PrefixStable, PrefixConfig, /*ExpectedRows=*/3,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/1, /*Col=*/1}});

  TestConfig ExtendedConfig = PrefixConfig;
  ExtendedConfig.Elements.push_back(
      {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/1,
       dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear});
  ExtendedConfig.Elements.push_back(
      {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
       dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear});
  ExtendedConfig.Elements.push_back(
      {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
       dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear});

  // Expected layout:
  // reg0: First.xyz       | WithFirst.w
  // reg1: Clip0.x         | Cull1[0].yz | Clip1.w
  // reg2: unused.x        | Cull1[1].yz | unused.w
  // reg3: AfterClipCull.x | unused.yzw
  expectPacking(PackingMethod::PrefixStable, ExtendedConfig, /*ExpectedRows=*/4,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/1},
                 {/*Row=*/1, /*Col=*/3},
                 {/*Row=*/0, /*Col=*/3},
                 {/*Row=*/3, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableRejectsClipCullOverflow) {
  // Clip and cull distances may use at most eight components, shared between
  // them, so nine components cannot be packed.

  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});
  expectPackingError(PackingMethod::PrefixStable, Config,
                     SignaturePackingError::ClipCullOverflow,
                     /*ExpectedElementIndex=*/2);
}

TEST_F(HLSLSemanticSignaturePackingTest,
       PrefixStableRejectsUnpackableClipCull) {
  // These clip and cull distances fit in eight components, but they cannot be
  // split across the two registers available to them.

  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});
  expectPackingError(PackingMethod::PrefixStable, Config,
                     SignaturePackingError::ClipCullOverflow,
                     /*ExpectedElementIndex=*/2);
}

//===----------------------------------------------------------------------===//
// Prefix-stable geometry stream tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableGeometryStreams) {
  // Each geometry shader output stream is packed into its own signature, so
  // elements of different streams never share a register. The reported number
  // of rows is the maximum used by any single stream.

  // struct Stream0 {
  //   float4 A : A;
  //   float2 C : C;
  // };
  // struct Stream1 {
  //   float4 B : B;
  // };
  // void GSMain(inout PointStream<Stream0> S0, inout PointStream<Stream1> S1);
  //
  // Elements are declared in the order A, B, C.
  TestConfig Config(Triple::EnvironmentType::Geometry, IOType::Out,
                    /*UseNative16BitTypes=*/false,
                    {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1,
                      /*Cols=*/4, dxil::ElementType::F32,
                      dxbc::PSV::InterpolationMode::Linear, /*SemanticIndex=*/0,
                      /*GSStream=*/0},
                     {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1,
                      /*Cols=*/4, dxil::ElementType::F32,
                      dxbc::PSV::InterpolationMode::Linear, /*SemanticIndex=*/0,
                      /*GSStream=*/1},
                     {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1,
                      /*Cols=*/2, dxil::ElementType::F32,
                      dxbc::PSV::InterpolationMode::Linear, /*SemanticIndex=*/0,
                      /*GSStream=*/0}});

  // Expected layout:
  // stream0 reg0: A.xyzw
  // stream0 reg1: C.xy | unused.zw
  // stream1 reg0: B.xyzw
  expectPacking(
      PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}});
}
} // namespace
