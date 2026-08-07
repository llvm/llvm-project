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
  static constexpr StringRef SignatureOverflowMessage =
      "signature elements do not fit in 32 rows";

  struct ElementConfig {
    dxbc::PSV::SemanticKind SemanticKind;
    uint32_t Rows;
    uint8_t Cols;
    dxil::ElementType CompType;
    dxbc::PSV::InterpolationMode InterpMode;
  };

  struct ExpectedLocation {
    uint32_t Row;
    uint8_t Col;
  };

  struct TestConfig {
    Triple::EnvironmentType ShaderStage;
    SemanticSignatureKind SignatureKind;
    bool UseNative16BitTypes;
    SmallVector<ElementConfig> Elements;

    TestConfig(Triple::EnvironmentType ShaderStage,
               SemanticSignatureKind SignatureKind, bool UseNative16BitTypes,
               std::initializer_list<ElementConfig> Elements)
        : ShaderStage(ShaderStage), SignatureKind(SignatureKind),
          UseNative16BitTypes(UseNative16BitTypes), Elements(Elements) {}
  };

  SmallVector<SemanticSignatureElement>
  makeSignature(const TestConfig &Config) {
    SmallVector<SemanticSignatureElement> Elements;
    for (const ElementConfig &Element : Config.Elements) {
      SmallVector<uint32_t> SemanticIndices;
      for (uint32_t Row = 0; Row != Element.Rows; ++Row)
        SemanticIndices.push_back(Row);

      Elements.push_back(SemanticSignatureElement{
          /*SigId=*/static_cast<uint32_t>(Elements.size()),
          /*SemanticName=*/"TEST",
          /*CompType=*/Element.CompType,
          /*SemanticKind=*/Element.SemanticKind,
          /*SemanticIndices=*/std::move(SemanticIndices),
          /*InterpMode=*/Element.InterpMode,
          /*Rows=*/Element.Rows,
          /*Cols=*/Element.Cols,
          /*StartRow=*/UnallocatedRow,
          /*StartCol=*/UnallocatedCol,
          /*UsageMask=*/0,
          /*DynIndexMask=*/0,
          /*GSStream=*/0,
      });
    }
    return Elements;
  }

  Error pack(SmallVectorImpl<SemanticSignatureElement> &Elements,
             const TestConfig &Config) {
    return packSignaturePrefixStable(Elements, Config.ShaderStage,
                                     Config.SignatureKind,
                                     Config.UseNative16BitTypes);
  }

  void expectPacking(const TestConfig &Config, unsigned ExpectedRows,
                     std::initializer_list<ExpectedLocation> Locations) {
    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    ASSERT_EQ(Elements.size(), Locations.size());

    ASSERT_THAT_ERROR(pack(Elements, Config), Succeeded());

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

  void expectPackingError(const TestConfig &Config, StringRef Message) {
    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    EXPECT_THAT_ERROR(pack(Elements, Config), FailedWithMessage(Message));
  }
};

//===----------------------------------------------------------------------===//
// Valid packing tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableEmptySignature) {
  // A signature without any elements uses no registers.

  // struct VSOut {};
  TestConfig Config(Triple::EnvironmentType::Vertex,
                    SemanticSignatureKind::Output,
                    /*UseNative16BitTypes=*/false, {});

  // Expected layout: no registers are used.
  expectPacking(Config, /*ExpectedRows=*/0, {});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableDeclarationOrder) {
  // Elements are visited in declaration order and are co-packed into a
  // register whenever the register has room left for them.

  // struct VSOut {
  //   float Fog      : FOG;
  //   float Alpha    : COLOR0;
  //   float2 Position : SV_Position;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, SemanticSignatureKind::Output,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Position, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: Fog.x | Alpha.y | Position.zw
  expectPacking(
      Config, /*ExpectedRows=*/1,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/0, /*Col=*/1}, {/*Row=*/0, /*Col=*/2}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableWhenAppended) {
  // Appending an element to a signature never moves the elements declared
  // before it; the appended element is only packed into the space they left.

  // struct Prefix {
  //   float3 A : A;
  //   float2 B : B;
  // };
  TestConfig PrefixConfig(
      Triple::EnvironmentType::Vertex, SemanticSignatureKind::Output,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: A.xyz | unused.w
  // reg1: B.xy  | unused.zw
  expectPacking(PrefixConfig, /*ExpectedRows=*/2,
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
      ExtendedConfig, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/0, /*Col=*/3}});
}

//===----------------------------------------------------------------------===//
// Boundary and stability tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableFillsAllRows) {
  // A signature may use all 32 rows.

  // struct VSOut {
  //   float4 A0  : A0;
  //   ...
  //   float4 A31 : A31;
  // };
  TestConfig Config(Triple::EnvironmentType::Vertex,
                    SemanticSignatureKind::Output,
                    /*UseNative16BitTypes=*/false, {});
  for (unsigned I = 0; I != MaxSignatureRows; ++I)
    Config.Elements.push_back({dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1,
                               /*Cols=*/MaxSignatureCols,
                               dxil::ElementType::F32,
                               dxbc::PSV::InterpolationMode::Linear});

  SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
  ASSERT_THAT_ERROR(pack(Elements, Config), Succeeded());

  for (unsigned I = 0; I != MaxSignatureRows; ++I) {
    EXPECT_EQ(Elements[I].StartRow, I) << "element " << I;
    EXPECT_EQ(Elements[I].StartCol, 0u) << "element " << I;
  }
}

//===----------------------------------------------------------------------===//
// Packing error tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, RejectsSignatureOverflow) {
  // A signature that requires more than 32 rows cannot be packed.

  // struct VSOut {
  //   float4 A0  : A0;
  //   ...
  //   float4 A32 : A32;
  // };
  TestConfig Config(Triple::EnvironmentType::Vertex,
                    SemanticSignatureKind::Output,
                    /*UseNative16BitTypes=*/false, {});
  for (unsigned I = 0; I != MaxSignatureRows + 1; ++I)
    Config.Elements.push_back({dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1,
                               /*Cols=*/MaxSignatureCols,
                               dxil::ElementType::F32,
                               dxbc::PSV::InterpolationMode::Linear});
  expectPackingError(Config, SignatureOverflowMessage);
}

} // namespace
