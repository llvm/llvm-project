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
  };

  struct ExpectedLocation {
    uint32_t Row;
    uint8_t Col;
  };

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
        SemanticIndices.push_back(Row);

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

  void expectPacking(const TestConfig &Config, unsigned ExpectedRows,
                     std::initializer_list<ExpectedLocation> Locations) {
    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    ASSERT_EQ(Elements.size(), Locations.size());

    ASSERT_THAT_ERROR(packStacked(Elements, Config), Succeeded());

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
    EXPECT_THAT_ERROR(packStacked(Elements, Config),
                      FailedWithMessage(Message));
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

} // namespace
