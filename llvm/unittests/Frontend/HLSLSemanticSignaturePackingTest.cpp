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

  // Denotes an element that is expected to be left unallocated because it is
  // not part of the packed signature.
  static constexpr ExpectedLocation Unallocated = {UnallocatedRow,
                                                   UnallocatedCol};

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

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableSystemValueOrdering) {
  // System values may be co-packed to the right of arbitrary values, and a
  // system generated value may be co-packed to the right of both.

  // struct PSIn {
  //   uint A             : A;
  //   uint RTIndex       : SV_RenderTargetArrayIndex;
  //   uint PrimitiveID   : SV_PrimitiveID;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Pixel, SemanticSignatureKind::Input,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::U32, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::RenderTargetArrayIndex, /*Rows=*/1,
        /*Cols=*/1, dxil::ElementType::U32,
        dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::PrimitiveID, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::U32, dxbc::PSV::InterpolationMode::Constant}});

  // Expected layout:
  // reg0: A.x | RTIndex.y | PrimitiveID.z | unused.w
  expectPacking(
      Config, /*ExpectedRows=*/1,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/0, /*Col=*/1}, {/*Row=*/0, /*Col=*/2}});
}

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
      Triple::EnvironmentType::Vertex, SemanticSignatureKind::Output,
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
  expectPacking(Config, /*ExpectedRows=*/3,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/0, /*Col=*/2},
                 {/*Row=*/0, /*Col=*/3},
                 {/*Row=*/2, /*Col=*/0}});
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
      Triple::EnvironmentType::Pixel, SemanticSignatureKind::Input,
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
  expectPacking(Config, /*ExpectedRows=*/4,
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
      Triple::EnvironmentType::Vertex, SemanticSignatureKind::Output,
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
      Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/0, /*Col=*/2}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableInterpolationMode) {
  // struct VSOut {
  //   float2 A                : A;
  //   nointerpolation float2 B : B;
  //   float2 C                : C;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, SemanticSignatureKind::Output,
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
      Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/0, /*Col=*/2}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableCompatible16BitTypes) {
  // struct VSOut {
  //   nointerpolation int16_t A    : A;
  //   nointerpolation float16_t3 B : B;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, SemanticSignatureKind::Output,
      /*UseNative16BitTypes=*/true,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::I16, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F16, dxbc::PSV::InterpolationMode::Constant}});

  // Expected layout:
  // reg0: A.x | B.yzw
  expectPacking(Config, /*ExpectedRows=*/1,
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
      Triple::EnvironmentType::Vertex, SemanticSignatureKind::Output,
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
      Config, /*ExpectedRows=*/2,
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
      Triple::EnvironmentType::Vertex, SemanticSignatureKind::Output,
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
      Config, /*ExpectedRows=*/2,
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
      Triple::EnvironmentType::Vertex, SemanticSignatureKind::Output,
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
      Config, /*ExpectedRows=*/2,
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
      Triple::EnvironmentType::Vertex, SemanticSignatureKind::Output,
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
      Config, /*ExpectedRows=*/2,
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
      Triple::EnvironmentType::Vertex, SemanticSignatureKind::Output,
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
      Config, /*ExpectedRows=*/3,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/2, /*Col=*/0}});
}

//===----------------------------------------------------------------------===//
// Boundary and stability tests
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Component ordering tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableArbitraryNotRightOfSV) {
  // Arbitrary values may never be placed to the right of a system value in the
  // same register, so B cannot co-pack with Position even though there is
  // space for it.

  // struct VSOut {
  //   float2 Position : SV_Position;
  //   float2 A        : A;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, SemanticSignatureKind::Output,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Position, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: Position.xy | unused.zw
  // reg1: A.xy        | unused.zw
  expectPacking(Config, /*ExpectedRows=*/2,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableSGVIsRightmost) {
  // Nothing may be placed to the right of a system generated value, so both A
  // and RTIndex are pushed into the next register.

  // struct PSIn {
  //   bool IsFrontFace : SV_IsFrontFace;
  //   uint A           : A;
  //   uint RTIndex     : SV_RenderTargetArrayIndex;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Pixel, SemanticSignatureKind::Input,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::IsFrontFace, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::I1, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::U32, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::RenderTargetArrayIndex, /*Rows=*/1,
        /*Cols=*/1, dxil::ElementType::U32,
        dxbc::PSV::InterpolationMode::Constant}});

  // Expected layout:
  // reg0: IsFrontFace.x | unused.yzw
  // reg1: A.x | RTIndex.y | unused.zw
  expectPacking(
      Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/1, /*Col=*/1}});
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableSkipsUnpackedElements) {
  // Semantics that are accessed through a dedicated intrinsic are left
  // unallocated and do not reserve any signature space.

  // struct PSIn {
  //   float2 A         : A;
  //   uint SampleIndex : SV_SampleIndex;
  //   float2 B         : B;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Pixel, SemanticSignatureKind::Input,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::SampleIndex, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::U32, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: A.xy | B.zw
  expectPacking(Config, /*ExpectedRows=*/1,
                {{/*Row=*/0, /*Col=*/0}, Unallocated, {/*Row=*/0, /*Col=*/2}});
}

//===----------------------------------------------------------------------===//
// Dynamic indexing tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableIndexedAfterSystemValue) {
  // A multi-row element is dynamically indexable, so it may not share any of
  // its rows with a system value, and it requires contiguous rows.

  // struct VSOut {
  //   float Position : SV_Position;
  //   float3 A[2]    : A;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, SemanticSignatureKind::Output,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Position, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/2, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: Position.x | unused.yzw
  // reg1: A[0].xyz   | unused.w
  // reg2: A[1].xyz   | unused.w
  expectPacking(Config, /*ExpectedRows=*/3,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}});
}

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

TEST_F(HLSLSemanticSignaturePackingTest, RejectsOverflowFromComponentOrdering) {
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
  TestConfig Config(Triple::EnvironmentType::Pixel,
                    SemanticSignatureKind::Input,
                    /*UseNative16BitTypes=*/false,
                    {{dxbc::PSV::SemanticKind::IsFrontFace, /*Rows=*/1,
                      /*Cols=*/1, dxil::ElementType::I1,
                      dxbc::PSV::InterpolationMode::Constant}});
  for (unsigned I = 0; I != MaxSignatureRows; ++I)
    Config.Elements.push_back({dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1,
                               /*Cols=*/3, dxil::ElementType::I32,
                               dxbc::PSV::InterpolationMode::Constant});
  expectPackingError(Config, SignatureOverflowMessage);
}

} // namespace
