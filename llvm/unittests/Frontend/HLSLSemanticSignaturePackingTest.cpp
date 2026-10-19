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
#include <initializer_list>
#include <string>

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
    Optimized,
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

  Expected<unsigned> pack(PackingMethod Method,
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
    case PackingMethod::Optimized:
      return packSignatureOptimized(Elements, Config.ShaderStage, Config.IOTy,
                                    Config.UseNative16BitTypes);
    }
    llvm_unreachable("invalid packing method");
  }

  void verifyPacking(PackingMethod Method, const TestConfig &Config,
                     unsigned ExpectedRows,
                     std::initializer_list<ExpectedLocation> Locations) {
    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    ASSERT_EQ(Elements.size(), Locations.size());

    Expected<unsigned> Rows = pack(Method, Elements, Config);
    ASSERT_THAT_EXPECTED(Rows, Succeeded());
    EXPECT_EQ(*Rows, ExpectedRows);

    unsigned Index = 0;
    for (ExpectedLocation Location : Locations) {
      EXPECT_EQ(Elements[Index].StartRow, Location.Row) << "element " << Index;
      EXPECT_EQ(Elements[Index].StartCol, Location.Col) << "element " << Index;
      ++Index;
    }
  }

  void verifyMetadata(const SemanticSignatureElement &Before,
                      const SemanticSignatureElement &After) {
    EXPECT_EQ(After.SigId, Before.SigId);
    EXPECT_EQ(After.SemanticName, Before.SemanticName);
    EXPECT_EQ(After.CompType, Before.CompType);
    EXPECT_EQ(After.SemanticKind, Before.SemanticKind);
    EXPECT_EQ(After.SemanticIndices, Before.SemanticIndices);
    EXPECT_EQ(After.InterpMode, Before.InterpMode);
    EXPECT_EQ(After.Rows, Before.Rows);
    EXPECT_EQ(After.Cols, Before.Cols);
    EXPECT_EQ(After.UsageMask, Before.UsageMask);
    EXPECT_EQ(After.DynIndexMask, Before.DynIndexMask);
    EXPECT_EQ(After.GSStream, Before.GSStream);
  }

  void verifyPackingError(PackingMethod Method, const TestConfig &Config,
                          SignaturePackingError::ErrorKind ExpectedKind,
                          unsigned ExpectedElementIndex) {
    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    Expected<unsigned> Rows = pack(Method, Elements, Config);
    if (Rows) {
      ADD_FAILURE() << "expected a SignaturePackingError";
      return;
    }
    handleAllErrors(
        Rows.takeError(),
        [&](const SignaturePackingError &PackingErr) {
          EXPECT_EQ(PackingErr.getErrorKind(), ExpectedKind);
          EXPECT_EQ(PackingErr.getElementIndex(), ExpectedElementIndex);
        },
        [](const ErrorInfoBase &Other) {
          ADD_FAILURE() << "expected a SignaturePackingError, got: "
                        << Other.message();
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

  for (PackingMethod Method :
       {PackingMethod::Stacked, PackingMethod::PrefixStable,
        PackingMethod::Optimized}) {
    Config.IOTy = Method == PackingMethod::Stacked ? IOType::In : IOType::Out;
    verifyPacking(Method, Config, /*ExpectedRows=*/0, {});
  }
}

TEST_F(HLSLSemanticSignaturePackingTest, SkipsNotAllocatedElements) {
  // Semantics accessed through dedicated intrinsics do not consume signature
  // rows and remain unallocated. The elements around them are packed as if the
  // unallocated element was not declared at all.

  // struct VSIn {
  //   float2 A    : A;
  //   uint ViewID : SV_ViewID;
  //   float3 B    : B;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::In,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ViewID, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::U32, dxbc::PSV::InterpolationMode::Undefined},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Expected layout:
  // reg0: A.xy  | unused.zw
  // reg1: B.xyz | unused.w
  verifyPacking(PackingMethod::Stacked, Config, /*ExpectedRows=*/2,
                {{/*Row=*/0, /*Col=*/0}, Unallocated, {/*Row=*/1, /*Col=*/0}});

  // ViewID is also not allocated in pixel inputs. The surrounding elements
  // still use the same rows when packed prefix-stably.
  Config.ShaderStage = Triple::Pixel;
  verifyPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
                {{/*Row=*/0, /*Col=*/0}, Unallocated, {/*Row=*/1, /*Col=*/0}});
  // Optimized packing places the wider B before A.
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/2,
                {{/*Row=*/1, /*Col=*/0}, Unallocated, {/*Row=*/0, /*Col=*/0}});
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
  verifyPacking(
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
  verifyPacking(PackingMethod::Stacked, Config, /*ExpectedRows=*/4,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/2, /*Col=*/0},
                 {/*Row=*/3, /*Col=*/0}});

  // Prefix-stable and optimized layout for the corresponding vertex output:
  // reg0: A.x | B.y | C.z | D.w
  Config.IOTy = IOType::Out;
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPacking(Method, Config, /*ExpectedRows=*/1,
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
  verifyPacking(
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
       {PackingMethod::Stacked, PackingMethod::PrefixStable,
        PackingMethod::Optimized}) {
    Config.IOTy = Method == PackingMethod::Stacked ? IOType::In : IOType::Out;
    verifyPacking(Method, Config, /*ExpectedRows=*/MaxSignatureRows,
                  {{/*Row=*/0, /*Col=*/0}});
  }
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
       {PackingMethod::Stacked, PackingMethod::PrefixStable,
        PackingMethod::Optimized}) {
    Config.IOTy = Method == PackingMethod::Stacked ? IOType::In : IOType::Out;
    verifyPackingError(Method, Config, SignaturePackingError::SignatureOverflow,
                       /*ExpectedElementIndex=*/MaxSignatureRows);
  }
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
       {PackingMethod::Stacked, PackingMethod::PrefixStable,
        PackingMethod::Optimized}) {
    Config.IOTy = Method == PackingMethod::Stacked ? IOType::In : IOType::Out;
    verifyPackingError(Method, Config, SignaturePackingError::SignatureOverflow,
                       /*ExpectedElementIndex=*/0);
  }
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
       {PackingMethod::Stacked, PackingMethod::PrefixStable,
        PackingMethod::Optimized}) {
    Config.IOTy = Method == PackingMethod::Stacked ? IOType::In : IOType::Out;
    verifyPackingError(Method, Config, SignaturePackingError::SignatureOverflow,
                       /*ExpectedElementIndex=*/1);
  }
}

//===----------------------------------------------------------------------===//
// Basic prefix-stable packing tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableSupportedSignatures) {
  const struct {
    Triple::EnvironmentType Stage;
    IOType IOTy;
  } Signatures[] = {{Triple::Vertex, IOType::Out},
                    {Triple::Hull, IOType::In},
                    {Triple::Hull, IOType::Out},
                    {Triple::Hull, IOType::PatchConstantOrPrimitive},
                    {Triple::Domain, IOType::In},
                    {Triple::Domain, IOType::Out},
                    {Triple::Domain, IOType::PatchConstantOrPrimitive},
                    {Triple::Geometry, IOType::In},
                    {Triple::Geometry, IOType::Out},
                    {Triple::Pixel, IOType::In},
                    {Triple::Mesh, IOType::Out},
                    {Triple::Mesh, IOType::PatchConstantOrPrimitive}};
  for (const auto &Signature : Signatures) {
    SCOPED_TRACE(static_cast<unsigned>(Signature.Stage));
    SCOPED_TRACE(static_cast<unsigned>(Signature.IOTy));
    TestConfig Config(
        Signature.Stage, Signature.IOTy,
        {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
          dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined}});
    for (PackingMethod Method :
         {PackingMethod::PrefixStable, PackingMethod::Optimized}) {
      SCOPED_TRACE(static_cast<unsigned>(Method));
      verifyPacking(Method, Config, /*ExpectedRows=*/1,
                    {{/*Row=*/0, /*Col=*/0}});
    }
  }
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableOnlyUnallocatedElements) {
  TestConfig Config(
      Triple::Pixel, IOType::In,
      {{dxbc::PSV::SemanticKind::ViewID, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::U32, dxbc::PSV::InterpolationMode::Undefined}});

  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPacking(Method, Config, /*ExpectedRows=*/0, {Unallocated});
}

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
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPacking(Method, PrefixConfig, /*ExpectedRows=*/2,
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
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPacking(Method, ExtendedConfig, /*ExpectedRows=*/2,
                  {{/*Row=*/0, /*Col=*/0},
                   {/*Row=*/1, /*Col=*/0},
                   {/*Row=*/0, /*Col=*/3}});
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

  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized}) {
    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    Expected<unsigned> Rows = pack(Method, Elements, Config);
    ASSERT_THAT_EXPECTED(Rows, Succeeded());
    EXPECT_EQ(*Rows, MaxSignatureRows);

    for (unsigned I = 0; I != MaxSignatureRows; ++I) {
      EXPECT_EQ(Elements[I].StartRow, I) << "element " << I;
      EXPECT_EQ(Elements[I].StartCol, 0u) << "element " << I;
    }
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

  // Prefix-stable layout:
  // reg0: A.xy | F.zw
  // reg1: B.xy | D.zw
  // reg2: C.xyz | G.w
  // reg3: E.x | unused.yzw
  verifyPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/4,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/2, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/2},
                 {/*Row=*/3, /*Col=*/0},
                 {/*Row=*/0, /*Col=*/2},
                 {/*Row=*/2, /*Col=*/3}});

  // Optimized layout:
  // reg0: E.x | unused.yzw
  // reg1: C.xyz | G.w
  // reg2: A.xy | F.zw
  // reg3: B.xy | D.zw
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/4,
                {{/*Row=*/2, /*Col=*/0},
                 {/*Row=*/3, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/3, /*Col=*/2},
                 {/*Row=*/0, /*Col=*/0},
                 {/*Row=*/2, /*Col=*/2},
                 {/*Row=*/1, /*Col=*/3}});
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
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPacking(Method, Config, /*ExpectedRows=*/2,
                  {{/*Row=*/0, /*Col=*/0},
                   {/*Row=*/1, /*Col=*/0},
                   {/*Row=*/0, /*Col=*/2}});
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

  // Prefix-stable layout:
  // reg0: A.xy | C.zw
  // reg1: B.xy | unused.zw
  verifyPacking(
      PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/0, /*Col=*/2}});

  // Optimized layout:
  // reg0: B.xy | unused.zw
  // reg1: A.xy | C.zw
  verifyPacking(
      PackingMethod::Optimized, Config, /*ExpectedRows=*/2,
      {{/*Row=*/1, /*Col=*/0}, {/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/2}});
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

  // Prefix-stable layout:
  // reg0: A.x | B.yzw
  verifyPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/1,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/0, /*Col=*/1}});

  // Optimized layout:
  // reg0: B.xyz | A.w
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/1,
                {{/*Row=*/0, /*Col=*/3}, {/*Row=*/0, /*Col=*/0}});
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
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPacking(Method, Config, /*ExpectedRows=*/2,
                  {{/*Row=*/0, /*Col=*/0},
                   {/*Row=*/0, /*Col=*/1},
                   {/*Row=*/1, /*Col=*/0}});
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
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPacking(Method, Config, /*ExpectedRows=*/2,
                  {{/*Row=*/0, /*Col=*/0},
                   {/*Row=*/0, /*Col=*/2},
                   {/*Row=*/1, /*Col=*/0}});
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

  // Prefix-stable layout:
  // reg0: A.xy | B.z | unused.w
  // reg1: C.x  | unused.yzw
  verifyPacking(
      PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/0, /*Col=*/2}, {/*Row=*/1, /*Col=*/0}});

  // Optimized layout:
  // reg0: A.xy | C.z | unused.w
  // reg1: B.x | unused.yzw
  verifyPacking(
      PackingMethod::Optimized, Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/0, /*Col=*/2}});
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

  // Prefix-stable layout:
  // reg0: A.xy | C.zw
  // reg1: B.xy | unused.zw
  verifyPacking(
      PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/0, /*Col=*/2}});

  // Optimized layout:
  // reg0: B.xy | A.zw
  // reg1: C.xy | unused.zw
  verifyPacking(
      PackingMethod::Optimized, Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/2}, {/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}});
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
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPacking(Method, Config, /*ExpectedRows=*/3,
                  {{/*Row=*/0, /*Col=*/0},
                   {/*Row=*/1, /*Col=*/0},
                   {/*Row=*/2, /*Col=*/0}});
}

//===----------------------------------------------------------------------===//
// Prefix-stable component ordering tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableSystemValueOrdering) {
  // System values may be co-packed to the right of arbitrary values, and a
  // system generated value may be co-packed to the right of both.

  // struct PSIn {
  //   uint A             : A;
  //   float Position     : SV_Position;
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
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPacking(Method, Config, /*ExpectedRows=*/1,
                  {{/*Row=*/0, /*Col=*/0},
                   {/*Row=*/0, /*Col=*/1},
                   {/*Row=*/0, /*Col=*/2}});
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

  // Prefix-stable layout:
  // reg0: Position.xy | unused.zw
  // reg1: A.xy        | unused.zw
  verifyPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}});

  // Optimized layout:
  // reg0: A.xy | Position.zw
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/1,
                {{/*Row=*/0, /*Col=*/2}, {/*Row=*/0, /*Col=*/0}});
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

  // Prefix-stable layout:
  // reg0: IsFrontFace.x | unused.yzw
  // reg1: A.x | Position.y | unused.zw
  verifyPacking(
      PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/1, /*Col=*/1}});

  // Optimized layout:
  // reg0: A.x | Position.y | IsFrontFace.z | unused.w
  verifyPacking(
      PackingMethod::Optimized, Config, /*ExpectedRows=*/1,
      {{/*Row=*/0, /*Col=*/2}, {/*Row=*/0, /*Col=*/0}, {/*Row=*/0, /*Col=*/1}});
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
  verifyPackingError(PackingMethod::PrefixStable, Config,
                     SignaturePackingError::SignatureOverflow,
                     /*ExpectedElementIndex=*/MaxSignatureRows);

  // Optimized layout:
  // reg0:    A0.xyz | IsFrontFace.w
  // reg1-31: A1-A31.xyz | unused.w
  SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
  Expected<unsigned> Rows = pack(PackingMethod::Optimized, Elements, Config);
  ASSERT_THAT_EXPECTED(Rows, Succeeded());
  EXPECT_EQ(*Rows, MaxSignatureRows);
  EXPECT_EQ(Elements[0].StartRow, 0u);
  EXPECT_EQ(Elements[0].StartCol, 3u);
  for (unsigned I = 0; I != MaxSignatureRows; ++I) {
    EXPECT_EQ(Elements[I + 1].StartRow, I) << "element " << I + 1;
    EXPECT_EQ(Elements[I + 1].StartCol, 0u) << "element " << I + 1;
  }
}

//===----------------------------------------------------------------------===//
// Prefix-stable dynamic indexing tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableIndexedRanges) {
  // An element with multiple rows occupies the same columns of a contiguous
  // range of rows, and other elements may be co-packed into the columns those
  // rows have left. B extends the indexed range of the rows it shares with A.
  // A system value cannot be placed in a dynamically indexable row, so
  // Position starts a new register after both arrays.

  // struct VSOut {
  //   float2 A[2]    : A;
  //   float B[3]     : B;
  //   float C        : C;
  //   float Position : SV_Position;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/2, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/3, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Position, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Prefix-stable layout:
  // reg0: A[0].xy | B[0].z | C.w
  // reg1: A[1].xy | B[1].z | unused.w
  // reg2: unused.xy | B[2].z | unused.w
  // reg3: Position.x | unused.yzw
  verifyPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/4,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/0, /*Col=*/2},
                 {/*Row=*/0, /*Col=*/3},
                 {/*Row=*/3, /*Col=*/0}});

  // Optimized layout: the longer indexed range is packed first.
  // reg0: B[0].x | A[0].yz | C.w
  // reg1: B[1].x | A[1].yz | unused.w
  // reg2: B[2].x | unused.yzw
  // reg3: Position.x | unused.yzw
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/4,
                {{/*Row=*/0, /*Col=*/1},
                 {/*Row=*/0, /*Col=*/0},
                 {/*Row=*/0, /*Col=*/3},
                 {/*Row=*/3, /*Col=*/0}});
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

  // Prefix-stable layout:
  // reg0: Position.x | unused.yzw
  // reg1: A[0].xyz   | unused.w
  // reg2: A[1].xyz   | unused.w
  verifyPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/3,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}});

  // Optimized layout:
  // reg0: A[0].xyz | unused.w
  // reg1: A[1].xyz | unused.w
  // reg2: Position.x | unused.yzw
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/3,
                {{/*Row=*/2, /*Col=*/0}, {/*Row=*/0, /*Col=*/0}});
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

  // Prefix-stable and optimized layout:
  // reg0: Data[0].xyz | TessFactor[0].w
  // reg1: Data[1].xyz | TessFactor[1].w
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPacking(Method, Config, /*ExpectedRows=*/2,
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

  // Prefix-stable layout:
  // reg0: TessFactor.x | unused.yzw
  // reg1: Data.xyz     | unused.w
  verifyPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}});

  // Optimized layout:
  // reg0: Data.xyz | TessFactor.w
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/1,
                {{/*Row=*/0, /*Col=*/3}, {/*Row=*/0, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest,
       PrefixStableSingleRowTessFactorAfterArbitrary) {
  // A single-row tess factor can follow arbitrary data in the same row.
  // Unlike an indexed tess factor, it need not occupy the last column.

  // struct PatchConstants {
  //   float2 Data      : DATA;
  //   float TessFactor : SV_TessFactor;
  // };
  TestConfig Config(
      Triple::Hull, IOType::PatchConstantOrPrimitive,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined},
       {dxbc::PSV::SemanticKind::TessFactor, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined}});

  // Expected layout:
  // reg0: Data.xy | TessFactor.z | unused.w
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPacking(Method, Config, /*ExpectedRows=*/1,
                  {{/*Row=*/0, /*Col=*/0}, {/*Row=*/0, /*Col=*/2}});
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

  // Prefix-stable layout:
  // reg0: Data[0].xyz | unused.w
  // reg1: Data[1].xyz | unused.w
  // reg2: Data[2].xyz | unused.w
  // reg3: unused.xyz  | TessFactor[0].w
  // reg4: unused.xyz  | TessFactor[1].w
  verifyPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/5,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/3, /*Col=*/3}});

  // Optimized layout:
  // reg0: unused.xyz  | TessFactor[0].w
  // reg1: unused.xyz  | TessFactor[1].w
  // reg2: Data[0].xyz | unused.w
  // reg3: Data[1].xyz | unused.w
  // reg4: Data[2].xyz | unused.w
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/5,
                {{/*Row=*/2, /*Col=*/0}, {/*Row=*/0, /*Col=*/3}});
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

  // Prefix-stable layout:
  // reg0: First.xyz       | WithFirst.w
  // reg1: Clip0.x         | Cull1.yzw
  // reg2: Cull0.x         | Clip1.yz | unused.w
  // reg3: AfterClipCull.x | unused.yzw
  verifyPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/4,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/1},
                 {/*Row=*/2, /*Col=*/0},
                 {/*Row=*/2, /*Col=*/1},
                 {/*Row=*/0, /*Col=*/3},
                 {/*Row=*/3, /*Col=*/0}});

  // Optimized layout:
  // reg0: First.xyz       | WithFirst.w
  // reg1: AfterClipCull.x | Clip1.yz | Cull0.w
  // reg2: Cull1.xyz       | Clip0.w
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/3,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/2, /*Col=*/3},
                 {/*Row=*/2, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/3},
                 {/*Row=*/1, /*Col=*/1},
                 {/*Row=*/0, /*Col=*/3},
                 {/*Row=*/1, /*Col=*/0}});
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

  // Prefix-stable layout:
  // reg0: First.xyz       | WithFirst.w
  // reg1: Clip0.x         | Cull1[0].yz | Clip1.w
  // reg2: unused.x        | Cull1[1].yz | unused.w
  // reg3: AfterClipCull.x | unused.yzw
  verifyPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/4,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/1},
                 {/*Row=*/1, /*Col=*/3},
                 {/*Row=*/0, /*Col=*/3},
                 {/*Row=*/3, /*Col=*/0}});

  // Optimized layout:
  // reg0: First.xyz       | WithFirst.w
  // reg1: AfterClipCull.x | Cull1[0].yz | Clip0.w
  // reg2: Clip1.x         | Cull1[1].yz | unused.w
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/3,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/3},
                 {/*Row=*/1, /*Col=*/1},
                 {/*Row=*/2, /*Col=*/0},
                 {/*Row=*/0, /*Col=*/3},
                 {/*Row=*/1, /*Col=*/0}});
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

  // Prefix-stable layout:
  // reg0: First.xyz       | WithFirst.w
  // reg1: Clip0.x         | Cull1[0].yz | Clip1[0].w
  // reg2: unused.x        | Cull1[1].yz | Clip1[1].w
  // reg3: AfterClipCull.x | unused.yzw
  verifyPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/4,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/1},
                 {/*Row=*/1, /*Col=*/3},
                 {/*Row=*/0, /*Col=*/3},
                 {/*Row=*/3, /*Col=*/0}});

  // Optimized layout:
  // reg0: First.xyz       | WithFirst.w
  // reg1: AfterClipCull.x | Cull1[0].yz | Clip1[0].w
  // reg2: Clip0.x         | Cull1[1].yz | Clip1[1].w
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/3,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/2, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/1},
                 {/*Row=*/1, /*Col=*/3},
                 {/*Row=*/0, /*Col=*/3},
                 {/*Row=*/1, /*Col=*/0}});
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

  // Prefix-stable and optimized layout:
  // reg0: Clip0.xyzw
  // reg1: Cull0.xyzw
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPacking(Method, Config, /*ExpectedRows=*/MaxClipCullRows,
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

  // Prefix-stable layout:
  // reg0:     Clip0.xyz | unused.w
  // reg1-30:  A[0-29].xyzw
  // reg31:    Cull0.xyz | unused.w
  verifyPacking(PackingMethod::PrefixStable, Config,
                /*ExpectedRows=*/MaxSignatureRows,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/31, /*Col=*/0}});

  // Optimized layout:
  // reg0-29: A[0-29].xyzw
  // reg30:   Clip0.xyz | unused.w
  // reg31:   Cull0.xyz | unused.w
  verifyPacking(PackingMethod::Optimized, Config,
                /*ExpectedRows=*/MaxSignatureRows,
                {{/*Row=*/30, /*Col=*/0},
                 {/*Row=*/0, /*Col=*/0},
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

  // Prefix-stable and optimized layout:
  // reg0: Clip0.xyz | unused.w
  // reg1: Clip1.xyz | unused.w
  // reg2: Cull0.xyz | unused.w
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPacking(Method, Config, /*ExpectedRows=*/3,
                  {{/*Row=*/0, /*Col=*/0},
                   {/*Row=*/1, /*Col=*/0},
                   {/*Row=*/2, /*Col=*/0}});
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

  // Prefix-stable layout:
  // reg0: First.xyz | unused.w
  // reg1: Clip0.x   | Cull1[0].yz | unused.w
  // reg2: unused.x  | Cull1[1].yz | unused.w
  verifyPacking(
      PackingMethod::PrefixStable, PrefixConfig, /*ExpectedRows=*/3,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}, {/*Row=*/1, /*Col=*/1}});

  // Optimized layout:
  // reg0: First.xyz | unused.w
  // reg1: Cull1[0].xy | Clip0.z | unused.w
  // reg2: Cull1[1].xy | unused.zw
  verifyPacking(
      PackingMethod::Optimized, PrefixConfig, /*ExpectedRows=*/3,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/2}, {/*Row=*/1, /*Col=*/0}});

  // struct Extended {
  //   float3 First         : First;
  //   float  Clip0         : SV_ClipDistance0;
  //   float2 Cull1[2]      : SV_CullDistance1;
  //   float  Clip1         : SV_ClipDistance1;
  //   float  WithFirst     : WithFirst;
  //   float  AfterClipCull : AfterClipCull;
  // };
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

  // The complete layout is covered by PrefixStableIndexedClipCull. Here,
  // check that each appended element leaves all earlier locations unchanged.
  SmallVector<SemanticSignatureElement> PreviousElements =
      makeSignature(PrefixConfig);
  ASSERT_THAT_EXPECTED(
      pack(PackingMethod::PrefixStable, PreviousElements, PrefixConfig),
      Succeeded());
  for (unsigned Count = PrefixConfig.Elements.size() + 1;
       Count <= ExtendedConfig.Elements.size(); ++Count) {
    SCOPED_TRACE(Count);
    TestConfig Config = ExtendedConfig;
    Config.Elements.resize(Count);
    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    ASSERT_THAT_EXPECTED(pack(PackingMethod::PrefixStable, Elements, Config),
                         Succeeded());
    for (unsigned I = 0; I != PreviousElements.size(); ++I) {
      EXPECT_EQ(Elements[I].StartRow, PreviousElements[I].StartRow)
          << "element " << I;
      EXPECT_EQ(Elements[I].StartCol, PreviousElements[I].StartCol)
          << "element " << I;
    }
    PreviousElements = std::move(Elements);
  }
}

TEST_F(HLSLSemanticSignaturePackingTest,
       PrefixStableRejectsBlockedClipCullExtension) {
  // The signature has room, but extending Clip's reserved row for the indexed
  // Cull element would overlap Color. Report adjacency rather than capacity.

  // struct VSOut {
  //   float Clip    : SV_ClipDistance0;
  //   float4 Color  : COLOR;
  //   float Cull[2] : SV_CullDistance0;
  // };
  TestConfig Config(
      Triple::Vertex, IOType::Out,
      {{dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/4,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/2, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});
  verifyPackingError(PackingMethod::PrefixStable, Config,
                     SignaturePackingError::ClipCullNotAdjacent,
                     /*ExpectedElementIndex=*/2);

  SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
  EXPECT_THAT_EXPECTED(
      pack(PackingMethod::PrefixStable, Elements, Config),
      FailedWithMessage("indexed clip/cull elements require adjacent signature "
                        "rows (element 2)"));
  EXPECT_EQ(Elements[0].StartRow, 0u);
  EXPECT_EQ(Elements[0].StartCol, 0u);
  EXPECT_EQ(Elements[1].StartRow, 1u);
  EXPECT_EQ(Elements[1].StartCol, 0u);
  EXPECT_EQ(Elements[2].StartRow, UnallocatedRow);
  EXPECT_EQ(Elements[2].StartCol, UnallocatedCol);

  // Optimized packing places Color first and reserves adjacent rows for Cull.
  // reg0: Color.xyzw
  // reg1: Cull[0].x | Clip.y | unused.zw
  // reg2: Cull[1].x | unused.yzw
  verifyPacking(
      PackingMethod::Optimized, Config, /*ExpectedRows=*/3,
      {{/*Row=*/1, /*Col=*/1}, {/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest,
       PrefixStableRejectsNonAdjacentClipCullRows) {
  // These seven clip/cull components fit in two rows, but Color separates the
  // reserved rows. Clip1 cannot span them without moving existing elements.

  // struct VSOut {
  //   float3 Clip0   : SV_ClipDistance0;
  //   float4 Color   : COLOR;
  //   float2 Cull0   : SV_CullDistance0;
  //   float Clip1[2] : SV_ClipDistance1;
  // };
  TestConfig Config(
      Triple::Vertex, IOType::Out,
      {{dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/4,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/2, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear,
        /*SemanticIndex=*/1}});
  verifyPackingError(PackingMethod::PrefixStable, Config,
                     SignaturePackingError::ClipCullNotAdjacent,
                     /*ExpectedElementIndex=*/3);

  SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
  EXPECT_THAT_EXPECTED(
      pack(PackingMethod::PrefixStable, Elements, Config),
      FailedWithMessage("indexed clip/cull elements require adjacent signature "
                        "rows (element 3)"));
  for (unsigned I = 0; I != 3; ++I) {
    EXPECT_EQ(Elements[I].StartRow, I) << "element " << I;
    EXPECT_EQ(Elements[I].StartCol, 0u) << "element " << I;
  }
  EXPECT_EQ(Elements[3].StartRow, UnallocatedRow);
  EXPECT_EQ(Elements[3].StartCol, UnallocatedCol);

  // Optimized packing places the indexed clip element before scalar clip/cull
  // values, so Color cannot separate its reserved rows.
  // reg0: Color.xyzw
  // reg1: Clip1[0].x | Clip0.yzw
  // reg2: Clip1[1].x | Cull0.yz | unused.w
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/3,
                {{/*Row=*/1, /*Col=*/1},
                 {/*Row=*/0, /*Col=*/0},
                 {/*Row=*/2, /*Col=*/1},
                 {/*Row=*/1, /*Col=*/0}});

  // Declaring Color last leaves adjacent clip/cull rows and the same
  // components fit. This is an adjacency failure, not a clip/cull overflow.
  ElementConfig Color = Config.Elements[1];
  Config.Elements.erase(Config.Elements.begin() + 1);
  Config.Elements.push_back(Color);
  verifyPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/3,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/0, /*Col=*/3},
                 {/*Row=*/2, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest,
       PrefixStableClipCullExtensionAtSignatureBoundary) {
  // A two-row clip/cull range can end at the last signature row, but cannot
  // extend beyond it. The latter is still a genuine capacity overflow.

  // struct VSOut {
  //   float4 Data[30] : DATA;
  //   float Clip      : SV_ClipDistance0;
  //   float Cull[2]   : SV_CullDistance0;
  // };
  TestConfig Config(
      Triple::Vertex, IOType::Out,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/MaxSignatureRows - 2,
        /*Cols=*/4, dxil::ElementType::F32,
        dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/2, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});
  verifyPacking(PackingMethod::PrefixStable, Config,
                /*ExpectedRows=*/MaxSignatureRows,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/MaxSignatureRows - 2, /*Col=*/0},
                 {/*Row=*/MaxSignatureRows - 2, /*Col=*/1}});

  // Optimized packing reserves Cull's two rows before placing scalar Clip.
  verifyPacking(PackingMethod::Optimized, Config,
                /*ExpectedRows=*/MaxSignatureRows,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/MaxSignatureRows - 2, /*Col=*/1},
                 {/*Row=*/MaxSignatureRows - 2, /*Col=*/0}});

  // Extending Data to 31 rows leaves no room for Cull's two-row range.
  ++Config.Elements[0].Rows;
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPackingError(Method, Config, SignaturePackingError::SignatureOverflow,
                       /*ExpectedElementIndex=*/2);
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableRejectsClipCullOverflow) {
  // Clip and cull distances may use at most eight components, shared between
  // them, so nine components cannot be packed.

  // struct VSOut {
  //   float3 Clip0 : SV_ClipDistance0;
  //   float3 Cull0 : SV_CullDistance0;
  //   float3 Clip1 : SV_ClipDistance1;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized}) {
    SCOPED_TRACE(static_cast<unsigned>(Method));
    verifyPackingError(Method, Config, SignaturePackingError::ClipCullOverflow,
                       /*ExpectedElementIndex=*/2);

    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    EXPECT_THAT_EXPECTED(pack(Method, Elements, Config),
                         FailedWithMessage("clip/cull elements do not fit in " +
                                           std::to_string(MaxClipCullRows) +
                                           " rows (element 2)"));
  }
}

TEST_F(HLSLSemanticSignaturePackingTest,
       PrefixStableRejectsUnpackableClipCull) {
  // These clip and cull distances fit in eight components, but they cannot be
  // split across the two registers available to them.

  // struct VSOut {
  //   float3 Clip0 : SV_ClipDistance0;
  //   float3 Cull0 : SV_CullDistance0;
  //   float2 Clip1 : SV_ClipDistance1;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Vertex, IOType::Out,
      /*UseNative16BitTypes=*/false,
      {{dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPackingError(Method, Config, SignaturePackingError::ClipCullOverflow,
                       /*ExpectedElementIndex=*/2);
}

//===----------------------------------------------------------------------===//
// Prefix-stable failure state tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest,
       PrefixStablePreservesPartialAllocation) {
  for (bool IsClipCull : {false, true}) {
    const auto Kind = IsClipCull ? dxbc::PSV::SemanticKind::ClipDistance
                                 : dxbc::PSV::SemanticKind::Arbitrary;
    const unsigned RowCount = IsClipCull ? MaxClipCullRows : MaxSignatureRows;
    TestConfig Config(
        Triple::Vertex, IOType::Out,
        {{Kind, RowCount, /*Cols=*/MaxSignatureCols, dxil::ElementType::F32,
          dxbc::PSV::InterpolationMode::Linear},
         {Kind, /*Rows=*/1, /*Cols=*/1, dxil::ElementType::F32,
          dxbc::PSV::InterpolationMode::Linear},
         {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
          dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});
    verifyPackingError(PackingMethod::PrefixStable, Config,
                       IsClipCull ? SignaturePackingError::ClipCullOverflow
                                  : SignaturePackingError::SignatureOverflow,
                       /*ExpectedElementIndex=*/1);

    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    EXPECT_THAT_EXPECTED(pack(PackingMethod::PrefixStable, Elements, Config),
                         Failed<SignaturePackingError>());
    EXPECT_EQ(Elements[0].StartRow, 0u);
    EXPECT_EQ(Elements[0].StartCol, 0u);
    for (unsigned I = 1; I != Elements.size(); ++I) {
      EXPECT_EQ(Elements[I].StartRow, UnallocatedRow) << "element " << I;
      EXPECT_EQ(Elements[I].StartCol, UnallocatedCol) << "element " << I;
    }
  }
}

TEST_F(HLSLSemanticSignaturePackingTest, OptimizedPreservesPartialAllocation) {
  // Declaration order is Small, Full, Last. Optimized packing places Full
  // first, then fails on Small. Report Small's original index (zero), not its
  // index in packing order (one). Preserve Full's allocation for ordinary
  // elements, but leave the entire clip/cull phase unallocated on failure.
  //
  // struct VSOut {
  //   float2 Small    : Small;
  //   float4 Full[32] : Full;
  //   float Last      : Last;
  // };
  // The clip/cull case uses clip-distance semantics and Full[2] instead.
  for (bool IsClipCull : {false, true}) {
    SCOPED_TRACE(IsClipCull);
    const auto Kind = IsClipCull ? dxbc::PSV::SemanticKind::ClipDistance
                                 : dxbc::PSV::SemanticKind::Arbitrary;
    const unsigned RowCount = IsClipCull ? MaxClipCullRows : MaxSignatureRows;
    TestConfig Config(
        Triple::Vertex, IOType::Out,
        {{Kind, /*Rows=*/1, /*Cols=*/2, dxil::ElementType::F32,
          dxbc::PSV::InterpolationMode::Linear},
         {Kind, RowCount, /*Cols=*/MaxSignatureCols, dxil::ElementType::F32,
          dxbc::PSV::InterpolationMode::Linear},
         {Kind, /*Rows=*/1, /*Cols=*/1, dxil::ElementType::F32,
          dxbc::PSV::InterpolationMode::Linear}});
    verifyPackingError(PackingMethod::Optimized, Config,
                       IsClipCull ? SignaturePackingError::ClipCullOverflow
                                  : SignaturePackingError::SignatureOverflow,
                       /*ExpectedElementIndex=*/0);

    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    EXPECT_THAT_EXPECTED(
        pack(PackingMethod::Optimized, Elements, Config),
        FailedWithMessage(
            std::string(IsClipCull ? "clip/cull elements do not fit in "
                                   : "signature elements do not fit in ") +
            std::to_string(RowCount) + " rows (element 0)"));
    EXPECT_EQ(Elements[1].StartRow, IsClipCull ? UnallocatedRow : 0u);
    EXPECT_EQ(Elements[1].StartCol, IsClipCull ? UnallocatedCol : 0u);
    for (unsigned I : {0u, 2u}) {
      EXPECT_EQ(Elements[I].StartRow, UnallocatedRow) << "element " << I;
      EXPECT_EQ(Elements[I].StartCol, UnallocatedCol) << "element " << I;
    }
    for (unsigned I = 0; I != Elements.size(); ++I) {
      EXPECT_EQ(Elements[I].SigId, I) << "element " << I;
      EXPECT_EQ(Elements[I].Rows, Config.Elements[I].Rows) << "element " << I;
      EXPECT_EQ(Elements[I].Cols, Config.Elements[I].Cols) << "element " << I;
    }
  }
}

TEST_F(HLSLSemanticSignaturePackingTest,
       OptimizedSingleRowClipCullFailureIsAtomic) {
  // struct VSOut {
  //   float2 Clip1    : SV_ClipDistance1;
  //   float4 Fill[31] : FILL;
  //   float Cull      : SV_CullDistance0;
  //   float3 Clip0    : SV_ClipDistance0;
  // };
  // The first bundle (Clip0 + Cull) fits in row 31, but Clip1's bundle fails.
  // Neither bundle is published; only the preceding Fill allocation remains.
  TestConfig Config(
      Triple::Vertex, IOType::Out,
      {{dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear,
        /*SemanticIndex=*/1},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/MaxSignatureRows - 1,
        /*Cols=*/4, dxil::ElementType::F32,
        dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});
  verifyPackingError(PackingMethod::Optimized, Config,
                     SignaturePackingError::SignatureOverflow,
                     /*ExpectedElementIndex=*/0);
  SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
  const SmallVector<SemanticSignatureElement> Before = Elements;
  EXPECT_THAT_EXPECTED(pack(PackingMethod::Optimized, Elements, Config),
                       FailedWithMessage("signature elements do not fit in " +
                                         std::to_string(MaxSignatureRows) +
                                         " rows (element 0)"));
  EXPECT_EQ(Elements[1].StartRow, 0u);
  EXPECT_EQ(Elements[1].StartCol, 0u);
  for (unsigned I : {0u, 2u, 3u}) {
    EXPECT_EQ(Elements[I].StartRow, UnallocatedRow) << "element " << I;
    EXPECT_EQ(Elements[I].StartCol, UnallocatedCol) << "element " << I;
  }
  for (unsigned I = 0; I != Elements.size(); ++I) {
    SCOPED_TRACE(I);
    verifyMetadata(Before[I], Elements[I]);
  }
}

TEST_F(HLSLSemanticSignaturePackingTest,
       OptimizedIndexedClipCullFailureIsAtomic) {
  // struct VSOut {
  //   float2 Cull     : SV_CullDistance0;
  //   float4 Fill[30] : FILL;
  //   float A[2]      : A;
  //   float2 Clip[2]  : SV_ClipDistance0;
  // };
  // The final pair fits Clip in yz but not Cull.xy. Do not retain Clip's
  // speculative allocation. The error identifies Clip, first in group order.
  TestConfig Config(
      Triple::Vertex, IOType::Out,
      {{dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/MaxSignatureRows - 2,
        /*Cols=*/4, dxil::ElementType::F32,
        dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/2, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/2, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});
  verifyPackingError(PackingMethod::Optimized, Config,
                     SignaturePackingError::SignatureOverflow,
                     /*ExpectedElementIndex=*/3);
  SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
  const SmallVector<SemanticSignatureElement> Before = Elements;
  EXPECT_THAT_EXPECTED(pack(PackingMethod::Optimized, Elements, Config),
                       FailedWithMessage("signature elements do not fit in " +
                                         std::to_string(MaxSignatureRows) +
                                         " rows (element 3)"));
  EXPECT_EQ(Elements[1].StartRow, 0u);
  EXPECT_EQ(Elements[1].StartCol, 0u);
  EXPECT_EQ(Elements[2].StartRow, MaxSignatureRows - 2);
  EXPECT_EQ(Elements[2].StartCol, 0u);
  for (unsigned I : {0u, 3u}) {
    EXPECT_EQ(Elements[I].StartRow, UnallocatedRow) << "element " << I;
    EXPECT_EQ(Elements[I].StartCol, UnallocatedCol) << "element " << I;
  }
  for (unsigned I = 0; I != Elements.size(); ++I) {
    SCOPED_TRACE(I);
    verifyMetadata(Before[I], Elements[I]);
  }
}

TEST_F(HLSLSemanticSignaturePackingTest,
       OptimizedClipCullFailureIsAtomicAcrossStreams) {
  // struct Stream0 {
  //   float Clip : SV_ClipDistance;
  // };
  //
  // struct Stream1 {
  //   float4 Fill[32]          : FILL;
  //   float Cull[Failure.Rows] : SV_CullDistance;
  // };
  const struct {
    unsigned Rows;
    unsigned Stream;
    SignaturePackingError::ErrorKind Kind;
  } Failures[] = {
      {1, 1, SignaturePackingError::SignatureOverflow},
      {3, 1, SignaturePackingError::ClipCullOverflow},
      {1, MaxGeometryStreams, SignaturePackingError::InvalidGeometryStream}};
  for (const auto &Failure : Failures) {
    SCOPED_TRACE(Failure.Kind);
    TestConfig Config(
        Triple::Geometry, IOType::Out,
        {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/MaxSignatureRows,
          /*Cols=*/4, dxil::ElementType::F32,
          dxbc::PSV::InterpolationMode::Linear,
          /*SemanticIndex=*/0, /*GSStream=*/1},
         {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/1,
          dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear,
          /*SemanticIndex=*/0, /*GSStream=*/0},
         {dxbc::PSV::SemanticKind::CullDistance, Failure.Rows, /*Cols=*/1,
          dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear,
          /*SemanticIndex=*/0, Failure.Stream}});
    // Stream 0 fits, but failure in stream 1 (or an invalid stream) must leave
    // every clip/cull element unallocated, without undoing the earlier Fill.
    //
    // Partial optimized layout after failure:
    // stream0: Clip unallocated
    // stream1 reg0-31: Fill[0-31].xyzw
    // Cull remains unallocated.
    verifyPackingError(PackingMethod::Optimized, Config, Failure.Kind,
                       /*ExpectedElementIndex=*/2);
    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    const SmallVector<SemanticSignatureElement> Before = Elements;
    EXPECT_THAT_EXPECTED(pack(PackingMethod::Optimized, Elements, Config),
                         Failed<SignaturePackingError>());
    EXPECT_EQ(Elements[0].StartRow, 0u);
    EXPECT_EQ(Elements[0].StartCol, 0u);
    for (unsigned I : {1u, 2u}) {
      EXPECT_EQ(Elements[I].StartRow, UnallocatedRow) << "element " << I;
      EXPECT_EQ(Elements[I].StartCol, UnallocatedCol) << "element " << I;
    }
    for (unsigned I = 0; I != Elements.size(); ++I) {
      SCOPED_TRACE(I);
      verifyMetadata(Before[I], Elements[I]);
    }
  }
}

TEST_F(HLSLSemanticSignaturePackingTest,
       OptimizedClipCullFailureSkipsLaterGroups) {
  // struct PSIn {
  //   nointerpolation float A       : A;
  //   nointerpolation float Cull[3] : SV_CullDistance;
  //   bool IsFrontFace              : SV_IsFrontFace;
  // };
  TestConfig Config(
      Triple::Pixel, IOType::In,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/MaxClipCullRows + 1,
        /*Cols=*/1, dxil::ElementType::F32,
        dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::IsFrontFace, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::I1, dxbc::PSV::InterpolationMode::Constant}});

  // Partial optimized layout after failure:
  // reg0: A.x | unused.yzw
  // Cull and IsFrontFace remain unallocated.
  verifyPackingError(PackingMethod::Optimized, Config,
                     SignaturePackingError::ClipCullOverflow,
                     /*ExpectedElementIndex=*/1);
  SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
  EXPECT_THAT_EXPECTED(pack(PackingMethod::Optimized, Elements, Config),
                       Failed<SignaturePackingError>());
  EXPECT_EQ(Elements[0].StartRow, 0u);
  EXPECT_EQ(Elements[0].StartCol, 0u);
  for (unsigned I : {1u, 2u}) {
    EXPECT_EQ(Elements[I].StartRow, UnallocatedRow) << "element " << I;
    EXPECT_EQ(Elements[I].StartCol, UnallocatedCol) << "element " << I;
  }
}

TEST_F(HLSLSemanticSignaturePackingTest,
       OptimizedLaterFailurePreservesClipCullAllocations) {
  // struct PSIn {
  //   nointerpolation float4 Fill[31] : FILL;
  //   nointerpolation float4 Cull     : SV_CullDistance;
  //   bool IsFrontFace                 : SV_IsFrontFace;
  // };
  TestConfig Config(
      Triple::Pixel, IOType::In,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/MaxSignatureRows - 1,
        /*Cols=*/4, dxil::ElementType::F32,
        dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/4,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::IsFrontFace, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::I1, dxbc::PSV::InterpolationMode::Constant}});

  // Partial optimized layout after failure:
  // reg0-30: Fill[0-30].xyzw
  // reg31: Cull.xyzw
  // IsFrontFace remains unallocated.
  verifyPackingError(PackingMethod::Optimized, Config,
                     SignaturePackingError::SignatureOverflow,
                     /*ExpectedElementIndex=*/2);
  SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
  EXPECT_THAT_EXPECTED(pack(PackingMethod::Optimized, Elements, Config),
                       Failed<SignaturePackingError>());
  EXPECT_EQ(Elements[0].StartRow, 0u);
  EXPECT_EQ(Elements[0].StartCol, 0u);
  EXPECT_EQ(Elements[1].StartRow, MaxSignatureRows - 1);
  EXPECT_EQ(Elements[1].StartCol, 0u);
  EXPECT_EQ(Elements[2].StartRow, UnallocatedRow);
  EXPECT_EQ(Elements[2].StartCol, UnallocatedCol);
}

//===----------------------------------------------------------------------===//
// Prefix-stable geometry stream tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableRejectsInvalidStreams) {
  // Validate stream indices before indexing packing state, including in
  // release builds. Only geometry outputs may use a nonzero stream.
  const struct {
    Triple::EnvironmentType Stage;
    IOType IOTy;
    unsigned InvalidStream;
  } Signatures[] = {{Triple::Geometry, IOType::Out, MaxGeometryStreams},
                    {Triple::Geometry, IOType::Out, ~uint32_t{0}},
                    {Triple::Geometry, IOType::In, 1},
                    {Triple::Vertex, IOType::Out, 1},
                    {Triple::Pixel, IOType::In, 1}};
  for (const auto &Signature : Signatures) {
    SCOPED_TRACE(static_cast<unsigned>(Signature.Stage));
    SCOPED_TRACE(static_cast<unsigned>(Signature.IOTy));
    SCOPED_TRACE(Signature.InvalidStream);
    TestConfig Config(
        Signature.Stage, Signature.IOTy,
        {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
          dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
         {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
          dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear,
          /*SemanticIndex=*/1, Signature.InvalidStream},
         {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/4,
          dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear,
          /*SemanticIndex=*/2}});
    for (PackingMethod Method :
         {PackingMethod::PrefixStable, PackingMethod::Optimized}) {
      SCOPED_TRACE(static_cast<unsigned>(Method));
      verifyPackingError(Method, Config,
                         SignaturePackingError::InvalidGeometryStream,
                         /*ExpectedElementIndex=*/1);

      SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
      EXPECT_THAT_EXPECTED(
          pack(Method, Elements, Config),
          FailedWithMessage(
              "signature element has an invalid geometry stream: expected an "
              "index less than " +
              std::to_string(MaxGeometryStreams) +
              " for geometry outputs, or zero otherwise (element 1)"));
      const bool IsOptimized = Method == PackingMethod::Optimized;
      EXPECT_EQ(Elements[0].StartRow, IsOptimized ? 1u : 0u);
      EXPECT_EQ(Elements[0].StartCol, 0u);
      EXPECT_EQ(Elements[1].StartRow, UnallocatedRow);
      EXPECT_EQ(Elements[1].StartCol, UnallocatedCol);
      // Optimized packing places elements that occupy a full register first.
      // This element is therefore allocated even though it follows the invalid
      // stream in source order.
      // The error still identifies the original element, not its sorted index.
      EXPECT_EQ(Elements[2].StartRow, IsOptimized ? 0u : UnallocatedRow);
      EXPECT_EQ(Elements[2].StartCol, IsOptimized ? 0u : UnallocatedCol);
    }
  }
}

TEST_F(HLSLSemanticSignaturePackingTest, PrefixStableFullGeometryStreams) {
  TestConfig Config(Triple::Geometry, IOType::Out, {});
  for (unsigned Stream = 0; Stream != MaxGeometryStreams; ++Stream)
    Config.Elements.push_back(
        {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/MaxSignatureRows,
         /*Cols=*/MaxSignatureCols, dxil::ElementType::F32,
         dxbc::PSV::InterpolationMode::Linear, /*SemanticIndex=*/0, Stream});

  // Every stream can fill its entire register space independently. Return
  // the maximum extent, not the sum of all streams' extents.
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPacking(Method, Config, /*ExpectedRows=*/MaxSignatureRows,
                  {{/*Row=*/0, /*Col=*/0},
                   {/*Row=*/0, /*Col=*/0},
                   {/*Row=*/0, /*Col=*/0},
                   {/*Row=*/0, /*Col=*/0}});
}

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

  // Prefix-stable and optimized layout:
  // stream0 reg0: A.xyzw
  // stream0 reg1: C.xy | unused.zw
  // stream1 reg0: B.xyzw
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized})
    verifyPacking(Method, Config, /*ExpectedRows=*/2,
                  {{/*Row=*/0, /*Col=*/0},
                   {/*Row=*/0, /*Col=*/0},
                   {/*Row=*/1, /*Col=*/0}});
}

//===----------------------------------------------------------------------===//
// Optimized clip/cull packing tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, OptimizedClipCullSharesArbitraryRow) {
  // struct VSOut { float A : A; float3 Clip : SV_ClipDistance; };
  TestConfig Config(
      Triple::Vertex, IOType::Out,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});
  verifyPacking(PackingMethod::PrefixStable, Config, /*ExpectedRows=*/2,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}});
  // Optimized layout: reg0: A.x | Clip.yzw.
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/1,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/0, /*Col=*/1}});
}

TEST_F(HLSLSemanticSignaturePackingTest, OptimizedClipCullFillsSignature) {
  // struct VSOut {
  //   float4 Fill[31] : FILL;
  //   float A         : A;
  //   float3 Clip     : SV_ClipDistance;
  // };
  TestConfig Config(
      Triple::Vertex, IOType::Out,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/MaxSignatureRows - 1,
        /*Cols=*/4, dxil::ElementType::F32,
        dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});
  verifyPackingError(PackingMethod::PrefixStable, Config,
                     SignaturePackingError::SignatureOverflow,
                     /*ExpectedElementIndex=*/2);
  // All 128 components fit, including A.x | Clip.yzw in the last row.
  verifyPacking(PackingMethod::Optimized, Config,
                /*ExpectedRows=*/MaxSignatureRows,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/MaxSignatureRows - 1, /*Col=*/0},
                 {/*Row=*/MaxSignatureRows - 1, /*Col=*/1}});
}

TEST_F(HLSLSemanticSignaturePackingTest,
       OptimizedClipCullKeepsScalarGroupsTogether) {
  // struct VSOut {
  //   float3 A[3] : A;
  //   float Clip0 : SV_ClipDistance0;
  //   float Clip1 : SV_ClipDistance1;
  //   float Cull0 : SV_CullDistance0;
  // };
  // Filling the three w gaps individually would use three clip/cull rows.
  TestConfig Config(
      Triple::Vertex, IOType::Out,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/3, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear,
        /*SemanticIndex=*/1},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/4,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/3, /*Col=*/0},
                 {/*Row=*/3, /*Col=*/1},
                 {/*Row=*/3, /*Col=*/2}});
}

TEST_F(HLSLSemanticSignaturePackingTest,
       OptimizedSingleRowClipCullCanUseNonAdjacentRows) {
  // struct VSOut {
  //   nointerpolation float2 A    : A;
  //   float2 B                    : B;
  //   centroid float2 C           : C;
  //   nointerpolation float2 Cull : SV_CullDistance;
  //   centroid float2 Clip        : SV_ClipDistance;
  // };
  // Three arbitrary float2 values establish distinct interpolation modes.
  // Cull fits after the first and Clip after the third; row 1 separates them.
  TestConfig Config(
      Triple::Vertex, IOType::Out,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::LinearCentroid},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::LinearCentroid}});

  // Optimized layout:
  // reg0: A.xy | Cull.zw
  // reg1: B.xy | unused.zw
  // reg2: C.xy | Clip.zw
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/3,
                {{/*Row=*/0, /*Col=*/0},
                 {/*Row=*/1, /*Col=*/0},
                 {/*Row=*/2, /*Col=*/0},
                 {/*Row=*/0, /*Col=*/2},
                 {/*Row=*/2, /*Col=*/2}});
}

TEST_F(HLSLSemanticSignaturePackingTest,
       OptimizedIndexedClipCullCannotShareSystemValueRows) {
  // struct VSOut {
  //   float Position : SV_Position;
  //   float Clip[2]  : SV_ClipDistance;
  // };
  // Position fixes its row's indexed range to empty. The Clip array must
  // start in the next row even though Position leaves three free columns.
  TestConfig Config(
      Triple::Vertex, IOType::Out,
      {{dxbc::PSV::SemanticKind::Position, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/2, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});

  // Optimized layout:
  // reg0: Position.x | unused.yzw
  // reg1: Clip[0].x | unused.yzw
  // reg2: Clip[1].x | unused.yzw
  verifyPacking(PackingMethod::Optimized, Config, /*ExpectedRows=*/3,
                {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest,
       OptimizedIndexedClipCullRetriesAdjacentPairs) {
  // struct VSOut {
  //   float3 A[2]   : A;
  //   float Clip[2] : SV_ClipDistance;
  //   float Cull    : SV_CullDistance;
  // };
  // The pair at row 0 fits Clip but not Cull. The next pair fits both.
  TestConfig Config(
      Triple::Vertex, IOType::Out,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/2, /*Cols=*/3,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/2, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});
  // Optimized layout:
  // reg0: A[0].xyz | unused.w
  // reg1: A[1].xyz | Clip[0].w
  // reg2: Cull.x | unused.yz | Clip[1].w
  verifyPacking(
      PackingMethod::Optimized, Config, /*ExpectedRows=*/3,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/1, /*Col=*/3}, {/*Row=*/2, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest,
       OptimizedClipCullRespectsRowCompatibility) {
  // struct VSOut {
  //   half A      : A;
  //   float3 Clip : SV_ClipDistance;
  // };
  //
  // The variants enable native 16-bit types and nointerpolation on A.
  for (bool Native16Bit : {false, true}) {
    SCOPED_TRACE(Native16Bit);
    for (bool DifferentInterp : {false, true}) {
      SCOPED_TRACE(DifferentInterp);
      TestConfig Config(
          Triple::Vertex, IOType::Out, Native16Bit,
          {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
            dxil::ElementType::F16,
            DifferentInterp ? dxbc::PSV::InterpolationMode::Constant
                            : dxbc::PSV::InterpolationMode::Linear},
           {dxbc::PSV::SemanticKind::ClipDistance, /*Rows=*/1, /*Cols=*/3,
            dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear}});
      const bool Separate = Native16Bit || DifferentInterp;

      // Compatible optimized layout:
      // reg0: A.x | Clip.yzw
      // Incompatible optimized layout:
      // reg0: A.x | unused.yzw
      // reg1: Clip.xyz | unused.w
      verifyPacking(PackingMethod::Optimized, Config,
                    /*ExpectedRows=*/Separate ? 2 : 1,
                    {{/*Row=*/0, /*Col=*/0},
                     {/*Row=*/Separate ? 1u : 0u,
                      /*Col=*/static_cast<uint8_t>(Separate ? 0 : 1)}});
    }
  }
}

TEST_F(HLSLSemanticSignaturePackingTest,
       OptimizedClipCullPrecedesSystemGeneratedValues) {
  // struct PSIn {
  //   nointerpolation uint A     : A;
  //   nointerpolation float Cull : SV_CullDistance;
  //   bool IsFrontFace           : SV_IsFrontFace;
  // };
  TestConfig Config(
      Triple::Pixel, IOType::In,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::U32, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::CullDistance, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Constant},
       {dxbc::PSV::SemanticKind::IsFrontFace, /*Rows=*/1, /*Cols=*/1,
        dxil::ElementType::I1, dxbc::PSV::InterpolationMode::Constant}});
  verifyPacking(
      PackingMethod::Optimized, Config, /*ExpectedRows=*/1,
      {{/*Row=*/0, /*Col=*/0}, {/*Row=*/0, /*Col=*/1}, {/*Row=*/0, /*Col=*/2}});
}

TEST_F(HLSLSemanticSignaturePackingTest,
       OptimizedClipCullPacksGeometryStreamsIndependently) {
  // Each of the four geometry output streams has this structure:
  //
  // struct StreamN {
  //   float4 Fill[31] : FILL;
  //   float A         : A;
  //   float3 Clip     : SV_ClipDistance;
  // };
  TestConfig Config(Triple::Geometry, IOType::Out, {});
  for (unsigned Stream = 0; Stream != MaxGeometryStreams; ++Stream) {
    Config.Elements.push_back({dxbc::PSV::SemanticKind::Arbitrary,
                               /*Rows=*/MaxSignatureRows - 1,
                               /*Cols=*/4, dxil::ElementType::F32,
                               dxbc::PSV::InterpolationMode::Linear,
                               /*SemanticIndex=*/0, Stream});
    Config.Elements.push_back({dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1,
                               /*Cols=*/1, dxil::ElementType::F32,
                               dxbc::PSV::InterpolationMode::Linear,
                               /*SemanticIndex=*/0, Stream});
    Config.Elements.push_back({dxbc::PSV::SemanticKind::ClipDistance,
                               /*Rows=*/1, /*Cols=*/3, dxil::ElementType::F32,
                               dxbc::PSV::InterpolationMode::Linear,
                               /*SemanticIndex=*/0, Stream});
  }

  // Optimized layout for each stream:
  // reg0-30: Fill[0-30].xyzw
  // reg31: A.x | Clip.yzw
  SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
  Expected<unsigned> Rows = pack(PackingMethod::Optimized, Elements, Config);
  ASSERT_THAT_EXPECTED(Rows, Succeeded());
  EXPECT_EQ(*Rows, MaxSignatureRows);
  for (unsigned Stream = 0; Stream != MaxGeometryStreams; ++Stream) {
    SCOPED_TRACE(Stream);
    EXPECT_EQ(Elements[3 * Stream].StartRow, 0u);
    EXPECT_EQ(Elements[3 * Stream].StartCol, 0u);
    EXPECT_EQ(Elements[3 * Stream + 1].StartRow, MaxSignatureRows - 1);
    EXPECT_EQ(Elements[3 * Stream + 1].StartCol, 0u);
    EXPECT_EQ(Elements[3 * Stream + 2].StartRow, MaxSignatureRows - 1);
    EXPECT_EQ(Elements[3 * Stream + 2].StartCol, 1u);
  }
}

//===----------------------------------------------------------------------===//
// Optimized ordering tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, PackingPreservesElementMetadata) {
  // struct VSOut {
  //   float2 Scalar   : SCALAR;
  //   float Array[32] : ARRAY;
  // };
  TestConfig Config(
      Triple::Vertex, IOType::Out,
      {{dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1, /*Cols=*/2,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Linear},
       {dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/MaxSignatureRows,
        /*Cols=*/1, dxil::ElementType::F32,
        dxbc::PSV::InterpolationMode::Linear}});

  // Prefix-stable layout:
  // reg0: Scalar.xy | Array[0].z | unused.w
  // reg1-31: unused.xy | Array[1-31].z | unused.w
  // Optimized layout:
  // reg0: Array[0].x | Scalar.yz | unused.w
  // reg1-31: Array[1-31].x | unused.yzw
  for (PackingMethod Method :
       {PackingMethod::PrefixStable, PackingMethod::Optimized}) {
    SCOPED_TRACE(static_cast<unsigned>(Method));
    SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
    Elements[0].SemanticName = "Scalar";
    Elements[1].SemanticName = "Array";
    Elements[0].UsageMask = 3;
    Elements[1].UsageMask = 1;
    Elements[1].DynIndexMask = 1;
    const SmallVector<SemanticSignatureElement> Before = Elements;
    const uint32_t *Indices = Elements[1].SemanticIndices.data();

    Expected<unsigned> Rows = pack(Method, Elements, Config);
    ASSERT_THAT_EXPECTED(Rows, Succeeded());
    EXPECT_EQ(*Rows, MaxSignatureRows);
    EXPECT_EQ(Elements[1].SemanticIndices.data(), Indices);
    for (unsigned I = 0; I != Elements.size(); ++I) {
      SCOPED_TRACE(I);
      verifyMetadata(Before[I], Elements[I]);
      EXPECT_NE(Elements[I].StartRow, UnallocatedRow);
      EXPECT_NE(Elements[I].StartCol, UnallocatedCol);
    }
  }
}

TEST_F(HLSLSemanticSignaturePackingTest, OptimizedUsesSignatureIDToBreakTies) {
  // struct VSOut {
  //   float A : A;
  //   float B : B;
  //   float C : C;
  // };
  //
  // Assign the elements signature IDs 2, 0, 1, respectively.
  TestConfig Config(Triple::Vertex, IOType::Out, {});
  for (unsigned I = 0; I != 3; ++I)
    Config.Elements.push_back({dxbc::PSV::SemanticKind::Arbitrary, /*Rows=*/1,
                               /*Cols=*/1, dxil::ElementType::F32,
                               dxbc::PSV::InterpolationMode::Linear});
  SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
  Elements[0].SigId = 2;
  Elements[1].SigId = 0;
  Elements[2].SigId = 1;

  // Optimized layout:
  // reg0: B.x | C.y | A.z | unused.w
  Expected<unsigned> Rows = pack(PackingMethod::Optimized, Elements, Config);
  ASSERT_THAT_EXPECTED(Rows, Succeeded());
  EXPECT_EQ(*Rows, 1u);
  for (unsigned I = 0; I != Elements.size(); ++I) {
    EXPECT_EQ(Elements[I].SigId, (I + 2) % 3);
    EXPECT_EQ(Elements[I].StartRow, 0u);
    EXPECT_EQ(Elements[I].StartCol, Elements[I].SigId);
  }
}

//===----------------------------------------------------------------------===//
// Indexed packing tests
//===----------------------------------------------------------------------===//

TEST_F(HLSLSemanticSignaturePackingTest, IndexedEmptySignature) {
  TestConfig Config(Triple::EnvironmentType::Pixel, IOType::Out, {});

  verifyPacking(PackingMethod::Indexed, Config, /*ExpectedRows=*/0, {});
}

TEST_F(HLSLSemanticSignaturePackingTest, IndexedUsesLastSignatureRow) {
  // The row extent includes the unused rows before the target's semantic index.
  TestConfig Config(
      Triple::EnvironmentType::Pixel, IOType::Out,
      {{dxbc::PSV::SemanticKind::Target, /*Rows=*/1, /*Cols=*/4,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined,
        /*SemanticIndex=*/MaxSignatureRows - 1}});

  verifyPacking(PackingMethod::Indexed, Config,
                /*ExpectedRows=*/MaxSignatureRows,
                {{/*Row=*/MaxSignatureRows - 1, /*Col=*/0}});
}

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
  verifyPacking(
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
  verifyPacking(PackingMethod::Indexed, Config, /*ExpectedRows=*/8,
                {{/*Row=*/1, /*Col=*/0}, {/*Row=*/7, /*Col=*/0}});
}

TEST_F(HLSLSemanticSignaturePackingTest,
       IndexedRejectsOutOfRangeSemanticIndex) {
  // A semantic index outside the 32-row signature cannot be allocated.

  // struct PSOut {
  //   float4 Color32 : SV_Target32;
  // };
  TestConfig Config(
      Triple::EnvironmentType::Pixel, IOType::Out,
      {{dxbc::PSV::SemanticKind::Target, /*Rows=*/1, /*Cols=*/4,
        dxil::ElementType::F32, dxbc::PSV::InterpolationMode::Undefined,
        /*SemanticIndex=*/MaxSignatureRows}});

  verifyPackingError(PackingMethod::Indexed, Config,
                     SignaturePackingError::SemanticIndexOutOfRange,
                     /*ExpectedElementIndex=*/0);

  SmallVector<SemanticSignatureElement> Elements = makeSignature(Config);
  EXPECT_THAT_EXPECTED(pack(PackingMethod::Indexed, Elements, Config),
                       FailedWithMessage("semantic index must be less than " +
                                         std::to_string(MaxSignatureRows) +
                                         " (element 0)"));
}

} // namespace
