//===- DXILResourceTest.cpp - Unit tests for DXILResource -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/Constants.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace dxil;

namespace {
// Helper to succinctly build resource shaped metadata for tests.
struct MDBuilder {
  LLVMContext &Context;
  Type *Int32Ty;
  Type *Int1Ty;

  MDBuilder(LLVMContext &Context, Type *Int32Ty, Type *Int1Ty)
      : Context(Context), Int32Ty(Int32Ty), Int1Ty(Int1Ty) {}

  template <typename... Ts>
  void appendMDs(SmallVectorImpl<Metadata *> &MDs, int V, Ts... More) {
    MDs.push_back(ConstantAsMetadata::get(
        Constant::getIntegerValue(Int32Ty, APInt(32, V))));
    appendMDs(MDs, More...);
  }
  template <typename... Ts>
  void appendMDs(SmallVectorImpl<Metadata *> &MDs, unsigned int V, Ts... More) {
    MDs.push_back(ConstantAsMetadata::get(
        Constant::getIntegerValue(Int32Ty, APInt(32, V))));
    appendMDs(MDs, More...);
  }
  template <typename... Ts>
  void appendMDs(SmallVectorImpl<Metadata *> &MDs, bool V, Ts... More) {
    MDs.push_back(ConstantAsMetadata::get(
        Constant::getIntegerValue(Int1Ty, APInt(1, V))));
    appendMDs(MDs, More...);
  }
  template <typename... Ts>
  void appendMDs(SmallVectorImpl<Metadata *> &MDs, Value *V, Ts... More) {
    MDs.push_back(ValueAsMetadata::get(V));
    appendMDs(MDs, More...);
  }
  template <typename... Ts>
  void appendMDs(SmallVectorImpl<Metadata *> &MDs, const char *V, Ts... More) {
    MDs.push_back(MDString::get(Context, V));
    appendMDs(MDs, More...);
  }
  template <typename... Ts>
  void appendMDs(SmallVectorImpl<Metadata *> &MDs, StringRef V, Ts... More) {
    MDs.push_back(MDString::get(Context, V));
    appendMDs(MDs, More...);
  }
  template <typename... Ts>
  void appendMDs(SmallVectorImpl<Metadata *> &MDs, std::nullptr_t V,
                 Ts... More) {
    MDs.push_back(nullptr);
    appendMDs(MDs, More...);
  }
  template <typename... Ts>
  void appendMDs(SmallVectorImpl<Metadata *> &MDs, MDTuple *V, Ts... More) {
    MDs.push_back(V);
    appendMDs(MDs, More...);
  }
  void appendMDs(SmallVectorImpl<Metadata *> &MDs) {
    // Base case, nothing to do.
  }

  template <typename... Ts> MDTuple *get(Ts... Data) {
    SmallVector<Metadata *> MDs;
    appendMDs(MDs, Data...);
    return MDNode::get(Context, MDs);
  }
};

testing::AssertionResult MDTupleEq(const char *LHSExpr, const char *RHSExpr,
                                   MDTuple *LHS, MDTuple *RHS) {
  if (LHS == RHS)
    return testing::AssertionSuccess();
  std::string LHSRepr, RHSRepr;
  raw_string_ostream LHSS(LHSRepr), RHSS(RHSRepr);
  LHS->printTree(LHSS);
  RHS->printTree(RHSS);

  return testing::AssertionFailure() << "Expected equality:\n"
                                     << "  " << LHSExpr << "\n"
                                     << "Which is:\n"
                                     << "  " << LHSRepr << "\n\n"
                                     << "  " << RHSExpr << "\n"
                                     << "Which is:\n"
                                     << "  " << RHSRepr;
}
#define EXPECT_MDEQ(X, Y) EXPECT_PRED_FORMAT2(MDTupleEq, X, Y)
} // namespace

TEST(DXILResource, AnnotationsAndMetadata) {
  LLVMContext Context;
  Type *Int1Ty = Type::getInt1Ty(Context);
  Type *Int32Ty = Type::getInt32Ty(Context);
  Type *FloatTy = Type::getFloatTy(Context);
  Type *DoubleTy = Type::getDoubleTy(Context);
  Type *Floatx4Ty = FixedVectorType::get(FloatTy, 4);
  Type *Floatx3Ty = FixedVectorType::get(FloatTy, 3);
  Type *Int32x2Ty = FixedVectorType::get(Int32Ty, 2);

  MDBuilder TestMD(Context, Int32Ty, Int1Ty);

  // ByteAddressBuffer Buffer0;
  Value *Symbol = UndefValue::get(
      StructType::create(Context, {Int32Ty}, "struct.ByteAddressBuffer"));
  ResourceInfo Resource = ResourceInfo::RawBuffer(Symbol, "Buffer0");
  Resource.bind(0, 0, 0, 1);
  std::pair<uint32_t, uint32_t> Props = Resource.getAnnotateProps();
  EXPECT_EQ(Props.first, 0x0000000bU);
  EXPECT_EQ(Props.second, 0U);
  MDTuple *MD = Resource.getAsMetadata(Context);
  EXPECT_MDEQ(MD, TestMD.get(0, Symbol, "Buffer0", 0, 0, 1, 11, 0, nullptr));

  // RWByteAddressBuffer BufferOut : register(u3, space2);
  Symbol = UndefValue::get(
      StructType::create(Context, {Int32Ty}, "struct.RWByteAddressBuffer"));
  Resource =
      ResourceInfo::RWRawBuffer(Symbol, "BufferOut",
                                /*GloballyCoherent=*/false, /*IsROV=*/false);
  Resource.bind(1, 2, 3, 1);
  Props = Resource.getAnnotateProps();
  EXPECT_EQ(Props.first, 0x0000100bU);
  EXPECT_EQ(Props.second, 0U);
  MD = Resource.getAsMetadata(Context);
  EXPECT_MDEQ(MD, TestMD.get(1, Symbol, "BufferOut", 2, 3, 1, 11, false, false,
                             false, nullptr));

  // struct BufType0 { int i; float f; double d; };
  // StructuredBuffer<BufType0> Buffer0 : register(t0);
  StructType *BufType0 =
      StructType::create(Context, {Int32Ty, FloatTy, DoubleTy}, "BufType0");
  Symbol = UndefValue::get(StructType::create(
      Context, {BufType0}, "class.StructuredBuffer<BufType>"));
  Resource = ResourceInfo::StructuredBuffer(Symbol, "Buffer0",
                                            /*Stride=*/16, Align(8));
  Resource.bind(0, 0, 0, 1);
  Props = Resource.getAnnotateProps();
  EXPECT_EQ(Props.first, 0x0000030cU);
  EXPECT_EQ(Props.second, 0x00000010U);
  MD = Resource.getAsMetadata(Context);
  EXPECT_MDEQ(
      MD, TestMD.get(0, Symbol, "Buffer0", 0, 0, 1, 12, 0, TestMD.get(1, 16)));

  // StructuredBuffer<float3> Buffer1 : register(t1);
  Symbol = UndefValue::get(StructType::create(
      Context, {Floatx3Ty}, "class.StructuredBuffer<vector<float, 3> >"));
  Resource = ResourceInfo::StructuredBuffer(Symbol, "Buffer1",
                                            /*Stride=*/12, {});
  Resource.bind(1, 0, 1, 1);
  Props = Resource.getAnnotateProps();
  EXPECT_EQ(Props.first, 0x0000000cU);
  EXPECT_EQ(Props.second, 0x0000000cU);
  MD = Resource.getAsMetadata(Context);
  EXPECT_MDEQ(
      MD, TestMD.get(1, Symbol, "Buffer1", 0, 1, 1, 12, 0, TestMD.get(1, 12)));

  // Texture2D<float4> ColorMapTexture : register(t2);
  Symbol = UndefValue::get(StructType::create(
      Context, {Floatx4Ty}, "class.Texture2D<vector<float, 4> >"));
  Resource =
      ResourceInfo::SRV(Symbol, "ColorMapTexture", dxil::ElementType::F32,
                        /*ElementCount=*/4, dxil::ResourceKind::Texture2D);
  Resource.bind(2, 0, 2, 1);
  Props = Resource.getAnnotateProps();
  EXPECT_EQ(Props.first, 0x00000002U);
  EXPECT_EQ(Props.second, 0x00000409U);
  MD = Resource.getAsMetadata(Context);
  EXPECT_MDEQ(MD, TestMD.get(2, Symbol, "ColorMapTexture", 0, 2, 1, 2, 0,
                             TestMD.get(0, 9)));

  // Texture2DMS<float, 8> DepthBuffer : register(t0);
  Symbol = UndefValue::get(
      StructType::create(Context, {FloatTy}, "class.Texture2DMS<float, 8>"));
  Resource =
      ResourceInfo::Texture2DMS(Symbol, "DepthBuffer", dxil::ElementType::F32,
                                /*ElementCount=*/1, /*SampleCount=*/8);
  Resource.bind(0, 0, 0, 1);
  Props = Resource.getAnnotateProps();
  EXPECT_EQ(Props.first, 0x00000003U);
  EXPECT_EQ(Props.second, 0x00080109U);
  MD = Resource.getAsMetadata(Context);
  EXPECT_MDEQ(MD, TestMD.get(0, Symbol, "DepthBuffer", 0, 0, 1, 3, 8,
                             TestMD.get(0, 9)));

  // FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMinMip;
  Symbol = UndefValue::get(
      StructType::create(Context, {Int32Ty}, "class.FeedbackTexture2D<0>"));
  Resource = ResourceInfo::FeedbackTexture2D(Symbol, "feedbackMinMip",
                                             SamplerFeedbackType::MinMip);
  Resource.bind(0, 0, 0, 1);
  Props = Resource.getAnnotateProps();
  EXPECT_EQ(Props.first, 0x00001011U);
  EXPECT_EQ(Props.second, 0U);
  MD = Resource.getAsMetadata(Context);
  EXPECT_MDEQ(MD, TestMD.get(0, Symbol, "feedbackMinMip", 0, 0, 1, 17, false,
                             false, false, TestMD.get(2, 0)));

  // FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIP_REGION_USED> feedbackMipRegion;
  Symbol = UndefValue::get(StructType::create(
      Context, {Int32Ty}, "class.FeedbackTexture2DArray<1>"));
  Resource = ResourceInfo::FeedbackTexture2DArray(
      Symbol, "feedbackMipRegion", SamplerFeedbackType::MipRegionUsed);
  Resource.bind(0, 0, 0, 1);
  Props = Resource.getAnnotateProps();
  EXPECT_EQ(Props.first, 0x00001012U);
  EXPECT_EQ(Props.second, 0x00000001U);
  MD = Resource.getAsMetadata(Context);
  EXPECT_MDEQ(MD, TestMD.get(0, Symbol, "feedbackMipRegion", 0, 0, 1, 18, false,
                             false, false, TestMD.get(2, 1)));

  // globallycoherent RWTexture2D<int2> OutputTexture : register(u0, space2);
  Symbol = UndefValue::get(StructType::create(
      Context, {Int32x2Ty}, "class.RWTexture2D<vector<int, 2> >"));
  Resource = ResourceInfo::UAV(Symbol, "OutputTexture", dxil::ElementType::I32,
                               /*ElementCount=*/2, /*GloballyCoherent=*/1,
                               /*IsROV=*/0, dxil::ResourceKind::Texture2D);
  Resource.bind(0, 2, 0, 1);
  Props = Resource.getAnnotateProps();
  EXPECT_EQ(Props.first, 0x00005002U);
  EXPECT_EQ(Props.second, 0x00000204U);
  MD = Resource.getAsMetadata(Context);
  EXPECT_MDEQ(MD, TestMD.get(0, Symbol, "OutputTexture", 2, 0, 1, 2, true,
                             false, false, TestMD.get(0, 4)));

  // RasterizerOrderedBuffer<float4> ROB;
  Symbol = UndefValue::get(
      StructType::create(Context, {Floatx4Ty},
                         "class.RasterizerOrderedBuffer<vector<float, 4> >"));
  Resource = ResourceInfo::UAV(Symbol, "ROB", dxil::ElementType::F32,
                               /*ElementCount=*/4, /*GloballyCoherent=*/0,
                               /*IsROV=*/1, dxil::ResourceKind::TypedBuffer);
  Resource.bind(0, 0, 0, 1);
  Props = Resource.getAnnotateProps();
  EXPECT_EQ(Props.first, 0x0000300aU);
  EXPECT_EQ(Props.second, 0x00000409U);
  MD = Resource.getAsMetadata(Context);
  EXPECT_MDEQ(MD, TestMD.get(0, Symbol, "ROB", 0, 0, 1, 10, false, false, true,
                             TestMD.get(0, 9)));

  // RWStructuredBuffer<ParticleMotion> g_OutputBuffer : register(u2);
  StructType *BufType1 = StructType::create(
      Context, {Floatx3Ty, FloatTy, Int32Ty}, "ParticleMotion");
  Symbol = UndefValue::get(StructType::create(
      Context, {BufType1}, "class.StructuredBuffer<ParticleMotion>"));
  Resource =
      ResourceInfo::RWStructuredBuffer(Symbol, "g_OutputBuffer", /*Stride=*/20,
                                       Align(4), /*GloballyCoherent=*/false,
                                       /*IsROV=*/false, /*HasCounter=*/true);
  Resource.bind(0, 0, 2, 1);
  Props = Resource.getAnnotateProps();
  EXPECT_EQ(Props.first, 0x0000920cU);
  EXPECT_EQ(Props.second, 0x00000014U);
  MD = Resource.getAsMetadata(Context);
  EXPECT_MDEQ(MD, TestMD.get(0, Symbol, "g_OutputBuffer", 0, 2, 1, 12, false,
                             true, false, TestMD.get(1, 20)));

  // RWTexture2DMSArray<uint,8> g_rw_t2dmsa;
  Symbol = UndefValue::get(StructType::create(
      Context, {Int32Ty}, "class.RWTexture2DMSArray<unsigned int, 8>"));
  Resource = ResourceInfo::RWTexture2DMSArray(
      Symbol, "g_rw_t2dmsa", dxil::ElementType::U32, /*ElementCount=*/1,
      /*SampleCount=*/8, /*GloballyCoherent=*/false);
  Resource.bind(0, 0, 0, 1);
  Props = Resource.getAnnotateProps();
  EXPECT_EQ(Props.first, 0x00001008U);
  EXPECT_EQ(Props.second, 0x00080105U);
  MD = Resource.getAsMetadata(Context);
  EXPECT_MDEQ(MD, TestMD.get(0, Symbol, "g_rw_t2dmsa", 0, 0, 1, 8, false, false,
                             false, TestMD.get(0, 5)));

  // cbuffer cb0 { float4 g_X; float4 g_Y; }
  Symbol = UndefValue::get(
      StructType::create(Context, {Floatx4Ty, Floatx4Ty}, "cb0"));
  Resource = ResourceInfo::CBuffer(Symbol, "cb0", /*Size=*/32);
  Resource.bind(0, 0, 0, 1);
  Props = Resource.getAnnotateProps();
  EXPECT_EQ(Props.first, 0x0000000dU);
  EXPECT_EQ(Props.second, 0x00000020U);
  MD = Resource.getAsMetadata(Context);
  EXPECT_MDEQ(MD, TestMD.get(0, Symbol, "cb0", 0, 0, 1, 32, nullptr));

  // SamplerState ColorMapSampler : register(s0);
  Symbol = UndefValue::get(
      StructType::create(Context, {Int32Ty}, "struct.SamplerState"));
  Resource = ResourceInfo::Sampler(Symbol, "ColorMapSampler",
                                   dxil::SamplerType::Default);
  Resource.bind(0, 0, 0, 1);
  Props = Resource.getAnnotateProps();
  EXPECT_EQ(Props.first, 0x0000000eU);
  EXPECT_EQ(Props.second, 0U);
  MD = Resource.getAsMetadata(Context);
  EXPECT_MDEQ(MD,
              TestMD.get(0, Symbol, "ColorMapSampler", 0, 0, 1, 0, nullptr));

  // SamplerComparisonState ShadowSampler {...};
  Resource = ResourceInfo::Sampler(Symbol, "CmpSampler",
                                   dxil::SamplerType::Comparison);
  Resource.bind(0, 0, 0, 1);
  Props = Resource.getAnnotateProps();
  EXPECT_EQ(Props.first, 0x0000800eU);
  EXPECT_EQ(Props.second, 0U);
  MD = Resource.getAsMetadata(Context);
  EXPECT_MDEQ(MD, TestMD.get(0, Symbol, "CmpSampler", 0, 0, 1, 1, nullptr));
}
