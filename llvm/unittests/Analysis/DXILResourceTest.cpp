//===- DXILResourceTest.cpp - Unit tests for DXILResource -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
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

  Metadata *toMD(unsigned int V) {
    return ConstantAsMetadata::get(
        Constant::getIntegerValue(Int32Ty, APInt(32, V)));
  }
  Metadata *toMD(int V) { return toMD(static_cast<unsigned int>(V)); }
  Metadata *toMD(bool V) {
    return ConstantAsMetadata::get(
        Constant::getIntegerValue(Int32Ty, APInt(1, V)));
  }
  Metadata *toMD(Value *V) { return ValueAsMetadata::get(V); }
  Metadata *toMD(const char *V) { return MDString::get(Context, V); }
  Metadata *toMD(StringRef V) { return MDString::get(Context, V); }
  Metadata *toMD(std::nullptr_t V) { return nullptr; }
  Metadata *toMD(MDTuple *V) { return V; }

  template <typename... Ts> MDTuple *get(Ts... Vs) {
    std::array<Metadata *, sizeof...(Vs)> MDs{toMD(std::forward<Ts>(Vs))...};
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
  // TODO: How am I supposed to get this?
  DataLayout DL("e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-"
                "f64:64-n8:16:32:64-v96:32");

  LLVMContext Context;
  Module M("AnnotationsAndMetadata", Context);
  M.setDataLayout(DL);

  Type *Int1Ty = Type::getInt1Ty(Context);
  Type *Int8Ty = Type::getInt8Ty(Context);
  Type *Int32Ty = Type::getInt32Ty(Context);
  Type *FloatTy = Type::getFloatTy(Context);
  Type *DoubleTy = Type::getDoubleTy(Context);
  Type *Floatx4Ty = FixedVectorType::get(FloatTy, 4);
  Type *Floatx3Ty = FixedVectorType::get(FloatTy, 3);
  Type *Int32x2Ty = FixedVectorType::get(Int32Ty, 2);

  MDBuilder TestMD(Context, Int32Ty, Int1Ty);

  // ByteAddressBuffer Buffer;
  ResourceTypeInfo RTI(llvm::TargetExtType::get(
      Context, "dx.RawBuffer", Int8Ty, {/*IsWriteable=*/0, /*IsROV=*/0}));
  EXPECT_EQ(RTI.getResourceClass(), ResourceClass::SRV);
  EXPECT_EQ(RTI.getResourceKind(), ResourceKind::RawBuffer);

  ResourceBindingInfo RBI(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1,
      RTI.getHandleTy());
  GlobalVariable *GV = RBI.createSymbol(M, RTI.createElementStruct(), "Buffer");
  std::pair<uint32_t, uint32_t> Props = RBI.getAnnotateProps(M, RTI);
  EXPECT_EQ(Props.first, 0x0000000bU);
  EXPECT_EQ(Props.second, 0U);
  MDTuple *MD = RBI.getAsMetadata(M, RTI);
  EXPECT_MDEQ(MD, TestMD.get(0, GV, "Buffer", 0, 0, 1, 11, 0, nullptr));

  // RWByteAddressBuffer BufferOut : register(u3, space2);
  RTI = ResourceTypeInfo(llvm::TargetExtType::get(
      Context, "dx.RawBuffer", Int8Ty, {/*IsWriteable=*/1, /*IsROV=*/0}));
  EXPECT_EQ(RTI.getResourceClass(), ResourceClass::UAV);
  EXPECT_EQ(RTI.getUAV().GloballyCoherent, false);
  EXPECT_EQ(RTI.getUAV().HasCounter, false);
  EXPECT_EQ(RTI.getUAV().IsROV, false);
  EXPECT_EQ(RTI.getResourceKind(), ResourceKind::RawBuffer);

  RBI = ResourceBindingInfo(
      /*RecordID=*/1, /*Space=*/2, /*LowerBound=*/3, /*Size=*/1,
      RTI.getHandleTy());
  GV = RBI.createSymbol(M, RTI.createElementStruct(), "BufferOut");
  Props = RBI.getAnnotateProps(M, RTI);
  EXPECT_EQ(Props.first, 0x0000100bU);
  EXPECT_EQ(Props.second, 0U);
  MD = RBI.getAsMetadata(M, RTI);
  EXPECT_MDEQ(MD, TestMD.get(1, GV, "BufferOut", 2, 3, 1, 11, false, false,
                             false, nullptr));

  // struct BufType0 { int i; float f; double d; };
  // StructuredBuffer<BufType0> Buffer0 : register(t0);
  StructType *BufType0 =
      StructType::create(Context, {Int32Ty, FloatTy, DoubleTy}, "BufType0");
  RTI = ResourceTypeInfo(llvm::TargetExtType::get(
      Context, "dx.RawBuffer", BufType0, {/*IsWriteable=*/0, /*IsROV=*/0}));
  EXPECT_EQ(RTI.getResourceClass(), ResourceClass::SRV);
  ASSERT_EQ(RTI.isStruct(), true);
  EXPECT_EQ(RTI.getStruct(DL).Stride, 16u);
  EXPECT_EQ(RTI.getStruct(DL).AlignLog2, Log2(Align(8)));
  EXPECT_EQ(RTI.getResourceKind(), ResourceKind::StructuredBuffer);

  RBI = ResourceBindingInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1,
      RTI.getHandleTy());
  GV = RBI.createSymbol(M, RTI.createElementStruct(), "Buffer0");
  Props = RBI.getAnnotateProps(M, RTI);
  EXPECT_EQ(Props.first, 0x0000030cU);
  EXPECT_EQ(Props.second, 0x00000010U);
  MD = RBI.getAsMetadata(M, RTI);
  EXPECT_MDEQ(MD,
              TestMD.get(0, GV, "Buffer0", 0, 0, 1, 12, 0, TestMD.get(1, 16)));

  // StructuredBuffer<float3> Buffer1 : register(t1);
  RTI = ResourceTypeInfo(llvm::TargetExtType::get(
      Context, "dx.RawBuffer", Floatx3Ty, {/*IsWriteable=*/0, /*IsROV=*/0}));
  EXPECT_EQ(RTI.getResourceClass(), ResourceClass::SRV);
  ASSERT_EQ(RTI.isStruct(), true);
  EXPECT_EQ(RTI.getStruct(DL).Stride, 12u);
  EXPECT_EQ(RTI.getStruct(DL).AlignLog2, 0u);
  EXPECT_EQ(RTI.getResourceKind(), ResourceKind::StructuredBuffer);

  RBI = ResourceBindingInfo(
      /*RecordID=*/1, /*Space=*/0, /*LowerBound=*/1, /*Size=*/1,
      RTI.getHandleTy());
  GV = RBI.createSymbol(M, RTI.createElementStruct(), "Buffer1");
  Props = RBI.getAnnotateProps(M, RTI);
  EXPECT_EQ(Props.first, 0x0000000cU);
  EXPECT_EQ(Props.second, 0x0000000cU);
  MD = RBI.getAsMetadata(M, RTI);
  EXPECT_MDEQ(MD,
              TestMD.get(1, GV, "Buffer1", 0, 1, 1, 12, 0, TestMD.get(1, 12)));

  // Texture2D<float4> ColorMapTexture : register(t2);
  RTI = ResourceTypeInfo(
      llvm::TargetExtType::get(Context, "dx.Texture", Floatx4Ty,
                               {/*IsWriteable=*/0, /*IsROV=*/0, /*IsSigned=*/0,
                                llvm::to_underlying(ResourceKind::Texture2D)}));
  EXPECT_EQ(RTI.getResourceClass(), ResourceClass::SRV);
  ASSERT_EQ(RTI.isTyped(), true);
  EXPECT_EQ(RTI.getTyped().ElementTy, ElementType::F32);
  EXPECT_EQ(RTI.getTyped().ElementCount, 4u);
  EXPECT_EQ(RTI.getResourceKind(), ResourceKind::Texture2D);

  RBI = ResourceBindingInfo(
      /*RecordID=*/2, /*Space=*/0, /*LowerBound=*/2, /*Size=*/1,
      RTI.getHandleTy());
  GV = RBI.createSymbol(M, RTI.createElementStruct(), "ColorMapTexture");
  Props = RBI.getAnnotateProps(M, RTI);
  EXPECT_EQ(Props.first, 0x00000002U);
  EXPECT_EQ(Props.second, 0x00000409U);
  MD = RBI.getAsMetadata(M, RTI);
  EXPECT_MDEQ(MD, TestMD.get(2, GV, "ColorMapTexture", 0, 2, 1, 2, 0,
                             TestMD.get(0, 9)));

  // Texture2DMS<float, 8> DepthBuffer : register(t0);
  RTI = ResourceTypeInfo(llvm::TargetExtType::get(
      Context, "dx.MSTexture", FloatTy,
      {/*IsWriteable=*/0, /*SampleCount=*/8,
       /*IsSigned=*/0, llvm::to_underlying(ResourceKind::Texture2DMS)}));
  EXPECT_EQ(RTI.getResourceClass(), ResourceClass::SRV);
  ASSERT_EQ(RTI.isTyped(), true);
  EXPECT_EQ(RTI.getTyped().ElementTy, ElementType::F32);
  EXPECT_EQ(RTI.getTyped().ElementCount, 1u);
  ASSERT_EQ(RTI.isMultiSample(), true);
  EXPECT_EQ(RTI.getMultiSampleCount(), 8u);
  EXPECT_EQ(RTI.getResourceKind(), ResourceKind::Texture2DMS);

  RBI = ResourceBindingInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1,
      RTI.getHandleTy());
  GV = RBI.createSymbol(M, RTI.createElementStruct(), "DepthBuffer");
  Props = RBI.getAnnotateProps(M, RTI);
  EXPECT_EQ(Props.first, 0x00000003U);
  EXPECT_EQ(Props.second, 0x00080109U);
  MD = RBI.getAsMetadata(M, RTI);
  EXPECT_MDEQ(
      MD, TestMD.get(0, GV, "DepthBuffer", 0, 0, 1, 3, 8, TestMD.get(0, 9)));

  // FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMinMip;
  RTI = ResourceTypeInfo(llvm::TargetExtType::get(
      Context, "dx.FeedbackTexture", {},
      {llvm::to_underlying(SamplerFeedbackType::MinMip),
       llvm::to_underlying(ResourceKind::FeedbackTexture2D)}));
  EXPECT_EQ(RTI.getResourceClass(), ResourceClass::UAV);
  ASSERT_EQ(RTI.isFeedback(), true);
  EXPECT_EQ(RTI.getFeedbackType(), SamplerFeedbackType::MinMip);
  EXPECT_EQ(RTI.getResourceKind(), ResourceKind::FeedbackTexture2D);

  RBI = ResourceBindingInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1,
      RTI.getHandleTy());
  GV = RBI.createSymbol(M, RTI.createElementStruct(), "feedbackMinMip");
  Props = RBI.getAnnotateProps(M, RTI);
  EXPECT_EQ(Props.first, 0x00001011U);
  EXPECT_EQ(Props.second, 0U);
  MD = RBI.getAsMetadata(M, RTI);
  EXPECT_MDEQ(MD, TestMD.get(0, GV, "feedbackMinMip", 0, 0, 1, 17, false, false,
                             false, TestMD.get(2, 0)));

  // FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIP_REGION_USED> feedbackMipRegion;
  RTI = ResourceTypeInfo(llvm::TargetExtType::get(
      Context, "dx.FeedbackTexture", {},
      {llvm::to_underlying(SamplerFeedbackType::MipRegionUsed),
       llvm::to_underlying(ResourceKind::FeedbackTexture2DArray)}));
  EXPECT_EQ(RTI.getResourceClass(), ResourceClass::UAV);
  ASSERT_EQ(RTI.isFeedback(), true);
  EXPECT_EQ(RTI.getFeedbackType(), SamplerFeedbackType::MipRegionUsed);
  EXPECT_EQ(RTI.getResourceKind(), ResourceKind::FeedbackTexture2DArray);

  RBI = ResourceBindingInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1,
      RTI.getHandleTy());
  GV = RBI.createSymbol(M, RTI.createElementStruct(), "feedbackMipRegion");
  Props = RBI.getAnnotateProps(M, RTI);
  EXPECT_EQ(Props.first, 0x00001012U);
  EXPECT_EQ(Props.second, 0x00000001U);
  MD = RBI.getAsMetadata(M, RTI);
  EXPECT_MDEQ(MD, TestMD.get(0, GV, "feedbackMipRegion", 0, 0, 1, 18, false,
                             false, false, TestMD.get(2, 1)));

  // globallycoherent RWTexture2D<int2> OutputTexture : register(u0, space2);
  RTI = ResourceTypeInfo(
      llvm::TargetExtType::get(Context, "dx.Texture", Int32x2Ty,
                               {/*IsWriteable=*/1,
                                /*IsROV=*/0, /*IsSigned=*/1,
                                llvm::to_underlying(ResourceKind::Texture2D)}),
      /*GloballyCoherent=*/true, /*HasCounter=*/false);

  EXPECT_EQ(RTI.getResourceClass(), ResourceClass::UAV);
  EXPECT_EQ(RTI.getUAV().GloballyCoherent, true);
  EXPECT_EQ(RTI.getUAV().HasCounter, false);
  EXPECT_EQ(RTI.getUAV().IsROV, false);
  EXPECT_EQ(RTI.getResourceKind(), ResourceKind::Texture2D);

  RBI = ResourceBindingInfo(
      /*RecordID=*/0, /*Space=*/2, /*LowerBound=*/0, /*Size=*/1,
      RTI.getHandleTy());
  GV = RBI.createSymbol(M, RTI.createElementStruct(), "OutputTexture");
  Props = RBI.getAnnotateProps(M, RTI);
  EXPECT_EQ(Props.first, 0x00005002U);
  EXPECT_EQ(Props.second, 0x00000204U);
  MD = RBI.getAsMetadata(M, RTI);
  EXPECT_MDEQ(MD, TestMD.get(0, GV, "OutputTexture", 2, 0, 1, 2, true, false,
                             false, TestMD.get(0, 4)));

  // RasterizerOrderedBuffer<float4> ROB;
  RTI = ResourceTypeInfo(llvm::TargetExtType::get(
      Context, "dx.TypedBuffer", Floatx4Ty,
      {/*IsWriteable=*/1, /*IsROV=*/1, /*IsSigned=*/0}));
  EXPECT_EQ(RTI.getResourceClass(), ResourceClass::UAV);
  EXPECT_EQ(RTI.getUAV().GloballyCoherent, false);
  EXPECT_EQ(RTI.getUAV().HasCounter, false);
  EXPECT_EQ(RTI.getUAV().IsROV, true);
  ASSERT_EQ(RTI.isTyped(), true);
  EXPECT_EQ(RTI.getTyped().ElementTy, ElementType::F32);
  EXPECT_EQ(RTI.getTyped().ElementCount, 4u);
  EXPECT_EQ(RTI.getResourceKind(), ResourceKind::TypedBuffer);

  RBI = ResourceBindingInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1,
      RTI.getHandleTy());
  GV = RBI.createSymbol(M, RTI.createElementStruct(), "ROB");
  Props = RBI.getAnnotateProps(M, RTI);
  EXPECT_EQ(Props.first, 0x0000300aU);
  EXPECT_EQ(Props.second, 0x00000409U);
  MD = RBI.getAsMetadata(M, RTI);
  EXPECT_MDEQ(MD, TestMD.get(0, GV, "ROB", 0, 0, 1, 10, false, false, true,
                             TestMD.get(0, 9)));

  // RWStructuredBuffer<ParticleMotion> g_OutputBuffer : register(u2);
  StructType *BufType1 = StructType::create(
      Context, {Floatx3Ty, FloatTy, Int32Ty}, "ParticleMotion");
  RTI = ResourceTypeInfo(
      llvm::TargetExtType::get(Context, "dx.RawBuffer", BufType1,
                               {/*IsWriteable=*/1, /*IsROV=*/0}),
      /*GloballyCoherent=*/false, /*HasCounter=*/true);
  EXPECT_EQ(RTI.getResourceClass(), ResourceClass::UAV);
  EXPECT_EQ(RTI.getUAV().GloballyCoherent, false);
  EXPECT_EQ(RTI.getUAV().HasCounter, true);
  EXPECT_EQ(RTI.getUAV().IsROV, false);
  ASSERT_EQ(RTI.isStruct(), true);
  EXPECT_EQ(RTI.getStruct(DL).Stride, 20u);
  EXPECT_EQ(RTI.getStruct(DL).AlignLog2, Log2(Align(4)));
  EXPECT_EQ(RTI.getResourceKind(), ResourceKind::StructuredBuffer);

  RBI = ResourceBindingInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/2, /*Size=*/1,
      RTI.getHandleTy());
  GV = RBI.createSymbol(M, RTI.createElementStruct(), "g_OutputBuffer");
  Props = RBI.getAnnotateProps(M, RTI);
  EXPECT_EQ(Props.first, 0x0000920cU);
  EXPECT_EQ(Props.second, 0x00000014U);
  MD = RBI.getAsMetadata(M, RTI);
  EXPECT_MDEQ(MD, TestMD.get(0, GV, "g_OutputBuffer", 0, 2, 1, 12, false, true,
                             false, TestMD.get(1, 20)));

  // RWTexture2DMSArray<uint, 8> g_rw_t2dmsa;
  RTI = ResourceTypeInfo(llvm::TargetExtType::get(
      Context, "dx.MSTexture", Int32Ty,
      {/*IsWriteable=*/1, /*SampleCount=*/8, /*IsSigned=*/0,
       llvm::to_underlying(ResourceKind::Texture2DMSArray)}));
  EXPECT_EQ(RTI.getResourceClass(), ResourceClass::UAV);
  EXPECT_EQ(RTI.getUAV().GloballyCoherent, false);
  EXPECT_EQ(RTI.getUAV().HasCounter, false);
  EXPECT_EQ(RTI.getUAV().IsROV, false);
  ASSERT_EQ(RTI.isTyped(), true);
  EXPECT_EQ(RTI.getTyped().ElementTy, ElementType::U32);
  EXPECT_EQ(RTI.getTyped().ElementCount, 1u);
  ASSERT_EQ(RTI.isMultiSample(), true);
  EXPECT_EQ(RTI.getMultiSampleCount(), 8u);
  EXPECT_EQ(RTI.getResourceKind(), ResourceKind::Texture2DMSArray);

  RBI = ResourceBindingInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1,
      RTI.getHandleTy());
  GV = RBI.createSymbol(M, RTI.createElementStruct(), "g_rw_t2dmsa");
  Props = RBI.getAnnotateProps(M, RTI);
  EXPECT_EQ(Props.first, 0x00001008U);
  EXPECT_EQ(Props.second, 0x00080105U);
  MD = RBI.getAsMetadata(M, RTI);
  EXPECT_MDEQ(MD, TestMD.get(0, GV, "g_rw_t2dmsa", 0, 0, 1, 8, false, false,
                             false, TestMD.get(0, 5)));

  // cbuffer cb0 { float4 g_X; float4 g_Y; }
  StructType *CBufType0 =
      StructType::create(Context, {Floatx4Ty, Floatx4Ty}, "cb0");
  RTI = ResourceTypeInfo(llvm::TargetExtType::get(Context, "dx.CBuffer",
                                                  CBufType0, {/*Size=*/32}));
  EXPECT_EQ(RTI.getResourceClass(), ResourceClass::CBuffer);
  EXPECT_EQ(RTI.getCBufferSize(DL), 32u);
  EXPECT_EQ(RTI.getResourceKind(), ResourceKind::CBuffer);

  RBI = ResourceBindingInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1,
      RTI.getHandleTy());
  GV = RBI.createSymbol(M, RTI.createElementStruct(), "");
  Props = RBI.getAnnotateProps(M, RTI);
  EXPECT_EQ(Props.first, 0x0000000dU);
  EXPECT_EQ(Props.second, 0x00000020U);
  MD = RBI.getAsMetadata(M, RTI);
  EXPECT_MDEQ(MD, TestMD.get(0, GV, "", 0, 0, 1, 32, nullptr));

  // SamplerState ColorMapSampler : register(s0);
  RTI = ResourceTypeInfo(llvm::TargetExtType::get(
      Context, "dx.Sampler", {},
      {llvm::to_underlying(dxil::SamplerType::Default)}));
  EXPECT_EQ(RTI.getResourceClass(), ResourceClass::Sampler);
  EXPECT_EQ(RTI.getSamplerType(), dxil::SamplerType::Default);
  EXPECT_EQ(RTI.getResourceKind(), ResourceKind::Sampler);

  RBI = ResourceBindingInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1,
      RTI.getHandleTy());
  GV = RBI.createSymbol(M, RTI.createElementStruct(), "ColorMapSampler");
  Props = RBI.getAnnotateProps(M, RTI);
  EXPECT_EQ(Props.first, 0x0000000eU);
  EXPECT_EQ(Props.second, 0U);
  MD = RBI.getAsMetadata(M, RTI);
  EXPECT_MDEQ(MD, TestMD.get(0, GV, "ColorMapSampler", 0, 0, 1, 0, nullptr));

  RTI = ResourceTypeInfo(llvm::TargetExtType::get(
      Context, "dx.Sampler", {},
      {llvm::to_underlying(dxil::SamplerType::Comparison)}));
  EXPECT_EQ(RTI.getResourceClass(), ResourceClass::Sampler);
  EXPECT_EQ(RTI.getSamplerType(), dxil::SamplerType::Comparison);
  EXPECT_EQ(RTI.getResourceKind(), ResourceKind::Sampler);

  RBI = ResourceBindingInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1,
      RTI.getHandleTy());
  GV = RBI.createSymbol(M, RTI.createElementStruct(), "CmpSampler");
  Props = RBI.getAnnotateProps(M, RTI);
  EXPECT_EQ(Props.first, 0x0000800eU);
  EXPECT_EQ(Props.second, 0U);
  MD = RBI.getAsMetadata(M, RTI);
  EXPECT_MDEQ(MD, TestMD.get(0, GV, "CmpSampler", 0, 0, 1, 1, nullptr));
}
