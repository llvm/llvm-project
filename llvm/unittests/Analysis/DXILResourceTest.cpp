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
  Value *DummyGV = UndefValue::get(PointerType::getUnqual(Context));

  // ByteAddressBuffer Buffer;
  auto *HandleTy = llvm::TargetExtType::get(Context, "dx.RawBuffer", Int8Ty,
                                            {/*IsWriteable=*/0, /*IsROV=*/0});
  ResourceInfo RI(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1, HandleTy);

  EXPECT_EQ(RI.getResourceClass(), ResourceClass::SRV);
  EXPECT_EQ(RI.getResourceKind(), ResourceKind::RawBuffer);

  std::pair<uint32_t, uint32_t> Props = RI.getAnnotateProps(M);
  EXPECT_EQ(Props.first, 0x0000000bU);
  EXPECT_EQ(Props.second, 0U);
  MDTuple *MD = RI.getAsMetadata(M);
  EXPECT_MDEQ(MD, TestMD.get(0, DummyGV, "", 0, 0, 1, 11, 0, nullptr));

  // RWByteAddressBuffer BufferOut : register(u3, space2);
  HandleTy = llvm::TargetExtType::get(Context, "dx.RawBuffer", Int8Ty,
                                      {/*IsWriteable=*/1, /*IsROV=*/0});
  RI = ResourceInfo(
      /*RecordID=*/1, /*Space=*/2, /*LowerBound=*/3, /*Size=*/1, HandleTy);

  EXPECT_EQ(RI.getResourceClass(), ResourceClass::UAV);
  EXPECT_EQ(RI.getUAV().GloballyCoherent, false);
  EXPECT_EQ(RI.getUAV().HasCounter, false);
  EXPECT_EQ(RI.getUAV().IsROV, false);
  EXPECT_EQ(RI.getResourceKind(), ResourceKind::RawBuffer);

  Props = RI.getAnnotateProps(M);
  EXPECT_EQ(Props.first, 0x0000100bU);
  EXPECT_EQ(Props.second, 0U);
  MD = RI.getAsMetadata(M);
  EXPECT_MDEQ(MD, TestMD.get(1, DummyGV, "", 2, 3, 1, 11, false, false, false,
                             nullptr));

  // struct BufType0 { int i; float f; double d; };
  // StructuredBuffer<BufType0> Buffer0 : register(t0);
  StructType *BufType0 =
      StructType::create(Context, {Int32Ty, FloatTy, DoubleTy}, "BufType0");
  HandleTy = llvm::TargetExtType::get(Context, "dx.RawBuffer", BufType0,
                                      {/*IsWriteable=*/0, /*IsROV=*/0});
  RI = ResourceInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1, HandleTy);

  EXPECT_EQ(RI.getResourceClass(), ResourceClass::SRV);
  ASSERT_EQ(RI.isStruct(), true);
  EXPECT_EQ(RI.getStruct(DL).Stride, 16u);
  EXPECT_EQ(RI.getStruct(DL).AlignLog2, Log2(Align(8)));
  EXPECT_EQ(RI.getResourceKind(), ResourceKind::StructuredBuffer);

  Props = RI.getAnnotateProps(M);
  EXPECT_EQ(Props.first, 0x0000030cU);
  EXPECT_EQ(Props.second, 0x00000010U);
  MD = RI.getAsMetadata(M);
  EXPECT_MDEQ(MD,
              TestMD.get(0, DummyGV, "", 0, 0, 1, 12, 0, TestMD.get(1, 16)));

  // StructuredBuffer<float3> Buffer1 : register(t1);
  HandleTy = llvm::TargetExtType::get(Context, "dx.RawBuffer", Floatx3Ty,
                                      {/*IsWriteable=*/0, /*IsROV=*/0});
  RI = ResourceInfo(
      /*RecordID=*/1, /*Space=*/0, /*LowerBound=*/1, /*Size=*/1, HandleTy);

  EXPECT_EQ(RI.getResourceClass(), ResourceClass::SRV);
  ASSERT_EQ(RI.isStruct(), true);
  EXPECT_EQ(RI.getStruct(DL).Stride, 12u);
  EXPECT_EQ(RI.getStruct(DL).AlignLog2, 0u);
  EXPECT_EQ(RI.getResourceKind(), ResourceKind::StructuredBuffer);

  Props = RI.getAnnotateProps(M);
  EXPECT_EQ(Props.first, 0x0000000cU);
  EXPECT_EQ(Props.second, 0x0000000cU);
  MD = RI.getAsMetadata(M);
  EXPECT_MDEQ(MD,
              TestMD.get(1, DummyGV, "", 0, 1, 1, 12, 0, TestMD.get(1, 12)));

  // Texture2D<float4> ColorMapTexture : register(t2);
  HandleTy =
      llvm::TargetExtType::get(Context, "dx.Texture", Floatx4Ty,
                               {/*IsWriteable=*/0, /*IsROV=*/0, /*IsSigned=*/0,
                                llvm::to_underlying(ResourceKind::Texture2D)});
  RI = ResourceInfo(
      /*RecordID=*/2, /*Space=*/0, /*LowerBound=*/2, /*Size=*/1, HandleTy);

  EXPECT_EQ(RI.getResourceClass(), ResourceClass::SRV);
  ASSERT_EQ(RI.isTyped(), true);
  EXPECT_EQ(RI.getTyped().ElementTy, ElementType::F32);
  EXPECT_EQ(RI.getTyped().ElementCount, 4u);
  EXPECT_EQ(RI.getResourceKind(), ResourceKind::Texture2D);

  Props = RI.getAnnotateProps(M);
  EXPECT_EQ(Props.first, 0x00000002U);
  EXPECT_EQ(Props.second, 0x00000409U);
  MD = RI.getAsMetadata(M);
  EXPECT_MDEQ(MD, TestMD.get(2, DummyGV, "", 0, 2, 1, 2, 0, TestMD.get(0, 9)));

  // Texture2DMS<float, 8> DepthBuffer : register(t0);
  HandleTy = llvm::TargetExtType::get(
      Context, "dx.MSTexture", FloatTy,
      {/*IsWriteable=*/0, /*SampleCount=*/8,
       /*IsSigned=*/0, llvm::to_underlying(ResourceKind::Texture2DMS)});
  RI = ResourceInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1, HandleTy);

  EXPECT_EQ(RI.getResourceClass(), ResourceClass::SRV);
  ASSERT_EQ(RI.isTyped(), true);
  EXPECT_EQ(RI.getTyped().ElementTy, ElementType::F32);
  EXPECT_EQ(RI.getTyped().ElementCount, 1u);
  ASSERT_EQ(RI.isMultiSample(), true);
  EXPECT_EQ(RI.getMultiSampleCount(), 8u);
  EXPECT_EQ(RI.getResourceKind(), ResourceKind::Texture2DMS);

  Props = RI.getAnnotateProps(M);
  EXPECT_EQ(Props.first, 0x00000003U);
  EXPECT_EQ(Props.second, 0x00080109U);
  MD = RI.getAsMetadata(M);
  EXPECT_MDEQ(MD, TestMD.get(0, DummyGV, "", 0, 0, 1, 3, 8, TestMD.get(0, 9)));

  // FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMinMip;
  HandleTy = llvm::TargetExtType::get(
      Context, "dx.FeedbackTexture", {},
      {llvm::to_underlying(SamplerFeedbackType::MinMip),
       llvm::to_underlying(ResourceKind::FeedbackTexture2D)});
  RI = ResourceInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1, HandleTy);

  EXPECT_EQ(RI.getResourceClass(), ResourceClass::UAV);
  ASSERT_EQ(RI.isFeedback(), true);
  EXPECT_EQ(RI.getFeedbackType(), SamplerFeedbackType::MinMip);
  EXPECT_EQ(RI.getResourceKind(), ResourceKind::FeedbackTexture2D);

  Props = RI.getAnnotateProps(M);
  EXPECT_EQ(Props.first, 0x00001011U);
  EXPECT_EQ(Props.second, 0U);
  MD = RI.getAsMetadata(M);
  EXPECT_MDEQ(MD, TestMD.get(0, DummyGV, "", 0, 0, 1, 17, false, false, false,
                             TestMD.get(2, 0)));

  // FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIP_REGION_USED> feedbackMipRegion;
  HandleTy = llvm::TargetExtType::get(
      Context, "dx.FeedbackTexture", {},
      {llvm::to_underlying(SamplerFeedbackType::MipRegionUsed),
       llvm::to_underlying(ResourceKind::FeedbackTexture2DArray)});
  RI = ResourceInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1, HandleTy);

  EXPECT_EQ(RI.getResourceClass(), ResourceClass::UAV);
  ASSERT_EQ(RI.isFeedback(), true);
  EXPECT_EQ(RI.getFeedbackType(), SamplerFeedbackType::MipRegionUsed);
  EXPECT_EQ(RI.getResourceKind(), ResourceKind::FeedbackTexture2DArray);

  Props = RI.getAnnotateProps(M);
  EXPECT_EQ(Props.first, 0x00001012U);
  EXPECT_EQ(Props.second, 0x00000001U);
  MD = RI.getAsMetadata(M);
  EXPECT_MDEQ(MD, TestMD.get(0, DummyGV, "", 0, 0, 1, 18, false, false, false,
                             TestMD.get(2, 1)));

  // globallycoherent RWTexture2D<int2> OutputTexture : register(u0, space2);
  HandleTy =
      llvm::TargetExtType::get(Context, "dx.Texture", Int32x2Ty,
                               {/*IsWriteable=*/1,
                                /*IsROV=*/0, /*IsSigned=*/1,
                                llvm::to_underlying(ResourceKind::Texture2D)});

  RI = ResourceInfo(
      /*RecordID=*/0, /*Space=*/2, /*LowerBound=*/0, /*Size=*/1, HandleTy,
      /*GloballyCoherent=*/true, /*HasCounter=*/false);

  EXPECT_EQ(RI.getResourceClass(), ResourceClass::UAV);
  EXPECT_EQ(RI.getUAV().GloballyCoherent, true);
  EXPECT_EQ(RI.getUAV().HasCounter, false);
  EXPECT_EQ(RI.getUAV().IsROV, false);
  EXPECT_EQ(RI.getResourceKind(), ResourceKind::Texture2D);

  Props = RI.getAnnotateProps(M);
  EXPECT_EQ(Props.first, 0x00005002U);
  EXPECT_EQ(Props.second, 0x00000204U);
  MD = RI.getAsMetadata(M);
  EXPECT_MDEQ(MD, TestMD.get(0, DummyGV, "", 2, 0, 1, 2, true, false, false,
                             TestMD.get(0, 4)));

  // RasterizerOrderedBuffer<float4> ROB;
  HandleTy = llvm::TargetExtType::get(
      Context, "dx.TypedBuffer", Floatx4Ty,
      {/*IsWriteable=*/1, /*IsROV=*/1, /*IsSigned=*/0});
  RI = ResourceInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1, HandleTy);

  EXPECT_EQ(RI.getResourceClass(), ResourceClass::UAV);
  EXPECT_EQ(RI.getUAV().GloballyCoherent, false);
  EXPECT_EQ(RI.getUAV().HasCounter, false);
  EXPECT_EQ(RI.getUAV().IsROV, true);
  ASSERT_EQ(RI.isTyped(), true);
  EXPECT_EQ(RI.getTyped().ElementTy, ElementType::F32);
  EXPECT_EQ(RI.getTyped().ElementCount, 4u);
  EXPECT_EQ(RI.getResourceKind(), ResourceKind::TypedBuffer);

  Props = RI.getAnnotateProps(M);
  EXPECT_EQ(Props.first, 0x0000300aU);
  EXPECT_EQ(Props.second, 0x00000409U);
  MD = RI.getAsMetadata(M);
  EXPECT_MDEQ(MD, TestMD.get(0, DummyGV, "", 0, 0, 1, 10, false, false, true,
                             TestMD.get(0, 9)));

  // RWStructuredBuffer<ParticleMotion> g_OutputBuffer : register(u2);
  StructType *BufType1 = StructType::create(
      Context, {Floatx3Ty, FloatTy, Int32Ty}, "ParticleMotion");
  HandleTy = llvm::TargetExtType::get(Context, "dx.RawBuffer", BufType1,
                                      {/*IsWriteable=*/1, /*IsROV=*/0});
  RI = ResourceInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/2, /*Size=*/1, HandleTy,
      /*GloballyCoherent=*/false, /*HasCounter=*/true);

  EXPECT_EQ(RI.getResourceClass(), ResourceClass::UAV);
  EXPECT_EQ(RI.getUAV().GloballyCoherent, false);
  EXPECT_EQ(RI.getUAV().HasCounter, true);
  EXPECT_EQ(RI.getUAV().IsROV, false);
  ASSERT_EQ(RI.isStruct(), true);
  EXPECT_EQ(RI.getStruct(DL).Stride, 20u);
  EXPECT_EQ(RI.getStruct(DL).AlignLog2, Log2(Align(4)));
  EXPECT_EQ(RI.getResourceKind(), ResourceKind::StructuredBuffer);

  Props = RI.getAnnotateProps(M);
  EXPECT_EQ(Props.first, 0x0000920cU);
  EXPECT_EQ(Props.second, 0x00000014U);
  MD = RI.getAsMetadata(M);
  EXPECT_MDEQ(MD, TestMD.get(0, DummyGV, "", 0, 2, 1, 12, false, true, false,
                             TestMD.get(1, 20)));

  // RWTexture2DMSArray<uint, 8> g_rw_t2dmsa;
  HandleTy = llvm::TargetExtType::get(
      Context, "dx.MSTexture", Int32Ty,
      {/*IsWriteable=*/1, /*SampleCount=*/8, /*IsSigned=*/0,
       llvm::to_underlying(ResourceKind::Texture2DMSArray)});
  RI = ResourceInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1, HandleTy);

  EXPECT_EQ(RI.getResourceClass(), ResourceClass::UAV);
  EXPECT_EQ(RI.getUAV().GloballyCoherent, false);
  EXPECT_EQ(RI.getUAV().HasCounter, false);
  EXPECT_EQ(RI.getUAV().IsROV, false);
  ASSERT_EQ(RI.isTyped(), true);
  EXPECT_EQ(RI.getTyped().ElementTy, ElementType::U32);
  EXPECT_EQ(RI.getTyped().ElementCount, 1u);
  ASSERT_EQ(RI.isMultiSample(), true);
  EXPECT_EQ(RI.getMultiSampleCount(), 8u);
  EXPECT_EQ(RI.getResourceKind(), ResourceKind::Texture2DMSArray);

  Props = RI.getAnnotateProps(M);
  EXPECT_EQ(Props.first, 0x00001008U);
  EXPECT_EQ(Props.second, 0x00080105U);
  MD = RI.getAsMetadata(M);
  EXPECT_MDEQ(MD, TestMD.get(0, DummyGV, "", 0, 0, 1, 8, false, false, false,
                             TestMD.get(0, 5)));

  // cbuffer cb0 { float4 g_X; float4 g_Y; }
  StructType *CBufType0 =
      StructType::create(Context, {Floatx4Ty, Floatx4Ty}, "cb0");
  HandleTy =
      llvm::TargetExtType::get(Context, "dx.CBuffer", CBufType0, {/*Size=*/32});
  RI = ResourceInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1, HandleTy);

  EXPECT_EQ(RI.getResourceClass(), ResourceClass::CBuffer);
  EXPECT_EQ(RI.getCBufferSize(DL), 32u);
  EXPECT_EQ(RI.getResourceKind(), ResourceKind::CBuffer);

  Props = RI.getAnnotateProps(M);
  EXPECT_EQ(Props.first, 0x0000000dU);
  EXPECT_EQ(Props.second, 0x00000020U);
  MD = RI.getAsMetadata(M);
  EXPECT_MDEQ(MD, TestMD.get(0, DummyGV, "", 0, 0, 1, 32, nullptr));

  // SamplerState ColorMapSampler : register(s0);
  HandleTy = llvm::TargetExtType::get(
      Context, "dx.Sampler", {},
      {llvm::to_underlying(dxil::SamplerType::Default)});
  RI = ResourceInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1, HandleTy);

  EXPECT_EQ(RI.getResourceClass(), ResourceClass::Sampler);
  EXPECT_EQ(RI.getSamplerType(), dxil::SamplerType::Default);
  EXPECT_EQ(RI.getResourceKind(), ResourceKind::Sampler);

  Props = RI.getAnnotateProps(M);
  EXPECT_EQ(Props.first, 0x0000000eU);
  EXPECT_EQ(Props.second, 0U);
  MD = RI.getAsMetadata(M);
  EXPECT_MDEQ(MD, TestMD.get(0, DummyGV, "", 0, 0, 1, 0, nullptr));

  HandleTy = llvm::TargetExtType::get(
      Context, "dx.Sampler", {},
      {llvm::to_underlying(dxil::SamplerType::Comparison)});
  RI = ResourceInfo(
      /*RecordID=*/0, /*Space=*/0, /*LowerBound=*/0, /*Size=*/1, HandleTy);

  EXPECT_EQ(RI.getResourceClass(), ResourceClass::Sampler);
  EXPECT_EQ(RI.getSamplerType(), dxil::SamplerType::Comparison);
  EXPECT_EQ(RI.getResourceKind(), ResourceKind::Sampler);

  Props = RI.getAnnotateProps(M);
  EXPECT_EQ(Props.first, 0x0000800eU);
  EXPECT_EQ(Props.second, 0U);
  MD = RI.getAsMetadata(M);
  EXPECT_MDEQ(MD, TestMD.get(0, DummyGV, "", 0, 0, 1, 1, nullptr));
}
