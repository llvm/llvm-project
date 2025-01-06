//===- DXILResource.cpp - Representations of DXIL resources ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DXILResource.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "dxil-resource"

using namespace llvm;
using namespace dxil;

static StringRef getResourceClassName(ResourceClass RC) {
  switch (RC) {
  case ResourceClass::SRV:
    return "SRV";
  case ResourceClass::UAV:
    return "UAV";
  case ResourceClass::CBuffer:
    return "CBuffer";
  case ResourceClass::Sampler:
    return "Sampler";
  }
  llvm_unreachable("Unhandled ResourceClass");
}

static StringRef getResourceKindName(ResourceKind RK) {
  switch (RK) {
  case ResourceKind::Texture1D:
    return "Texture1D";
  case ResourceKind::Texture2D:
    return "Texture2D";
  case ResourceKind::Texture2DMS:
    return "Texture2DMS";
  case ResourceKind::Texture3D:
    return "Texture3D";
  case ResourceKind::TextureCube:
    return "TextureCube";
  case ResourceKind::Texture1DArray:
    return "Texture1DArray";
  case ResourceKind::Texture2DArray:
    return "Texture2DArray";
  case ResourceKind::Texture2DMSArray:
    return "Texture2DMSArray";
  case ResourceKind::TextureCubeArray:
    return "TextureCubeArray";
  case ResourceKind::TypedBuffer:
    return "TypedBuffer";
  case ResourceKind::RawBuffer:
    return "RawBuffer";
  case ResourceKind::StructuredBuffer:
    return "StructuredBuffer";
  case ResourceKind::CBuffer:
    return "CBuffer";
  case ResourceKind::Sampler:
    return "Sampler";
  case ResourceKind::TBuffer:
    return "TBuffer";
  case ResourceKind::RTAccelerationStructure:
    return "RTAccelerationStructure";
  case ResourceKind::FeedbackTexture2D:
    return "FeedbackTexture2D";
  case ResourceKind::FeedbackTexture2DArray:
    return "FeedbackTexture2DArray";
  case ResourceKind::NumEntries:
  case ResourceKind::Invalid:
    return "<invalid>";
  }
  llvm_unreachable("Unhandled ResourceKind");
}

static StringRef getElementTypeName(ElementType ET) {
  switch (ET) {
  case ElementType::I1:
    return "i1";
  case ElementType::I16:
    return "i16";
  case ElementType::U16:
    return "u16";
  case ElementType::I32:
    return "i32";
  case ElementType::U32:
    return "u32";
  case ElementType::I64:
    return "i64";
  case ElementType::U64:
    return "u64";
  case ElementType::F16:
    return "f16";
  case ElementType::F32:
    return "f32";
  case ElementType::F64:
    return "f64";
  case ElementType::SNormF16:
    return "snorm_f16";
  case ElementType::UNormF16:
    return "unorm_f16";
  case ElementType::SNormF32:
    return "snorm_f32";
  case ElementType::UNormF32:
    return "unorm_f32";
  case ElementType::SNormF64:
    return "snorm_f64";
  case ElementType::UNormF64:
    return "unorm_f64";
  case ElementType::PackedS8x32:
    return "p32i8";
  case ElementType::PackedU8x32:
    return "p32u8";
  case ElementType::Invalid:
    return "<invalid>";
  }
  llvm_unreachable("Unhandled ElementType");
}

static StringRef getSamplerTypeName(SamplerType ST) {
  switch (ST) {
  case SamplerType::Default:
    return "Default";
  case SamplerType::Comparison:
    return "Comparison";
  case SamplerType::Mono:
    return "Mono";
  }
  llvm_unreachable("Unhandled SamplerType");
}

static StringRef getSamplerFeedbackTypeName(SamplerFeedbackType SFT) {
  switch (SFT) {
  case SamplerFeedbackType::MinMip:
    return "MinMip";
  case SamplerFeedbackType::MipRegionUsed:
    return "MipRegionUsed";
  }
  llvm_unreachable("Unhandled SamplerFeedbackType");
}

static dxil::ElementType toDXILElementType(Type *Ty, bool IsSigned) {
  // TODO: Handle unorm, snorm, and packed.
  Ty = Ty->getScalarType();

  if (Ty->isIntegerTy()) {
    switch (Ty->getIntegerBitWidth()) {
    case 16:
      return IsSigned ? ElementType::I16 : ElementType::U16;
    case 32:
      return IsSigned ? ElementType::I32 : ElementType::U32;
    case 64:
      return IsSigned ? ElementType::I64 : ElementType::U64;
    case 1:
    default:
      return ElementType::Invalid;
    }
  } else if (Ty->isFloatTy()) {
    return ElementType::F32;
  } else if (Ty->isDoubleTy()) {
    return ElementType::F64;
  } else if (Ty->isHalfTy()) {
    return ElementType::F16;
  }

  return ElementType::Invalid;
}

ResourceTypeInfo::ResourceTypeInfo(TargetExtType *HandleTy,
                                   const dxil::ResourceClass RC_,
                                   const dxil::ResourceKind Kind_,
                                   bool GloballyCoherent, bool HasCounter)
    : HandleTy(HandleTy), GloballyCoherent(GloballyCoherent),
      HasCounter(HasCounter) {
  // If we're provided a resource class and kind, trust them.
  if (Kind_ != dxil::ResourceKind::Invalid) {
    RC = RC_;
    Kind = Kind_;
    return;
  }

  if (auto *Ty = dyn_cast<RawBufferExtType>(HandleTy)) {
    RC = Ty->isWriteable() ? ResourceClass::UAV : ResourceClass::SRV;
    Kind = Ty->isStructured() ? ResourceKind::StructuredBuffer
                              : ResourceKind::RawBuffer;
  } else if (auto *Ty = dyn_cast<TypedBufferExtType>(HandleTy)) {
    RC = Ty->isWriteable() ? ResourceClass::UAV : ResourceClass::SRV;
    Kind = ResourceKind::TypedBuffer;
  } else if (auto *Ty = dyn_cast<TextureExtType>(HandleTy)) {
    RC = Ty->isWriteable() ? ResourceClass::UAV : ResourceClass::SRV;
    Kind = Ty->getDimension();
  } else if (auto *Ty = dyn_cast<MSTextureExtType>(HandleTy)) {
    RC = Ty->isWriteable() ? ResourceClass::UAV : ResourceClass::SRV;
    Kind = Ty->getDimension();
  } else if (auto *Ty = dyn_cast<FeedbackTextureExtType>(HandleTy)) {
    RC = ResourceClass::UAV;
    Kind = Ty->getDimension();
  } else if (isa<CBufferExtType>(HandleTy)) {
    RC = ResourceClass::CBuffer;
    Kind = ResourceKind::CBuffer;
  } else if (isa<SamplerExtType>(HandleTy)) {
    RC = ResourceClass::Sampler;
    Kind = ResourceKind::Sampler;
  } else
    llvm_unreachable("Unknown handle type");
}

static void formatTypeName(SmallString<64> &Dest, StringRef Name,
                           bool isWriteable, bool isROV) {
  Dest = isWriteable ? (isROV ? "RasterizerOrdered" : "RW") : "";
  Dest += Name;
}

StructType *ResourceTypeInfo::createElementStruct() {
  SmallString<64> TypeName;

  switch (Kind) {
  case ResourceKind::Texture1D:
  case ResourceKind::Texture2D:
  case ResourceKind::Texture3D:
  case ResourceKind::TextureCube:
  case ResourceKind::Texture1DArray:
  case ResourceKind::Texture2DArray:
  case ResourceKind::TextureCubeArray: {
    auto *RTy = cast<TextureExtType>(HandleTy);
    formatTypeName(TypeName, getResourceKindName(Kind), RTy->isWriteable(),
                   RTy->isROV());
    return StructType::create(RTy->getResourceType(), TypeName);
  }
  case ResourceKind::Texture2DMS:
  case ResourceKind::Texture2DMSArray: {
    auto *RTy = cast<MSTextureExtType>(HandleTy);
    formatTypeName(TypeName, getResourceKindName(Kind), RTy->isWriteable(),
                   /*IsROV=*/false);
    return StructType::create(RTy->getResourceType(), TypeName);
  }
  case ResourceKind::TypedBuffer: {
    auto *RTy = cast<TypedBufferExtType>(HandleTy);
    formatTypeName(TypeName, getResourceKindName(Kind), RTy->isWriteable(),
                   RTy->isROV());
    return StructType::create(RTy->getResourceType(), TypeName);
  }
  case ResourceKind::RawBuffer: {
    auto *RTy = cast<RawBufferExtType>(HandleTy);
    formatTypeName(TypeName, "ByteAddressBuffer", RTy->isWriteable(),
                   RTy->isROV());
    return StructType::create(Type::getInt32Ty(HandleTy->getContext()),
                              TypeName);
  }
  case ResourceKind::StructuredBuffer: {
    auto *RTy = cast<RawBufferExtType>(HandleTy);
    formatTypeName(TypeName, "StructuredBuffer", RTy->isWriteable(),
                   RTy->isROV());
    return StructType::create(RTy->getResourceType(), TypeName);
  }
  case ResourceKind::FeedbackTexture2D:
  case ResourceKind::FeedbackTexture2DArray: {
    auto *RTy = cast<FeedbackTextureExtType>(HandleTy);
    TypeName = formatv("{0}<{1}>", getResourceKindName(Kind),
                       llvm::to_underlying(RTy->getFeedbackType()));
    return StructType::create(Type::getInt32Ty(HandleTy->getContext()),
                              TypeName);
  }
  case ResourceKind::CBuffer:
    return StructType::create(HandleTy->getContext(), "cbuffer");
  case ResourceKind::Sampler: {
    auto *RTy = cast<SamplerExtType>(HandleTy);
    TypeName = formatv("SamplerState<{0}>",
                       llvm::to_underlying(RTy->getSamplerType()));
    return StructType::create(Type::getInt32Ty(HandleTy->getContext()),
                              TypeName);
  }
  case ResourceKind::TBuffer:
  case ResourceKind::RTAccelerationStructure:
    llvm_unreachable("Unhandled resource kind");
  case ResourceKind::Invalid:
  case ResourceKind::NumEntries:
    llvm_unreachable("Invalid resource kind");
  }
  llvm_unreachable("Unhandled ResourceKind enum");
}

bool ResourceTypeInfo::isUAV() const { return RC == ResourceClass::UAV; }

bool ResourceTypeInfo::isCBuffer() const {
  return RC == ResourceClass::CBuffer;
}

bool ResourceTypeInfo::isSampler() const {
  return RC == ResourceClass::Sampler;
}

bool ResourceTypeInfo::isStruct() const {
  return Kind == ResourceKind::StructuredBuffer;
}

bool ResourceTypeInfo::isTyped() const {
  switch (Kind) {
  case ResourceKind::Texture1D:
  case ResourceKind::Texture2D:
  case ResourceKind::Texture2DMS:
  case ResourceKind::Texture3D:
  case ResourceKind::TextureCube:
  case ResourceKind::Texture1DArray:
  case ResourceKind::Texture2DArray:
  case ResourceKind::Texture2DMSArray:
  case ResourceKind::TextureCubeArray:
  case ResourceKind::TypedBuffer:
    return true;
  case ResourceKind::RawBuffer:
  case ResourceKind::StructuredBuffer:
  case ResourceKind::FeedbackTexture2D:
  case ResourceKind::FeedbackTexture2DArray:
  case ResourceKind::CBuffer:
  case ResourceKind::Sampler:
  case ResourceKind::TBuffer:
  case ResourceKind::RTAccelerationStructure:
    return false;
  case ResourceKind::Invalid:
  case ResourceKind::NumEntries:
    llvm_unreachable("Invalid resource kind");
  }
  llvm_unreachable("Unhandled ResourceKind enum");
}

bool ResourceTypeInfo::isFeedback() const {
  return Kind == ResourceKind::FeedbackTexture2D ||
         Kind == ResourceKind::FeedbackTexture2DArray;
}

bool ResourceTypeInfo::isMultiSample() const {
  return Kind == ResourceKind::Texture2DMS ||
         Kind == ResourceKind::Texture2DMSArray;
}

static bool isROV(dxil::ResourceKind Kind, TargetExtType *Ty) {
  switch (Kind) {
  case ResourceKind::Texture1D:
  case ResourceKind::Texture2D:
  case ResourceKind::Texture3D:
  case ResourceKind::TextureCube:
  case ResourceKind::Texture1DArray:
  case ResourceKind::Texture2DArray:
  case ResourceKind::TextureCubeArray:
    return cast<TextureExtType>(Ty)->isROV();
  case ResourceKind::TypedBuffer:
    return cast<TypedBufferExtType>(Ty)->isROV();
  case ResourceKind::RawBuffer:
  case ResourceKind::StructuredBuffer:
    return cast<RawBufferExtType>(Ty)->isROV();
  case ResourceKind::Texture2DMS:
  case ResourceKind::Texture2DMSArray:
  case ResourceKind::FeedbackTexture2D:
  case ResourceKind::FeedbackTexture2DArray:
    return false;
  case ResourceKind::CBuffer:
  case ResourceKind::Sampler:
  case ResourceKind::TBuffer:
  case ResourceKind::RTAccelerationStructure:
  case ResourceKind::Invalid:
  case ResourceKind::NumEntries:
    llvm_unreachable("Resource cannot be ROV");
  }
  llvm_unreachable("Unhandled ResourceKind enum");
}

ResourceTypeInfo::UAVInfo ResourceTypeInfo::getUAV() const {
  assert(isUAV() && "Not a UAV");
  return {GloballyCoherent, HasCounter, isROV(Kind, HandleTy)};
}

uint32_t ResourceTypeInfo::getCBufferSize(const DataLayout &DL) const {
  assert(isCBuffer() && "Not a CBuffer");
  return cast<CBufferExtType>(HandleTy)->getCBufferSize();
}

dxil::SamplerType ResourceTypeInfo::getSamplerType() const {
  assert(isSampler() && "Not a Sampler");
  return cast<SamplerExtType>(HandleTy)->getSamplerType();
}

ResourceTypeInfo::StructInfo
ResourceTypeInfo::getStruct(const DataLayout &DL) const {
  assert(isStruct() && "Not a Struct");

  Type *ElTy = cast<RawBufferExtType>(HandleTy)->getResourceType();

  uint32_t Stride = DL.getTypeAllocSize(ElTy);
  MaybeAlign Alignment;
  if (auto *STy = dyn_cast<StructType>(ElTy))
    Alignment = DL.getStructLayout(STy)->getAlignment();
  uint32_t AlignLog2 = Alignment ? Log2(*Alignment) : 0;
  return {Stride, AlignLog2};
}

static std::pair<Type *, bool> getTypedElementType(dxil::ResourceKind Kind,
                                                   TargetExtType *Ty) {
  switch (Kind) {
  case ResourceKind::Texture1D:
  case ResourceKind::Texture2D:
  case ResourceKind::Texture3D:
  case ResourceKind::TextureCube:
  case ResourceKind::Texture1DArray:
  case ResourceKind::Texture2DArray:
  case ResourceKind::TextureCubeArray: {
    auto *RTy = cast<TextureExtType>(Ty);
    return {RTy->getResourceType(), RTy->isSigned()};
  }
  case ResourceKind::Texture2DMS:
  case ResourceKind::Texture2DMSArray: {
    auto *RTy = cast<MSTextureExtType>(Ty);
    return {RTy->getResourceType(), RTy->isSigned()};
  }
  case ResourceKind::TypedBuffer: {
    auto *RTy = cast<TypedBufferExtType>(Ty);
    return {RTy->getResourceType(), RTy->isSigned()};
  }
  case ResourceKind::RawBuffer:
  case ResourceKind::StructuredBuffer:
  case ResourceKind::FeedbackTexture2D:
  case ResourceKind::FeedbackTexture2DArray:
  case ResourceKind::CBuffer:
  case ResourceKind::Sampler:
  case ResourceKind::TBuffer:
  case ResourceKind::RTAccelerationStructure:
  case ResourceKind::Invalid:
  case ResourceKind::NumEntries:
    llvm_unreachable("Resource is not typed");
  }
  llvm_unreachable("Unhandled ResourceKind enum");
}

ResourceTypeInfo::TypedInfo ResourceTypeInfo::getTyped() const {
  assert(isTyped() && "Not typed");

  auto [ElTy, IsSigned] = getTypedElementType(Kind, HandleTy);
  dxil::ElementType ET = toDXILElementType(ElTy, IsSigned);
  uint32_t Count = 1;
  if (auto *VTy = dyn_cast<FixedVectorType>(ElTy))
    Count = VTy->getNumElements();
  return {ET, Count};
}

dxil::SamplerFeedbackType ResourceTypeInfo::getFeedbackType() const {
  assert(isFeedback() && "Not Feedback");
  return cast<FeedbackTextureExtType>(HandleTy)->getFeedbackType();
}
uint32_t ResourceTypeInfo::getMultiSampleCount() const {
  assert(isMultiSample() && "Not MultiSampled");
  return cast<MSTextureExtType>(HandleTy)->getSampleCount();
}

bool ResourceTypeInfo::operator==(const ResourceTypeInfo &RHS) const {
  return std::tie(HandleTy, GloballyCoherent, HasCounter) ==
         std::tie(RHS.HandleTy, RHS.GloballyCoherent, RHS.HasCounter);
}

bool ResourceTypeInfo::operator<(const ResourceTypeInfo &RHS) const {
  // An empty datalayout is sufficient for sorting purposes.
  DataLayout DummyDL;
  if (std::tie(RC, Kind) < std::tie(RHS.RC, RHS.Kind))
    return true;
  if (isCBuffer() && RHS.isCBuffer() &&
      getCBufferSize(DummyDL) < RHS.getCBufferSize(DummyDL))
    return true;
  if (isSampler() && RHS.isSampler() && getSamplerType() < RHS.getSamplerType())
    return true;
  if (isUAV() && RHS.isUAV() && getUAV() < RHS.getUAV())
    return true;
  if (isStruct() && RHS.isStruct() &&
      getStruct(DummyDL) < RHS.getStruct(DummyDL))
    return true;
  if (isFeedback() && RHS.isFeedback() &&
      getFeedbackType() < RHS.getFeedbackType())
    return true;
  if (isTyped() && RHS.isTyped() && getTyped() < RHS.getTyped())
    return true;
  if (isMultiSample() && RHS.isMultiSample() &&
      getMultiSampleCount() < RHS.getMultiSampleCount())
    return true;
  return false;
}

void ResourceTypeInfo::print(raw_ostream &OS, const DataLayout &DL) const {
  OS << "  Class: " << getResourceClassName(RC) << "\n"
     << "  Kind: " << getResourceKindName(Kind) << "\n";

  if (isCBuffer()) {
    OS << "  CBuffer size: " << getCBufferSize(DL) << "\n";
  } else if (isSampler()) {
    OS << "  Sampler Type: " << getSamplerTypeName(getSamplerType()) << "\n";
  } else {
    if (isUAV()) {
      UAVInfo UAVFlags = getUAV();
      OS << "  Globally Coherent: " << UAVFlags.GloballyCoherent << "\n"
         << "  HasCounter: " << UAVFlags.HasCounter << "\n"
         << "  IsROV: " << UAVFlags.IsROV << "\n";
    }
    if (isMultiSample())
      OS << "  Sample Count: " << getMultiSampleCount() << "\n";

    if (isStruct()) {
      StructInfo Struct = getStruct(DL);
      OS << "  Buffer Stride: " << Struct.Stride << "\n";
      OS << "  Alignment: " << Struct.AlignLog2 << "\n";
    } else if (isTyped()) {
      TypedInfo Typed = getTyped();
      OS << "  Element Type: " << getElementTypeName(Typed.ElementTy) << "\n"
         << "  Element Count: " << Typed.ElementCount << "\n";
    } else if (isFeedback())
      OS << "  Feedback Type: " << getSamplerFeedbackTypeName(getFeedbackType())
         << "\n";
  }
}

GlobalVariable *ResourceBindingInfo::createSymbol(Module &M, StructType *Ty,
                                                  StringRef Name) {
  assert(!Symbol && "Symbol has already been created");
  Symbol = new GlobalVariable(M, Ty, /*isConstant=*/true,
                              GlobalValue::ExternalLinkage,
                              /*Initializer=*/nullptr, Name);
  return Symbol;
}

MDTuple *ResourceBindingInfo::getAsMetadata(Module &M,
                                            dxil::ResourceTypeInfo &RTI) const {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();

  SmallVector<Metadata *, 11> MDVals;

  Type *I32Ty = Type::getInt32Ty(Ctx);
  Type *I1Ty = Type::getInt1Ty(Ctx);
  auto getIntMD = [&I32Ty](uint32_t V) {
    return ConstantAsMetadata::get(
        Constant::getIntegerValue(I32Ty, APInt(32, V)));
  };
  auto getBoolMD = [&I1Ty](uint32_t V) {
    return ConstantAsMetadata::get(
        Constant::getIntegerValue(I1Ty, APInt(1, V)));
  };

  MDVals.push_back(getIntMD(Binding.RecordID));
  assert(Symbol && "Cannot yet create useful resource metadata without symbol");
  MDVals.push_back(ValueAsMetadata::get(Symbol));
  MDVals.push_back(MDString::get(Ctx, Symbol->getName()));
  MDVals.push_back(getIntMD(Binding.Space));
  MDVals.push_back(getIntMD(Binding.LowerBound));
  MDVals.push_back(getIntMD(Binding.Size));

  if (RTI.isCBuffer()) {
    MDVals.push_back(getIntMD(RTI.getCBufferSize(DL)));
    MDVals.push_back(nullptr);
  } else if (RTI.isSampler()) {
    MDVals.push_back(getIntMD(llvm::to_underlying(RTI.getSamplerType())));
    MDVals.push_back(nullptr);
  } else {
    MDVals.push_back(getIntMD(llvm::to_underlying(RTI.getResourceKind())));

    if (RTI.isUAV()) {
      ResourceTypeInfo::UAVInfo UAVFlags = RTI.getUAV();
      MDVals.push_back(getBoolMD(UAVFlags.GloballyCoherent));
      MDVals.push_back(getBoolMD(UAVFlags.HasCounter));
      MDVals.push_back(getBoolMD(UAVFlags.IsROV));
    } else {
      // All SRVs include sample count in the metadata, but it's only meaningful
      // for multi-sampled textured. Also, UAVs can be multisampled in SM6.7+,
      // but this just isn't reflected in the metadata at all.
      uint32_t SampleCount =
          RTI.isMultiSample() ? RTI.getMultiSampleCount() : 0;
      MDVals.push_back(getIntMD(SampleCount));
    }

    // Further properties are attached to a metadata list of tag-value pairs.
    SmallVector<Metadata *> Tags;
    if (RTI.isStruct()) {
      Tags.push_back(
          getIntMD(llvm::to_underlying(ExtPropTags::StructuredBufferStride)));
      Tags.push_back(getIntMD(RTI.getStruct(DL).Stride));
    } else if (RTI.isTyped()) {
      Tags.push_back(getIntMD(llvm::to_underlying(ExtPropTags::ElementType)));
      Tags.push_back(getIntMD(llvm::to_underlying(RTI.getTyped().ElementTy)));
    } else if (RTI.isFeedback()) {
      Tags.push_back(
          getIntMD(llvm::to_underlying(ExtPropTags::SamplerFeedbackKind)));
      Tags.push_back(getIntMD(llvm::to_underlying(RTI.getFeedbackType())));
    }
    MDVals.push_back(Tags.empty() ? nullptr : MDNode::get(Ctx, Tags));
  }

  return MDNode::get(Ctx, MDVals);
}

std::pair<uint32_t, uint32_t>
ResourceBindingInfo::getAnnotateProps(Module &M,
                                      dxil::ResourceTypeInfo &RTI) const {
  const DataLayout &DL = M.getDataLayout();

  uint32_t ResourceKind = llvm::to_underlying(RTI.getResourceKind());
  uint32_t AlignLog2 = RTI.isStruct() ? RTI.getStruct(DL).AlignLog2 : 0;
  bool IsUAV = RTI.isUAV();
  ResourceTypeInfo::UAVInfo UAVFlags =
      IsUAV ? RTI.getUAV() : ResourceTypeInfo::UAVInfo{};
  bool IsROV = IsUAV && UAVFlags.IsROV;
  bool IsGloballyCoherent = IsUAV && UAVFlags.GloballyCoherent;
  uint8_t SamplerCmpOrHasCounter = 0;
  if (IsUAV)
    SamplerCmpOrHasCounter = UAVFlags.HasCounter;
  else if (RTI.isSampler())
    SamplerCmpOrHasCounter = RTI.getSamplerType() == SamplerType::Comparison;

  // TODO: Document this format. Currently the only reference is the
  // implementation of dxc's DxilResourceProperties struct.
  uint32_t Word0 = 0;
  Word0 |= ResourceKind & 0xFF;
  Word0 |= (AlignLog2 & 0xF) << 8;
  Word0 |= (IsUAV & 1) << 12;
  Word0 |= (IsROV & 1) << 13;
  Word0 |= (IsGloballyCoherent & 1) << 14;
  Word0 |= (SamplerCmpOrHasCounter & 1) << 15;

  uint32_t Word1 = 0;
  if (RTI.isStruct())
    Word1 = RTI.getStruct(DL).Stride;
  else if (RTI.isCBuffer())
    Word1 = RTI.getCBufferSize(DL);
  else if (RTI.isFeedback())
    Word1 = llvm::to_underlying(RTI.getFeedbackType());
  else if (RTI.isTyped()) {
    ResourceTypeInfo::TypedInfo Typed = RTI.getTyped();
    uint32_t CompType = llvm::to_underlying(Typed.ElementTy);
    uint32_t CompCount = Typed.ElementCount;
    uint32_t SampleCount = RTI.isMultiSample() ? RTI.getMultiSampleCount() : 0;

    Word1 |= (CompType & 0xFF) << 0;
    Word1 |= (CompCount & 0xFF) << 8;
    Word1 |= (SampleCount & 0xFF) << 16;
  }

  return {Word0, Word1};
}

void ResourceBindingInfo::print(raw_ostream &OS, dxil::ResourceTypeInfo &RTI,
                                const DataLayout &DL) const {
  if (Symbol) {
    OS << "  Symbol: ";
    Symbol->printAsOperand(OS);
    OS << "\n";
  }

  OS << "  Binding:\n"
     << "    Record ID: " << Binding.RecordID << "\n"
     << "    Space: " << Binding.Space << "\n"
     << "    Lower Bound: " << Binding.LowerBound << "\n"
     << "    Size: " << Binding.Size << "\n";

  RTI.print(OS, DL);
}

//===----------------------------------------------------------------------===//

bool DXILResourceTypeMap::invalidate(Module &M, const PreservedAnalyses &PA,
                                     ModuleAnalysisManager::Invalidator &Inv) {
  // Passes that introduce resource types must explicitly invalidate this pass.
  auto PAC = PA.getChecker<DXILResourceTypeAnalysis>();
  return !PAC.preservedWhenStateless();
}

//===----------------------------------------------------------------------===//

void DXILBindingMap::populate(Module &M, DXILResourceTypeMap &DRTM) {
  SmallVector<std::tuple<CallInst *, ResourceBindingInfo, ResourceTypeInfo>>
      CIToInfos;

  for (Function &F : M.functions()) {
    if (!F.isDeclaration())
      continue;
    LLVM_DEBUG(dbgs() << "Function: " << F.getName() << "\n");
    Intrinsic::ID ID = F.getIntrinsicID();
    switch (ID) {
    default:
      continue;
    case Intrinsic::dx_resource_handlefrombinding: {
      auto *HandleTy = cast<TargetExtType>(F.getReturnType());
      ResourceTypeInfo &RTI = DRTM[HandleTy];

      for (User *U : F.users())
        if (CallInst *CI = dyn_cast<CallInst>(U)) {
          LLVM_DEBUG(dbgs() << "  Visiting: " << *U << "\n");
          uint32_t Space =
              cast<ConstantInt>(CI->getArgOperand(0))->getZExtValue();
          uint32_t LowerBound =
              cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
          uint32_t Size =
              cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
          ResourceBindingInfo RBI = ResourceBindingInfo{
              /*RecordID=*/0, Space, LowerBound, Size, HandleTy};

          CIToInfos.emplace_back(CI, RBI, RTI);
        }

      break;
    }
    }
  }

  llvm::stable_sort(CIToInfos, [](auto &LHS, auto &RHS) {
    const auto &[LCI, LRBI, LRTI] = LHS;
    const auto &[RCI, RRBI, RRTI] = RHS;
    // Sort by resource class first for grouping purposes, and then by the
    // binding and type so we can remove duplicates.
    ResourceClass LRC = LRTI.getResourceClass();
    ResourceClass RRC = RRTI.getResourceClass();

    return std::tie(LRC, LRBI, LRTI) < std::tie(RRC, RRBI, RRTI);
  });
  for (auto [CI, RBI, RTI] : CIToInfos) {
    if (Infos.empty() || RBI != Infos.back())
      Infos.push_back(RBI);
    CallMap[CI] = Infos.size() - 1;
  }

  unsigned Size = Infos.size();
  // In DXC, Record ID is unique per resource type. Match that.
  FirstUAV = FirstCBuffer = FirstSampler = Size;
  uint32_t NextID = 0;
  for (unsigned I = 0, E = Size; I != E; ++I) {
    ResourceBindingInfo &RBI = Infos[I];
    ResourceTypeInfo &RTI = DRTM[RBI.getHandleTy()];
    if (RTI.isUAV() && FirstUAV == Size) {
      FirstUAV = I;
      NextID = 0;
    } else if (RTI.isCBuffer() && FirstCBuffer == Size) {
      FirstCBuffer = I;
      NextID = 0;
    } else if (RTI.isSampler() && FirstSampler == Size) {
      FirstSampler = I;
      NextID = 0;
    }

    // Adjust the resource binding to use the next ID.
    RBI.setBindingID(NextID++);
  }
}

void DXILBindingMap::print(raw_ostream &OS, DXILResourceTypeMap &DRTM,
                           const DataLayout &DL) const {
  for (unsigned I = 0, E = Infos.size(); I != E; ++I) {
    OS << "Binding " << I << ":\n";
    const dxil::ResourceBindingInfo &RBI = Infos[I];
    RBI.print(OS, DRTM[RBI.getHandleTy()], DL);
    OS << "\n";
  }

  for (const auto &[CI, Index] : CallMap) {
    OS << "Call bound to " << Index << ":";
    CI->print(OS);
    OS << "\n";
  }
}

//===----------------------------------------------------------------------===//

AnalysisKey DXILResourceTypeAnalysis::Key;
AnalysisKey DXILResourceBindingAnalysis::Key;

DXILBindingMap DXILResourceBindingAnalysis::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  DXILBindingMap Data;
  DXILResourceTypeMap &DRTM = AM.getResult<DXILResourceTypeAnalysis>(M);
  Data.populate(M, DRTM);
  return Data;
}

PreservedAnalyses
DXILResourceBindingPrinterPass::run(Module &M, ModuleAnalysisManager &AM) {
  DXILBindingMap &DBM = AM.getResult<DXILResourceBindingAnalysis>(M);
  DXILResourceTypeMap &DRTM = AM.getResult<DXILResourceTypeAnalysis>(M);

  DBM.print(OS, DRTM, M.getDataLayout());
  return PreservedAnalyses::all();
}

void DXILResourceTypeWrapperPass::anchor() {}

DXILResourceTypeWrapperPass::DXILResourceTypeWrapperPass() : ImmutablePass(ID) {
  initializeDXILResourceTypeWrapperPassPass(*PassRegistry::getPassRegistry());
}

INITIALIZE_PASS(DXILResourceTypeWrapperPass, "dxil-resource-type",
                "DXIL Resource Type Analysis", false, true)
char DXILResourceTypeWrapperPass::ID = 0;

ModulePass *llvm::createDXILResourceTypeWrapperPassPass() {
  return new DXILResourceTypeWrapperPass();
}

DXILResourceBindingWrapperPass::DXILResourceBindingWrapperPass()
    : ModulePass(ID) {
  initializeDXILResourceBindingWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

DXILResourceBindingWrapperPass::~DXILResourceBindingWrapperPass() = default;

void DXILResourceBindingWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequiredTransitive<DXILResourceTypeWrapperPass>();
  AU.setPreservesAll();
}

bool DXILResourceBindingWrapperPass::runOnModule(Module &M) {
  Map.reset(new DXILBindingMap());

  DRTM = &getAnalysis<DXILResourceTypeWrapperPass>().getResourceTypeMap();
  Map->populate(M, *DRTM);

  return false;
}

void DXILResourceBindingWrapperPass::releaseMemory() { Map.reset(); }

void DXILResourceBindingWrapperPass::print(raw_ostream &OS,
                                           const Module *M) const {
  if (!Map) {
    OS << "No resource map has been built!\n";
    return;
  }
  Map->print(OS, *DRTM, M->getDataLayout());
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
void DXILResourceBindingWrapperPass::dump() const { print(dbgs(), nullptr); }
#endif

INITIALIZE_PASS(DXILResourceBindingWrapperPass, "dxil-resource-binding",
                "DXIL Resource Binding Analysis", false, true)
char DXILResourceBindingWrapperPass::ID = 0;

ModulePass *llvm::createDXILResourceBindingWrapperPassPass() {
  return new DXILResourceBindingWrapperPass();
}
