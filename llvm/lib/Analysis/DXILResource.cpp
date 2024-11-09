//===- DXILResource.cpp - Representations of DXIL resources ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DXILResource.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

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

bool ResourceInfo::isUAV() const { return RC == ResourceClass::UAV; }

bool ResourceInfo::isCBuffer() const { return RC == ResourceClass::CBuffer; }

bool ResourceInfo::isSampler() const { return RC == ResourceClass::Sampler; }

bool ResourceInfo::isStruct() const {
  return Kind == ResourceKind::StructuredBuffer;
}

bool ResourceInfo::isTyped() const {
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

bool ResourceInfo::isFeedback() const {
  return Kind == ResourceKind::FeedbackTexture2D ||
         Kind == ResourceKind::FeedbackTexture2DArray;
}

bool ResourceInfo::isMultiSample() const {
  return Kind == ResourceKind::Texture2DMS ||
         Kind == ResourceKind::Texture2DMSArray;
}

ResourceInfo ResourceInfo::SRV(Value *Symbol, StringRef Name,
                               ElementType ElementTy, uint32_t ElementCount,
                               ResourceKind Kind) {
  ResourceInfo RI(ResourceClass::SRV, Kind, Symbol, Name);
  assert(RI.isTyped() && !(RI.isStruct() || RI.isMultiSample()) &&
         "Invalid ResourceKind for SRV constructor.");
  RI.setTyped(ElementTy, ElementCount);
  return RI;
}

ResourceInfo ResourceInfo::RawBuffer(Value *Symbol, StringRef Name) {
  ResourceInfo RI(ResourceClass::SRV, ResourceKind::RawBuffer, Symbol, Name);
  return RI;
}

ResourceInfo ResourceInfo::StructuredBuffer(Value *Symbol, StringRef Name,
                                            uint32_t Stride,
                                            MaybeAlign Alignment) {
  ResourceInfo RI(ResourceClass::SRV, ResourceKind::StructuredBuffer, Symbol,
                  Name);
  RI.setStruct(Stride, Alignment);
  return RI;
}

ResourceInfo ResourceInfo::Texture2DMS(Value *Symbol, StringRef Name,
                                       ElementType ElementTy,
                                       uint32_t ElementCount,
                                       uint32_t SampleCount) {
  ResourceInfo RI(ResourceClass::SRV, ResourceKind::Texture2DMS, Symbol, Name);
  RI.setTyped(ElementTy, ElementCount);
  RI.setMultiSample(SampleCount);
  return RI;
}

ResourceInfo ResourceInfo::Texture2DMSArray(Value *Symbol, StringRef Name,
                                            ElementType ElementTy,
                                            uint32_t ElementCount,
                                            uint32_t SampleCount) {
  ResourceInfo RI(ResourceClass::SRV, ResourceKind::Texture2DMSArray, Symbol,
                  Name);
  RI.setTyped(ElementTy, ElementCount);
  RI.setMultiSample(SampleCount);
  return RI;
}

ResourceInfo ResourceInfo::UAV(Value *Symbol, StringRef Name,
                               ElementType ElementTy, uint32_t ElementCount,
                               bool GloballyCoherent, bool IsROV,
                               ResourceKind Kind) {
  ResourceInfo RI(ResourceClass::UAV, Kind, Symbol, Name);
  assert(RI.isTyped() && !(RI.isStruct() || RI.isMultiSample()) &&
         "Invalid ResourceKind for UAV constructor.");
  RI.setTyped(ElementTy, ElementCount);
  RI.setUAV(GloballyCoherent, /*HasCounter=*/false, IsROV);
  return RI;
}

ResourceInfo ResourceInfo::RWRawBuffer(Value *Symbol, StringRef Name,
                                       bool GloballyCoherent, bool IsROV) {
  ResourceInfo RI(ResourceClass::UAV, ResourceKind::RawBuffer, Symbol, Name);
  RI.setUAV(GloballyCoherent, /*HasCounter=*/false, IsROV);
  return RI;
}

ResourceInfo ResourceInfo::RWStructuredBuffer(Value *Symbol, StringRef Name,
                                              uint32_t Stride,
                                              MaybeAlign Alignment,
                                              bool GloballyCoherent, bool IsROV,
                                              bool HasCounter) {
  ResourceInfo RI(ResourceClass::UAV, ResourceKind::StructuredBuffer, Symbol,
                  Name);
  RI.setStruct(Stride, Alignment);
  RI.setUAV(GloballyCoherent, HasCounter, IsROV);
  return RI;
}

ResourceInfo ResourceInfo::RWTexture2DMS(Value *Symbol, StringRef Name,
                                         ElementType ElementTy,
                                         uint32_t ElementCount,
                                         uint32_t SampleCount,
                                         bool GloballyCoherent) {
  ResourceInfo RI(ResourceClass::UAV, ResourceKind::Texture2DMS, Symbol, Name);
  RI.setTyped(ElementTy, ElementCount);
  RI.setUAV(GloballyCoherent, /*HasCounter=*/false, /*IsROV=*/false);
  RI.setMultiSample(SampleCount);
  return RI;
}

ResourceInfo ResourceInfo::RWTexture2DMSArray(Value *Symbol, StringRef Name,
                                              ElementType ElementTy,
                                              uint32_t ElementCount,
                                              uint32_t SampleCount,
                                              bool GloballyCoherent) {
  ResourceInfo RI(ResourceClass::UAV, ResourceKind::Texture2DMSArray, Symbol,
                  Name);
  RI.setTyped(ElementTy, ElementCount);
  RI.setUAV(GloballyCoherent, /*HasCounter=*/false, /*IsROV=*/false);
  RI.setMultiSample(SampleCount);
  return RI;
}

ResourceInfo ResourceInfo::FeedbackTexture2D(Value *Symbol, StringRef Name,
                                             SamplerFeedbackType FeedbackTy) {
  ResourceInfo RI(ResourceClass::UAV, ResourceKind::FeedbackTexture2D, Symbol,
                  Name);
  RI.setUAV(/*GloballyCoherent=*/false, /*HasCounter=*/false, /*IsROV=*/false);
  RI.setFeedback(FeedbackTy);
  return RI;
}

ResourceInfo
ResourceInfo::FeedbackTexture2DArray(Value *Symbol, StringRef Name,
                                     SamplerFeedbackType FeedbackTy) {
  ResourceInfo RI(ResourceClass::UAV, ResourceKind::FeedbackTexture2DArray,
                  Symbol, Name);
  RI.setUAV(/*GloballyCoherent=*/false, /*HasCounter=*/false, /*IsROV=*/false);
  RI.setFeedback(FeedbackTy);
  return RI;
}

ResourceInfo ResourceInfo::CBuffer(Value *Symbol, StringRef Name,
                                   uint32_t Size) {
  ResourceInfo RI(ResourceClass::CBuffer, ResourceKind::CBuffer, Symbol, Name);
  RI.setCBuffer(Size);
  return RI;
}

ResourceInfo ResourceInfo::Sampler(Value *Symbol, StringRef Name,
                                   SamplerType SamplerTy) {
  ResourceInfo RI(ResourceClass::Sampler, ResourceKind::Sampler, Symbol, Name);
  RI.setSampler(SamplerTy);
  return RI;
}

bool ResourceInfo::operator==(const ResourceInfo &RHS) const {
  if (std::tie(Symbol, Name, Binding, RC, Kind) !=
      std::tie(RHS.Symbol, RHS.Name, RHS.Binding, RHS.RC, RHS.Kind))
    return false;
  if (isCBuffer() && RHS.isCBuffer() && CBufferSize != RHS.CBufferSize)
    return false;
  if (isSampler() && RHS.isSampler() && SamplerTy != RHS.SamplerTy)
    return false;
  if (isUAV() && RHS.isUAV() && UAVFlags != RHS.UAVFlags)
    return false;
  if (isStruct() && RHS.isStruct() && Struct != RHS.Struct)
    return false;
  if (isFeedback() && RHS.isFeedback() && Feedback != RHS.Feedback)
    return false;
  if (isTyped() && RHS.isTyped() && Typed != RHS.Typed)
    return false;
  if (isMultiSample() && RHS.isMultiSample() && MultiSample != RHS.MultiSample)
    return false;
  return true;
}

bool ResourceInfo::operator<(const ResourceInfo &RHS) const {
  // Skip the symbol to avoid non-determinism, and the name to keep a consistent
  // ordering even when we strip reflection data.
  if (std::tie(Binding, RC, Kind) < std::tie(RHS.Binding, RHS.RC, RHS.Kind))
    return true;
  if (isCBuffer() && RHS.isCBuffer() && CBufferSize < RHS.CBufferSize)
    return true;
  if (isSampler() && RHS.isSampler() && SamplerTy < RHS.SamplerTy)
    return true;
  if (isUAV() && RHS.isUAV() && UAVFlags < RHS.UAVFlags)
    return true;
  if (isStruct() && RHS.isStruct() && Struct < RHS.Struct)
    return true;
  if (isFeedback() && RHS.isFeedback() && Feedback < RHS.Feedback)
    return true;
  if (isTyped() && RHS.isTyped() && Typed < RHS.Typed)
    return true;
  if (isMultiSample() && RHS.isMultiSample() && MultiSample < RHS.MultiSample)
    return true;
  return false;
}

MDTuple *ResourceInfo::getAsMetadata(LLVMContext &Ctx) const {
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
  MDVals.push_back(ValueAsMetadata::get(Symbol));
  MDVals.push_back(MDString::get(Ctx, Name));
  MDVals.push_back(getIntMD(Binding.Space));
  MDVals.push_back(getIntMD(Binding.LowerBound));
  MDVals.push_back(getIntMD(Binding.Size));

  if (isCBuffer()) {
    MDVals.push_back(getIntMD(CBufferSize));
    MDVals.push_back(nullptr);
  } else if (isSampler()) {
    MDVals.push_back(getIntMD(llvm::to_underlying(SamplerTy)));
    MDVals.push_back(nullptr);
  } else {
    MDVals.push_back(getIntMD(llvm::to_underlying(Kind)));

    if (isUAV()) {
      MDVals.push_back(getBoolMD(UAVFlags.GloballyCoherent));
      MDVals.push_back(getBoolMD(UAVFlags.HasCounter));
      MDVals.push_back(getBoolMD(UAVFlags.IsROV));
    } else {
      // All SRVs include sample count in the metadata, but it's only meaningful
      // for multi-sampled textured. Also, UAVs can be multisampled in SM6.7+,
      // but this just isn't reflected in the metadata at all.
      uint32_t SampleCount = isMultiSample() ? MultiSample.Count : 0;
      MDVals.push_back(getIntMD(SampleCount));
    }

    // Further properties are attached to a metadata list of tag-value pairs.
    SmallVector<Metadata *> Tags;
    if (isStruct()) {
      Tags.push_back(
          getIntMD(llvm::to_underlying(ExtPropTags::StructuredBufferStride)));
      Tags.push_back(getIntMD(Struct.Stride));
    } else if (isTyped()) {
      Tags.push_back(getIntMD(llvm::to_underlying(ExtPropTags::ElementType)));
      Tags.push_back(getIntMD(llvm::to_underlying(Typed.ElementTy)));
    } else if (isFeedback()) {
      Tags.push_back(
          getIntMD(llvm::to_underlying(ExtPropTags::SamplerFeedbackKind)));
      Tags.push_back(getIntMD(llvm::to_underlying(Feedback.Type)));
    }
    MDVals.push_back(Tags.empty() ? nullptr : MDNode::get(Ctx, Tags));
  }

  return MDNode::get(Ctx, MDVals);
}

std::pair<uint32_t, uint32_t> ResourceInfo::getAnnotateProps() const {
  uint32_t ResourceKind = llvm::to_underlying(Kind);
  uint32_t AlignLog2 = isStruct() ? Struct.AlignLog2 : 0;
  bool IsUAV = isUAV();
  bool IsROV = IsUAV && UAVFlags.IsROV;
  bool IsGloballyCoherent = IsUAV && UAVFlags.GloballyCoherent;
  uint8_t SamplerCmpOrHasCounter = 0;
  if (IsUAV)
    SamplerCmpOrHasCounter = UAVFlags.HasCounter;
  else if (isSampler())
    SamplerCmpOrHasCounter = SamplerTy == SamplerType::Comparison;

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
  if (isStruct())
    Word1 = Struct.Stride;
  else if (isCBuffer())
    Word1 = CBufferSize;
  else if (isFeedback())
    Word1 = llvm::to_underlying(Feedback.Type);
  else if (isTyped()) {
    uint32_t CompType = llvm::to_underlying(Typed.ElementTy);
    uint32_t CompCount = Typed.ElementCount;
    uint32_t SampleCount = isMultiSample() ? MultiSample.Count : 0;

    Word1 |= (CompType & 0xFF) << 0;
    Word1 |= (CompCount & 0xFF) << 8;
    Word1 |= (SampleCount & 0xFF) << 16;
  }

  return {Word0, Word1};
}

void ResourceInfo::print(raw_ostream &OS) const {
  OS << "  Symbol: ";
  Symbol->printAsOperand(OS);
  OS << "\n";

  OS << "  Name: \"" << Name << "\"\n"
     << "  Binding:\n"
     << "    Record ID: " << Binding.RecordID << "\n"
     << "    Space: " << Binding.Space << "\n"
     << "    Lower Bound: " << Binding.LowerBound << "\n"
     << "    Size: " << Binding.Size << "\n"
     << "  Class: " << getResourceClassName(RC) << "\n"
     << "  Kind: " << getResourceKindName(Kind) << "\n";

  if (isCBuffer()) {
    OS << "  CBuffer size: " << CBufferSize << "\n";
  } else if (isSampler()) {
    OS << "  Sampler Type: " << getSamplerTypeName(SamplerTy) << "\n";
  } else {
    if (isUAV()) {
      OS << "  Globally Coherent: " << UAVFlags.GloballyCoherent << "\n"
         << "  HasCounter: " << UAVFlags.HasCounter << "\n"
         << "  IsROV: " << UAVFlags.IsROV << "\n";
    }
    if (isMultiSample())
      OS << "  Sample Count: " << MultiSample.Count << "\n";

    if (isStruct()) {
      OS << "  Buffer Stride: " << Struct.Stride << "\n";
      OS << "  Alignment: " << Struct.AlignLog2 << "\n";
    } else if (isTyped()) {
      OS << "  Element Type: " << getElementTypeName(Typed.ElementTy) << "\n"
         << "  Element Count: " << Typed.ElementCount << "\n";
    } else if (isFeedback())
      OS << "  Feedback Type: " << getSamplerFeedbackTypeName(Feedback.Type)
         << "\n";
  }
}

//===----------------------------------------------------------------------===//
// ResourceMapper

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

namespace {

class ResourceMapper {
  Module &M;
  LLVMContext &Context;
  SmallVector<std::pair<CallInst *, dxil::ResourceInfo>> Resources;

public:
  ResourceMapper(Module &M) : M(M), Context(M.getContext()) {}

  void diagnoseHandle(CallInst *CI, const Twine &Msg,
                      DiagnosticSeverity Severity = DS_Error) {
    std::string S;
    raw_string_ostream SS(S);
    CI->printAsOperand(SS);
    DiagnosticInfoUnsupported Diag(*CI->getFunction(), Msg + ": " + SS.str(),
                                   CI->getDebugLoc(), Severity);
    Context.diagnose(Diag);
  }

  ResourceInfo *mapBufferType(CallInst *CI, TargetExtType *HandleTy,
                              bool IsTyped) {
    if (HandleTy->getNumTypeParameters() != 1 ||
        HandleTy->getNumIntParameters() != (IsTyped ? 3 : 2)) {
      diagnoseHandle(CI, Twine("Invalid buffer target type"));
      return nullptr;
    }

    Type *ElTy = HandleTy->getTypeParameter(0);
    unsigned IsWriteable = HandleTy->getIntParameter(0);
    unsigned IsROV = HandleTy->getIntParameter(1);
    bool IsSigned = IsTyped && HandleTy->getIntParameter(2);

    ResourceClass RC = IsWriteable ? ResourceClass::UAV : ResourceClass::SRV;
    ResourceKind Kind;
    if (IsTyped)
      Kind = ResourceKind::TypedBuffer;
    else if (ElTy->isIntegerTy(8))
      Kind = ResourceKind::RawBuffer;
    else
      Kind = ResourceKind::StructuredBuffer;

    // TODO: We need to lower to a typed pointer, can we smuggle the type
    // through?
    Value *Symbol = UndefValue::get(PointerType::getUnqual(Context));
    // TODO: We don't actually keep track of the name right now...
    StringRef Name = "";

    // Note that we return a pointer into the vector's storage. This is okay as
    // long as we don't add more elements until we're done with the pointer.
    auto &Pair =
        Resources.emplace_back(CI, ResourceInfo{RC, Kind, Symbol, Name});
    ResourceInfo *RI = &Pair.second;

    if (RI->isUAV())
      // TODO: We need analysis for GloballyCoherent and HasCounter
      RI->setUAV(false, false, IsROV);

    if (RI->isTyped()) {
      dxil::ElementType ET = toDXILElementType(ElTy, IsSigned);
      uint32_t Count = 1;
      if (auto *VTy = dyn_cast<FixedVectorType>(ElTy))
        Count = VTy->getNumElements();
      RI->setTyped(ET, Count);
    } else if (RI->isStruct()) {
      const DataLayout &DL = M.getDataLayout();

      // This mimics what DXC does. Notably, we only ever set the alignment if
      // the type is actually a struct type.
      uint32_t Stride = DL.getTypeAllocSize(ElTy);
      MaybeAlign Alignment;
      if (auto *STy = dyn_cast<StructType>(ElTy))
        Alignment = DL.getStructLayout(STy)->getAlignment();
      RI->setStruct(Stride, Alignment);
    }

    return RI;
  }

  ResourceInfo *mapHandleIntrin(CallInst *CI) {
    FunctionType *FTy = CI->getFunctionType();
    Type *RetTy = FTy->getReturnType();
    auto *HandleTy = dyn_cast<TargetExtType>(RetTy);
    if (!HandleTy) {
      diagnoseHandle(CI, "dx.handle.fromBinding requires target type");
      return nullptr;
    }

    StringRef TypeName = HandleTy->getName();
    if (TypeName == "dx.TypedBuffer") {
      return mapBufferType(CI, HandleTy, /*IsTyped=*/true);
    } else if (TypeName == "dx.RawBuffer") {
      return mapBufferType(CI, HandleTy, /*IsTyped=*/false);
    } else if (TypeName == "dx.CBuffer") {
      // TODO: implement
      diagnoseHandle(CI, "dx.CBuffer handles are not implemented yet");
      return nullptr;
    } else if (TypeName == "dx.Sampler") {
      // TODO: implement
      diagnoseHandle(CI, "dx.Sampler handles are not implemented yet");
      return nullptr;
    } else if (TypeName == "dx.Texture") {
      // TODO: implement
      diagnoseHandle(CI, "dx.Texture handles are not implemented yet");
      return nullptr;
    }

    diagnoseHandle(CI, "Invalid target(dx) type");
    return nullptr;
  }

  ResourceInfo *mapHandleFromBinding(CallInst *CI) {
    assert(CI->getIntrinsicID() == Intrinsic::dx_handle_fromBinding &&
           "Must be dx.handle.fromBinding intrinsic");

    ResourceInfo *RI = mapHandleIntrin(CI);
    if (!RI)
      return nullptr;

    uint32_t Space = cast<ConstantInt>(CI->getArgOperand(0))->getZExtValue();
    uint32_t LowerBound =
        cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
    uint32_t Size = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();

    // We use a binding ID of zero for now - these will be filled in later.
    RI->bind(0U, Space, LowerBound, Size);

    return RI;
  }

  DXILResourceMap mapResources() {
    for (Function &F : M.functions()) {
      if (!F.isDeclaration())
        continue;
      LLVM_DEBUG(dbgs() << "Function: " << F.getName() << "\n");
      Intrinsic::ID ID = F.getIntrinsicID();
      switch (ID) {
      default:
        // TODO: handle `dx.op` functions.
        continue;
      case Intrinsic::dx_handle_fromBinding:
        for (User *U : F.users()) {
          LLVM_DEBUG(dbgs() << "  Visiting: " << *U << "\n");
          if (CallInst *CI = dyn_cast<CallInst>(U))
            mapHandleFromBinding(CI);
        }
        break;
      }
    }

    return DXILResourceMap(std::move(Resources));
  }
};

} // namespace

DXILResourceMap::DXILResourceMap(
    SmallVectorImpl<std::pair<CallInst *, dxil::ResourceInfo>> &&CIToRI) {
  if (CIToRI.empty())
    return;

  llvm::stable_sort(CIToRI, [](auto &LHS, auto &RHS) {
    // Sort by resource class first for grouping purposes, and then by the rest
    // of the fields so that we can remove duplicates.
    ResourceClass LRC = LHS.second.getResourceClass();
    ResourceClass RRC = RHS.second.getResourceClass();
    return std::tie(LRC, LHS.second) < std::tie(RRC, RHS.second);
  });
  for (auto [CI, RI] : CIToRI) {
    if (Resources.empty() || RI != Resources.back())
      Resources.push_back(RI);
    CallMap[CI] = Resources.size() - 1;
  }

  unsigned Size = Resources.size();
  // In DXC, Record ID is unique per resource type. Match that.
  FirstUAV = FirstCBuffer = FirstSampler = Size;
  uint32_t NextID = 0;
  for (unsigned I = 0, E = Size; I != E; ++I) {
    ResourceInfo &RI = Resources[I];
    if (RI.isUAV() && FirstUAV == Size) {
      FirstUAV = I;
      NextID = 0;
    } else if (RI.isCBuffer() && FirstCBuffer == Size) {
      FirstCBuffer = I;
      NextID = 0;
    } else if (RI.isSampler() && FirstSampler == Size) {
      FirstSampler = I;
      NextID = 0;
    }

    // Adjust the resource binding to use the next ID.
    const ResourceInfo::ResourceBinding &Binding = RI.getBinding();
    RI.bind(NextID++, Binding.Space, Binding.LowerBound, Binding.Size);
  }
}

void DXILResourceMap::print(raw_ostream &OS) const {
  for (unsigned I = 0, E = Resources.size(); I != E; ++I) {
    OS << "Binding " << I << ":\n";
    Resources[I].print(OS);
    OS << "\n";
  }

  for (const auto &[CI, Index] : CallMap) {
    OS << "Call bound to " << Index << ":";
    CI->print(OS);
    OS << "\n";
  }
}

//===----------------------------------------------------------------------===//
// DXILResourceAnalysis and DXILResourcePrinterPass

// Provide an explicit template instantiation for the static ID.
AnalysisKey DXILResourceAnalysis::Key;

DXILResourceMap DXILResourceAnalysis::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  DXILResourceMap Data = ResourceMapper(M).mapResources();
  return Data;
}

PreservedAnalyses DXILResourcePrinterPass::run(Module &M,
                                               ModuleAnalysisManager &AM) {
  DXILResourceMap &DRM = AM.getResult<DXILResourceAnalysis>(M);
  DRM.print(OS);
  return PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
// DXILResourceWrapperPass

DXILResourceWrapperPass::DXILResourceWrapperPass() : ModulePass(ID) {
  initializeDXILResourceWrapperPassPass(*PassRegistry::getPassRegistry());
}

DXILResourceWrapperPass::~DXILResourceWrapperPass() = default;

void DXILResourceWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

bool DXILResourceWrapperPass::runOnModule(Module &M) {
  ResourceMap.reset(new DXILResourceMap(ResourceMapper(M).mapResources()));
  return false;
}

void DXILResourceWrapperPass::releaseMemory() { ResourceMap.reset(); }

void DXILResourceWrapperPass::print(raw_ostream &OS, const Module *) const {
  if (!ResourceMap) {
    OS << "No resource map has been built!\n";
    return;
  }
  ResourceMap->print(OS);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
void DXILResourceWrapperPass::dump() const { print(dbgs(), nullptr); }
#endif

INITIALIZE_PASS(DXILResourceWrapperPass, DEBUG_TYPE, "DXIL Resource analysis",
                false, true)
char DXILResourceWrapperPass::ID = 0;

ModulePass *llvm::createDXILResourceWrapperPassPass() {
  return new DXILResourceWrapperPass();
}
