//===- DXILResource.cpp - DXIL Resource helper objects --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects for working with DXIL Resources.
///
//===----------------------------------------------------------------------===//

#include "DXILResource.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

using namespace llvm;
using namespace llvm::dxil;

GlobalVariable *FrontendResource::getGlobalVariable() {
  return cast<GlobalVariable>(
      cast<ConstantAsMetadata>(Entry->getOperand(0))->getValue());
}

StringRef FrontendResource::getSourceType() {
  return cast<MDString>(Entry->getOperand(1))->getString();
}

Constant *FrontendResource::getID() {
  return cast<ConstantAsMetadata>(Entry->getOperand(2))->getValue();
}

void Resources::collectUAVs(Module &M) {
  NamedMDNode *Entry = M.getNamedMetadata("hlsl.uavs");
  if (!Entry || Entry->getNumOperands() == 0)
    return;

  uint32_t Counter = 0;
  for (auto *UAV : Entry->operands()) {
    UAVs.push_back(UAVResource(Counter++, FrontendResource(cast<MDNode>(UAV))));
  }
}

void Resources::collect(Module &M) { collectUAVs(M); }

ResourceBase::ResourceBase(uint32_t I, FrontendResource R)
    : ID(I), GV(R.getGlobalVariable()), Name(""), Space(0), LowerBound(0),
      RangeSize(1) {
  if (auto *ArrTy = dyn_cast<ArrayType>(GV->getInitializer()->getType()))
    RangeSize = ArrTy->getNumElements();
}

StringRef ResourceBase::getComponentTypeName(ComponentType CompType) {
  switch (CompType) {
  case ComponentType::LastEntry:
  case ComponentType::Invalid:
    return "invalid";
  case ComponentType::I1:
    return "i1";
  case ComponentType::I16:
    return "i16";
  case ComponentType::U16:
    return "u16";
  case ComponentType::I32:
    return "i32";
  case ComponentType::U32:
    return "u32";
  case ComponentType::I64:
    return "i64";
  case ComponentType::U64:
    return "u64";
  case ComponentType::F16:
    return "f16";
  case ComponentType::F32:
    return "f32";
  case ComponentType::F64:
    return "f64";
  case ComponentType::SNormF16:
    return "snorm_f16";
  case ComponentType::UNormF16:
    return "unorm_f16";
  case ComponentType::SNormF32:
    return "snorm_f32";
  case ComponentType::UNormF32:
    return "unorm_f32";
  case ComponentType::SNormF64:
    return "snorm_f64";
  case ComponentType::UNormF64:
    return "unorm_f64";
  case ComponentType::PackedS8x32:
    return "p32i8";
  case ComponentType::PackedU8x32:
    return "p32u8";
  }
}

void ResourceBase::printComponentType(Kinds Kind, ComponentType CompType,
                                      unsigned Alignment, raw_ostream &OS) {
  switch (Kind) {
  default:
    // TODO: add vector size.
    OS << right_justify(getComponentTypeName(CompType), Alignment);
    break;
  case Kinds::RawBuffer:
    OS << right_justify("byte", Alignment);
    break;
  case Kinds::StructuredBuffer:
    OS << right_justify("struct", Alignment);
    break;
  case Kinds::CBuffer:
  case Kinds::Sampler:
    OS << right_justify("NA", Alignment);
    break;
  case Kinds::Invalid:
  case Kinds::NumEntries:
    break;
  }
}

StringRef ResourceBase::getKindName(Kinds Kind) {
  switch (Kind) {
  case Kinds::NumEntries:
  case Kinds::Invalid:
    return "invalid";
  case Kinds::Texture1D:
    return "1d";
  case Kinds::Texture2D:
    return "2d";
  case Kinds::Texture2DMS:
    return "2dMS";
  case Kinds::Texture3D:
    return "3d";
  case Kinds::TextureCube:
    return "cube";
  case Kinds::Texture1DArray:
    return "1darray";
  case Kinds::Texture2DArray:
    return "2darray";
  case Kinds::Texture2DMSArray:
    return "2darrayMS";
  case Kinds::TextureCubeArray:
    return "cubearray";
  case Kinds::TypedBuffer:
    return "buf";
  case Kinds::RawBuffer:
    return "rawbuf";
  case Kinds::StructuredBuffer:
    return "structbuf";
  case Kinds::CBuffer:
    return "cbuffer";
  case Kinds::Sampler:
    return "sampler";
  case Kinds::TBuffer:
    return "tbuffer";
  case Kinds::RTAccelerationStructure:
    return "ras";
  case Kinds::FeedbackTexture2D:
    return "fbtex2d";
  case Kinds::FeedbackTexture2DArray:
    return "fbtex2darray";
  }
}

void ResourceBase::printKind(Kinds Kind, unsigned Alignment, raw_ostream &OS,
                             bool SRV, bool HasCounter, uint32_t SampleCount) {
  switch (Kind) {
  default:
    OS << right_justify(getKindName(Kind), Alignment);
    break;

  case Kinds::RawBuffer:
  case Kinds::StructuredBuffer:
    if (SRV)
      OS << right_justify("r/o", Alignment);
    else {
      if (!HasCounter)
        OS << right_justify("r/w", Alignment);
      else
        OS << right_justify("r/w+cnt", Alignment);
    }
    break;
  case Kinds::TypedBuffer:
    OS << right_justify("buf", Alignment);
    break;
  case Kinds::Texture2DMS:
  case Kinds::Texture2DMSArray: {
    std::string DimName = getKindName(Kind).str();
    if (SampleCount)
      DimName += std::to_string(SampleCount);
    OS << right_justify(DimName, Alignment);
  } break;
  case Kinds::CBuffer:
  case Kinds::Sampler:
    OS << right_justify("NA", Alignment);
    break;
  case Kinds::Invalid:
  case Kinds::NumEntries:
    break;
  }
}

void ResourceBase::print(raw_ostream &OS, StringRef IDPrefix,
                         StringRef BindingPrefix) const {
  std::string ResID = IDPrefix.str();
  ResID += std::to_string(ID);
  OS << right_justify(ResID, 8);

  std::string Bind = BindingPrefix.str();
  Bind += std::to_string(LowerBound);
  if (Space)
    Bind += ",space" + std::to_string(Space);

  OS << right_justify(Bind, 15);
  if (RangeSize != UINT_MAX)
    OS << right_justify(std::to_string(RangeSize), 6) << "\n";
  else
    OS << right_justify("unbounded", 6) << "\n";
}

UAVResource::UAVResource(uint32_t I, FrontendResource R)
    : ResourceBase(I, R), Shape(Kinds::Invalid), GloballyCoherent(false),
      HasCounter(false), IsROV(false), ExtProps() {
  parseSourceType(R.getSourceType());
}

void UAVResource::print(raw_ostream &OS) const {
  OS << "; " << left_justify(Name, 31);

  OS << right_justify("UAV", 10);

  printComponentType(
      Shape, ExtProps.ElementType.value_or(ComponentType::Invalid), 8, OS);

  // FIXME: support SampleCount.
  // See https://github.com/llvm/llvm-project/issues/58175
  printKind(Shape, 12, OS, /*SRV*/ false, HasCounter);
  // Print the binding part.
  ResourceBase::print(OS, "U", "u");
}

// FIXME: Capture this in HLSL source. I would go do this right now, but I want
// to get this in first so that I can make sure to capture all the extra
// information we need to remove the source type string from here (See issue:
// https://github.com/llvm/llvm-project/issues/57991).
void UAVResource::parseSourceType(StringRef S) {
  IsROV = S.startswith("RasterizerOrdered");
  if (IsROV)
    S = S.substr(strlen("RasterizerOrdered"));
  if (S.startswith("RW"))
    S = S.substr(strlen("RW"));

  // Note: I'm deliberately not handling any of the Texture buffer types at the
  // moment. I want to resolve the issue above before adding Texture or Sampler
  // support.
  Shape = StringSwitch<ResourceBase::Kinds>(S)
              .StartsWith("Buffer<", Kinds::TypedBuffer)
              .StartsWith("ByteAddressBuffer<", Kinds::RawBuffer)
              .StartsWith("StructuredBuffer<", Kinds::StructuredBuffer)
              .Default(Kinds::Invalid);
  assert(Shape != Kinds::Invalid && "Unsupported buffer type");

  S = S.substr(S.find("<") + 1);

  constexpr size_t PrefixLen = StringRef("vector<").size();
  if (S.startswith("vector<"))
    S = S.substr(PrefixLen, S.find(",") - PrefixLen);
  else
    S = S.substr(0, S.find(">"));

  ComponentType ElTy = StringSwitch<ResourceBase::ComponentType>(S)
                           .Case("bool", ComponentType::I1)
                           .Case("int16_t", ComponentType::I16)
                           .Case("uint16_t", ComponentType::U16)
                           .Case("int32_t", ComponentType::I32)
                           .Case("uint32_t", ComponentType::U32)
                           .Case("int64_t", ComponentType::I64)
                           .Case("uint64_t", ComponentType::U64)
                           .Case("half", ComponentType::F16)
                           .Case("float", ComponentType::F32)
                           .Case("double", ComponentType::F64)
                           .Default(ComponentType::Invalid);
  if (ElTy != ComponentType::Invalid)
    ExtProps.ElementType = ElTy;
}

MDNode *ResourceBase::ExtendedProperties::write(LLVMContext &Ctx) const {
  IRBuilder<> B(Ctx);
  SmallVector<Metadata *> Entries;
  if (ElementType) {
    Entries.emplace_back(
        ConstantAsMetadata::get(B.getInt32(TypedBufferElementType)));
    Entries.emplace_back(ConstantAsMetadata::get(
        B.getInt32(static_cast<uint32_t>(*ElementType))));
  }
  if (Entries.empty())
    return nullptr;
  return MDNode::get(Ctx, Entries);
}

void ResourceBase::write(LLVMContext &Ctx,
                         MutableArrayRef<Metadata *> Entries) const {
  IRBuilder<> B(Ctx);
  Entries[0] = ConstantAsMetadata::get(B.getInt32(ID));
  Entries[1] = ConstantAsMetadata::get(GV);
  Entries[2] = MDString::get(Ctx, Name);
  Entries[3] = ConstantAsMetadata::get(B.getInt32(Space));
  Entries[4] = ConstantAsMetadata::get(B.getInt32(LowerBound));
  Entries[5] = ConstantAsMetadata::get(B.getInt32(RangeSize));
}

MDNode *UAVResource::write() const {
  auto &Ctx = GV->getContext();
  IRBuilder<> B(Ctx);
  Metadata *Entries[11];
  ResourceBase::write(Ctx, Entries);
  Entries[6] =
      ConstantAsMetadata::get(B.getInt32(static_cast<uint32_t>(Shape)));
  Entries[7] = ConstantAsMetadata::get(B.getInt1(GloballyCoherent));
  Entries[8] = ConstantAsMetadata::get(B.getInt1(HasCounter));
  Entries[9] = ConstantAsMetadata::get(B.getInt1(IsROV));
  Entries[10] = ExtProps.write(Ctx);
  return MDNode::get(Ctx, Entries);
}

void Resources::write(Module &M) const {
  Metadata *ResourceMDs[4] = {nullptr, nullptr, nullptr, nullptr};
  SmallVector<Metadata *> UAVMDs;
  for (auto &UAV : UAVs)
    UAVMDs.emplace_back(UAV.write());

  if (!UAVMDs.empty())
    ResourceMDs[1] = MDNode::get(M.getContext(), UAVMDs);

  NamedMDNode *DXResMD = M.getOrInsertNamedMetadata("dx.resources");
  DXResMD->addOperand(MDNode::get(M.getContext(), ResourceMDs));

  NamedMDNode *Entry = M.getNamedMetadata("hlsl.uavs");
  if (Entry)
    Entry->eraseFromParent();
}

void Resources::print(raw_ostream &O) const {
  O << ";\n"
    << "; Resource Bindings:\n"
    << ";\n"
    << "; Name                                 Type  Format         Dim      "
       "ID      HLSL Bind  Count\n"
    << "; ------------------------------ ---------- ------- ----------- "
       "------- -------------- ------\n";

  for (auto &UAV : UAVs)
    UAV.print(O);
}

void Resources::dump() const { print(dbgs()); }
