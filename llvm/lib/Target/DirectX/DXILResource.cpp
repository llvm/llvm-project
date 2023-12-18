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
#include "CBufferDataLayout.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

using namespace llvm;
using namespace llvm::dxil;
using namespace llvm::hlsl;

template <typename T> void ResourceTable<T>::collect(Module &M) {
  NamedMDNode *Entry = M.getNamedMetadata(MDName);
  if (!Entry || Entry->getNumOperands() == 0)
    return;

  uint32_t Counter = 0;
  for (auto *Res : Entry->operands()) {
    Data.push_back(T(Counter++, FrontendResource(cast<MDNode>(Res))));
  }
}

template <> void ResourceTable<ConstantBuffer>::collect(Module &M) {
  NamedMDNode *Entry = M.getNamedMetadata(MDName);
  if (!Entry || Entry->getNumOperands() == 0)
    return;

  uint32_t Counter = 0;
  for (auto *Res : Entry->operands()) {
    Data.push_back(
        ConstantBuffer(Counter++, FrontendResource(cast<MDNode>(Res))));
  }
  // FIXME: share CBufferDataLayout with CBuffer load lowering.
  //   See https://github.com/llvm/llvm-project/issues/58381
  CBufferDataLayout CBDL(M.getDataLayout(), /*IsLegacy*/ true);
  for (auto &CB : Data)
    CB.setSize(CBDL);
}

void Resources::collect(Module &M) {
  UAVs.collect(M);
  CBuffers.collect(M);
}

ResourceBase::ResourceBase(uint32_t I, FrontendResource R)
    : ID(I), GV(R.getGlobalVariable()), Name(""), Space(R.getSpace()),
      LowerBound(R.getResourceIndex()), RangeSize(1) {
  if (auto *ArrTy = dyn_cast<ArrayType>(GV->getValueType()))
    RangeSize = ArrTy->getNumElements();
}

StringRef ResourceBase::getElementTypeName(ElementType ElTy) {
  switch (ElTy) {
  case ElementType::Invalid:
    return "invalid";
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
  }
  llvm_unreachable("All ElementType enums are handled in switch");
}

void ResourceBase::printElementType(Kinds Kind, ElementType ElTy,
                                    unsigned Alignment, raw_ostream &OS) {
  switch (Kind) {
  default:
    // TODO: add vector size.
    OS << right_justify(getElementTypeName(ElTy), Alignment);
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
  llvm_unreachable("All Kinds enums are handled in switch");
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

void UAVResource::print(raw_ostream &OS) const {
  OS << "; " << left_justify(Name, 31);

  OS << right_justify("UAV", 10);

  printElementType(Shape, ExtProps.ElementType.value_or(ElementType::Invalid),
                   8, OS);

  // FIXME: support SampleCount.
  // See https://github.com/llvm/llvm-project/issues/58175
  printKind(Shape, 12, OS, /*SRV*/ false, HasCounter);
  // Print the binding part.
  ResourceBase::print(OS, "U", "u");
}

ConstantBuffer::ConstantBuffer(uint32_t I, hlsl::FrontendResource R)
    : ResourceBase(I, R) {}

void ConstantBuffer::setSize(CBufferDataLayout &DL) {
  CBufferSizeInBytes = DL.getTypeAllocSizeInBytes(GV->getValueType());
}

void ConstantBuffer::print(raw_ostream &OS) const {
  OS << "; " << left_justify(Name, 31);

  OS << right_justify("cbuffer", 10);

  printElementType(Kinds::CBuffer, ElementType::Invalid, 8, OS);

  printKind(Kinds::CBuffer, 12, OS, /*SRV*/ false, /*HasCounter*/ false);
  // Print the binding part.
  ResourceBase::print(OS, "CB", "cb");
}

template <typename T> void ResourceTable<T>::print(raw_ostream &OS) const {
  for (auto &Res : Data)
    Res.print(OS);
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

MDNode *ConstantBuffer::write() const {
  auto &Ctx = GV->getContext();
  IRBuilder<> B(Ctx);
  Metadata *Entries[7];
  ResourceBase::write(Ctx, Entries);

  Entries[6] = ConstantAsMetadata::get(B.getInt32(CBufferSizeInBytes));
  return MDNode::get(Ctx, Entries);
}

template <typename T> MDNode *ResourceTable<T>::write(Module &M) const {
  if (Data.empty())
    return nullptr;
  SmallVector<Metadata *> MDs;
  for (auto &Res : Data)
    MDs.emplace_back(Res.write());

  NamedMDNode *Entry = M.getNamedMetadata(MDName);
  if (Entry)
    Entry->eraseFromParent();

  return MDNode::get(M.getContext(), MDs);
}

void Resources::write(Module &M) const {
  Metadata *ResourceMDs[4] = {nullptr, nullptr, nullptr, nullptr};

  ResourceMDs[1] = UAVs.write(M);

  ResourceMDs[2] = CBuffers.write(M);

  bool HasResource = ResourceMDs[0] != nullptr || ResourceMDs[1] != nullptr ||
                     ResourceMDs[2] != nullptr || ResourceMDs[3] != nullptr;

  if (HasResource) {
    NamedMDNode *DXResMD = M.getOrInsertNamedMetadata("dx.resources");
    DXResMD->addOperand(MDNode::get(M.getContext(), ResourceMDs));
  }

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

  CBuffers.print(O);
  UAVs.print(O);
}

void Resources::dump() const { print(dbgs()); }
