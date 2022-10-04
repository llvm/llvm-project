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

void Resources::collectUAVs() {
  NamedMDNode *Entry = Mod.getNamedMetadata("hlsl.uavs");
  if (!Entry || Entry->getNumOperands() == 0)
    return;

  uint32_t Counter = 0;
  for (auto *UAV : Entry->operands()) {
    UAVs.push_back(UAVResource(Counter++, FrontendResource(cast<MDNode>(UAV))));
  }
}

ResourceBase::ResourceBase(uint32_t I, FrontendResource R)
    : ID(I), GV(R.getGlobalVariable()), Name(""), Space(0), LowerBound(0),
      RangeSize(1) {
  if (auto *ArrTy = dyn_cast<ArrayType>(GV->getInitializer()->getType()))
    RangeSize = ArrTy->getNumElements();
}

UAVResource::UAVResource(uint32_t I, FrontendResource R)
    : ResourceBase(I, R), Shape(Kinds::Invalid), GloballyCoherent(false),
      HasCounter(false), IsROV(false), ExtProps() {
  parseSourceType(R.getSourceType());
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

MDNode *ResourceBase::ExtendedProperties::write(LLVMContext &Ctx) {
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
                         MutableArrayRef<Metadata *> Entries) {
  IRBuilder<> B(Ctx);
  Entries[0] = ConstantAsMetadata::get(B.getInt32(ID));
  Entries[1] = ConstantAsMetadata::get(GV);
  Entries[2] = MDString::get(Ctx, Name);
  Entries[3] = ConstantAsMetadata::get(B.getInt32(Space));
  Entries[4] = ConstantAsMetadata::get(B.getInt32(LowerBound));
  Entries[5] = ConstantAsMetadata::get(B.getInt32(RangeSize));
}

MDNode *UAVResource::write() {
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

void Resources::write() {
  Metadata *ResourceMDs[4] = {nullptr, nullptr, nullptr, nullptr};
  SmallVector<Metadata *> UAVMDs;
  for (auto &UAV : UAVs)
    UAVMDs.emplace_back(UAV.write());

  if (!UAVMDs.empty())
    ResourceMDs[1] = MDNode::get(Mod.getContext(), UAVMDs);

  NamedMDNode *DXResMD = Mod.getOrInsertNamedMetadata("dx.resources");
  DXResMD->addOperand(MDNode::get(Mod.getContext(), ResourceMDs));

  NamedMDNode *Entry = Mod.getNamedMetadata("hlsl.uavs");
  if (Entry)
    Entry->eraseFromParent();
}
