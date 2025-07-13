//===- RootSignatureMetadata.h - HLSL Root Signature helpers --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file implements a library for working with HLSL Root Signatures
/// and their metadata representation.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/RootSignatureMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/ScopedPrinter.h"

namespace llvm {
namespace hlsl {
namespace rootsig {

static const EnumEntry<dxil::ResourceClass> ResourceClassNames[] = {
    {"CBV", dxil::ResourceClass::CBuffer},
    {"SRV", dxil::ResourceClass::SRV},
    {"UAV", dxil::ResourceClass::UAV},
    {"Sampler", dxil::ResourceClass::Sampler},
};

static std::optional<StringRef> getResourceName(dxil::ResourceClass Class) {
  for (const auto &ClassEnum : ResourceClassNames)
    if (ClassEnum.Value == Class)
      return ClassEnum.Name;
  return std::nullopt;
}

namespace {

// We use the OverloadVisit with std::visit to ensure the compiler catches if a
// new RootElement variant type is added but it's metadata generation isn't
// handled.
template <class... Ts> struct OverloadedVisit : Ts... {
  using Ts::operator()...;
};
template <class... Ts> OverloadedVisit(Ts...) -> OverloadedVisit<Ts...>;

} // namespace

MDNode *MetadataBuilder::BuildRootSignature() {
  const auto Visitor = OverloadedVisit{
      [this](const dxbc::RootFlags &Flags) -> MDNode * {
        return BuildRootFlags(Flags);
      },
      [this](const RootConstants &Constants) -> MDNode * {
        return BuildRootConstants(Constants);
      },
      [this](const RootDescriptor &Descriptor) -> MDNode * {
        return BuildRootDescriptor(Descriptor);
      },
      [this](const DescriptorTableClause &Clause) -> MDNode * {
        return BuildDescriptorTableClause(Clause);
      },
      [this](const DescriptorTable &Table) -> MDNode * {
        return BuildDescriptorTable(Table);
      },
      [this](const StaticSampler &Sampler) -> MDNode * {
        return BuildStaticSampler(Sampler);
      },
  };

  for (const RootElement &Element : Elements) {
    MDNode *ElementMD = std::visit(Visitor, Element);
    assert(ElementMD != nullptr &&
           "Root Element must be initialized and validated");
    GeneratedMetadata.push_back(ElementMD);
  }

  return MDNode::get(Ctx, GeneratedMetadata);
}

MDNode *MetadataBuilder::BuildRootFlags(const dxbc::RootFlags &Flags) {
  IRBuilder<> Builder(Ctx);
  Metadata *Operands[] = {
      MDString::get(Ctx, "RootFlags"),
      ConstantAsMetadata::get(Builder.getInt32(llvm::to_underlying(Flags))),
  };
  return MDNode::get(Ctx, Operands);
}

MDNode *MetadataBuilder::BuildRootConstants(const RootConstants &Constants) {
  IRBuilder<> Builder(Ctx);
  Metadata *Operands[] = {
      MDString::get(Ctx, "RootConstants"),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Constants.Visibility))),
      ConstantAsMetadata::get(Builder.getInt32(Constants.Reg.Number)),
      ConstantAsMetadata::get(Builder.getInt32(Constants.Space)),
      ConstantAsMetadata::get(Builder.getInt32(Constants.Num32BitConstants)),
  };
  return MDNode::get(Ctx, Operands);
}

MDNode *MetadataBuilder::BuildRootDescriptor(const RootDescriptor &Descriptor) {
  IRBuilder<> Builder(Ctx);
  std::optional<StringRef> ResName = getResourceName(
      dxil::ResourceClass(llvm::to_underlying(Descriptor.Type)));
  assert(ResName && "Provided an invalid Resource Class");
  llvm::SmallString<7> Name({"Root", *ResName});
  Metadata *Operands[] = {
      MDString::get(Ctx, Name),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Descriptor.Visibility))),
      ConstantAsMetadata::get(Builder.getInt32(Descriptor.Reg.Number)),
      ConstantAsMetadata::get(Builder.getInt32(Descriptor.Space)),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Descriptor.Flags))),
  };
  return MDNode::get(Ctx, Operands);
}

MDNode *MetadataBuilder::BuildDescriptorTable(const DescriptorTable &Table) {
  IRBuilder<> Builder(Ctx);
  SmallVector<Metadata *> TableOperands;
  // Set the mandatory arguments
  TableOperands.push_back(MDString::get(Ctx, "DescriptorTable"));
  TableOperands.push_back(ConstantAsMetadata::get(
      Builder.getInt32(llvm::to_underlying(Table.Visibility))));

  // Remaining operands are references to the table's clauses. The in-memory
  // representation of the Root Elements created from parsing will ensure that
  // the previous N elements are the clauses for this table.
  assert(Table.NumClauses <= GeneratedMetadata.size() &&
         "Table expected all owned clauses to be generated already");
  // So, add a refence to each clause to our operands
  TableOperands.append(GeneratedMetadata.end() - Table.NumClauses,
                       GeneratedMetadata.end());
  // Then, remove those clauses from the general list of Root Elements
  GeneratedMetadata.pop_back_n(Table.NumClauses);

  return MDNode::get(Ctx, TableOperands);
}

MDNode *MetadataBuilder::BuildDescriptorTableClause(
    const DescriptorTableClause &Clause) {
  IRBuilder<> Builder(Ctx);
  std::optional<StringRef> ResName =
      getResourceName(dxil::ResourceClass(llvm::to_underlying(Clause.Type)));
  assert(ResName && "Provided an invalid Resource Class");
  Metadata *Operands[] = {
      MDString::get(Ctx, *ResName),
      ConstantAsMetadata::get(Builder.getInt32(Clause.NumDescriptors)),
      ConstantAsMetadata::get(Builder.getInt32(Clause.Reg.Number)),
      ConstantAsMetadata::get(Builder.getInt32(Clause.Space)),
      ConstantAsMetadata::get(Builder.getInt32(Clause.Offset)),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Clause.Flags))),
  };
  return MDNode::get(Ctx, Operands);
}

MDNode *MetadataBuilder::BuildStaticSampler(const StaticSampler &Sampler) {
  IRBuilder<> Builder(Ctx);
  Metadata *Operands[] = {
      MDString::get(Ctx, "StaticSampler"),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Sampler.Filter))),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Sampler.AddressU))),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Sampler.AddressV))),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Sampler.AddressW))),
      ConstantAsMetadata::get(llvm::ConstantFP::get(llvm::Type::getFloatTy(Ctx),
                                                    Sampler.MipLODBias)),
      ConstantAsMetadata::get(Builder.getInt32(Sampler.MaxAnisotropy)),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Sampler.CompFunc))),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Sampler.BorderColor))),
      ConstantAsMetadata::get(
          llvm::ConstantFP::get(llvm::Type::getFloatTy(Ctx), Sampler.MinLOD)),
      ConstantAsMetadata::get(
          llvm::ConstantFP::get(llvm::Type::getFloatTy(Ctx), Sampler.MaxLOD)),
      ConstantAsMetadata::get(Builder.getInt32(Sampler.Reg.Number)),
      ConstantAsMetadata::get(Builder.getInt32(Sampler.Space)),
      ConstantAsMetadata::get(
          Builder.getInt32(llvm::to_underlying(Sampler.Visibility))),
  };
  return MDNode::get(Ctx, Operands);
}

} // namespace rootsig
} // namespace hlsl
} // namespace llvm
