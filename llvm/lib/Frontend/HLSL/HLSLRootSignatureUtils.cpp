//===- HLSLRootSignatureUtils.cpp - HLSL Root Signature helpers -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helpers for working with HLSL Root Signatures.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/HLSLRootSignatureUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/bit.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"

namespace llvm {
namespace hlsl {
namespace rootsig {

static raw_ostream &operator<<(raw_ostream &OS, const Register &Reg) {
  switch (Reg.ViewType) {
  case RegisterType::BReg:
    OS << "b";
    break;
  case RegisterType::TReg:
    OS << "t";
    break;
  case RegisterType::UReg:
    OS << "u";
    break;
  case RegisterType::SReg:
    OS << "s";
    break;
  }
  OS << Reg.Number;
  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS,
                               const ShaderVisibility &Visibility) {
  switch (Visibility) {
  case ShaderVisibility::All:
    OS << "All";
    break;
  case ShaderVisibility::Vertex:
    OS << "Vertex";
    break;
  case ShaderVisibility::Hull:
    OS << "Hull";
    break;
  case ShaderVisibility::Domain:
    OS << "Domain";
    break;
  case ShaderVisibility::Geometry:
    OS << "Geometry";
    break;
  case ShaderVisibility::Pixel:
    OS << "Pixel";
    break;
  case ShaderVisibility::Amplification:
    OS << "Amplification";
    break;
  case ShaderVisibility::Mesh:
    OS << "Mesh";
    break;
  }

  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS, const ClauseType &Type) {
  switch (Type) {
  case ClauseType::CBuffer:
    OS << "CBV";
    break;
  case ClauseType::SRV:
    OS << "SRV";
    break;
  case ClauseType::UAV:
    OS << "UAV";
    break;
  case ClauseType::Sampler:
    OS << "Sampler";
    break;
  }

  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS,
                               const DescriptorRangeFlags &Flags) {
  bool FlagSet = false;
  unsigned Remaining = llvm::to_underlying(Flags);
  while (Remaining) {
    unsigned Bit = 1u << llvm::countr_zero(Remaining);
    if (Remaining & Bit) {
      if (FlagSet)
        OS << " | ";

      switch (static_cast<DescriptorRangeFlags>(Bit)) {
      case DescriptorRangeFlags::DescriptorsVolatile:
        OS << "DescriptorsVolatile";
        break;
      case DescriptorRangeFlags::DataVolatile:
        OS << "DataVolatile";
        break;
      case DescriptorRangeFlags::DataStaticWhileSetAtExecute:
        OS << "DataStaticWhileSetAtExecute";
        break;
      case DescriptorRangeFlags::DataStatic:
        OS << "DataStatic";
        break;
      case DescriptorRangeFlags::DescriptorsStaticKeepingBufferBoundsChecks:
        OS << "DescriptorsStaticKeepingBufferBoundsChecks";
        break;
      default:
        OS << "invalid: " << Bit;
        break;
      }

      FlagSet = true;
    }
    Remaining &= ~Bit;
  }

  if (!FlagSet)
    OS << "None";

  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const DescriptorTable &Table) {
  OS << "DescriptorTable(numClauses = " << Table.NumClauses
     << ", visibility = " << Table.Visibility << ")";

  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const DescriptorTableClause &Clause) {
  OS << Clause.Type << "(" << Clause.Reg
     << ", numDescriptors = " << Clause.NumDescriptors
     << ", space = " << Clause.Space << ", offset = ";
  if (Clause.Offset == DescriptorTableOffsetAppend)
    OS << "DescriptorTableOffsetAppend";
  else
    OS << Clause.Offset;
  OS << ", flags = " << Clause.Flags << ")";

  return OS;
}

void dumpRootElements(raw_ostream &OS, ArrayRef<RootElement> Elements) {
  OS << "RootElements{";
  bool First = true;
  for (const RootElement &Element : Elements) {
    if (!First)
      OS << ",";
    OS << " ";
    if (const auto &Clause = std::get_if<DescriptorTableClause>(&Element))
      OS << *Clause;
    if (const auto &Table = std::get_if<DescriptorTable>(&Element))
      OS << *Table;
    First = false;
  }
  OS << "}";
}

namespace {

// We use the OverloadBuild with std::visit to ensure the compiler catches if a
// new RootElement variant type is added but it's metadata generation isn't
// handled.
template <class... Ts> struct OverloadedBuild : Ts... {
  using Ts::operator()...;
};
template <class... Ts> OverloadedBuild(Ts...) -> OverloadedBuild<Ts...>;

} // namespace

MDNode *MetadataBuilder::BuildRootSignature() {
  const auto Visitor = OverloadedBuild{
      [this](const RootFlags &Flags) -> MDNode * {
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

MDNode *MetadataBuilder::BuildRootFlags(const RootFlags &Flags) {
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
  llvm::SmallString<7> Name;
  llvm::raw_svector_ostream OS(Name);
  OS << "Root" << ClauseType(llvm::to_underlying(Descriptor.Type));

  Metadata *Operands[] = {
      MDString::get(Ctx, OS.str()),
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
  std::string Name;
  llvm::raw_string_ostream OS(Name);
  OS << Clause.Type;
  return MDNode::get(
      Ctx, {
               MDString::get(Ctx, OS.str()),
               ConstantAsMetadata::get(Builder.getInt32(Clause.NumDescriptors)),
               ConstantAsMetadata::get(Builder.getInt32(Clause.Reg.Number)),
               ConstantAsMetadata::get(Builder.getInt32(Clause.Space)),
               ConstantAsMetadata::get(Builder.getInt32(Clause.Offset)),
               ConstantAsMetadata::get(
                   Builder.getInt32(llvm::to_underlying(Clause.Flags))),
           });
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
