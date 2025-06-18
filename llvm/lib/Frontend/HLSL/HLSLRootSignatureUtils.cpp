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
#include "llvm/Support/ScopedPrinter.h"

namespace llvm {
namespace hlsl {
namespace rootsig {

template <typename T>
static std::optional<StringRef> getEnumName(const T Value,
                                            ArrayRef<EnumEntry<T>> Enums) {
  for (const auto &EnumItem : Enums)
    if (EnumItem.Value == Value)
      return EnumItem.Name;
  return std::nullopt;
}

template <typename T>
static raw_ostream &printEnum(raw_ostream &OS, const T Value,
                              ArrayRef<EnumEntry<T>> Enums) {
  auto MaybeName = getEnumName(Value, Enums);
  if (MaybeName)
    OS << *MaybeName;
  return OS;
}

template <typename T>
static raw_ostream &printFlags(raw_ostream &OS, const T Value,
                               ArrayRef<EnumEntry<T>> Flags) {
  bool FlagSet = false;
  unsigned Remaining = llvm::to_underlying(Value);
  while (Remaining) {
    unsigned Bit = 1u << llvm::countr_zero(Remaining);
    if (Remaining & Bit) {
      if (FlagSet)
        OS << " | ";

      auto MaybeFlag = getEnumName(T(Bit), Flags);
      if (MaybeFlag)
        OS << *MaybeFlag;
      else
        OS << "invalid: " << Bit;

      FlagSet = true;
    }
    Remaining &= ~Bit;
  }

  if (!FlagSet)
    OS << "None";
  return OS;
}

static const EnumEntry<RegisterType> RegisterNames[] = {
    {"b", RegisterType::BReg},
    {"t", RegisterType::TReg},
    {"u", RegisterType::UReg},
    {"s", RegisterType::SReg},
};

static raw_ostream &operator<<(raw_ostream &OS, const Register &Reg) {
  printEnum(OS, Reg.ViewType, ArrayRef(RegisterNames));
  OS << Reg.Number;

  return OS;
}

static const EnumEntry<ShaderVisibility> VisibilityNames[] = {
    {"All", ShaderVisibility::All},
    {"Vertex", ShaderVisibility::Vertex},
    {"Hull", ShaderVisibility::Hull},
    {"Domain", ShaderVisibility::Domain},
    {"Geometry", ShaderVisibility::Geometry},
    {"Pixel", ShaderVisibility::Pixel},
    {"Amplification", ShaderVisibility::Amplification},
    {"Mesh", ShaderVisibility::Mesh},
};

static raw_ostream &operator<<(raw_ostream &OS,
                               const ShaderVisibility &Visibility) {
  printEnum(OS, Visibility, ArrayRef(VisibilityNames));

  return OS;
}

static const EnumEntry<SamplerFilter> SamplerFilterNames[] = {
    {"MinMagMipPoint", SamplerFilter::MinMagMipPoint},
    {"MinMagPointMipLinear", SamplerFilter::MinMagPointMipLinear},
    {"MinPointMagLinearMipPoint", SamplerFilter::MinPointMagLinearMipPoint},
    {"MinPointMagMipLinear", SamplerFilter::MinPointMagMipLinear},
    {"MinLinearMagMipPoint", SamplerFilter::MinLinearMagMipPoint},
    {"MinLinearMagPointMipLinear", SamplerFilter::MinLinearMagPointMipLinear},
    {"MinMagLinearMipPoint", SamplerFilter::MinMagLinearMipPoint},
    {"MinMagMipLinear", SamplerFilter::MinMagMipLinear},
    {"Anisotropic", SamplerFilter::Anisotropic},
    {"ComparisonMinMagMipPoint", SamplerFilter::ComparisonMinMagMipPoint},
    {"ComparisonMinMagPointMipLinear",
     SamplerFilter::ComparisonMinMagPointMipLinear},
    {"ComparisonMinPointMagLinearMipPoint",
     SamplerFilter::ComparisonMinPointMagLinearMipPoint},
    {"ComparisonMinPointMagMipLinear",
     SamplerFilter::ComparisonMinPointMagMipLinear},
    {"ComparisonMinLinearMagMipPoint",
     SamplerFilter::ComparisonMinLinearMagMipPoint},
    {"ComparisonMinLinearMagPointMipLinear",
     SamplerFilter::ComparisonMinLinearMagPointMipLinear},
    {"ComparisonMinMagLinearMipPoint",
     SamplerFilter::ComparisonMinMagLinearMipPoint},
    {"ComparisonMinMagMipLinear", SamplerFilter::ComparisonMinMagMipLinear},
    {"ComparisonAnisotropic", SamplerFilter::ComparisonAnisotropic},
    {"MinimumMinMagMipPoint", SamplerFilter::MinimumMinMagMipPoint},
    {"MinimumMinMagPointMipLinear", SamplerFilter::MinimumMinMagPointMipLinear},
    {"MinimumMinPointMagLinearMipPoint",
     SamplerFilter::MinimumMinPointMagLinearMipPoint},
    {"MinimumMinPointMagMipLinear", SamplerFilter::MinimumMinPointMagMipLinear},
    {"MinimumMinLinearMagMipPoint", SamplerFilter::MinimumMinLinearMagMipPoint},
    {"MinimumMinLinearMagPointMipLinear",
     SamplerFilter::MinimumMinLinearMagPointMipLinear},
    {"MinimumMinMagLinearMipPoint", SamplerFilter::MinimumMinMagLinearMipPoint},
    {"MinimumMinMagMipLinear", SamplerFilter::MinimumMinMagMipLinear},
    {"MinimumAnisotropic", SamplerFilter::MinimumAnisotropic},
    {"MaximumMinMagMipPoint", SamplerFilter::MaximumMinMagMipPoint},
    {"MaximumMinMagPointMipLinear", SamplerFilter::MaximumMinMagPointMipLinear},
    {"MaximumMinPointMagLinearMipPoint",
     SamplerFilter::MaximumMinPointMagLinearMipPoint},
    {"MaximumMinPointMagMipLinear", SamplerFilter::MaximumMinPointMagMipLinear},
    {"MaximumMinLinearMagMipPoint", SamplerFilter::MaximumMinLinearMagMipPoint},
    {"MaximumMinLinearMagPointMipLinear",
     SamplerFilter::MaximumMinLinearMagPointMipLinear},
    {"MaximumMinMagLinearMipPoint", SamplerFilter::MaximumMinMagLinearMipPoint},
    {"MaximumMinMagMipLinear", SamplerFilter::MaximumMinMagMipLinear},
    {"MaximumAnisotropic", SamplerFilter::MaximumAnisotropic},
};

static raw_ostream &operator<<(raw_ostream &OS, const SamplerFilter &Filter) {
  printEnum(OS, Filter, ArrayRef(SamplerFilterNames));

  return OS;
}

static const EnumEntry<TextureAddressMode> TextureAddressModeNames[] = {
    {"Wrap", TextureAddressMode::Wrap},
    {"Mirror", TextureAddressMode::Mirror},
    {"Clamp", TextureAddressMode::Clamp},
    {"Border", TextureAddressMode::Border},
    {"MirrorOnce", TextureAddressMode::MirrorOnce},
};

static raw_ostream &operator<<(raw_ostream &OS,
                               const TextureAddressMode &Address) {
  printEnum(OS, Address, ArrayRef(TextureAddressModeNames));

  return OS;
}

static const EnumEntry<ComparisonFunc> ComparisonFuncNames[] = {
    {"Never", ComparisonFunc::Never},
    {"Less", ComparisonFunc::Less},
    {"Equal", ComparisonFunc::Equal},
    {"LessEqual", ComparisonFunc::LessEqual},
    {"Greater", ComparisonFunc::Greater},
    {"NotEqual", ComparisonFunc::NotEqual},
    {"GreaterEqual", ComparisonFunc::GreaterEqual},
    {"Always", ComparisonFunc::Always},
};

static raw_ostream &operator<<(raw_ostream &OS,
                               const ComparisonFunc &CompFunc) {
  printEnum(OS, CompFunc, ArrayRef(ComparisonFuncNames));

  return OS;
}

static const EnumEntry<StaticBorderColor> StaticBorderColorNames[] = {
    {"TransparentBlack", StaticBorderColor::TransparentBlack},
    {"OpaqueBlack", StaticBorderColor::OpaqueBlack},
    {"OpaqueWhite", StaticBorderColor::OpaqueWhite},
    {"OpaqueBlackUint", StaticBorderColor::OpaqueBlackUint},
    {"OpaqueWhiteUint", StaticBorderColor::OpaqueWhiteUint},
};

static raw_ostream &operator<<(raw_ostream &OS,
                               const StaticBorderColor &BorderColor) {
  printEnum(OS, BorderColor, ArrayRef(StaticBorderColorNames));

  return OS;
}

static const EnumEntry<dxil::ResourceClass> ResourceClassNames[] = {
    {"CBV", dxil::ResourceClass::CBuffer},
    {"SRV", dxil::ResourceClass::SRV},
    {"UAV", dxil::ResourceClass::UAV},
    {"Sampler", dxil::ResourceClass::Sampler},
};

static raw_ostream &operator<<(raw_ostream &OS, const ClauseType &Type) {
  printEnum(OS, dxil::ResourceClass(llvm::to_underlying(Type)),
            ArrayRef(ResourceClassNames));

  return OS;
}

static const EnumEntry<RootDescriptorFlags> RootDescriptorFlagNames[] = {
    {"DataVolatile", RootDescriptorFlags::DataVolatile},
    {"DataStaticWhileSetAtExecute",
     RootDescriptorFlags::DataStaticWhileSetAtExecute},
    {"DataStatic", RootDescriptorFlags::DataStatic},
};

static raw_ostream &operator<<(raw_ostream &OS,
                               const RootDescriptorFlags &Flags) {
  printFlags(OS, Flags, ArrayRef(RootDescriptorFlagNames));

  return OS;
}

static const EnumEntry<DescriptorRangeFlags> DescriptorRangeFlagNames[] = {
    {"DescriptorsVolatile", DescriptorRangeFlags::DescriptorsVolatile},
    {"DataVolatile", DescriptorRangeFlags::DataVolatile},
    {"DataStaticWhileSetAtExecute",
     DescriptorRangeFlags::DataStaticWhileSetAtExecute},
    {"DataStatic", DescriptorRangeFlags::DataStatic},
    {"DescriptorsStaticKeepingBufferBoundsChecks",
     DescriptorRangeFlags::DescriptorsStaticKeepingBufferBoundsChecks},
};

static raw_ostream &operator<<(raw_ostream &OS,
                               const DescriptorRangeFlags &Flags) {
  printFlags(OS, Flags, ArrayRef(DescriptorRangeFlagNames));

  return OS;
}

static const EnumEntry<RootFlags> RootFlagNames[] = {
    {"AllowInputAssemblerInputLayout",
     RootFlags::AllowInputAssemblerInputLayout},
    {"DenyVertexShaderRootAccess", RootFlags::DenyVertexShaderRootAccess},
    {"DenyHullShaderRootAccess", RootFlags::DenyHullShaderRootAccess},
    {"DenyDomainShaderRootAccess", RootFlags::DenyDomainShaderRootAccess},
    {"DenyGeometryShaderRootAccess", RootFlags::DenyGeometryShaderRootAccess},
    {"DenyPixelShaderRootAccess", RootFlags::DenyPixelShaderRootAccess},
    {"AllowStreamOutput", RootFlags::AllowStreamOutput},
    {"LocalRootSignature", RootFlags::LocalRootSignature},
    {"DenyAmplificationShaderRootAccess",
     RootFlags::DenyAmplificationShaderRootAccess},
    {"DenyMeshShaderRootAccess", RootFlags::DenyMeshShaderRootAccess},
    {"CBVSRVUAVHeapDirectlyIndexed", RootFlags::CBVSRVUAVHeapDirectlyIndexed},
    {"SamplerHeapDirectlyIndexed", RootFlags::SamplerHeapDirectlyIndexed},
};

raw_ostream &operator<<(raw_ostream &OS, const RootFlags &Flags) {
  OS << "RootFlags(";
  printFlags(OS, Flags, ArrayRef(RootFlagNames));
  OS << ")";

  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const RootConstants &Constants) {
  OS << "RootConstants(num32BitConstants = " << Constants.Num32BitConstants
     << ", " << Constants.Reg << ", space = " << Constants.Space
     << ", visibility = " << Constants.Visibility << ")";

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

raw_ostream &operator<<(raw_ostream &OS, const RootDescriptor &Descriptor) {
  ClauseType Type = ClauseType(llvm::to_underlying(Descriptor.Type));
  OS << "Root" << Type << "(" << Descriptor.Reg
     << ", space = " << Descriptor.Space
     << ", visibility = " << Descriptor.Visibility
     << ", flags = " << Descriptor.Flags << ")";

  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const StaticSampler &Sampler) {
  OS << "StaticSampler(" << Sampler.Reg << ", filter = " << Sampler.Filter
     << ", addressU = " << Sampler.AddressU
     << ", addressV = " << Sampler.AddressV
     << ", addressW = " << Sampler.AddressW
     << ", mipLODBias = " << Sampler.MipLODBias
     << ", maxAnisotropy = " << Sampler.MaxAnisotropy
     << ", comparisonFunc = " << Sampler.CompFunc
     << ", borderColor = " << Sampler.BorderColor
     << ", minLOD = " << Sampler.MinLOD << ", maxLOD = " << Sampler.MaxLOD
     << ", space = " << Sampler.Space << ", visibility = " << Sampler.Visibility
     << ")";
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
  std::optional<StringRef> TypeName =
      getEnumName(dxil::ResourceClass(llvm::to_underlying(Descriptor.Type)),
                  ArrayRef(ResourceClassNames));
  assert(TypeName && "Provided an invalid Resource Class");
  llvm::SmallString<7> Name({"Root", *TypeName});
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
  std::optional<StringRef> Name =
      getEnumName(dxil::ResourceClass(llvm::to_underlying(Clause.Type)),
                  ArrayRef(ResourceClassNames));
  assert(Name && "Provided an invalid Resource Class");
  Metadata *Operands[] = {
      MDString::get(Ctx, *Name),
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

std::optional<const RangeInfo *>
ResourceRange::getOverlapping(const RangeInfo &Info) const {
  MapT::const_iterator Interval = Intervals.find(Info.LowerBound);
  if (!Interval.valid() || Info.UpperBound < Interval.start())
    return std::nullopt;
  return Interval.value();
}

const RangeInfo *ResourceRange::lookup(uint32_t X) const {
  return Intervals.lookup(X, nullptr);
}

std::optional<const RangeInfo *> ResourceRange::insert(const RangeInfo &Info) {
  uint32_t LowerBound = Info.LowerBound;
  uint32_t UpperBound = Info.UpperBound;

  std::optional<const RangeInfo *> Res = std::nullopt;
  MapT::iterator Interval = Intervals.begin();

  while (true) {
    if (UpperBound < LowerBound)
      break;

    Interval.advanceTo(LowerBound);
    if (!Interval.valid()) // No interval found
      break;

    // Let Interval = [x;y] and [LowerBound;UpperBound] = [a;b] and note that
    // a <= y implicitly from Intervals.find(LowerBound)
    if (UpperBound < Interval.start())
      break; // found interval does not overlap with inserted one

    if (!Res.has_value()) // Update to be the first found intersection
      Res = Interval.value();

    if (Interval.start() <= LowerBound && UpperBound <= Interval.stop()) {
      // x <= a <= b <= y implies that [a;b] is covered by [x;y]
      //  -> so we don't need to insert this, report an overlap
      return Res;
    } else if (LowerBound <= Interval.start() &&
               Interval.stop() <= UpperBound) {
      // a <= x <= y <= b implies that [x;y] is covered by [a;b]
      //  -> so remove the existing interval that we will cover with the
      //  overwrite
      Interval.erase();
    } else if (LowerBound < Interval.start() && UpperBound <= Interval.stop()) {
      // a < x <= b <= y implies that [a; x] is not covered but [x;b] is
      //  -> so set b = x - 1 such that [a;x-1] is now the interval to insert
      UpperBound = Interval.start() - 1;
    } else if (Interval.start() <= LowerBound && Interval.stop() < UpperBound) {
      // a < x <= b <= y implies that [y; b] is not covered but [a;y] is
      //  -> so set a = y + 1 such that [y+1;b] is now the interval to insert
      LowerBound = Interval.stop() + 1;
    }
  }

  assert(LowerBound <= UpperBound && "Attempting to insert an empty interval");
  Intervals.insert(LowerBound, UpperBound, &Info);
  return Res;
}

} // namespace rootsig
} // namespace hlsl
} // namespace llvm
