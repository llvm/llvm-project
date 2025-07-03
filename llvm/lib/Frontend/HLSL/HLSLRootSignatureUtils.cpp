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

static raw_ostream &operator<<(raw_ostream &OS,
                               const llvm::dxbc::ShaderVisibility &Visibility) {
  printEnum(OS, Visibility, dxbc::getShaderVisibility());

  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS,
                               const llvm::dxbc::SamplerFilter &Filter) {
  printEnum(OS, Filter, dxbc::getSamplerFilters());

  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS,
                               const dxbc::TextureAddressMode &Address) {
  printEnum(OS, Address, dxbc::getTextureAddressModes());

  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS,
                               const dxbc::ComparisonFunc &CompFunc) {
  printEnum(OS, CompFunc, dxbc::getComparisonFuncs());

  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS,
                               const dxbc::StaticBorderColor &BorderColor) {
  printEnum(OS, BorderColor, dxbc::getStaticBorderColors());

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

static raw_ostream &operator<<(raw_ostream &OS,
                               const dxbc::RootDescriptorFlags &Flags) {
  printFlags(OS, Flags, dxbc::getRootDescriptorFlags());

  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS,
                               const llvm::dxbc::DescriptorRangeFlags &Flags) {
  printFlags(OS, Flags, dxbc::getDescriptorRangeFlags());

  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const dxbc::RootFlags &Flags) {
  OS << "RootFlags(";
  printFlags(OS, Flags, dxbc::getRootFlags());
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
  OS << Clause.Type << "(" << Clause.Reg << ", numDescriptors = ";
  if (Clause.NumDescriptors == NumDescriptorsUnbounded)
    OS << "unbounded";
  else
    OS << Clause.NumDescriptors;
  OS << ", space = " << Clause.Space << ", offset = ";
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

namespace {

// We use the OverloadVisit with std::visit to ensure the compiler catches if a
// new RootElement variant type is added but it's operator<< isn't handled.
template <class... Ts> struct OverloadedVisit : Ts... {
  using Ts::operator()...;
};
template <class... Ts> OverloadedVisit(Ts...) -> OverloadedVisit<Ts...>;

} // namespace

raw_ostream &operator<<(raw_ostream &OS, const RootElement &Element) {
  const auto Visitor = OverloadedVisit{
      [&OS](const dxbc::RootFlags &Flags) { OS << Flags; },
      [&OS](const RootConstants &Constants) { OS << Constants; },
      [&OS](const RootDescriptor &Descriptor) { OS << Descriptor; },
      [&OS](const DescriptorTableClause &Clause) { OS << Clause; },
      [&OS](const DescriptorTable &Table) { OS << Table; },
      [&OS](const StaticSampler &Sampler) { OS << Sampler; },
  };
  std::visit(Visitor, Element);
  return OS;
}

void dumpRootElements(raw_ostream &OS, ArrayRef<RootElement> Elements) {
  OS << " RootElements{";
  bool First = true;
  for (const RootElement &Element : Elements) {
    if (!First)
      OS << ",";
    OS << " " << Element;
    First = false;
  }
  OS << "}";
}

} // namespace rootsig
} // namespace hlsl
} // namespace llvm
