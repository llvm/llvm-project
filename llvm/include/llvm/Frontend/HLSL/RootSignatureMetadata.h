//===- RootSignatureMetadata.h - HLSL Root Signature helpers --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains a library for working with HLSL Root Signatures and
/// their metadata representation.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_ROOTSIGNATUREMETADATA_H
#define LLVM_FRONTEND_HLSL_ROOTSIGNATUREMETADATA_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/HLSL/HLSLRootSignature.h"
#include "llvm/IR/Constants.h"
#include "llvm/MC/DXContainerRootSignature.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class LLVMContext;
class MDNode;
class Metadata;

namespace hlsl {
namespace rootsig {

enum class RSErrorKind {
  Validation,
  AppendAfterUnboundedRange,
  ShaderRegisterOverflow,
  OffsetOverflow,
  SamplerMixin,
  GenericMetadata,
  InvalidMetadataFormat,
  InvalidMetadataValue
};

template <typename T>
void formatImpl(raw_string_ostream &Buff,
                std::integral_constant<RSErrorKind, RSErrorKind::Validation>,
                StringRef ParamName, T Value);

void formatImpl(
    raw_string_ostream &Buff,
    std::integral_constant<RSErrorKind, RSErrorKind::AppendAfterUnboundedRange>,
    dxil::ResourceClass Type, uint32_t Register, uint32_t Space);

void formatImpl(
    raw_string_ostream &Buff,
    std::integral_constant<RSErrorKind, RSErrorKind::ShaderRegisterOverflow>,
    dxil::ResourceClass Type, uint32_t Register, uint32_t Space);

void formatImpl(
    raw_string_ostream &Buff,
    std::integral_constant<RSErrorKind, RSErrorKind::OffsetOverflow>,
    dxil::ResourceClass Type, uint32_t Register, uint32_t Space);

void formatImpl(raw_string_ostream &Buff,
                std::integral_constant<RSErrorKind, RSErrorKind::SamplerMixin>,
                dxil::ResourceClass Type, uint32_t Location);

void formatImpl(
    raw_string_ostream &Buff,
    std::integral_constant<RSErrorKind, RSErrorKind::InvalidMetadataFormat>,
    StringRef ElementName);

void formatImpl(
    raw_string_ostream &Buff,
    std::integral_constant<RSErrorKind, RSErrorKind::InvalidMetadataValue>,
    StringRef ParamName);

void formatImpl(
    raw_string_ostream &Buff,
    std::integral_constant<RSErrorKind, RSErrorKind::GenericMetadata>,
    StringRef Message, MDNode *MD);

template <typename... ArgsTs>
inline void formatImpl(raw_string_ostream &Buff, RSErrorKind Kind,
                       ArgsTs... Args) {
  switch (Kind) {
  case RSErrorKind::Validation:
    return formatImpl(
        Buff, std::integral_constant<RSErrorKind, RSErrorKind::Validation>(),
        Args...);
  case RSErrorKind::AppendAfterUnboundedRange:
    return formatImpl(
        Buff,
        std::integral_constant<RSErrorKind,
                               RSErrorKind::AppendAfterUnboundedRange>(),
        Args...);
  case RSErrorKind::ShaderRegisterOverflow:
    return formatImpl(
        Buff,
        std::integral_constant<RSErrorKind,
                               RSErrorKind::ShaderRegisterOverflow>(),
        Args...);
  case RSErrorKind::OffsetOverflow:
    return formatImpl(
        Buff,
        std::integral_constant<RSErrorKind, RSErrorKind::OffsetOverflow>(),
        Args...);
  case RSErrorKind::SamplerMixin:
    return formatImpl(
        Buff, std::integral_constant<RSErrorKind, RSErrorKind::SamplerMixin>(),
        Args...);
  case RSErrorKind::InvalidMetadataFormat:
    return formatImpl(
        Buff,
        std::integral_constant<RSErrorKind,
                               RSErrorKind::InvalidMetadataFormat>(),
        Args...);
  case RSErrorKind::InvalidMetadataValue:
    return formatImpl(
        Buff,
        std::integral_constant<RSErrorKind,
                               RSErrorKind::InvalidMetadataValue>(),
        Args...);
  case RSErrorKind::GenericMetadata:
    return formatImpl(
        Buff,
        std::integral_constant<RSErrorKind, RSErrorKind::GenericMetadata>(),
        Args...);
  }
}

template <typename... ArgsTs>
static llvm::Error createRSError(RSErrorKind Kind, ArgsTs... Args) {
  std::string Msg;
  raw_string_ostream Buff(Msg);
  formatImpl(Buff, Kind, Args...);
  return createStringError(std::move(Buff.str()), inconvertibleErrorCode());
}

class MetadataBuilder {
public:
  MetadataBuilder(llvm::LLVMContext &Ctx, ArrayRef<RootElement> Elements)
      : Ctx(Ctx), Elements(Elements) {}

  /// Iterates through elements and dispatches onto the correct Build* method
  ///
  /// Accumulates the root signature and returns the Metadata node that is just
  /// a list of all the elements
  LLVM_ABI MDNode *BuildRootSignature();

private:
  /// Define the various builders for the different metadata types
  MDNode *BuildRootFlags(const dxbc::RootFlags &Flags);
  MDNode *BuildRootConstants(const RootConstants &Constants);
  MDNode *BuildRootDescriptor(const RootDescriptor &Descriptor);
  MDNode *BuildDescriptorTable(const DescriptorTable &Table);
  MDNode *BuildDescriptorTableClause(const DescriptorTableClause &Clause);
  MDNode *BuildStaticSampler(const StaticSampler &Sampler);

  llvm::LLVMContext &Ctx;
  ArrayRef<RootElement> Elements;
  SmallVector<Metadata *> GeneratedMetadata;
};

enum class RootSignatureElementKind {
  Error = 0,
  RootFlags = 1,
  RootConstants = 2,
  SRV = 3,
  UAV = 4,
  CBV = 5,
  DescriptorTable = 6,
  StaticSamplers = 7
};

class MetadataParser {
public:
  MetadataParser(MDNode *Root) : Root(Root) {}

  LLVM_ABI llvm::Expected<llvm::mcdxbc::RootSignatureDesc>
  ParseRootSignature(uint32_t Version);

private:
  llvm::Error parseRootFlags(mcdxbc::RootSignatureDesc &RSD,
                             MDNode *RootFlagNode);
  llvm::Error parseRootConstants(mcdxbc::RootSignatureDesc &RSD,
                                 MDNode *RootConstantNode);
  llvm::Error parseRootDescriptors(mcdxbc::RootSignatureDesc &RSD,
                                   MDNode *RootDescriptorNode,
                                   RootSignatureElementKind ElementKind);
  llvm::Error parseDescriptorRange(mcdxbc::DescriptorTable &Table,
                                   MDNode *RangeDescriptorNode);
  llvm::Error parseDescriptorTable(mcdxbc::RootSignatureDesc &RSD,
                                   MDNode *DescriptorTableNode);
  llvm::Error parseRootSignatureElement(mcdxbc::RootSignatureDesc &RSD,
                                        MDNode *Element);
  llvm::Error parseStaticSampler(mcdxbc::RootSignatureDesc &RSD,
                                 MDNode *StaticSamplerNode);

  llvm::Error validateRootSignature(const llvm::mcdxbc::RootSignatureDesc &RSD);

  MDNode *Root;
};

} // namespace rootsig
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_ROOTSIGNATUREMETADATA_H
