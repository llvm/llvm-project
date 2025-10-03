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

enum class ErrorKind {
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
                std::integral_constant<ErrorKind, ErrorKind::Validation>,
                StringRef ParamName, T Value) {
  Buff << "Invalid value for: " << ParamName << ":" << Value;
}

void formatImpl(
    raw_string_ostream &Buff,
    std::integral_constant<ErrorKind, ErrorKind::AppendAfterUnboundedRange>,
    dxil::ResourceClass Type, uint32_t Register, uint32_t Space) {
  Buff << "Range " << getResourceClassName(Type) << "(register=" << Register
       << ", space=" << Space << ") "
       << "cannot be appended after an unbounded range ";
}

void formatImpl(
    raw_string_ostream &Buff,
    std::integral_constant<ErrorKind, ErrorKind::ShaderRegisterOverflow>,
    dxil::ResourceClass Type, uint32_t Register, uint32_t Space) {
  Buff << "Overflow for shader register range: " << getResourceClassName(Type)
       << "(register=" << Register << ", space=" << Space << ").";
}

void formatImpl(raw_string_ostream &Buff,
                std::integral_constant<ErrorKind, ErrorKind::OffsetOverflow>,
                dxil::ResourceClass Type, uint32_t Register, uint32_t Space) {
  Buff << "Offset overflow for descriptor range: " << getResourceClassName(Type)
       << "(register=" << Register << ", space=" << Space << ").";
}

void formatImpl(raw_string_ostream &Buff,
                std::integral_constant<ErrorKind, ErrorKind::SamplerMixin>,
                dxil::ResourceClass Type, uint32_t Location) {
  Buff << "Samplers cannot be mixed with other "
       << "resource types in a descriptor table, " << getResourceClassName(Type)
       << "(location=" << Location << ")";
}

void formatImpl(
    raw_string_ostream &Buff,
    std::integral_constant<ErrorKind, ErrorKind::InvalidMetadataFormat>,
    StringRef ElementName) {
  Buff << "Invalid format for  " << ElementName;
}

void formatImpl(
    raw_string_ostream &Buff,
    std::integral_constant<ErrorKind, ErrorKind::InvalidMetadataValue>,
    StringRef ParamName) {
  Buff << "Invalid value for " << ParamName;
}

void formatImpl(raw_string_ostream &Buff,
                std::integral_constant<ErrorKind, ErrorKind::GenericMetadata>,
                StringRef Message, MDNode *MD) {
  Buff << Message;
  if (MD) {
    Buff << "\n";
    MD->printTree(Buff);
  }
}

template <typename... ArgsTs>
static void formatErrMsg(raw_string_ostream &Buff, ErrorKind Kind,
                         ArgsTs... Args) {
  switch (Kind) {
  case ErrorKind::Validation:
    formatImpl(Buff, std::integral_constant<ErrorKind, ErrorKind::Validation>{},
               Args...);
    break;
  case ErrorKind::AppendAfterUnboundedRange:
    formatImpl(Buff,
               std::integral_constant<ErrorKind,
                                      ErrorKind::AppendAfterUnboundedRange>{},
               Args...);
    break;
  case ErrorKind::ShaderRegisterOverflow:
    formatImpl(
        Buff,
        std::integral_constant<ErrorKind, ErrorKind::ShaderRegisterOverflow>{},
        Args...);
    break;
  case ErrorKind::OffsetOverflow:
    formatImpl(Buff,
               std::integral_constant<ErrorKind, ErrorKind::OffsetOverflow>{},
               Args...);
    break;
  case ErrorKind::SamplerMixin:
    formatImpl(Buff,
               std::integral_constant<ErrorKind, ErrorKind::SamplerMixin>{},
               Args...);
    break;
  case ErrorKind::GenericMetadata:
    formatImpl(Buff,
               std::integral_constant<ErrorKind, ErrorKind::GenericMetadata>{},
               Args...);
    break;

  case ErrorKind::InvalidMetadataFormat:
    formatImpl(
        Buff,
        std::integral_constant<ErrorKind, ErrorKind::InvalidMetadataFormat>{},
        Args...);
    break;

  case ErrorKind::InvalidMetadataValue:
    formatImpl(
        Buff,
        std::integral_constant<ErrorKind, ErrorKind::InvalidMetadataValue>{},
        Args...);
    break;
  }
}

template <typename... ArgsTs>
static llvm::Error createRSError(ErrorKind Kind, ArgsTs... Args) {
  std::string Msg;
  raw_string_ostream Buff(Msg);
  formatErrMsg(Buff, Kind, Args...);
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
