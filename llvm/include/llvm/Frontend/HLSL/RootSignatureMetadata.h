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
class RootSignatureValidationError
    : public ErrorInfo<RootSignatureValidationError> {
public:
  static char ID;
  std::string Msg;

  RootSignatureValidationError(const Twine &Msg) : Msg(Msg.str()) {}

  void log(raw_ostream &OS) const override { OS << Msg; }

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
};

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
