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

#include "llvm/Frontend/HLSL/HLSLRootSignature.h"
#include "llvm/IR/Constants.h"
#include "llvm/MC/DXContainerRootSignature.h"

namespace llvm {
class LLVMContext;
class MDNode;
class Metadata;

namespace hlsl {
namespace rootsig {

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

  LLVM_ABI bool ParseRootSignature(LLVMContext *Ctx,
                                   mcdxbc::RootSignatureDesc &RSD);

private:
  bool parseRootFlags(LLVMContext *Ctx, mcdxbc::RootSignatureDesc &RSD,
                      MDNode *RootFlagNode);
  bool parseRootConstants(LLVMContext *Ctx, mcdxbc::RootSignatureDesc &RSD,
                          MDNode *RootConstantNode);
  bool parseRootDescriptors(LLVMContext *Ctx, mcdxbc::RootSignatureDesc &RSD,
                            MDNode *RootDescriptorNode,
                            RootSignatureElementKind ElementKind);
  bool parseDescriptorRange(LLVMContext *Ctx, mcdxbc::DescriptorTable &Table,
                            MDNode *RangeDescriptorNode);
  bool parseDescriptorTable(LLVMContext *Ctx, mcdxbc::RootSignatureDesc &RSD,
                            MDNode *DescriptorTableNode);
  bool parseRootSignatureElement(LLVMContext *Ctx,
                                 mcdxbc::RootSignatureDesc &RSD,
                                 MDNode *Element);
  bool parseStaticSampler(LLVMContext *Ctx, mcdxbc::RootSignatureDesc &RSD,
                          MDNode *StaticSamplerNode);

  bool validateRootSignature(LLVMContext *Ctx,
                             const llvm::mcdxbc::RootSignatureDesc &RSD);

  MDNode *Root;
};

} // namespace rootsig
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_ROOTSIGNATUREMETADATA_H
