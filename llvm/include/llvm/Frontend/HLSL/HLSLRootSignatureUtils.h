//===- HLSLRootSignatureUtils.h - HLSL Root Signature helpers -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper obejcts for working with HLSL Root
/// Signatures.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_HLSLROOTSIGNATUREUTILS_H
#define LLVM_FRONTEND_HLSL_HLSLROOTSIGNATUREUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Frontend/HLSL/HLSLRootSignature.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
class LLVMContext;
class MDNode;
class Metadata;

namespace hlsl {
namespace rootsig {

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const RootFlags &Flags);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const RootConstants &Constants);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const DescriptorTableClause &Clause);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const DescriptorTable &Table);

LLVM_ABI void dumpRootElements(raw_ostream &OS, ArrayRef<RootElement> Elements);

class MetadataBuilder {
public:
  MetadataBuilder(llvm::LLVMContext &Ctx, ArrayRef<RootElement> Elements)
      : Ctx(Ctx), Elements(Elements) {}

  /// Iterates through the elements and dispatches onto the correct Build method
  ///
  /// Accumulates the root signature and returns the Metadata node that is just
  /// a list of all the elements
  LLVM_ABI MDNode *BuildRootSignature();

private:
  /// Define the various builders for the different metadata types
  MDNode *BuildRootFlags(const RootFlags &Flags);
  MDNode *BuildRootConstants(const RootConstants &Constants);
  MDNode *BuildRootDescriptor(const RootDescriptor &Descriptor);
  MDNode *BuildDescriptorTable(const DescriptorTable &Table);
  MDNode *BuildDescriptorTableClause(const DescriptorTableClause &Clause);
  MDNode *BuildStaticSampler(const StaticSampler &Sampler);

  llvm::LLVMContext &Ctx;
  ArrayRef<RootElement> Elements;
  SmallVector<Metadata *> GeneratedMetadata;
};

} // namespace rootsig
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLROOTSIGNATUREUTILS_H
