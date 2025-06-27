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
#include "llvm/ADT/IntervalMap.h"
#include "llvm/Frontend/HLSL/HLSLRootSignature.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
class LLVMContext;
class MDNode;
class Metadata;

namespace hlsl {
namespace rootsig {

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const dxbc::RootFlags &Flags);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const RootConstants &Constants);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const DescriptorTableClause &Clause);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const DescriptorTable &Table);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const RootDescriptor &Descriptor);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const StaticSampler &StaticSampler);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const RootElement &Element);

LLVM_ABI void dumpRootElements(raw_ostream &OS, ArrayRef<RootElement> Elements);

} // namespace rootsig
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLROOTSIGNATUREUTILS_H
