//===- RootSignatureValidations.h - HLSL Root Signature helpers -----------===//
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

#ifndef LLVM_FRONTEND_HLSL_ROOTSIGNATUREVALIDATIONS_H
#define LLVM_FRONTEND_HLSL_ROOTSIGNATUREVALIDATIONS_H

#include "llvm/ADT/IntervalMap.h"
#include "llvm/Frontend/HLSL/HLSLRootSignature.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace hlsl {
namespace rootsig {

// Basic verification of RootElements

LLVM_ABI bool verifyRootFlag(uint32_t Flags);
LLVM_ABI bool verifyVersion(uint32_t Version);
LLVM_ABI bool verifyRegisterValue(uint32_t RegisterValue);
LLVM_ABI bool verifyRegisterSpace(uint32_t RegisterSpace);
LLVM_ABI bool verifyRootDescriptorFlag(uint32_t Version, uint32_t FlagsVal);
LLVM_ABI bool verifyRangeType(uint32_t Type);
LLVM_ABI bool verifyDescriptorRangeFlag(uint32_t Version, uint32_t Type,
                                        uint32_t FlagsVal);
LLVM_ABI bool verifyNumDescriptors(uint32_t NumDescriptors);
LLVM_ABI bool verifySamplerFilter(uint32_t Value);
LLVM_ABI bool verifyAddress(uint32_t Address);
LLVM_ABI bool verifyMipLODBias(float MipLODBias);
LLVM_ABI bool verifyMaxAnisotropy(uint32_t MaxAnisotropy);
LLVM_ABI bool verifyComparisonFunc(uint32_t ComparisonFunc);
LLVM_ABI bool verifyBorderColor(uint32_t BorderColor);
LLVM_ABI bool verifyLOD(float LOD);

LLVM_ABI bool verifyBoundOffset(uint32_t Offset);
LLVM_ABI bool verifyNoOverflowedOffset(uint64_t Offset);
LLVM_ABI uint64_t computeRangeBound(uint32_t Offset, uint32_t Size);

} // namespace rootsig
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_ROOTSIGNATUREVALIDATIONS_H
