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
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Frontend/HLSL/HLSLRootSignature.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
namespace hlsl {
namespace rootsig {

// Basic verification of RootElements

LLVM_ABI bool verifyRootFlag(uint32_t Flags);
LLVM_ABI bool verifyVersion(dxbc::RootSignatureVersion Version);
LLVM_ABI bool verifyRegisterValue(uint32_t RegisterValue);
LLVM_ABI bool verifyRegisterSpace(uint32_t RegisterSpace);
LLVM_ABI bool verifyRootDescriptorFlag(dxbc::RootSignatureVersion Version,
                                       uint32_t FlagsVal);
LLVM_ABI bool verifyRangeType(dxbc::DescriptorRangeType Type);
LLVM_ABI bool verifyDescriptorRangeFlag(dxbc::RootSignatureVersion Version,
                                        dxbc::DescriptorRangeType Type,
                                        uint32_t FlagsVal);
LLVM_ABI bool verifyNumDescriptors(uint32_t NumDescriptors);
LLVM_ABI bool verifySamplerFilter(dxbc::SamplerFilter Value);
LLVM_ABI bool verifyAddress(dxbc::TextureAddressMode Address);
LLVM_ABI bool verifyMipLODBias(float MipLODBias);
LLVM_ABI bool verifyMaxAnisotropy(uint32_t MaxAnisotropy);
LLVM_ABI bool verifyComparisonFunc(dxbc::ComparisonFunc ComparisonFunc);
LLVM_ABI bool verifyBorderColor(dxbc::StaticBorderColor BorderColor);
LLVM_ABI bool verifyLOD(float LOD);

} // namespace rootsig
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_ROOTSIGNATUREVALIDATIONS_H
