//===--- SIProgramInfo.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines struct to track resource usage and hardware flags for kernels and
/// entry functions.
///
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SIPROGRAMINFO_H
#define LLVM_LIB_TARGET_AMDGPU_SIPROGRAMINFO_H

#include "llvm/IR/CallingConv.h"
#include <cstdint>

namespace llvm {

/// Track resource usage for kernels / entry functions.
struct SIProgramInfo {
    // Fields set in PGM_RSRC1 pm4 packet.
    uint32_t VGPRBlocks = 0;
    uint32_t SGPRBlocks = 0;
    uint32_t Priority = 0;
    uint32_t FloatMode = 0;
    uint32_t Priv = 0;
    uint32_t DX10Clamp = 0;
    uint32_t DebugMode = 0;
    uint32_t IEEEMode = 0;
    uint32_t WgpMode = 0; // GFX10+
    uint32_t MemOrdered = 0; // GFX10+
    uint64_t ScratchSize = 0;

    // Fields set in PGM_RSRC2 pm4 packet.
    uint32_t LDSBlocks = 0;
    uint32_t ScratchBlocks = 0;

    uint64_t ComputePGMRSrc2 = 0;
    uint64_t ComputePGMRSrc3GFX90A = 0;

    uint32_t NumVGPR = 0;
    uint32_t NumArchVGPR = 0;
    uint32_t NumAccVGPR = 0;
    uint32_t AccumOffset = 0;
    uint32_t TgSplit = 0;
    uint32_t NumSGPR = 0;
    uint32_t LDSSize = 0;
    bool FlatUsed = false;

    // Number of SGPRs that meets number of waves per execution unit request.
    uint32_t NumSGPRsForWavesPerEU = 0;

    // Number of VGPRs that meets number of waves per execution unit request.
    uint32_t NumVGPRsForWavesPerEU = 0;

    // Final occupancy.
    uint32_t Occupancy = 0;

    // Whether there is recursion, dynamic allocas, indirect calls or some other
    // reason there may be statically unknown stack usage.
    bool DynamicCallStack = false;

    // Bonus information for debugging.
    bool VCCUsed = false;

    SIProgramInfo() = default;

    /// Compute the value of the ComputePGMRsrc1 register.
    uint64_t getComputePGMRSrc1() const;
    uint64_t getPGMRSrc1(CallingConv::ID CC) const;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SIPROGRAMINFO_H
