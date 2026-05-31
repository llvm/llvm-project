//===- AIRLibWriter.h - .metallib container writer ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_AIRLIBWRITER_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_AIRLIBWRITER_H

#include "PointeeTypeMap.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

namespace llvm {
namespace metal {

/// Platform tag stored in the .metallib `PLAT` field. The format has no public
/// spec; values are empirically observed: macOS uses 0x81, iOS uses 0x82.
enum class AIRPlatform : uint8_t {
  macOS = 0x81,
  iOS = 0x82,
};

/// Default OS major recorded in the metallib `OSVR` field. Apple's metallib
/// writer pins this to 26 regardless of host SDK.
inline constexpr uint16_t DefaultAIRLibOSMajor = 26;

/// Default AIR Shading Language version recorded in the metallib `LANG`
/// field (3.2 == latest stable at time of writing).
inline constexpr uint16_t DefaultAIRLangMajor = 3;
inline constexpr uint16_t DefaultAIRLangMinor = 2;

struct AIRLibOptions {
  AIRPlatform Platform = AIRPlatform::macOS;
  uint16_t OSMajor = DefaultAIRLibOSMajor;
  uint16_t OSMinor = 0;
  uint16_t OSPatch = 0;
  uint16_t AIRMajor = DefaultAIRLangMajor;
  uint16_t AIRMinor = DefaultAIRLangMinor;
};

// Write the module as a metallib to the output stream.
//
// The PointeeTypeMap is critical: LLVM's Module has opaque pointers,
// but AIR's bitcode format requires typed POINTER records. The writer
// uses the map to emit:
// ptr addrspace(1) %buf → float addrspace(1)* %buf (in bitcode)
//
// Without the map, all pointers would be opaque and AIR would reject
// the metallib.
//
// Pipeline:
// Module + PointeeTypeMap
// → custom bitcode (typed POINTER records, not LLVM's BitcodeWriter)
// → bitcode wrapper (0x0B17C0DE magic)
// → metallib container (MTLB header + 4 sections)
bool writeAIRLib(llvm::Module &M, PointeeTypeMap &PTM,
                   llvm::raw_ostream &OS, const AIRLibOptions &Opts = {});

// Convenience: write to a byte vector.
std::vector<uint8_t> serializeAIRLib(llvm::Module &M,
                                       PointeeTypeMap &PTM,
                                       const AIRLibOptions &Opts = {});

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRWRITER_AIRLIBWRITER_H
