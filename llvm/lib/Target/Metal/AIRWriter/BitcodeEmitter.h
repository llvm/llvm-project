//===- BitcodeEmitter.h - Metal v1 bitcode emitter --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_AIRWRITER_BITCODEEMITTER_H
#define LLVM_LIB_TARGET_METAL_AIRWRITER_BITCODEEMITTER_H

#include "PointeeTypeMap.h"
#include "llvm/IR/Module.h"
#include <vector>

namespace llvm {
namespace metal {

/// Emit LLVM Module as bitcode with Metal-compatible typed pointers.
///
/// LLVM 17+ only supports opaque pointers in-memory, but Metal's GPU JIT
/// requires typed POINTER records (code 8) in bitcode. This emitter walks
/// the Module and uses the PointeeTypeMap to emit typed pointer type records
/// instead of opaque pointer records (code 25).
///
/// This replaces LLVM's WriteBitcodeToFile for Metal targets.
std::vector<uint8_t> emitMetalBitcode(llvm::Module &M, PointeeTypeMap &PTM);

} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_AIRWRITER_BITCODEEMITTER_H
