//===- AIRCrossBufferStoreSeparate.h - Re-separate device stores -*-C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Work around AGX-1: the Apple AGX GPU JIT corrupts a device (AS1) store for
/// warp 0 when two device stores to DIFFERENT buffer arguments at the SAME
/// per-thread element offset are program-adjacent (straight-line). At -O0 each
/// masked store lives in its own predicated basic block, so the cross-buffer
/// pairs are never straight-line-adjacent and the JIT cannot coalesce them ->
/// correct. At -O1+ the mid-end flattens all stores into one straight-line
/// block -> the AGX trigger. This late pass detects straight-line same-offset
/// cross-buffer device-store adjacency and restores the O0-style control-flow
/// separation by sinking each conflicting store into its own single-entry
/// block, so the JIT's address-based store scheduler never sees the pair in one
/// straight-line run. See AGX_BUGS.md (AGX-1).
///
/// No-op by construction on single-output kernels (GEMM/dot/conv/mm): they have
/// no cross-buffer same-offset stores, detection finds nothing, IR unchanged.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRCROSSBUFFERSTORESEPARATE_H
#define LLVM_LIB_TARGET_AIR_AIRCROSSBUFFERSTORESEPARATE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRCrossBufferStoreSeparatePass
    : public PassInfoMixin<AIRCrossBufferStoreSeparatePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class AIRCrossBufferStoreSeparateLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRCrossBufferStoreSeparateLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRCROSSBUFFERSTORESEPARATE_H
