//===- AIRLowerAtomicRMW.h - Lower atomicrmw to AIR intrinsics *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Lower `atomicrmw` instructions to `air.atomic.*` intrinsic calls.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRLOWERATOMICRMW_H
#define LLVM_LIB_TARGET_AIR_AIRLOWERATOMICRMW_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

/// The lock table of the Metal pipeline's 64- and 128-bit atomics: a device
/// buffer parameter of this name at this location index, of
/// kAtomicLockCount 32-bit words the host binds zeroed.
inline constexpr const char *kAtomicLockTableName = "air_atomic_locks";
inline constexpr unsigned kAtomicLockTableIndex = 30;
inline constexpr unsigned kAtomicLockCount = 1024;

class AIRLowerAtomicRMWPass : public RequiredPassInfoMixin<AIRLowerAtomicRMWPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class AIRLowerAtomicRMWLegacy : public ModulePass {
  bool OrderingHints;

public:
  bool runOnModule(Module &M) override;
  /// With \p OrderingHints, ordered atomics keep their ordering as
  /// air::AtomicOrderingMD (for consumers other than Metal).
  AIRLowerAtomicRMWLegacy(bool OrderingHints = false)
      : ModulePass(ID), OrderingHints(OrderingHints) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRLOWERATOMICRMW_H
