//===- AIRPrepare.h - Pre-serialization IR normalization ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Final IR-level normalizations needed before .metallib serialization.
/// Performs the IR-mutating typed-pointer work the writer cannot do on its
/// own (the side-table half lives in the AIRWriter as `buildPointeeTypeMap`).
///
/// Specifically:
///   * Normalize `getelementptr i1, ...` to `getelementptr i8, ...` because
///     the AIR GPU JIT has no i1 memory type.
///   * Convert pointer-typed phi nodes to i64 phis (with ptrtoint/inttoptr
///     bridges) when a block exceeds the JIT's ~63 ptr-phi limit, or when
///     any ptr phi in the function has an undef/poison incoming value.
///   * Insert ptrtoint+inttoptr "typed-pointer transitions" before
///     `air.atomic.global.*` intrinsic calls so the writer can give that
///     fresh SSA pointer a different pointee type (e.g. i32 / f32) than
///     the upstream GEP-result pointer (typically float).
///
/// Must run LAST in the AIR pipeline, immediately before the
/// AIRWriterPass.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRPREPARE_H
#define LLVM_LIB_TARGET_AIR_AIRPREPARE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class AIRPreparePass : public PassInfoMixin<AIRPreparePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class AIRPrepareLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override;
  AIRPrepareLegacy() : ModulePass(ID) {}
  static char ID;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRPREPARE_H
