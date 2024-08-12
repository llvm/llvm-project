//===-- PGOCtxProfLowering.h - Contextual PGO Instr. Lowering ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the PGOCtxProfLoweringPass class.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_PGOCTXPROFLOWERING_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_PGOCTXPROFLOWERING_H

#include "llvm/IR/PassManager.h"
namespace llvm {
class Type;

class PGOCtxProfLoweringPass : public PassInfoMixin<PGOCtxProfLoweringPass> {
public:
  explicit PGOCtxProfLoweringPass() = default;
  // True if contextual instrumentation is enabled.
  static bool isCtxIRPGOInstrEnabled();

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

/// Assign a GUID to functions as metadata. GUID calculation takes linkage into
/// account, which may change especially through and after thinlto. By
/// pre-computing and assigning as metadata, this mechanism is resilient to such
/// changes (as well as name changes e.g. suffix ".llvm." additions). It's
/// arguably a much simpler mechanism than PGO's current GV-based one, and can
/// be made available more broadly.

// FIXME(mtrofin): we can generalize this mechanism to calculate a GUID early in
// the pass pipeline, associate it with any Global Value, and then use it for
// PGO and ThinLTO.
// At that point, this should be moved elsewhere.
class AssignUniqueIDPass : public PassInfoMixin<AssignUniqueIDPass> {
public:
  explicit AssignUniqueIDPass() = default;
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  static const char *GUIDMetadataName;
  // This should become GlobalValue::getGUID
  static uint64_t getGUID(const Function &F);
};

} // namespace llvm
#endif
