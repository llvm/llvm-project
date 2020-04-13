//===- Debugify.h - Attach synthetic debug info to everything -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file Interface to the `debugify` synthetic debug info testing utility.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORM_UTILS_DEBUGIFY_H
#define LLVM_TRANSFORM_UTILS_DEBUGIFY_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class DIBuilder;

/// Add synthesized debug information to a module.
///
/// \param M The module to add debug information to.
/// \param Functions A range of functions to add debug information to.
/// \param Banner A prefix string to add to debug/error messages.
/// \param ApplyToMF A call back that will add debug information to the
///                  MachineFunction for a Function. If nullptr, then the
///                  MachineFunction (if any) will not be modified.
bool applyDebugifyMetadata(
    Module &M, iterator_range<Module::iterator> Functions, StringRef Banner,
    std::function<bool(DIBuilder &, Function &)> ApplyToMF);

/// Strip out all of the metadata and debug info inserted by debugify. If no
/// llvm.debugify module-level named metadata is present, this is a no-op.
/// Returns true if any change was made.
bool stripDebugifyMetadata(Module &M);

} // namespace llvm

llvm::ModulePass *createDebugifyModulePass();
llvm::FunctionPass *createDebugifyFunctionPass();

struct NewPMDebugifyPass : public llvm::PassInfoMixin<NewPMDebugifyPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

/// Track how much `debugify` information has been lost.
struct DebugifyStatistics {
  /// Number of missing dbg.values.
  unsigned NumDbgValuesMissing = 0;

  /// Number of dbg.values expected.
  unsigned NumDbgValuesExpected = 0;

  /// Number of instructions with empty debug locations.
  unsigned NumDbgLocsMissing = 0;

  /// Number of instructions expected to have debug locations.
  unsigned NumDbgLocsExpected = 0;

  /// Get the ratio of missing/expected dbg.values.
  float getMissingValueRatio() const {
    return float(NumDbgValuesMissing) / float(NumDbgLocsExpected);
  }

  /// Get the ratio of missing/expected instructions with locations.
  float getEmptyLocationRatio() const {
    return float(NumDbgLocsMissing) / float(NumDbgLocsExpected);
  }
};

/// Map pass names to a per-pass DebugifyStatistics instance.
using DebugifyStatsMap = llvm::MapVector<llvm::StringRef, DebugifyStatistics>;

llvm::ModulePass *
createCheckDebugifyModulePass(bool Strip = false,
                              llvm::StringRef NameOfWrappedPass = "",
                              DebugifyStatsMap *StatsMap = nullptr);

llvm::FunctionPass *
createCheckDebugifyFunctionPass(bool Strip = false,
                                llvm::StringRef NameOfWrappedPass = "",
                                DebugifyStatsMap *StatsMap = nullptr);

struct NewPMCheckDebugifyPass
    : public llvm::PassInfoMixin<NewPMCheckDebugifyPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

#endif // LLVM_TRANSFORM_UTILS_DEBUGIFY_H
