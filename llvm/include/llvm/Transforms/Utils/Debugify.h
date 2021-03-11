//===- Debugify.h - Check debug info preservation in optimizations --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file Interface to the `debugify` synthetic/original debug info testing
/// utility.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_DEBUGIFY_H
#define LLVM_TRANSFORMS_UTILS_DEBUGIFY_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"

using DebugFnMap = llvm::DenseMap<llvm::StringRef, const llvm::DISubprogram *>;
using DebugInstMap = llvm::DenseMap<const llvm::Instruction *, bool>;
using WeakInstValueMap =
    llvm::DenseMap<const llvm::Instruction *, llvm::WeakVH>;

/// Used to track the Debug Info Metadata information.
struct DebugInfoPerPass {
  // This maps a function name to its associated DISubprogram.
  DebugFnMap DIFunctions;
  // This maps an instruction and the info about whether it has !dbg attached.
  DebugInstMap DILocations;
  // This tracks value (instruction) deletion. If an instruction gets deleted,
  // WeakVH nulls itself.
  WeakInstValueMap InstToDelete;
};

/// Map pass names to a per-pass DebugInfoPerPass instance.
using DebugInfoPerPassMap = llvm::MapVector<llvm::StringRef, DebugInfoPerPass>;

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

/// Collect original debug information before a pass.
///
/// \param M The module to collect debug information from.
/// \param Functions A range of functions to collect debug information from.
/// \param DIPreservationMap A map to collect the DI metadata.
/// \param Banner A prefix string to add to debug/error messages.
/// \param NameOfWrappedPass A name of a pass to add to debug/error messages.
bool collectDebugInfoMetadata(Module &M,
                              iterator_range<Module::iterator> Functions,
                              DebugInfoPerPassMap &DIPreservationMap,
                              StringRef Banner, StringRef NameOfWrappedPass);

/// Check original debug information after a pass.
///
/// \param M The module to collect debug information from.
/// \param Functions A range of functions to collect debug information from.
/// \param DIPreservationMap A map used to check collected the DI metadata.
/// \param Banner A prefix string to add to debug/error messages.
/// \param NameOfWrappedPass A name of a pass to add to debug/error messages.
bool checkDebugInfoMetadata(Module &M,
                            iterator_range<Module::iterator> Functions,
                            DebugInfoPerPassMap &DIPreservationMap,
                            StringRef Banner, StringRef NameOfWrappedPass,
                            StringRef OrigDIVerifyBugsReportFilePath);
} // namespace llvm

/// Used to check whether we track synthetic or original debug info.
enum class DebugifyMode { NoDebugify, SyntheticDebugInfo, OriginalDebugInfo };

llvm::ModulePass *createDebugifyModulePass(
    enum DebugifyMode Mode = DebugifyMode::SyntheticDebugInfo,
    llvm::StringRef NameOfWrappedPass = "",
    DebugInfoPerPassMap *DIPreservationMap = nullptr);
llvm::FunctionPass *createDebugifyFunctionPass(
    enum DebugifyMode Mode = DebugifyMode::SyntheticDebugInfo,
    llvm::StringRef NameOfWrappedPass = "",
    DebugInfoPerPassMap *DIPreservationMap = nullptr);

struct NewPMDebugifyPass : public llvm::PassInfoMixin<NewPMDebugifyPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

/// Track how much `debugify` information (in the `synthetic` mode only)
/// has been lost.
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

llvm::ModulePass *createCheckDebugifyModulePass(
    bool Strip = false, llvm::StringRef NameOfWrappedPass = "",
    DebugifyStatsMap *StatsMap = nullptr,
    enum DebugifyMode Mode = DebugifyMode::SyntheticDebugInfo,
    DebugInfoPerPassMap *DIPreservationMap = nullptr,
    llvm::StringRef OrigDIVerifyBugsReportFilePath = "");

llvm::FunctionPass *createCheckDebugifyFunctionPass(
    bool Strip = false, llvm::StringRef NameOfWrappedPass = "",
    DebugifyStatsMap *StatsMap = nullptr,
    enum DebugifyMode Mode = DebugifyMode::SyntheticDebugInfo,
    DebugInfoPerPassMap *DIPreservationMap = nullptr,
    llvm::StringRef OrigDIVerifyBugsReportFilePath = "");

struct NewPMCheckDebugifyPass
    : public llvm::PassInfoMixin<NewPMCheckDebugifyPass> {
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

namespace llvm {
void exportDebugifyStats(StringRef Path, const DebugifyStatsMap &Map);

struct DebugifyEachInstrumentation {
  DebugifyStatsMap StatsMap;

  void registerCallbacks(PassInstrumentationCallbacks &PIC);
};

/// DebugifyCustomPassManager wraps each pass with the debugify passes if
/// needed.
/// NOTE: We support legacy custom pass manager only.
/// TODO: Add New PM support for custom pass manager.
class DebugifyCustomPassManager : public legacy::PassManager {
  StringRef OrigDIVerifyBugsReportFilePath;
  DebugifyStatsMap *DIStatsMap = nullptr;
  DebugInfoPerPassMap *DIPreservationMap = nullptr;
  enum DebugifyMode Mode = DebugifyMode::NoDebugify;

public:
  using super = legacy::PassManager;

  void add(Pass *P) override {
    // Wrap each pass with (-check)-debugify passes if requested, making
    // exceptions for passes which shouldn't see -debugify instrumentation.
    bool WrapWithDebugify = Mode != DebugifyMode::NoDebugify &&
                            !P->getAsImmutablePass() && !isIRPrintingPass(P) &&
                            !isBitcodeWriterPass(P);
    if (!WrapWithDebugify) {
      super::add(P);
      return;
    }

    // Either apply -debugify/-check-debugify before/after each pass and collect
    // debug info loss statistics, or collect and check original debug info in
    // the optimizations.
    PassKind Kind = P->getPassKind();
    StringRef Name = P->getPassName();

    // TODO: Implement Debugify for LoopPass.
    switch (Kind) {
    case PT_Function:
      super::add(createDebugifyFunctionPass(Mode, Name, DIPreservationMap));
      super::add(P);
      super::add(createCheckDebugifyFunctionPass(
          isSyntheticDebugInfo(), Name, DIStatsMap, Mode, DIPreservationMap,
          OrigDIVerifyBugsReportFilePath));
      break;
    case PT_Module:
      super::add(createDebugifyModulePass(Mode, Name, DIPreservationMap));
      super::add(P);
      super::add(createCheckDebugifyModulePass(
          isSyntheticDebugInfo(), Name, DIStatsMap, Mode, DIPreservationMap,
          OrigDIVerifyBugsReportFilePath));
      break;
    default:
      super::add(P);
      break;
    }
  }

  // Used within DebugifyMode::SyntheticDebugInfo mode.
  void setDIStatsMap(DebugifyStatsMap &StatMap) { DIStatsMap = &StatMap; }
  // Used within DebugifyMode::OriginalDebugInfo mode.
  void setDIPreservationMap(DebugInfoPerPassMap &PerPassMap) {
    DIPreservationMap = &PerPassMap;
  }
  void setOrigDIVerifyBugsReportFilePath(StringRef BugsReportFilePath) {
    OrigDIVerifyBugsReportFilePath = BugsReportFilePath;
  }
  StringRef getOrigDIVerifyBugsReportFilePath() const {
    return OrigDIVerifyBugsReportFilePath;
  }

  void setDebugifyMode(enum DebugifyMode M) { Mode = M; }

  bool isSyntheticDebugInfo() const {
    return Mode == DebugifyMode::SyntheticDebugInfo;
  }
  bool isOriginalDebugInfoMode() const {
    return Mode == DebugifyMode::OriginalDebugInfo;
  }

  const DebugifyStatsMap &getDebugifyStatsMap() const { return *DIStatsMap; }
  DebugInfoPerPassMap &getDebugInfoPerPassMap() { return *DIPreservationMap; }
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_DEBUGIFY_H
