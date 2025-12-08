//===--- MitigationAnalysis.h - Emit LLVM Code from ASTs for a Module -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This processes mitigation metadata to create a report on enablement
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MITIGATIONANALYSIS_H
#define LLVM_ANALYSIS_MITIGATIONANALYSIS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/JSON.h"

#include <string>

namespace llvm {

enum class MitigationAnalysisSummaryType {
  NONE = 0,
  EMBED = 1 << 0,
  JSON = 1 << 1,
};

struct MitigationAnalysisOptions {
  MitigationAnalysisSummaryType SummaryType =
      MitigationAnalysisSummaryType::NONE;
  std::string OutputRoot = "/tmp";
  std::string OutputUnitName;
  std::string LibCXXPrefix = "std::";

  MitigationAnalysisOptions() = default;
  MitigationAnalysisOptions(MitigationAnalysisSummaryType ST, StringRef Root,
                            StringRef UnitName, StringRef Prefix)
      : SummaryType(ST), OutputRoot(Root), OutputUnitName(UnitName),
        LibCXXPrefix(Prefix) {}

  StringRef getOutputRoot() const { return OutputRoot; }
  StringRef getOutputUnitName() const { return OutputUnitName; }
  StringRef getLibCXXPrefix() const { return LibCXXPrefix; }
};

class MitigationAnalysisPass
    : public AnalysisInfoMixin<MitigationAnalysisPass> {
  friend AnalysisInfoMixin<MitigationAnalysisPass>;
  static AnalysisKey Key;

  static constexpr const char *kMitigationAnalysisDebugType =
      "mitigation_analysis";

public:
  MitigationAnalysisPass(MitigationAnalysisOptions Opts = {}) : Options(Opts) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  MitigationAnalysisOptions Options;

  void writeJsonToFile(const llvm::json::Value &JsonValue);
  void getHardenedAccessFunctions(Module &M,
                                  StringMap<bool> &HardenedCXXFunctions);
  bool compareDemangledFunctionName(StringRef MangledName,
                                    StringRef CompareName);
};

MitigationAnalysisOptions getMitigationAnalysisOptions();

} // end namespace llvm

#endif // LLVM_ANALYSIS_MITIGATIONANALYSIS_H
