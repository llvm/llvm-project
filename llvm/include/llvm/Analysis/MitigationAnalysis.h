#ifndef LLVM_ANALYSIS_MITIGATIONANALYSIS_H
#define LLVM_ANALYSIS_MITIGATIONANALYSIS_H

#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"

namespace llvm {

enum class MitigationAnalysisSummary {
  NONE = 0,
  SHORT = 1 << 0,
  FUNCTION = 1 << 1,
};

class MitigationAnalysis : public AnalysisInfoMixin<MitigationAnalysis> {
  friend AnalysisInfoMixin<MitigationAnalysis>;
  static AnalysisKey Key;

  static constexpr const char *kMitigationAnalysisDebugType =
      "mitigation_analysis";

public:
  MitigationAnalysis(
      MitigationAnalysisSummary Summary = MitigationAnalysisSummary::NONE);

  using Result = PreservedAnalyses;
  Result run(Module &M, ModuleAnalysisManager &AM);

private:
  MitigationAnalysisSummary summary_;
};

} // end namespace llvm

#endif // LLVM_ANALYSIS_MITIGATIONANALYSIS_H
