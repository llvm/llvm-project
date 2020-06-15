#ifndef LLVM_INLINEFEATURESANALYSIS_H_
#define LLVM_INLINEFEATURESANALYSIS_H_

#include "llvm/IR/PassManager.h"

namespace llvm {
class Function;

class InlineFeaturesAnalysis
    : public AnalysisInfoMixin<InlineFeaturesAnalysis> {
public:
  static AnalysisKey Key;
  struct Result {
    /// Number of basic blocks
    int64_t BasicBlockCount = 0;

    /// Number of blocks reached from a conditional instruction, or that are
    /// 'cases' of a SwitchInstr.
    // FIXME: We may want to replace this with a more meaningful metric, like
    // number of conditionally executed blocks:
    // 'if (a) s();' would be counted here as 2 blocks, just like
    // 'if (a) s(); else s2(); s3();' would.
    int64_t BlocksReachedFromConditionalInstruction = 0;

    /// Number of uses of this function, plus 1 if the function is callable
    /// outside the module.
    int64_t Uses = 0;

    /// Number of direct calls made from this function to other functions
    /// defined in this module.
    int64_t DirectCallsToDefinedFunctions = 0;
  };
  Result run(const Function &F, FunctionAnalysisManager &FAM);
};

} // namespace llvm
#endif // LLVM_INLINEFEATURESANALYSIS_H_