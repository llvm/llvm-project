#ifndef LLVM_TRANSFORMS_UTILS_DEBUG_RECORD_COUNTER_H
#define LLVM_TRANSFORMS_UTILS_DEBUG_RECORD_COUNTER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

    class DebugRecordCounterPass : public PassInfoMixin<DebugRecordCounterPass> {
    public:
        PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
    };

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_DEBUG_RECORD_COUNTER_H
