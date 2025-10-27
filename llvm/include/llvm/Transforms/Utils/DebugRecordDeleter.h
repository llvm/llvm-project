#ifndef LLVM_TRANSFORMS_UTILS_DEBUG_RECORD_DELETER_H
#define LLVM_TRANSFORMS_UTILS_DEBUG_RECORD_DELETER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

    class DebugRecordDeleterPass : public PassInfoMixin<DebugRecordDeleterPass> {
    public:
        PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
    };

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_DEBUG_RECORD_DELETER_H
