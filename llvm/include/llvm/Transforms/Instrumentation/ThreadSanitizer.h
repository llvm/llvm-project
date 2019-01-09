#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_THREADSANITIZER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_THREADSANITIZER_H

#include "llvm/IR/PassManager.h"

namespace llvm {
// Insert ThreadSanitizer (race detection) instrumentation
FunctionPass *createThreadSanitizerLegacyPassPass();

/// A function pass for tsan instrumentation.
///
/// Instruments functions to detect race conditions reads. This function pass
/// inserts calls to runtime library functions. If the functions aren't declared
/// yet, the pass inserts the declarations. Otherwise the existing globals are
struct ThreadSanitizerPass : public PassInfoMixin<ThreadSanitizerPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};
}
#endif /* LLVM_TRANSFORMS_INSTRUMENTATION_THREADSANITIZER_H */
