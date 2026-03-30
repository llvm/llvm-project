//===-- DebugRecordCountPass.h ------------------*- C++ -*-===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_DEBUGRECORDCOUNTPASS_H
#define LLVM_TRANSFORMS_UTILS_DEBUGRECORDCOUNTPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class DebugRecordCountPass : public PassInfoMixin<DebugRecordCountPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_DEBUGRECORDCOUNTPASS_H
