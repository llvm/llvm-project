//===-- DeleteDebugRecordPass.h ------------------*- C++ -*-===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_DELETEDEBUGRECORDPASS_H
#define LLVM_TRANSFORMS_UTILS_DELETEDEBUGRECORDPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class DeleteDebugRecordPass : public PassInfoMixin<DeleteDebugRecordPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_DELETEDEBUGRECORDPASS_H
