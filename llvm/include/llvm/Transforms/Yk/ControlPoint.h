#ifndef LLVM_TRANSFORMS_YK_CONTROLPOINT_H
#define LLVM_TRANSFORMS_YK_CONTROLPOINT_H

#include "llvm/IR/PassManager.h"

// The name of the "dummy function" that the user puts in their interpreter
// implementation and that we will replace with our own code.
#define YK_DUMMY_CONTROL_POINT "yk_control_point"

// The name of the new control point replacing the user's dummy control point.
#define YK_NEW_CONTROL_POINT "yk_new_control_point"

namespace llvm {
class YkControlPointPass : public PassInfoMixin<YkControlPointPass> {
public:
  explicit YkControlPointPass();
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
} // namespace llvm

#endif
