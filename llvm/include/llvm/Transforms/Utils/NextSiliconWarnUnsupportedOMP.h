//===- NextSiliconWarnUnsupportedOMP.h ------------------------------------===//
//
// Emit a warning if a call to an OMP function not supported on NextSilicon
// device is encountered.
//
// Don't immediately issue an error, as the function call may not be
// scheduled to device.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_NEXTSILICONWARNUNSUPPORTEDOMP_H
#define LLVM_TRANSFORMS_UTILS_NEXTSILICONWARNUNSUPPORTEDOMP_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;
class ModulePass;

class NextSiliconWarnUnsupportedOMPPass
    : public PassInfoMixin<NextSiliconWarnUnsupportedOMPPass> {
public:
  NextSiliconWarnUnsupportedOMPPass() {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_NEXTSILICONWARNUNSUPPORTEDOMP_H
