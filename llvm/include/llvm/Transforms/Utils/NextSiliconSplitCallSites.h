//===- NextSiliconSplitCallSites.h - Split blocks with multiple calls -----===//
//
// This file provides the prototypes and definitions related to the call site
// splitting pass.
//
// The call site splitting pass ensures that each call site is in a separate
// basic block. If there are multiple call sites in the same block, the block is
// split accordingly. The pass is aware of nextsilicon metadata and propagates
// them accordingly.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_NEXTSILICONSPLITCALLSITES_H
#define LLVM_TRANSFORMS_UTILS_NEXTSILICONSPLITCALLSITES_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;
class ModulePass;

class NextSiliconSplitCallSitesPass
    : public PassInfoMixin<NextSiliconSplitCallSitesPass> {
public:
  NextSiliconSplitCallSitesPass() {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_NEXTSILICONSPLITCALLSITES_H
