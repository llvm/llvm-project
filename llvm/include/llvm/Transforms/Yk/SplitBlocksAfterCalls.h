#ifndef LLVM_TRANSFORMS_YK_SPLITBLOCKSAFTERCALLS_H
#define LLVM_TRANSFORMS_YK_SPLITBLOCKSAFTERCALLS_H

#include "llvm/Pass.h"

namespace llvm {
ModulePass *createYkSplitBlocksAfterCallsPass();
} // namespace llvm

#endif
