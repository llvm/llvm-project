#ifndef LLVM_TRANSFORMS_YK_NOCALLSINENTRYBLOCKS_H
#define LLVM_TRANSFORMS_YK_NOCALLSINENTRYBLOCKS_H

#include "llvm/Pass.h"

namespace llvm {
ModulePass *createYkNoCallsInEntryBlocksPass();
} // namespace llvm

#endif
