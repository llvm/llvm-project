#ifndef LLVM_TRANSFORMS_YK_LINKAGE_H
#define LLVM_TRANSFORMS_YK_LINKAGE_H

#include "llvm/Pass.h"

namespace llvm {
ModulePass *createYkLinkagePass();
} // namespace llvm

#endif
