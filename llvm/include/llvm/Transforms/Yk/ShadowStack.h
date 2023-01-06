#ifndef LLVM_TRANSFORMS_YK_SHADOWSTACK_H
#define LLVM_TRANSFORMS_YK_SHADOWSTACK_H

#include "llvm/Pass.h"

namespace llvm {
ModulePass *createYkShadowStackPass();
} // namespace llvm

#endif
