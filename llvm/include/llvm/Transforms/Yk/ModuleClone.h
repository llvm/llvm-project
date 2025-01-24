#ifndef LLVM_TRANSFORMS_YK_MODULE_CLONE_H
#define LLVM_TRANSFORMS_YK_MODULE_CLONE_H

#include "llvm/Pass.h"

#define YK_UNOPT_PREFIX "__yk_unopt_"
#define YK_UNOPT_MAIN "__yk_unopt_main"
#define YK_CLONE_MODULE_CP_COUNT 2

namespace llvm {
ModulePass *createYkModuleClonePass();
} // namespace llvm

#endif
