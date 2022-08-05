#ifndef LLVM_TRANSFORMS_YK_BLOCKDISAMBIGUATE_H
#define LLVM_TRANSFORMS_YK_BLOCKDISAMBIGUATE_H

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
ModulePass *createYkBlockDisambiguatePass();
} // namespace llvm

#endif
