#ifndef LLVM_TRANSFORMS_YK_BASIC_BLOCK_TRACER_NOOP_H
#define LLVM_TRANSFORMS_YK_BASIC_BLOCK_TRACER_NOOP_H

#include "llvm/Pass.h"

namespace llvm {
ModulePass *createYkBasicBlockTracerNoopPass();
} // namespace llvm

#endif // LLVM_TRANSFORMS_YK_BASIC_BLOCK_TRACER_NOOP_H
