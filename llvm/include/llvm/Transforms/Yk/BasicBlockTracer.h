#ifndef LLVM_TRANSFORMS_YK_HELLOWORLD_H
#define LLVM_TRANSFORMS_YK_HELLOWORLD_H

#include "llvm/Pass.h"

// The name of the trace function
#define YK_TRACE_FUNCTION "yk_trace_basicblock"

namespace llvm {
ModulePass *createYkBasicBlockTracerPass();
} // namespace llvm

#endif
