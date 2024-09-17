#ifndef LLVM_TRANSFORMS_YK_BASIC_BLOCK_TRACER_H
#define LLVM_TRANSFORMS_YK_BASIC_BLOCK_TRACER_H

#include "llvm/Pass.h"

// The name of the trace function
#define YK_TRACE_FUNCTION "__yk_trace_basicblock"

namespace llvm {
ModulePass *createYkBasicBlockTracerPass();
} // namespace llvm

#endif
