// RUN: mlir-opt %s -verify-diagnostics

// The call stack is the one untyped shape: the tag and the context sizes are
// parameters the parser already type-checked.

// expected-error @below {{call stack of a memprof MemInfoBlock must not be empty}}
#mib = #llvm.memprof_mib<call_stack = [], alloc_type = "cold">
