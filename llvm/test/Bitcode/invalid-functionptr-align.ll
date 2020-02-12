; Bitcode with invalid function pointer alignment.

; RUN: not llvm-dis %s.bc -o - 2>&1 | FileCheck %s

CHECK: LLVM ERROR: Alignment is neither 0 nor a power of 2
