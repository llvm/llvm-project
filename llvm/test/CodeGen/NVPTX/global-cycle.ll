; RUN: not --crash llc < %s -mtriple=nvptx64 -mcpu=sm_20 2>&1 | FileCheck %s

; Self-references are allowed, but a real dependency cycle between distinct
; globals must still be rejected by the emission-order walk.

; CHECK: LLVM ERROR: Circular dependency found in global variable set

@a = addrspace(1) global ptr addrspace(1) @b
@b = addrspace(1) global ptr addrspace(1) @a