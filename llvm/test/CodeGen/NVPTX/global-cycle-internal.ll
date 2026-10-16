; RUN: not --crash llc < %s -mtriple=nvptx64 -mcpu=sm_20 2>&1 | FileCheck %s

; A PTX .extern declaration cannot be resolved by a static definition, so an
; internal-only cycle cannot be emitted for ptxas versions that reject forward
; references in initializers.

; CHECK: LLVM ERROR: Circular dependency found in global variable set

@a = internal addrspace(1) global ptr addrspace(1) @b
@b = internal addrspace(1) global ptr addrspace(1) @a
