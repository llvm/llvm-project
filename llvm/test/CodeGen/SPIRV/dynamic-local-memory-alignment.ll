; RUN: not opt -S -mtriple=spirv64-intel -passes=spirv-lower-dynamic-local-memory %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv64-intel %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not opt -S -mtriple=spirv64-unknown-unknown -passes=spirv-lower-dynamic-local-memory %s -o /dev/null 2>&1 | FileCheck %s
; RUN: opt -S -mtriple=spirv64-amd-amdhsa -passes=spirv-lower-dynamic-local-memory %s | FileCheck %s --check-prefix=UNCHANGED

; Local memory arguments are only aligned to 32 bytes, so dynamic local memory
; requiring more cannot be lowered to one.

; CHECK: LLVM ERROR: dynamic local memory 'buf' requires an alignment of 64 bytes, but at most 32 bytes are supported

; Not lowered, so not rejected, for other targets.
; UNCHANGED: @buf = external addrspace(3) global [0 x i32], align 64

@buf = external addrspace(3) global [0 x i32], align 64

define spir_kernel void @kernel(ptr addrspace(1) %out) addrspace(9) {
  %v = load i32, ptr addrspace(3) @buf
  store i32 %v, ptr addrspace(1) %out
  ret void
}
