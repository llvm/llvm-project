; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %s -o - \
; RUN:   2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Cannot lower a pointer cast without equal-width numeric types for Logical SPIR-V

@Values = external addrspace(3) global [8 x i8]

define void @byteArray() {
  store i32 42, ptr addrspace(3) getelementptr inbounds
    (i8, ptr addrspace(3) @Values, i64 4), align 4
  ret void
}
