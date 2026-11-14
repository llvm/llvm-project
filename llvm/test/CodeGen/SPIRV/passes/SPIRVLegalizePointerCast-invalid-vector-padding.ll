; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %s -o - \
; RUN:   2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Cannot lower an access to vector padding for Logical SPIR-V

target datalayout = "e-p:64:64-i64:64-v96:128"
@Values = external addrspace(3) global [2 x <3 x i32>]

define void @vectorPadding() {
  store i8 42, ptr addrspace(3) getelementptr inbounds
    (i8, ptr addrspace(3) @Values, i64 12), align 1
  ret void
}
