; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %s -o - \
; RUN:   2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Cannot lower a byte offset within a scalar or padding for Logical SPIR-V

@Value = external addrspace(3) global { i32, i8 }

define void @structPadding() {
  store i8 42, ptr addrspace(3) getelementptr inbounds
    (i8, ptr addrspace(3) @Value, i64 6), align 1
  ret void
}
