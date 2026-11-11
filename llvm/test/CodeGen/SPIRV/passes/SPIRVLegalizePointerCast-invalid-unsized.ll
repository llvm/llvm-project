; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %s -o - \
; RUN:   2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Cannot lower byte addressing with an unsized type for Logical SPIR-V

%Opaque = type opaque
@Value = external addrspace(3) global %Opaque

define void @unsized() {
  store i32 42, ptr addrspace(3) getelementptr
    (i8, ptr addrspace(3) @Value, i64 4), align 4
  ret void
}
