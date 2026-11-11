; RUN: not opt -S \
; RUN:   -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' \
; RUN:   -mtriple=spirv-unknown-vulkan-compute %s -o - \
; RUN:   2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Cannot lower an access to vector padding for Logical SPIR-V (in function 'multipleErrors')
; CHECK-NEXT: Cannot lower a byte offset within a scalar or padding for Logical SPIR-V (in function 'multipleErrors')
; CHECK-NEXT: Cannot lower byte addressing with an unsized type for Logical SPIR-V (in function 'unsizedAfterErrors')
; CHECK-NOT: Cannot lower

target datalayout = "e-p:64:64-i64:64-v96:128"
@Vectors = external addrspace(3) global [2 x <3 x i32>]
@Scalars = external addrspace(3) global [4 x i32]
%Opaque = type opaque
@Unsized = external addrspace(3) global %Opaque

define void @multipleErrors() {
  %padding = getelementptr inbounds i8, ptr addrspace(3) @Vectors, i64 12
  %dependent = getelementptr i8, ptr addrspace(3) %padding, i64 1
  store i8 42, ptr addrspace(3) %dependent, align 1
  store i8 43, ptr addrspace(3) getelementptr inbounds
    (i8, ptr addrspace(3) @Scalars, i64 1), align 1
  ret void
}

define void @unsizedAfterErrors() {
  store i32 42, ptr addrspace(3) getelementptr
    (i8, ptr addrspace(3) @Unsized, i64 4), align 4
  ret void
}
