; RUN: opt -S -passes=openmp-opt < %s | FileCheck %s
; Make sure nvvm.annotations isn't introduced into the module

; CHECK-NOT: nvvm

define amdgpu_kernel void @foo() "kernel" {
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 7, !"openmp", i32 50}
