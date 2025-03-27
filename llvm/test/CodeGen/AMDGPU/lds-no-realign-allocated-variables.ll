; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

; Can't have a second variable without absolute_symbol showing it is realigned as
; there is a fatal error on mixing absolute and non-absolute symbols

; CHECK: @lds.dont_realign = internal addrspace(3) global i64 poison, align 2, !absolute_symbol !0
@lds.dont_realign = internal addrspace(3) global i64 poison, align 2, !absolute_symbol !0

; CHECK: void @use_variables
define amdgpu_kernel void @use_variables(i64 %val) {
  store i64 %val, ptr addrspace(3) @lds.dont_realign, align 2
  ret void
}

!0 = !{i32 2, i32 3}
