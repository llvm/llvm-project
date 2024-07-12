; RUN: opt -mtriple=amdgcn-- -codegenprepare -S < %s | FileCheck -check-prefix=OPT %s
; RUN: llc -mtriple=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=SI-LLC %s

; OPT-LABEL: @test(
; OPT: mul nsw i32
; OPT-NEXT: sext

; SI-LLC-LABEL: {{^}}test:
; SI-LLC: s_mul_i32
; SI-LLC-NOT: mul
define amdgpu_kernel void @test(ptr addrspace(1) nocapture readonly %in, i32 %a, i8 %b) {
entry:
  %0 = mul nsw i32 %a, 3
  %1 = sext i32 %0 to i64
  %2 = getelementptr i8, ptr addrspace(1) %in, i64 %1
  store i8 %b, ptr addrspace(1) %2
  ret void
}
