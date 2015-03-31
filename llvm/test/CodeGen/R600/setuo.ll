; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}main:
; CHECK: v_cmp_u_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], [[SREG:s[0-9]+]], [[SREG]]
; CHECK-NEXT: v_cndmask_b32_e64 {{v[0-9]+}}, 0, 1.0, [[CMP]]
define void @main(float %p) {
main_body:
  %c = fcmp une float %p, %p
  %r = select i1 %c, float 1.000000e+00, float 0.000000e+00
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 0, float %r, float %r, float %r, float %r)
  ret void
}

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
