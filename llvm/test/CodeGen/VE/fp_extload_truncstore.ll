; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define double @func_fp32fp64(float* %a) {
; CHECK-LABEL: func_fp32fp64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ldu %s0, (,%s0)
; CHECK-NEXT:    cvt.d.s %s0, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %a.val = load float, float* %a, align 4
  %a.asd = fpext float %a.val to double
  ret double %a.asd
}

define void @func_fp64fp32(float* %fl.ptr, double %val) {
; CHECK-LABEL: func_fp64fp32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    cvt.s.d %s1, %s1
; CHECK-NEXT:    stu %s1, (,%s0)
; CHECK-NEXT:    or %s11, 0, %s9
  %val.asf = fptrunc double %val to float
  store float %val.asf, float* %fl.ptr
  ret void
}
