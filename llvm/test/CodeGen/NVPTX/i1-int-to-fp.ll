; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}

; CHECK-LABEL: foo
; CHECK: setp
; CHECK: selp
; CHECK: cvt.rn.f32.u32
define float @foo(i1 %a) {
  %ret = uitofp i1 %a to float
  ret float %ret
}

; CHECK-LABEL: foo2
; CHECK: setp
; CHECK: selp
; CHECK: cvt.rn.f32.s32
define float @foo2(i1 %a) {
  %ret = sitofp i1 %a to float
  ret float %ret
}

; CHECK-LABEL: foo3
; CHECK: setp
; CHECK: selp
; CHECK: cvt.rn.f64.u32
define double @foo3(i1 %a) {
  %ret = uitofp i1 %a to double
  ret double %ret
}

; CHECK-LABEL: foo4
; CHECK: setp
; CHECK: selp
; CHECK: cvt.rn.f64.s32
define double @foo4(i1 %a) {
  %ret = sitofp i1 %a to double
  ret double %ret
}
