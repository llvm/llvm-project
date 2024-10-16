; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK-LABEL: foo
; CHECK: setp.eq.b16 %[[P:p[0-9]+]], %{{.*}}, 1;
; CHECK: selp.u32 %[[R:r[0-9]+]], 1, 0, %[[P]];
; CHECK: cvt.rn.f32.u32 %f{{.*}}, %[[R]]
define float @foo(i1 %a) {
  %ret = uitofp i1 %a to float
  ret float %ret
}

; CHECK-LABEL: foo2
; CHECK: setp.eq.b16 %[[P:p[0-9]+]], %{{.*}}, 1;
; CHECK: selp.s32 %[[R:r[0-9]+]], -1, 0, %[[P]];
; CHECK: cvt.rn.f32.s32 %f{{.*}}, %[[R]]
define float @foo2(i1 %a) {
  %ret = sitofp i1 %a to float
  ret float %ret
}

; CHECK-LABEL: foo3
; CHECK: setp.eq.b16 %[[P:p[0-9]+]], %{{.*}}, 1;
; CHECK: selp.u32 %[[R:r[0-9]+]], 1, 0, %[[P]];
; CHECK: cvt.rn.f64.u32 %fd{{.*}}, %[[R]]
define double @foo3(i1 %a) {
  %ret = uitofp i1 %a to double
  ret double %ret
}

; CHECK-LABEL: foo4
; CHECK: setp.eq.b16 %[[P:p[0-9]+]], %{{.*}}, 1;
; CHECK: selp.s32 %[[R:r[0-9]+]], -1, 0, %[[P]];
; CHECK: cvt.rn.f64.s32 %fd{{.*}}, %[[R]]
define double @foo4(i1 %a) {
  %ret = sitofp i1 %a to double
  ret double %ret
}

; CHECK-LABEL: foo5
; CHECK: setp.eq.b16 %[[P:p[0-9]+]], %{{.*}}, 1;
; CHECK: selp.u32 %[[R:r[0-9]+]], 1, 0, %[[P]];
; CHECK: cvt.rn.f16.u32 %{{.*}}, %[[R]]
define half @foo5(i1 %a) {
  %ret = uitofp i1 %a to half
  ret half %ret
}

; CHECK-LABEL: foo6
; CHECK: setp.eq.b16 %[[P:p[0-9]+]], %{{.*}}, 1;
; CHECK: selp.s32 %[[R:r[0-9]+]], -1, 0, %[[P]];
; CHECK: cvt.rn.f16.s32 %{{.*}}, %[[R]]
define half @foo6(i1 %a) {
  %ret = sitofp i1 %a to half
  ret half %ret
}
