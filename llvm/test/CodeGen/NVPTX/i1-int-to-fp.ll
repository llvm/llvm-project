; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 | FileCheck %s
; RUN: %if ptxas-sm_90 && ptxas-isa-7.8 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 | %ptxas-verify -arch=sm_90 %}

; CHECK-LABEL: foo
; CHECK: setp.ne.b16 %[[P:p[0-9]+]], %{{.*}}, 0;
; CHECK-NEXT: selp.f32 %[[R:r[0-9]+]], 0f3F800000, 0f00000000, %[[P]];
; CHECK-NEXT: st.param.b32 [func_retval0], %[[R]];
define float @foo(i1 %a) {
  %ret = uitofp i1 %a to float
  ret float %ret
}

; CHECK-LABEL: foo2
; CHECK: setp.ne.b16 %[[P:p[0-9]+]], %{{.*}}, 0;
; CHECK-NEXT: selp.f32 %[[R:r[0-9]+]], 0fBF800000, 0f00000000, %[[P]];
; CHECK-NEXT: st.param.b32 [func_retval0], %[[R]];
define float @foo2(i1 %a) {
  %ret = sitofp i1 %a to float
  ret float %ret
}

; CHECK-LABEL: foo3
; CHECK: setp.ne.b16 %[[P:p[0-9]+]], %{{.*}}, 0;
; CHECK-NEXT: selp.f64 %[[R:rd[0-9]+]], 0d3FF0000000000000, 0d0000000000000000, %[[P]];
; CHECK-NEXT: st.param.b64 [func_retval0], %[[R]];
define double @foo3(i1 %a) {
  %ret = uitofp i1 %a to double
  ret double %ret
}

; CHECK-LABEL: foo4
; CHECK: setp.ne.b16 %[[P:p[0-9]+]], %{{.*}}, 0;
; CHECK-NEXT: selp.f64 %[[R:rd[0-9]+]], 0dBFF0000000000000, 0d0000000000000000, %[[P]];
; CHECK-NEXT: st.param.b64 [func_retval0], %[[R]];
define double @foo4(i1 %a) {
  %ret = sitofp i1 %a to double
  ret double %ret
}

; CHECK-LABEL: foo5
; CHECK: setp.ne.b16 %[[P:p[0-9]+]], %{{.*}}, 0;
; CHECK-NEXT: selp.b16 %[[R:rs[0-9]+]], 0x3C00, 0x0000, %[[P]];
; CHECK-NEXT: st.param.b16 [func_retval0], %[[R]];
define half @foo5(i1 %a) {
  %ret = uitofp i1 %a to half
  ret half %ret
}

; CHECK-LABEL: foo6
; CHECK: setp.ne.b16 %[[P:p[0-9]+]], %{{.*}}, 0;
; CHECK-NEXT: selp.b16 %[[R:rs[0-9]+]], 0xBC00, 0x0000, %[[P]];
; CHECK-NEXT: st.param.b16 [func_retval0], %[[R]];
define half @foo6(i1 %a) {
  %ret = sitofp i1 %a to half
  ret half %ret
}

; CHECK-LABEL: foo7
; CHECK: setp.ne.b16 %[[P:p[0-9]+]], %{{.*}}, 0;
; CHECK-NEXT: selp.b16 %[[R:rs[0-9]+]], 0xBF80, 0x0000, %[[P]];
; CHECK-NEXT: st.param.b16 [func_retval0], %[[R]];
define bfloat @foo7(i1 %a) {
  %ret = sitofp i1 %a to bfloat
  ret bfloat %ret
}
