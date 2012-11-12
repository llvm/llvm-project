; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

;;; Need 4-byte alignment on float* passed byval
define ptx_device void @t1(float* byval %x) {
; CHECK: .func t1
; CHECK: .param .align 4 .b8 t1_param_0[4]
  ret void
}


;;; Need 8-byte alignment on double* passed byval
define ptx_device void @t2(double* byval %x) {
; CHECK: .func t2
; CHECK: .param .align 8 .b8 t2_param_0[8]
  ret void
}


;;; Need 4-byte alignment on float2* passed byval
%struct.float2 = type { float, float }
define ptx_device void @t3(%struct.float2* byval %x) {
; CHECK: .func t3
; CHECK: .param .align 4 .b8 t3_param_0[8]
  ret void
}
