; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}

;;; Need 4-byte alignment on float* passed byval
define ptx_device void @t1(float* byval(float) %x) {
; CHECK: .func t1
; CHECK: .param .align 4 .b8 t1_param_0[4]
  ret void
}


;;; Need 8-byte alignment on double* passed byval
define ptx_device void @t2(double* byval(double) %x) {
; CHECK: .func t2
; CHECK: .param .align 8 .b8 t2_param_0[8]
  ret void
}


;;; Need 4-byte alignment on float2* passed byval
%struct.float2 = type { float, float }
define ptx_device void @t3(%struct.float2* byval(%struct.float2) %x) {
; CHECK: .func t3
; CHECK: .param .align 4 .b8 t3_param_0[8]
  ret void
}

;;; Need at least 4-byte alignment in order to avoid miscompilation by
;;; ptxas for sm_50+
define ptx_device void @t4(i8* byval(i8) %x) {
; CHECK: .func t4
; CHECK: .param .align 4 .b8 t4_param_0[1]
  ret void
}

;;; Make sure we adjust alignment at the call site as well.
define ptx_device void @t5(i8* align 2 byval(i8) %x) {
; CHECK: .func t5
; CHECK: .param .align 4 .b8 t5_param_0[1]
; CHECK: {
; CHECK: .param .align 4 .b8 param0[1];
; CHECK: call.uni
  call void @t4(i8* byval(i8) %x)
  ret void
}
