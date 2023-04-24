; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefixes=CHECK,NOALIGN4
; RUN: llc < %s -march=nvptx -mcpu=sm_20 -nvptx-force-min-byval-param-align | FileCheck %s --check-prefixes=CHECK,ALIGN4
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 -nvptx-force-min-byval-param-align | %ptxas-verify %}

;;; Need 4-byte alignment on ptr passed byval
define ptx_device void @t1(ptr byval(float) %x) {
; CHECK: .func t1
; CHECK: .param .align 4 .b8 t1_param_0[4]
  ret void
}


;;; Need 8-byte alignment on ptr passed byval
define ptx_device void @t2(ptr byval(double) %x) {
; CHECK: .func t2
; CHECK: .param .align 8 .b8 t2_param_0[8]
  ret void
}


;;; Need 4-byte alignment on float2* passed byval
%struct.float2 = type { float, float }
define ptx_device void @t3(ptr byval(%struct.float2) %x) {
; CHECK: .func t3
; CHECK: .param .align 4 .b8 t3_param_0[8]
  ret void
}

define ptx_device void @t4(ptr byval(i8) %x) {
; CHECK: .func t4
; NOALIGN4: .param .align 1 .b8 t4_param_0[1]
; ALIGN4: .param .align 4 .b8 t4_param_0[1]
  ret void
}

;;; Make sure we adjust alignment at the call site as well.
define ptx_device void @t5(ptr align 2 byval(i8) %x) {
; CHECK: .func t5
; NOALIGN4: .param .align 2 .b8 t5_param_0[1]
; ALIGN4: .param .align 4 .b8 t5_param_0[1]
; CHECK: {
; NOALIGN4: .param .align 1 .b8 param0[1];
; ALIGN4:   .param .align 4 .b8 param0[1];
; CHECK: call.uni
  call void @t4(ptr byval(i8) %x)
  ret void
}

;;; Make sure we adjust alignment for a function prototype
;;; in case of an inderect call.

declare ptr @getfp(i32 %n)
%struct.half2 = type { half, half }
define ptx_device void @t6() {
; CHECK: .func t6
  %fp = call ptr @getfp(i32 0)
; CHECK: prototype_2 : .callprototype ()_ (.param .align 8 .b8 _[8]);
  call void %fp(ptr byval(double) null);

  %fp2 = call ptr @getfp(i32 1)
; NOALIGN4: prototype_4 : .callprototype ()_ (.param .align 2 .b8 _[4]);
; ALIGN4: prototype_4 : .callprototype ()_ (.param .align 4 .b8 _[4]);
  call void %fp(ptr byval(%struct.half2) null);

  %fp3 = call ptr @getfp(i32 2)
; NOALIGN4: prototype_6 : .callprototype ()_ (.param .align 1 .b8 _[1]);
; ALIGN4: prototype_6 : .callprototype ()_ (.param .align 4 .b8 _[1]);
  call void %fp(ptr byval(i8) null);
  ret void
}
