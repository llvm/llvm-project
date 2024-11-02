; RUN: llc < %s -march=nvptx64 -mcpu=sm_50 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_50 -verify-machineinstrs | %ptxas-verify %}

; calls with a bitcasted function symbol should be fine, but in combination with
; a byval attribute were causing a segfault during isel. This testcase was
; reduced from a SYCL kernel using aggregate types which ended up being passed
; `byval`

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%"class.complex" = type { %"class.sycl::_V1::detail::half_impl::half", %"class.sycl::_V1::detail::half_impl::half" }
%"class.sycl::_V1::detail::half_impl::half" = type { half }
%complex_half = type { half, half }

; CHECK: .param .align 2 .b8 param2[4];
; CHECK: st.param.b16   [param2+0], %rs1;
; CHECK: st.param.b16   [param2+2], %rs2;
; CHECK: .param .align 2 .b8 retval0[4];
; CHECK-NEXT: prototype_0 : .callprototype (.param .align 2 .b8 _[4]) _ (.param .b32 _, .param .b32 _, .param .align 2 .b8 _[4]);
; CHECK-NEXT: call (retval0),
define weak_odr void @foo() {
entry:
  %call.i.i.i = tail call %"class.complex" @_Z20__spirv_GroupCMulKHRjjN5__spv12complex_halfE(i32 0, i32 0, ptr byval(%"class.complex") null)
  ret void
}

;; Function pointers can escape, so we have to use a conservative
;; alignment for a function that has address taken.
;;
declare ptr @usefp(ptr %fp)
; CHECK: .func callee(
; CHECK-NEXT: .param .align 2 .b8 callee_param_0[4]
define internal void @callee(ptr byval(%"class.complex") %byval_arg) {
  ret void
}
define void @boom() {
  %fp = call ptr @usefp(ptr @callee)
  ; CHECK: .param .align 2 .b8 param0[4];
  ; CHECK: st.param.b16 [param0+0], %rs1;
  ; CHECK: st.param.b16 [param0+2], %rs2;
  ; CHECK: .callprototype ()_ (.param .align 2 .b8 _[4]);
  call void %fp(ptr byval(%"class.complex") null)
  ret void
}

declare %complex_half @_Z20__spirv_GroupCMulKHRjjN5__spv12complex_halfE(i32, i32, ptr byval(%"class.complex"))
