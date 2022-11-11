; RUN: llc < %s -march=nvptx -mcpu=sm_50 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_50 -verify-machineinstrs | %ptxas-verify %}

; calls with a bitcasted function symbol should be fine, but in combination with
; a byval attribute were causing a segfault during isel. This testcase was
; reduced from a SYCL kernel using aggregate types which ended up being passed
; `byval`

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%"class.complex" = type { %"class.sycl::_V1::detail::half_impl::half", %"class.sycl::_V1::detail::half_impl::half" }
%"class.sycl::_V1::detail::half_impl::half" = type { half }
%complex_half = type { half, half }

define weak_odr void @foo() {
entry:
  %call.i.i.i = tail call %"class.complex" bitcast (%complex_half ()* @_Z20__spirv_GroupCMulKHRjjN5__spv12complex_halfE to %"class.complex" (i32, i32, %"class.complex"*)*)(i32 0, i32 0, %"class.complex"* byval(%"class.complex") null)
  ret void
}

declare %complex_half @_Z20__spirv_GroupCMulKHRjjN5__spv12complex_halfE()

; CHECK: .param .align 4 .b8 param2[4];
; CHECK: st.param.v2.b16         [param2+0], {%h2, %h1};
; CHECK: .param .align 2 .b8 retval0[4];
; CHECK: call.uni (retval0),
; CHECK-NEXT: _Z20__spirv_GroupCMulKHRjjN5__spv12complex_halfE,
