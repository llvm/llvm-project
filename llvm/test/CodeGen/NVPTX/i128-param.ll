; RUN: llc < %s -O0 -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -O0 -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK-LABEL: .visible .func callee(
; CHECK-NEXT: .param .align 16 .b8 callee_param_0[16],
; CHECK-NEXT: .param .align 16 .b8 callee_param_1[16],
define void @callee(i128, i128, ptr) {
  ; CHECK-DAG: ld.param.v2.u64 {%[[REG0:rd[0-9]+]], %[[REG1:rd[0-9]+]]}, [callee_param_0];
  ; CHECK-DAG: ld.param.v2.u64 {%[[REG2:rd[0-9]+]], %[[REG3:rd[0-9]+]]}, [callee_param_1];

  ; CHECK:      mul.lo.s64 %[[REG4:rd[0-9]+]], %[[REG0]], %[[REG3]];
	; CHECK-NEXT: mul.hi.u64 %[[REG5:rd[0-9]+]], %[[REG0]], %[[REG2]];
	; CHECK-NEXT: add.s64    %[[REG6:rd[0-9]+]], %[[REG5]], %[[REG4]];
	; CHECK-NEXT: mul.lo.s64 %[[REG7:rd[0-9]+]], %[[REG1]], %[[REG2]];
	; CHECK-NEXT: add.s64    %[[REG8:rd[0-9]+]], %[[REG6]], %[[REG7]];
	; CHECK-NEXT: mul.lo.s64 %[[REG9:rd[0-9]+]], %[[REG0]], %[[REG2]];
  %a = mul i128 %0, %1

  store i128 %a, ptr %2
  ret void
}

; CHECK-LABEL: .visible .entry caller_kernel(
; CHECK-NEXT: .param .align 16 .b8 caller_kernel_param_0[16],
; CHECK-NEXT: .param .align 16 .b8 caller_kernel_param_1[16],
define ptx_kernel void @caller_kernel(i128, i128, ptr) {
start:
  ; CHECK-DAG: ld.param.v2.u64 {%[[REG0:rd[0-9]+]], %[[REG1:rd[0-9]+]]}, [caller_kernel_param_0];
  ; CHECK-DAG: ld.param.v2.u64 {%[[REG2:rd[0-9]+]], %[[REG3:rd[0-9]+]]}, [caller_kernel_param_1];

  ; CHECK:      { // callseq [[CALLSEQ_ID:[0-9]]], 0
	; CHECK:      .param .align 16 .b8 param0[16];
	; CHECK-NEXT: st.param.v2.b64 	[param0], {%[[REG0]], %[[REG1]]}
	; CHECK:      .param .align 16 .b8 param1[16];
	; CHECK-NEXT: st.param.v2.b64 	[param1], {%[[REG2]], %[[REG3]]}
	; CHECK:      } // callseq [[CALLSEQ_ID]]
  call void @callee(i128 %0, i128 %1, ptr %2)

  ret void
}

; CHECK-LABEL: .visible .func caller_func(
; CHECK-NEXT: .param .align 16 .b8 caller_func_param_0[16],
; CHECK-NEXT: .param .align 16 .b8 caller_func_param_1[16],
define void @caller_func(i128, i128, ptr) {
start:
  ; CHECK-DAG: ld.param.v2.u64 {%[[REG0:rd[0-9]+]], %[[REG1:rd[0-9]+]]}, [caller_func_param_0]
  ; CHECK-DAG: ld.param.v2.u64 {%[[REG2:rd[0-9]+]], %[[REG3:rd[0-9]+]]}, [caller_func_param_1]

  ; CHECK: { // callseq [[CALLSEQ_ID:[0-9]]], 0
	; CHECK: .param .align 16 .b8 param0[16];
	; CHECK: st.param.v2.b64 	[param0], {%[[REG0]], %[[REG1]]}
	; CHECK: .param .align 16 .b8 param1[16];
  ; CHECK: st.param.v2.b64 	[param1], {%[[REG2]], %[[REG3]]}
	; CHECK: } // callseq [[CALLSEQ_ID]]
  call void @callee(i128 %0, i128 %1, ptr %2)

  ret void
}
