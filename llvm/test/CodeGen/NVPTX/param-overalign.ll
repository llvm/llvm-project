; RUN: llc < %s -march=nvptx | FileCheck %s
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -march=nvptx -verify-machineinstrs | %ptxas-verify %}

target triple = "nvptx64-nvidia-cuda"

%struct.float2 = type { float, float }

; CHECK-LABEL: .visible .func  (.param .b32 func_retval0) callee_md
; CHECK-NEXT: (
; CHECK-NEXT:         .param .align 8 .b8 callee_md_param_0[8]
; CHECK-NEXT: )
; CHECK-NEXT: ;

; CHECK-LABEL: .visible .func  (.param .b32 func_retval0) callee
; CHECK-NEXT: (
; CHECK-NEXT:         .param .align 8 .b8 callee_param_0[8]
; CHECK-NEXT: )
; CHECK-NEXT: ;

define float @caller_md(float %a, float %b) {
; CHECK-LABEL: .visible .func  (.param .b32 func_retval0) caller_md(
; CHECK-NEXT:         .param .b32 caller_md_param_0,
; CHECK-NEXT:         .param .b32 caller_md_param_1
; CHECK-NEXT: )
; CHECK-NEXT: {

; CHECK:         ld.param.f32 %f1, [caller_md_param_0];
; CHECK-NEXT:    ld.param.f32 %f2, [caller_md_param_1];
; CHECK-NEXT:    {
; CHECK-NEXT:    .param .align 8 .b8 param0[8];
; CHECK-NEXT:    st.param.v2.f32 [param0+0], {%f1, %f2};
; CHECK-NEXT:    .param .b32 retval0;
; CHECK-NEXT:    call.uni (retval0),
; CHECK-NEXT:    callee_md,
; CHECK-NEXT:    (
; CHECK-NEXT:    param0
; CHECK-NEXT:    );
; CHECK-NEXT:    ld.param.f32 %f3, [retval0+0];
; CHECK-NEXT:    }
; CHECK-NEXT:    st.param.f32 [func_retval0+0], %f3;
; CHECK-NEXT:    ret;
  %s1 = insertvalue %struct.float2 poison, float %a, 0
  %s2 = insertvalue %struct.float2 %s1, float %b, 1
  %r = call float @callee_md(%struct.float2 %s2)
  ret float %r
}

define float @callee_md(%struct.float2 %a) {
; CHECK-LABEL: .visible .func  (.param .b32 func_retval0) callee_md(
; CHECK-NEXT:         .param .align 8 .b8 callee_md_param_0[8]
; CHECK-NEXT: )
; CHECK-NEXT: {

; CHECK:         ld.param.v2.f32 {%f1, %f2}, [callee_md_param_0];
; CHECK-NEXT:    add.rn.f32 %f3, %f1, %f2;
; CHECK-NEXT:    st.param.f32 [func_retval0+0], %f3;
; CHECK-NEXT:    ret;
  %v0 = extractvalue %struct.float2 %a, 0
  %v1 = extractvalue %struct.float2 %a, 1
  %2 = fadd float %v0, %v1
  ret float %2
}

define float @caller(float %a, float %b) {
; CHECK-LABEL: .visible .func  (.param .b32 func_retval0) caller(
; CHECK-NEXT:         .param .b32 caller_param_0,
; CHECK-NEXT:         .param .b32 caller_param_1
; CHECK-NEXT: )
; CHECK-NEXT: {

; CHECK:         ld.param.f32 %f1, [caller_param_0];
; CHECK-NEXT:    ld.param.f32 %f2, [caller_param_1];
; CHECK-NEXT:    {
; CHECK-NEXT:    .param .align 8 .b8 param0[8];
; CHECK-NEXT:    st.param.v2.f32 [param0+0], {%f1, %f2};
; CHECK-NEXT:    .param .b32 retval0;
; CHECK-NEXT:    call.uni (retval0),
; CHECK-NEXT:    callee,
; CHECK-NEXT:    (
; CHECK-NEXT:    param0
; CHECK-NEXT:    );
; CHECK-NEXT:    ld.param.f32 %f3, [retval0+0];
; CHECK-NEXT:    }
; CHECK-NEXT:    st.param.f32 [func_retval0+0], %f3;
; CHECK-NEXT:    ret;
  %s1 = insertvalue %struct.float2 poison, float %a, 0
  %s2 = insertvalue %struct.float2 %s1, float %b, 1
  %r = call float @callee(%struct.float2 %s2)
  ret float %r
}

define float @callee(%struct.float2 alignstack(8) %a ) {
; CHECK-LABEL: .visible .func  (.param .b32 func_retval0) callee(
; CHECK-NEXT:         .param .align 8 .b8 callee_param_0[8]
; CHECK-NEXT: )
; CHECK-NEXT: {

; CHECK:         ld.param.v2.f32 {%f1, %f2}, [callee_param_0];
; CHECK-NEXT:    add.rn.f32 %f3, %f1, %f2;
; CHECK-NEXT:    st.param.f32 [func_retval0+0], %f3;
; CHECK-NEXT:    ret;
  %v0 = extractvalue %struct.float2 %a, 0
  %v1 = extractvalue %struct.float2 %a, 1
  %2 = fadd float %v0, %v1
  ret float %2
}

!nvvm.annotations = !{!0}
!0 = !{ptr @callee_md, !"align", i32 u0x00010008}
