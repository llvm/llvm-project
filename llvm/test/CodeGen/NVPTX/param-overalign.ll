; RUN: llc < %s -mtriple=nvptx | FileCheck %s
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -mtriple=nvptx -verify-machineinstrs | %ptxas-verify %}

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

; CHECK:         ld.param.b32 %r1, [caller_md_param_0];
; CHECK-NEXT:    ld.param.b32 %r2, [caller_md_param_1];
; CHECK-NEXT:    {
; CHECK-NEXT:    .param .align 8 .b8 param0[8];
; CHECK-NEXT:    st.param.v2.b32 [param0], {%r1, %r2};
; CHECK-NEXT:    .param .b32 retval0;
; CHECK-NEXT:    call.uni (retval0),
; CHECK-NEXT:    callee_md,
; CHECK-NEXT:    (
; CHECK-NEXT:    param0
; CHECK-NEXT:    );
; CHECK-NEXT:    ld.param.b32 %r3, [retval0];
; CHECK-NEXT:    }
; CHECK-NEXT:    st.param.b32 [func_retval0], %r3;
; CHECK-NEXT:    ret;
  %s1 = insertvalue %struct.float2 poison, float %a, 0
  %s2 = insertvalue %struct.float2 %s1, float %b, 1
  %r = call float @callee_md(%struct.float2 %s2)
  ret float %r
}

define float @callee_md(%struct.float2 alignstack(8) %a) {
; CHECK-LABEL: .visible .func  (.param .b32 func_retval0) callee_md(
; CHECK-NEXT:         .param .align 8 .b8 callee_md_param_0[8]
; CHECK-NEXT: )
; CHECK-NEXT: {

; CHECK:         ld.param.v2.b32 {%r1, %r2}, [callee_md_param_0];
; CHECK-NEXT:    add.rn.f32 %r3, %r1, %r2;
; CHECK-NEXT:    st.param.b32 [func_retval0], %r3;
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

; CHECK:         ld.param.b32 %r1, [caller_param_0];
; CHECK-NEXT:    ld.param.b32 %r2, [caller_param_1];
; CHECK-NEXT:    {
; CHECK-NEXT:    .param .align 8 .b8 param0[8];
; CHECK-NEXT:    st.param.v2.b32 [param0], {%r1, %r2};
; CHECK-NEXT:    .param .b32 retval0;
; CHECK-NEXT:    call.uni (retval0),
; CHECK-NEXT:    callee,
; CHECK-NEXT:    (
; CHECK-NEXT:    param0
; CHECK-NEXT:    );
; CHECK-NEXT:    ld.param.b32 %r3, [retval0];
; CHECK-NEXT:    }
; CHECK-NEXT:    st.param.b32 [func_retval0], %r3;
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

; CHECK:         ld.param.v2.b32 {%r1, %r2}, [callee_param_0];
; CHECK-NEXT:    add.rn.f32 %r3, %r1, %r2;
; CHECK-NEXT:    st.param.b32 [func_retval0], %r3;
; CHECK-NEXT:    ret;
  %v0 = extractvalue %struct.float2 %a, 0
  %v1 = extractvalue %struct.float2 %a, 1
  %2 = fadd float %v0, %v1
  ret float %2
}

define alignstack(8) %struct.float2 @aligned_return(%struct.float2 %a ) {
; CHECK-LABEL: .visible .func  (.param .align 8 .b8 func_retval0[8]) aligned_return(
; CHECK-NEXT:         .param .align 4 .b8 aligned_return_param_0[8]
; CHECK-NEXT: )
; CHECK-NEXT: {
  ret %struct.float2 %a
}
