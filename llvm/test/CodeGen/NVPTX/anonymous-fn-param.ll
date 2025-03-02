; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; Check that parameter names we generate in the function signature and the name
; we use when we refer to the parameter in the function body do match.

; CHECK:      .func (.param .b32 func_retval0) __unnamed_1(
; CHECK-NEXT: .param .b32 __unnamed_1_param_0
; CHECK:      ld.param.u32 {{%r[0-9]+}}, [__unnamed_1_param_0];

define internal i32 @0(i32 %a) {
entry:
  %r = add i32 %a, 1
  ret i32 %r
}

; CHECK:      .func (.param .b32 func_retval0) __unnamed_2(
; CHECK-NEXT: .param .b32 __unnamed_2_param_0
; CHECK:      ld.param.u32 {{%r[0-9]+}}, [__unnamed_2_param_0];

define internal i32 @1(i32 %a) {
entry:
  %r = add i32 %a, 1
  ret i32 %r
}
