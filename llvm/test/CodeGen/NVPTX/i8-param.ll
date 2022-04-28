; RUN: llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

; CHECK: .visible .func  (.param .b32 func_retval0) callee
define i8 @callee(i8 %a) {
; CHECK: ld.param.u8
  %ret = add i8 %a, 42
; CHECK: st.param.b32
  ret i8 %ret
}

; CHECK: .visible .func caller
define void @caller(i8* %a) {
; CHECK: ld.u8
  %val = load i8, i8* %a
  %ret = tail call i8 @callee(i8 %val)
; CHECK: ld.param.b32
  store i8 %ret, i8* %a
  ret void
}

  
