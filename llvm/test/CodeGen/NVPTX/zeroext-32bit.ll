; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_30 -verify-machineinstrs | %ptxas-verify %}

; The zeroext attribute below should be silently ignored because
; we can pass a 32-bit integer across a function call without
; needing to extend it.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-cuda"

; CHECK-LABEL: .visible .func zeroext_test
; CHECK-NOT: cvt.u32.u16
define void @zeroext_test()  {
  tail call void @call1(i32 zeroext 0)
  ret void
}

declare void @call1(i32 zeroext)

; CHECK-LABEL: .visible .func signext_test
; CHECK-NOT: cvt.s32.s16
define void @signext_test()  {
  tail call void @call2(i32 zeroext 0)
  ret void
}

declare void @call2(i32 zeroext)
