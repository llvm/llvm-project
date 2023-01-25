; RUN: llc < %s | FileCheck %s

; Make sure this doesn't crash (and the output is sane).
; CHECK: // %__except.ret
; CHECK-NEXT: mov     x0, xzr

target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-pc-windows-msvc19.11.0"

define i64 @f(ptr %hwnd, i32 %message, i64 %wparam, i64 %lparam) personality ptr @__C_specific_handler {
entry:
  %call = invoke i64 @callee(ptr %hwnd, i32 %message, i64 %wparam, i64 %lparam)
          to label %__try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %__except.ret] unwind to caller

__except.ret:                                     ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr @filt]
  catchret from %1 to label %__try.cont

__try.cont:                                       ; preds = %__except.ret, %entry
  %rv.0 = phi i64 [ 0, %__except.ret ], [ %call, %entry ]
  ret i64 %rv.0
}

declare dso_local i64 @callee(ptr, i32, i64, i64)
declare i32 @filt(ptr, ptr nocapture readnone)
declare dso_local i32 @__C_specific_handler(...)
