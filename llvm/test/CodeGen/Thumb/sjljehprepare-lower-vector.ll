; RUN: llc -mtriple=thumbv7-apple-ios < %s
; SjLjEHPrepare shouldn't crash when lowering vectors.

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios"

define ptr @foo(<4 x i32> %c) personality ptr @baz {
entry:
  invoke void @bar ()
    to label %unreachable unwind label %handler

unreachable:
  unreachable

handler:
  %tmp = landingpad { ptr, i32 }
  cleanup
  resume { ptr, i32 } undef
}

declare void @bar()
declare i32 @baz(...)

