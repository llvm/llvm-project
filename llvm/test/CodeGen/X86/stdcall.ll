; RUN: llc < %s | FileCheck %s
; PR5851

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:128:128-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-mingw32"

%0 = type { void (...)* }

@B = global %0 { void (...)* bitcast (void ()* @MyFunc to void (...)*) }, align 4
; CHECK: _B:
; CHECK: .long _MyFunc@0

define internal x86_stdcallcc void @MyFunc() nounwind {
entry:
  ret void
}
