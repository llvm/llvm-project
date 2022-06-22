; RUN: llc < %s -mattr=-sse,-sse2 | FileCheck %s
; PR3402
target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

%struct.ktermios = type { i32, i32, i32, i32, i8, [19 x i8], i32, i32 }

; CHECK-NOT: xmm
; CHECK-NOT: ymm
define void @foo() nounwind {
entry:
  %termios = alloca %struct.ktermios, align 8
  call void @llvm.memset.p0.i64(ptr align 8 %termios, i8 0, i64 44, i1 false)
  call void @bar(ptr %termios) nounwind
  ret void
}

declare void @bar(ptr)

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) nounwind
