; RUN: opt < %s -passes=instcombine -disable-output

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@buffer = external global [64 x float]

declare void @use(ptr)

define void @f() {
  call void @use(ptr getelementptr (i8, ptr getelementptr (i8, ptr @buffer, i64 and (i64 sub (i64 0, i64 ptrtoint (ptr @buffer to i64)), i64 63)), i64 64))
  ret void
}
