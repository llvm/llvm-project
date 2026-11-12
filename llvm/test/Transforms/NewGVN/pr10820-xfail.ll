; XFAIL: *
; RUN: opt < %s -passes=newgvn -S | FileCheck %s
; NewGVN fails this due to missing load coercion
target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-f128:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@g = external global i31

define void @main() nounwind uwtable {
entry:
; CHECK: store i32
  store i32 402662078, ptr @g, align 8
; CHECK-NOT: load i31
  %0 = load i31, ptr @g, align 8
; CHECK: store i31
  store i31 %0, ptr undef, align 1
  unreachable
}
