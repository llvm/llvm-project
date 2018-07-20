; RUN: opt -S -codegenprepare -disable-complex-addr-modes=false  %s | FileCheck %s --check-prefix=CHECK
target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @test() {
entry:
  %0 = getelementptr inbounds i64, i64 * null, i64 undef
  br label %start

start:
  %val1 = phi i64 * [ %0, %entry ], [ %val4, %exit ]
  %val2 = phi i64 * [ null, %entry ], [ %val5, %exit ]
  br i1 false, label %slowpath, label %exit

slowpath:
  %elem1 = getelementptr inbounds i64, i64 * undef, i64 undef
  br label %exit

exit:
; CHECK: sunkaddr
  %val3 = phi i64 * [ undef, %slowpath ], [ %val2, %start ]
  %val4 = phi i64 * [ %elem1, %slowpath ], [ %val1, %start ]
  %val5 = phi i64 * [ undef, %slowpath ], [ %val2, %start ]
  %loadx = load i64, i64 * %val4, align 8
  br label %start
}
