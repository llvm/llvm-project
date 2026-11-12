; RUN: opt < %s -passes=globalopt -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g_data = internal unnamed_addr global <{ [8 x i16], [8 x i16] }> <{ [8 x i16] [i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16], [8 x i16] zeroinitializer }>, align 16
; We cannot SRA here due to the second gep meaning the access to g_data may be to either element
; CHECK: @g_data = internal unnamed_addr constant <{ [8 x i16], [8 x i16] }>

define i16 @test(i64 %a1) {
entry:
  %g1 = getelementptr inbounds <{ [8 x i16], [8 x i16] }>, ptr @g_data, i64 0, i32 0
  %arrayidx.i = getelementptr inbounds [8 x i16], ptr %g1, i64 0, i64 %a1
  %r = load i16, ptr %arrayidx.i, align 2
  ret i16 %r
}
