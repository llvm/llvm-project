; RUN: opt -licm %s -disable-output

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"


; PR8068
@g_12 = external global i8, align 1
define void @test1() nounwind ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.cond, %bb.nph
  store i8 0, i8* @g_12, align 1
  %tmp6 = load i8* @g_12, align 1
  br label %for.cond

for.cond:                                         ; preds = %for.body
  store i8 %tmp6, i8* @g_12, align 1
  br i1 false, label %for.cond.for.end10_crit_edge, label %for.body

for.cond.for.end10_crit_edge:                     ; preds = %for.cond
  br label %for.end10

for.end10:                                        ; preds = %for.cond.for.end10_crit_edge, %entry
  ret void
}

; PR8067
@g_8 = external global i32, align 4

define void @test2() noreturn nounwind ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %tmp7 = load i32* @g_8, align 4
  store i32* @g_8, i32** undef, align 16
  store i32 undef, i32* @g_8, align 4
  br label %for.body
}
