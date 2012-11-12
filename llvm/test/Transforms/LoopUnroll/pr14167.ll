; RUN: opt < %s -S -loop-unroll -unroll-runtime | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-bgq-linux"

define void @test1() nounwind {
; Ensure that we don't crash when the trip count == -1.
; CHECK: @test1
entry:
  br label %for.cond2.preheader

for.cond2.preheader:                              ; preds = %for.end, %entry
  br i1 false, label %middle.block, label %vector.ph

vector.ph:                                        ; preds = %for.cond2.preheader
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  br i1 undef, label %middle.block.loopexit, label %vector.body

middle.block.loopexit:                            ; preds = %vector.body
  br label %middle.block

middle.block:                                     ; preds = %middle.block.loopexit, %for.cond2.preheader
  br i1 true, label %for.end, label %scalar.preheader

scalar.preheader:                                 ; preds = %middle.block
  br label %for.body4

for.body4:                                        ; preds = %for.body4, %scalar.preheader
  %indvars.iv = phi i64 [ 16000, %scalar.preheader ], [ %indvars.iv.next, %for.body4 ]
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 16000
  br i1 %exitcond, label %for.body4, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.body4
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %middle.block
  br i1 undef, label %for.cond2.preheader, label %for.end15

for.end15:                                        ; preds = %for.end
  ret void
}
