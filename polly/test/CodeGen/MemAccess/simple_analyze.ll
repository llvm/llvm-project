;RUN: opt %loadPolly -polly-import-jscop -analyze -polly-import-jscop-dir=%S -polly-import-jscop-postfix=transformed < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i386-pc-linux-gnu"

@A = common global [100 x i32] zeroinitializer, align 4
@B = common global [100 x i32] zeroinitializer, align 4

define i32 @simple() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %arrayidx = getelementptr [100 x i32]* @A, i32 0, i32 %0
  %exitcond1 = icmp ne i32 %0, 12
  br i1 %exitcond1, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i32 %0, i32* %arrayidx
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc11, %for.end
  %1 = phi i32 [ 0, %for.end ], [ %inc13, %for.inc11 ]
  %arrayidx10 = getelementptr [100 x i32]* @B, i32 0, i32 %1
  %exitcond = icmp ne i32 %1, 12
  br i1 %exitcond, label %for.body7, label %for.end14

for.body7:                                        ; preds = %for.cond4
  store i32 %1, i32* %arrayidx10
  br label %for.inc11

for.inc11:                                        ; preds = %for.body7
  %inc13 = add nsw i32 %1, 1
  br label %for.cond4

for.end14:                                        ; preds = %for.cond4
  ret i32 0
}
; CHECK: New access function '{ Stmt_for_body7[i0] -> MemRef_B[0] }'detected in JSCOP file
; CHECK: New access function '{ Stmt_for_body[i0] -> MemRef_A[0] }'detected in JSCOP file
