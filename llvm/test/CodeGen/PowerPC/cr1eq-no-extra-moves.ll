; RUN: llc < %s | FileCheck %s
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32"
target triple = "powerpc-unknown-linux"

@.str = private unnamed_addr constant [3 x i8] c"%i\00", align 1

define void @test(i32 %count) nounwind {
entry:
; CHECK: crxor 6, 6, 6
  %call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([3 x i8]* @.str, i32 0, i32 0), i32 1) nounwind
  %cmp2 = icmp sgt i32 %count, 0
  br i1 %cmp2, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.03 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
; CHECK: crxor 6, 6, 6
  %call1 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([3 x i8]* @.str, i32 0, i32 0), i32 1) nounwind
  %inc = add nsw i32 %i.03, 1
  %exitcond = icmp eq i32 %inc, %count
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

declare i32 @printf(i8* nocapture, ...) nounwind
