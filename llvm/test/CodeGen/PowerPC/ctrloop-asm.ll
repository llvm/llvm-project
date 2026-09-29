; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-freebsd10.0"

define void @test1(i32 %c) nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  call void asm sideeffect "", "~{r5}"() nounwind
  %inc = add nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %inc, 2048
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
; CHECK: @test1
; CHECK: mtctr
}

define void @test2(i32 %c) nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  call void asm sideeffect "", "~{ctr}"() nounwind
  %inc = add nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %inc, 2048
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
; CHECK: @test2
; CHECK-NOT: mtctr
}

