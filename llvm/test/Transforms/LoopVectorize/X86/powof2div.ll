; RUN: opt < %s  -loop-vectorize -mtriple=x86_64-unknown-linux-gnu -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.anon = type { [100 x i32], i32, [100 x i32] }

@Foo = common global %struct.anon zeroinitializer, align 4

; CHECK-LABEL: @foo(
; CHECK: load <4 x i32>, <4 x i32>*
; CHECK: sdiv <4 x i32>
; CHECK: store <4 x i32>

define void @foo(){
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds %struct.anon, %struct.anon* @Foo, i64 0, i32 2, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %div = sdiv i32 %0, 2
  %arrayidx2 = getelementptr inbounds %struct.anon, %struct.anon* @Foo, i64 0, i32 0, i64 %indvars.iv
  store i32 %div, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

