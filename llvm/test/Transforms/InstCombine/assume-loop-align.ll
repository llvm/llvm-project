; RUN: opt -domtree -instcombine -loops -S < %s | FileCheck %s
; Note: The -loops above can be anything that requires the domtree, and is
; necessary to work around a pass-manager bug.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @foo(i32* %a, i32* %b) #0 {
entry:
  %ptrint = ptrtoint i32* %a to i64
  %maskedptr = and i64 %ptrint, 63
  %maskcond = icmp eq i64 %maskedptr, 0
  tail call void @llvm.assume(i1 %maskcond)
  %ptrint1 = ptrtoint i32* %b to i64
  %maskedptr2 = and i64 %ptrint1, 63
  %maskcond3 = icmp eq i64 %maskedptr2, 0
  tail call void @llvm.assume(i1 %maskcond3)
  br label %for.body

; CHECK-LABEL: @foo
; CHECK: load i32, i32* {{.*}} align 64
; CHECK: store i32 {{.*}}  align 64
; CHECK: ret

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, 1
  %arrayidx5 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  store i32 %add, i32* %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 16
  %1 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %1, 1648
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

; Function Attrs: nounwind
declare void @llvm.assume(i1) #1

attributes #0 = { nounwind uwtable }
attributes #1 = { nounwind }

