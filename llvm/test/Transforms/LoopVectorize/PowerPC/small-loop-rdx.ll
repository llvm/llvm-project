; RUN: opt < %s -passes=loop-vectorize -S | FileCheck %s

; CHECK: vector.body:
; CHECK: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NEXT: fadd
; CHECK-NOT: fadd
; CHECK: middle.block

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-ibm-linux-gnu"

define void @test(ptr nocapture readonly %arr, i32 signext %len) #0 {
entry:
  %cmp4 = icmp sgt i32 %len, 0
  br i1 %cmp4, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %0 = add i32 %len, -1
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %redx.05 = phi double [ 0.000000e+00, %for.body.lr.ph ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %arr, i64 %indvars.iv
  %1 = load double, ptr %arrayidx, align 8
  %add = fadd fast double %1, %redx.05
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %0
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  %add.lcssa = phi double [ %add, %for.body ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %redx.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %add.lcssa, %for.end.loopexit ]
  ret void
}
