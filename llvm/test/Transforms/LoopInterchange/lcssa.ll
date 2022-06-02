; RUN: opt < %s -basic-aa -loop-interchange -pass-remarks-missed='loop-interchange' -verify-loop-lcssa -pass-remarks-output=%t -S
; RUN: FileCheck --input-file %t --check-prefix REMARK %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "powerpc64le-unknown-linux-gnu"

@A = common global [100 x [100 x i32]] zeroinitializer
@C = common global [100 x [100 x i32]] zeroinitializer
@X = common global i32 0
@Y = common global i64 0
@F = common global float 0.0

; We cannot interchange this loop at the moment, because iv.outer.next is
; produced in the outer loop latch and used in the loop exit block. If the inner
; loop body is not executed, the outer loop latch won't be executed either
; after interchanging.
; REMARK: UnsupportedExitPHI
; REMARK-NEXT: lcssa_01

define void @lcssa_01() {
entry:
  %cmp21 = icmp sgt i64 100, 1
  br i1 %cmp21, label %outer.ph, label %for.end16

outer.ph:                                         ; preds = %entry
  %cmp218 = icmp sgt i64 100, 1
  br label %outer.header

outer.header:                                     ; preds = %outer.inc, %outer.ph
  %iv.outer = phi i64 [ 1, %outer.ph ], [ %iv.outer.next, %outer.inc ]
  br i1 %cmp218, label %for.body3, label %outer.inc

for.body3:                                        ; preds = %for.body3, %outer.header
  %iv.inner = phi i64 [ %iv.inner.next, %for.body3 ], [ 1, %outer.header ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %iv.inner, i64 %iv.outer
  %vA = load i32, i32* %arrayidx5
  %arrayidx9 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @C, i64 0, i64 %iv.inner, i64 %iv.outer
  %vC = load i32, i32* %arrayidx9
  %add = add nsw i32 %vA, %vC
  store i32 %add, i32* %arrayidx5
  %iv.inner.next = add nuw nsw i64 %iv.inner, 1
  %exitcond = icmp eq i64 %iv.inner.next, 100
  br i1 %exitcond, label %outer.inc, label %for.body3

outer.inc:                                        ; preds = %for.body3, %outer.header
  %iv.outer.next = add nsw i64 %iv.outer, 1
  %cmp = icmp eq i64 %iv.outer.next, 100
  br i1 %cmp, label %outer.header, label %for.exit

for.exit:                                         ; preds = %outer.inc
  %iv.outer.next.lcssa = phi i64 [ %iv.outer.next, %outer.inc ]
  store i64 %iv.outer.next.lcssa, i64* @Y
  br label %for.end16

for.end16:                                        ; preds = %for.exit, %entry
  ret void
}

; REMARK: UnsupportedExitPHI
; REMARK-NEXT: lcssa_02
define void @lcssa_02() {
entry:
  %cmp21 = icmp sgt i64 100, 1
  br i1 %cmp21, label %outer.ph, label %for.end16

outer.ph:                                         ; preds = %entry
  %cmp218 = icmp sgt i64 100, 1
  br label %outer.header

outer.header:                                     ; preds = %outer.inc, %outer.ph
  %iv.outer = phi i64 [ 1, %outer.ph ], [ %iv.outer.next, %outer.inc ]
  br i1 %cmp218, label %for.body3, label %outer.inc

for.body3:                                        ; preds = %for.body3, %outer.header
  %iv.inner = phi i64 [ %iv.inner.next, %for.body3 ], [ 1, %outer.header ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %iv.inner, i64 %iv.outer
  %vA = load i32, i32* %arrayidx5
  %arrayidx9 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @C, i64 0, i64 %iv.inner, i64 %iv.outer
  %vC = load i32, i32* %arrayidx9
  %add = add nsw i32 %vA, %vC
  store i32 %add, i32* %arrayidx5
  %iv.inner.next = add nuw nsw i64 %iv.inner, 1
  %exitcond = icmp eq i64 %iv.inner.next, 100
  br i1 %exitcond, label %outer.inc, label %for.body3

outer.inc:                                        ; preds = %for.body3, %outer.header
  %iv.inner.end = phi i64 [ 0, %outer.header ], [ %iv.inner.next, %for.body3 ]
  %iv.outer.next = add nsw i64 %iv.outer, 1
  %cmp = icmp eq i64 %iv.outer.next, 100
  br i1 %cmp, label %outer.header, label %for.exit

for.exit:                                         ; preds = %outer.inc
  %iv.inner.end.lcssa = phi i64 [ %iv.inner.end, %outer.inc ]
  store i64 %iv.inner.end.lcssa, i64* @Y
  br label %for.end16

for.end16:                                        ; preds = %for.exit, %entry
  ret void
}

; REMARK: Interchanged
; REMARK-NEXT: lcssa_03
define void @lcssa_03() {
entry:
  br label %outer.header

outer.header:                                     ; preds = %outer.inc, %entry
  %iv.outer = phi i64 [ 1, %entry ], [ %iv.outer.next, %outer.inc ]
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %outer.header
  %iv.inner = phi i64 [ %iv.inner.next, %for.body3 ], [ 1, %outer.header ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %iv.inner, i64 %iv.outer
  %vA = load i32, i32* %arrayidx5
  %arrayidx9 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @C, i64 0, i64 %iv.inner, i64 %iv.outer
  %vC = load i32, i32* %arrayidx9
  %add = add nsw i32 %vA, %vC
  store i32 %add, i32* %arrayidx5
  %iv.inner.next = add nuw nsw i64 %iv.inner, 1
  %exitcond = icmp eq i64 %iv.inner.next, 100
  br i1 %exitcond, label %outer.inc, label %for.body3

outer.inc:                                        ; preds = %for.body3
  %iv.inner.lcssa = phi i64 [ %iv.inner, %for.body3 ]
  %iv.outer.next = add nsw i64 %iv.outer, 1
  %cmp = icmp eq i64 %iv.outer.next, 100
  br i1 %cmp, label %outer.header, label %for.exit

for.exit:                                         ; preds = %outer.inc
  %iv.inner.lcssa.lcssa = phi i64 [ %iv.inner.lcssa, %outer.inc ]
  store i64 %iv.inner.lcssa.lcssa, i64* @Y
  br label %for.end16

for.end16:                                        ; preds = %for.exit
  ret void
}

; Loops with floating point reductions are interchanged with fastmath.
; REMARK: Interchanged
; REMARK-NEXT: lcssa_04

define void @lcssa_04() {
entry:
  br label %outer.header

outer.header:                                     ; preds = %outer.inc, %entry
  %iv.outer = phi i64 [ 1, %entry ], [ %iv.outer.next, %outer.inc ]
  %float.outer = phi float [ 1.000000e+00, %entry ], [ %float.outer.next, %outer.inc ]
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %outer.header
  %iv.inner = phi i64 [ %iv.inner.next, %for.body3 ], [ 1, %outer.header ]
  %float.inner = phi float [ %float.inner.next, %for.body3 ], [ %float.outer, %outer.header ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %iv.inner, i64 %iv.outer
  %vA = load i32, i32* %arrayidx5
  %arrayidx9 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @C, i64 0, i64 %iv.inner, i64 %iv.outer
  %vC = load i32, i32* %arrayidx9
  %add = add nsw i32 %vA, %vC
  %float.inner.next = fadd fast float %float.inner, 1.000000e+00
  store i32 %add, i32* %arrayidx5
  %iv.inner.next = add nuw nsw i64 %iv.inner, 1
  %exitcond = icmp eq i64 %iv.inner.next, 100
  br i1 %exitcond, label %outer.inc, label %for.body3

outer.inc:                                        ; preds = %for.body3
  %float.outer.next = phi float [ %float.inner.next, %for.body3 ]
  %iv.outer.next = add nsw i64 %iv.outer, 1
  %cmp = icmp eq i64 %iv.outer.next, 100
  br i1 %cmp, label %outer.header, label %for.exit

for.exit:                                         ; preds = %outer.inc
  %float.outer.lcssa = phi float [ %float.outer.next, %outer.inc ]
  store float %float.outer.lcssa, float* @F
  br label %for.end16

for.end16:                                        ; preds = %for.exit
  ret void
}

; PHI node in inner latch with multiple predecessors.
; REMARK: Interchanged
; REMARK-NEXT: lcssa_05

define void @lcssa_05(i32* %ptr) {
entry:
  br label %outer.header

outer.header:                                     ; preds = %outer.inc, %entry
  %iv.outer = phi i64 [ 1, %entry ], [ %iv.outer.next, %outer.inc ]
  br label %for.body3

for.body3:                                        ; preds = %bb3, %outer.header
  %iv.inner = phi i64 [ %iv.inner.next, %bb3 ], [ 1, %outer.header ]
  br i1 undef, label %bb2, label %bb3

bb2:                                              ; preds = %for.body3
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %iv.inner, i64 %iv.outer
  %vA = load i32, i32* %arrayidx5
  %arrayidx9 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @C, i64 0, i64 %iv.inner, i64 %iv.outer
  %vC = load i32, i32* %arrayidx9
  %add = add nsw i32 %vA, %vC
  br label %bb3

bb3:                                              ; preds = %bb2, %for.body3
  %addp = phi i32 [ %add, %bb2 ], [ 0, %for.body3 ]
  store i32 %addp, i32* %ptr
  %iv.inner.next = add nuw nsw i64 %iv.inner, 1
  %exitcond = icmp eq i64 %iv.inner.next, 100
  br i1 %exitcond, label %outer.inc, label %for.body3

outer.inc:                                        ; preds = %bb3
  %iv.inner.lcssa = phi i64 [ %iv.inner, %bb3 ]
  %iv.outer.next = add nsw i64 %iv.outer, 1
  %cmp = icmp eq i64 %iv.outer.next, 100
  br i1 %cmp, label %outer.header, label %for.exit

for.exit:                                         ; preds = %outer.inc
  %iv.inner.lcssa.lcssa = phi i64 [ %iv.inner.lcssa, %outer.inc ]
  store i64 %iv.inner.lcssa.lcssa, i64* @Y
  br label %for.end16

for.end16:                                        ; preds = %for.exit
  ret void
}

; REMARK: UnsupportedExitPHI
; REMARK-NEXT: lcssa_06

define void @lcssa_06(i64* %ptr, i32* %ptr1) {
entry:
  br label %outer.header

outer.header:                                     ; preds = %outer.inc, %entry
  %iv.outer = phi i64 [ 1, %entry ], [ %iv.outer.next, %outer.inc ]
  br i1 undef, label %for.body3, label %outer.inc

for.body3:                                        ; preds = %for.body3, %outer.header
  %iv.inner = phi i64 [ %iv.inner.next, %for.body3 ], [ 1, %outer.header ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %iv.inner, i64 %iv.outer
  %vA = load i32, i32* %arrayidx5
  %arrayidx9 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @C, i64 0, i64 %iv.inner, i64 %iv.outer
  %vC = load i32, i32* %arrayidx9
  %add = add nsw i32 %vA, %vC
  store i32 %add, i32* %ptr1
  %iv.inner.next = add nuw nsw i64 %iv.inner, 1
  %exitcond = icmp eq i64 %iv.inner.next, 100
  br i1 %exitcond, label %outer.inc, label %for.body3

outer.inc:                                        ; preds = %for.body3, %outer.header
  %sv = phi i64 [ 0, %outer.header ], [ 1, %for.body3 ]
  %iv.outer.next = add nsw i64 %iv.outer, 1
  %cmp = icmp eq i64 %iv.outer.next, 100
  br i1 %cmp, label %outer.header, label %for.exit

for.exit:                                         ; preds = %outer.inc
  %sv.lcssa = phi i64 [ %sv, %outer.inc ]
  store i64 %sv.lcssa, i64* @Y
  br label %for.end16

for.end16:                                        ; preds = %for.exit
  ret void
}

; REMARK: Interchanged
; REMARK-NEXT: lcssa_07
define void @lcssa_07() {
entry:
  br label %outer.header

outer.header:                                     ; preds = %outer.inc, %entry
  %iv.outer = phi i64 [ 1, %entry ], [ %iv.outer.next, %outer.inc ]
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %outer.header
  %iv.inner = phi i64 [ %iv.inner.next, %for.body3 ], [ 1, %outer.header ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %iv.inner, i64 %iv.outer
  %vA = load i32, i32* %arrayidx5
  %arrayidx9 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @C, i64 0, i64 %iv.inner, i64 %iv.outer
  %vC = load i32, i32* %arrayidx9
  %add = add nsw i32 %vA, %vC
  store i32 %add, i32* %arrayidx5
  %iv.inner.next = add nuw nsw i64 %iv.inner, 1
  %exitcond = icmp eq i64 %iv.inner.next, 100
  br i1 %exitcond, label %outer.bb, label %for.body3

outer.bb:                                         ; preds = %for.body3
  %iv.inner.lcssa = phi i64 [ %iv.inner, %for.body3 ]
  br label %outer.inc

outer.inc:                                        ; preds = %outer.bb
  %iv.outer.next = add nsw i64 %iv.outer, 1
  %cmp = icmp eq i64 %iv.outer.next, 100
  br i1 %cmp, label %outer.header, label %for.exit

for.exit:                                         ; preds = %outer.inc
  %iv.inner.lcssa.lcssa = phi i64 [ %iv.inner.lcssa, %outer.inc ]
  store i64 %iv.inner.lcssa.lcssa, i64* @Y
  br label %for.end16

for.end16:                                        ; preds = %for.exit
  ret void
}

; Should not crash when the outer header branches to
; both the inner loop and the outer latch, and there
; is an lcssa phi node outside the loopnest.
; REMARK: Interchanged
; REMARK-NEXT: lcssa_08
define i64 @lcssa_08([100 x [100 x i64]]* %Arr) {
entry:
  br label %for1.header

for1.header:                                         ; preds = %for1.inc, %entry
  %indvars.iv23 = phi i64 [ 0, %entry ], [ %indvars.iv.next24, %for1.inc ]
  br i1 undef, label %for2, label %for1.inc

for2:                                        ; preds = %for2, %for1.header
  %indvars.iv = phi i64 [ 0, %for1.header ], [ %indvars.iv.next.3, %for2 ]
  %arrayidx = getelementptr inbounds [100 x [100 x i64]], [100 x [100 x i64]]* %Arr, i64 0, i64 %indvars.iv, i64 %indvars.iv23
  %lv = load i64, i64* %arrayidx, align 4
  %indvars.iv.next.3 = add nuw nsw i64 %indvars.iv, 1
  %exit1 = icmp eq i64 %indvars.iv.next.3, 100
  br i1 %exit1, label %for1.inc, label %for2

for1.inc:                                ; preds = %for2, %for1.header
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %exit2 = icmp eq i64 %indvars.iv.next24, 100
  br i1 %exit2, label %for1.loopexit, label %for1.header

for1.loopexit:                                 ; preds = %for1.inc
  %sum.outer.lcssa = phi i64 [ %indvars.iv23, %for1.inc ]
  ret i64 %sum.outer.lcssa
}

