; RUN: opt < %s -basicaa -loop-interchange -pass-remarks-missed='loop-interchange' -pass-remarks-output=%t
; RUN: cat %t |  FileCheck --check-prefix REMARK %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

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

define void @lcssa_01(){
entry:
  %cmp21 = icmp sgt i64 100, 1
  br i1 %cmp21, label %outer.ph, label %for.end16

outer.ph:
  %cmp218 = icmp sgt i64 100, 1
  br label %outer.header

outer.header:
  %iv.outer= phi i64 [ 1, %outer.ph ], [ %iv.outer.next, %outer.inc ]
  br i1 %cmp218, label %for.body3, label %outer.inc

for.body3:
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

outer.inc:
  %iv.outer.next = add nsw i64 %iv.outer, 1
  %cmp = icmp eq i64 %iv.outer.next, 100
  br i1 %cmp, label %outer.header, label %for.exit

for.exit:
  store i64 %iv.outer.next, i64 * @Y
  br label %for.end16

for.end16:
  ret void
}

; REMARK: UnsupportedExitPHI
; REMARK-NEXT: lcssa_02
define void @lcssa_02(){
entry:
  %cmp21 = icmp sgt i64 100, 1
  br i1 %cmp21, label %outer.ph, label %for.end16

outer.ph:
  %cmp218 = icmp sgt i64 100, 1
  br label %outer.header

outer.header:
  %iv.outer= phi i64 [ 1, %outer.ph ], [ %iv.outer.next, %outer.inc ]
  br i1 %cmp218, label %for.body3, label %outer.inc

for.body3:
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

outer.inc:
  %iv.inner.end = phi i64 [ 0, %outer.header ], [ %iv.inner.next, %for.body3 ]
  %iv.outer.next = add nsw i64 %iv.outer, 1
  %cmp = icmp eq i64 %iv.outer.next, 100
  br i1 %cmp, label %outer.header, label %for.exit

for.exit:
  store i64 %iv.inner.end, i64 * @Y
  br label %for.end16

for.end16:
  ret void
}


; REMARK: Interchanged
; REMARK-NEXT: lcssa_03
define void @lcssa_03(){
entry:
  br label %outer.header

outer.header:
  %iv.outer= phi i64 [ 1, %entry ], [ %iv.outer.next, %outer.inc ]
  br label %for.body3

for.body3:
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

outer.inc:
  %iv.outer.next = add nsw i64 %iv.outer, 1
  %cmp = icmp eq i64 %iv.outer.next, 100
  br i1 %cmp, label %outer.header, label %for.exit

for.exit:
  store i64 %iv.inner, i64 * @Y
  br label %for.end16

for.end16:
  ret void
}

; FIXME: We currently do not support LCSSA phi nodes involving floating point
;        types, as we fail to detect floating point reductions for now.
; REMARK: UnsupportedPHIOuter
; REMARK-NEXT: lcssa_04
define void @lcssa_04(){
entry:
  br label %outer.header

outer.header:
  %iv.outer= phi i64 [ 1, %entry ], [ %iv.outer.next, %outer.inc ]
  %float.outer= phi float [ 1.0, %entry ], [ 2.0, %outer.inc ]
  br label %for.body3

for.body3:
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

outer.inc:
  %iv.outer.next = add nsw i64 %iv.outer, 1
  %cmp = icmp eq i64 %iv.outer.next, 100
  br i1 %cmp, label %outer.header, label %for.exit

for.exit:
  store float %float.outer, float* @F
  br label %for.end16

for.end16:
  ret void
}
