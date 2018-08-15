; RUN: opt -S -loop-vectorize -force-vector-interleave=1 -force-vector-width=2 < %s 2>&1 | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"

@f = common global i32 0, align 4
@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@c = common global i32 0, align 4
@a = common global i32 0, align 4
@b = common global i32 0, align 4
@e = common global i32 0, align 4

; It has a value that is used outside of the loop
; and is not a recognized reduction variable "tmp17".
; However, tmp17 is a non-header phi which is an allowed exit.

; CHECK-LABEL: @test1(
; CHECK: %vec.ind = phi <2 x i32>
; CHECK: [[CMP:%[a-zA-Z0-9.]+]] = icmp sgt <2 x i32> %vec.ind, <i32 10, i32 10>
; CHECK: %predphi = select <2 x i1> [[CMP]], <2 x i32> <i32 1, i32 1>, <2 x i32> zeroinitializer

; CHECK-LABEL: middle.block:
; CHECK:          [[E1:%[a-zA-Z0-9.]+]] = extractelement <2 x i32> %predphi, i32 1

; CHECK-LABEL: f1.exit.loopexit:
; CHECK:          %.lcssa = phi i32 [ %tmp17, %bb16 ], [ [[E1]], %middle.block ]

define i32 @test1()  {
bb:
  %b.promoted = load i32, i32* @b, align 4
  br label %.lr.ph.i

.lr.ph.i:
  %tmp8 = phi i32 [ %tmp18, %bb16 ], [ %b.promoted, %bb ]
  %tmp2 = icmp sgt i32 %tmp8, 10
  br i1 %tmp2, label %bb16, label %bb10

bb10:
  br label %bb16

bb16:
  %tmp17 = phi i32 [ 0, %bb10 ], [ 1, %.lr.ph.i ]
  %tmp18 = add nsw i32 %tmp8, 1
  %tmp19 = icmp slt i32 %tmp18, 4
  br i1 %tmp19, label %.lr.ph.i, label %f1.exit.loopexit

f1.exit.loopexit:
  %.lcssa = phi i32 [ %tmp17, %bb16 ]
  ret i32 %.lcssa
}

; non-hdr phi depends on header phi.
; CHECK-LABEL: @test2(
; CHECK: %vec.ind = phi <2 x i32>
; CHECK: [[CMP:%[a-zA-Z0-9.]+]] = icmp sgt <2 x i32> %vec.ind, <i32 10, i32 10>
; CHECK: %predphi = select <2 x i1> [[CMP]], <2 x i32> <i32 1, i32 1>, <2 x i32> %vec.ind

; CHECK-LABEL: middle.block:
; CHECK:          [[E1:%[a-zA-Z0-9.]+]] = extractelement <2 x i32> %predphi, i32 1

; CHECK-LABEL: f1.exit.loopexit:
; CHECK:          %.lcssa = phi i32 [ %tmp17, %bb16 ], [ [[E1]], %middle.block ]
define i32 @test2()  {
bb:
  %b.promoted = load i32, i32* @b, align 4
  br label %.lr.ph.i

.lr.ph.i:
  %tmp8 = phi i32 [ %tmp18, %bb16 ], [ %b.promoted, %bb ]
  %tmp2 = icmp sgt i32 %tmp8, 10
  br i1 %tmp2, label %bb16, label %bb10

bb10:
  br label %bb16

bb16:
  %tmp17 = phi i32 [ %tmp8, %bb10 ], [ 1, %.lr.ph.i ]
  %tmp18 = add nsw i32 %tmp8, 1
  %tmp19 = icmp slt i32 %tmp18, 4
  br i1 %tmp19, label %.lr.ph.i, label %f1.exit.loopexit

f1.exit.loopexit:
  %.lcssa = phi i32 [ %tmp17, %bb16 ]
  ret i32 %.lcssa
}

; more than 2 incoming values for tmp17 phi that is used outside loop.
; CHECK-LABEL: test3(
; CHECK-LABEL: vector.body:
; CHECK:          %predphi = select <2 x i1> %{{.*}}, <2 x i32> <i32 1, i32 1>, <2 x i32> zeroinitializer
; CHECK:          %predphi1 = select <2 x i1> %{{.*}}, <2 x i32> <i32 2, i32 2>, <2 x i32> %predphi

; CHECK-LABEL: middle.block:
; CHECK:          [[E1:%[a-zA-Z0-9.]+]] = extractelement <2 x i32> %predphi1, i32 1

; CHECK-LABEL: f1.exit.loopexit:
; CHECK:          phi i32 [ %tmp17, %bb16 ], [ [[E1]], %middle.block ]
define i32 @test3(i32 %N)  {
bb:
  %b.promoted = load i32, i32* @b, align 4
  br label %.lr.ph.i

.lr.ph.i:
  %tmp8 = phi i32 [ %tmp18, %bb16 ], [ %b.promoted, %bb ]
  %tmp2 = icmp sgt i32 %tmp8, 10
  br i1 %tmp2, label %bb16, label %bb10

bb10:
  %cmp = icmp sgt i32 %tmp8, %N
  br i1  %cmp, label %bb12, label %bb16

bb12:
  br label %bb16

bb16:
  %tmp17 = phi i32 [ 0, %bb10 ], [ 1, %.lr.ph.i ], [ 2, %bb12 ]
  %tmp18 = add nsw i32 %tmp8, 1
  %tmp19 = icmp slt i32 %tmp18, 4
  br i1 %tmp19, label %.lr.ph.i, label %f1.exit.loopexit

f1.exit.loopexit:
  %.lcssa = phi i32 [ %tmp17, %bb16 ]
  ret i32 %.lcssa
}

; more than one incoming value for outside user: %.lcssa
; CHECK-LABEL: test4(
; CHECK-LABEL: vector.body:
; CHECK:          %predphi = select <2 x i1>

; CHECK-LABEL: middle.block:
; CHECK:          [[E1:%[a-zA-Z0-9.]+]] = extractelement <2 x i32> %predphi, i32 1

; CHECK-LABEL: f1.exit.loopexit.loopexit:
; CHECK:          %tmp17.lcssa = phi i32 [ %tmp17, %bb16 ], [ [[E1]], %middle.block ]
; CHECK-NEXT:     br label %f1.exit.loopexit

; CHECK-LABEL: f1.exit.loopexit:
; CHECK:          %.lcssa = phi i32 [ 2, %bb ], [ %tmp17.lcssa, %f1.exit.loopexit.loopexit ]
define i32 @test4(i32 %N)  {
bb:
  %b.promoted = load i32, i32* @b, align 4
  %icmp = icmp slt i32 %b.promoted, %N
  br i1 %icmp, label %f1.exit.loopexit, label %.lr.ph.i

.lr.ph.i:
  %tmp8 = phi i32 [ %tmp18, %bb16 ], [ %b.promoted, %bb ]
  %tmp2 = icmp sgt i32 %tmp8, 10
  br i1 %tmp2, label %bb16, label %bb10

bb10:
  br label %bb16

bb16:
  %tmp17 = phi i32 [ 0, %bb10 ], [ 1, %.lr.ph.i ]
  %tmp18 = add nsw i32 %tmp8, 1
  %tmp19 = icmp slt i32 %tmp18, 4
  br i1 %tmp19, label %.lr.ph.i, label %f1.exit.loopexit

f1.exit.loopexit:
  %.lcssa = phi i32 [ %tmp17, %bb16 ], [ 2, %bb ]
  ret i32 %.lcssa
}

; non hdr phi that depends on reduction and is used outside the loop.
; reduction phis are only allowed to have bump or reduction operations as the inside user, so we should
; not vectorize this.
; CHECK-LABEL: reduction_sum(
; CHECK-NOT: <2 x i32>
define i32 @reduction_sum(i32 %n, i32* noalias nocapture %A, i32* noalias nocapture %B) nounwind uwtable readonly noinline ssp {
entry:
  %c1 = icmp sgt i32 %n, 0
  br i1 %c1, label %header, label %._crit_edge

header:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb16 ], [ 0, %entry ]
  %sum.02 = phi i32 [ %c9, %bb16 ], [ 0, %entry ]
  %c2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %c3 = load i32, i32* %c2, align 4
  %c4 = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %c5 = load i32, i32* %c4, align 4
  %tmp2 = icmp sgt i32 %sum.02, 10
  br i1 %tmp2, label %bb16, label %bb10

bb10:
  br label %bb16

bb16:
  %tmp17 = phi i32 [ %sum.02, %bb10 ], [ 1, %header ]
  %c6 = trunc i64 %indvars.iv to i32
  %c7 = add i32 %sum.02, %c6
  %c8 = add i32 %c7, %c3
  %c9 = add i32 %c8, %c5
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %header

._crit_edge:                                      ; preds = %.lr.ph, %0
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %c9, %bb16 ]
  %nonhdr.lcssa = phi i32 [ 1, %entry], [ %tmp17, %bb16 ]
  ret i32 %sum.0.lcssa
}

; invalid cyclic dependency with header phi iv, which prevents iv from being
; recognized as induction var.
; cannot vectorize.
; CHECK-LABEL: cyclic_dep_with_indvar(
; CHECK-NOT: <2 x i32>
define i32 @cyclic_dep_with_indvar()  {
bb:
  %b.promoted = load i32, i32* @b, align 4
  br label %.lr.ph.i

.lr.ph.i:
  %iv = phi i32 [ %ivnext, %bb16 ], [ %b.promoted, %bb ]
  %tmp2 = icmp sgt i32 %iv, 10
  br i1 %tmp2, label %bb16, label %bb10

bb10:
  br label %bb16

bb16:
  %tmp17 = phi i32 [ 0, %bb10 ], [ %iv, %.lr.ph.i ]
  %ivnext = add nsw i32 %tmp17, 1
  %tmp19 = icmp slt i32 %ivnext, 4
  br i1 %tmp19, label %.lr.ph.i, label %f1.exit.loopexit

f1.exit.loopexit:
  %.lcssa = phi i32 [ %tmp17, %bb16 ]
  ret i32 %.lcssa
}

; non-reduction phi 'tmp17' used outside loop has cyclic dependence with %x.05 phi
; cannot vectorize.
; CHECK-LABEL: not_valid_reduction(
; CHECK-NOT: <2 x i32>
define i32 @not_valid_reduction(i32 %n, i32* noalias nocapture %A) nounwind uwtable readonly {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %latch ], [ 0, %entry ]
  %x.05 = phi i32 [ %tmp17, %latch ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp0 = load i32, i32* %arrayidx, align 4
  %tmp2 = icmp sgt i64 %indvars.iv, 10
  %sub = sub nsw i32 %x.05, %tmp0
  br i1 %tmp2, label %bb16, label %bb10

bb10:
  br label %bb16

bb16:
  %tmp17 = phi i32 [ 1, %bb10 ], [ %sub, %for.body ]
  br label %latch

latch:
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %x.0.lcssa = phi i32 [ 0, %entry ], [ %tmp17 , %latch ]
  ret i32 %x.0.lcssa
}
