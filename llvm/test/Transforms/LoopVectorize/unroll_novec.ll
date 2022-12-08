; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-width=1 -force-target-num-scalar-regs=16 -force-target-max-scalar-interleave=8 -force-target-instruction-cost=1 -small-loop-cost=40 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@a = common global [2048 x i32] zeroinitializer, align 16

; This is the loop.
;  for (i=0; i<n; i++){
;    a[i] += i;
;  }
;CHECK-LABEL: @inc(
;CHECK: load i32, i32*
;CHECK: load i32, i32*
;CHECK: load i32, i32*
;CHECK: load i32, i32*
;CHECK-NOT: load i32, i32*
;CHECK: add nsw i32
;CHECK: add nsw i32
;CHECK: add nsw i32
;CHECK: add nsw i32
;CHECK-NOT: add nsw i32
;CHECK: store i32
;CHECK: store i32
;CHECK: store i32
;CHECK: store i32
;CHECK-NOT: store i32
;CHECK: add nuw i64 %{{.*}}, 4
;CHECK: ret void
define void @inc(i32 %n) nounwind uwtable noinline ssp {
  %1 = icmp sgt i32 %n, 0
  br i1 %1, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %0, %.lr.ph
  %indvars.iv = phi i64 [ %indvars.iv.next, %.lr.ph ], [ 0, %0 ]
  %2 = getelementptr inbounds [2048 x i32], [2048 x i32]* @a, i64 0, i64 %indvars.iv
  %3 = load i32, i32* %2, align 4
  %4 = trunc i64 %indvars.iv to i32
  %5 = add nsw i32 %3, %4
  store i32 %5, i32* %2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %._crit_edge, label %.lr.ph

._crit_edge:                                      ; preds = %.lr.ph, %0
  ret void
}

