; RUN: opt -S -loop-vectorize -force-vector-interleave=1 -force-vector-width=2 < %s 2>&1 | FileCheck %s

; CHECK: remark: {{.*}}: loop not vectorized: value could not be identified as an induction or reduction variable

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"

@f = common global i32 0, align 4
@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@c = common global i32 0, align 4
@a = common global i32 0, align 4
@b = common global i32 0, align 4
@e = common global i32 0, align 4

; We used to vectorize this loop. But it has a value that is used outside of the
; and is not a recognized reduction variable "tmp17".

; CHECK-LABEL: @main(
; CHECK-NOT: <2 x i32>

define i32 @main()  {
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
