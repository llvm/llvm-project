; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' < %s

; This test case helps to determine wether SCEVRemoveMax::remove produces
; an infinite loop and a segmentation fault, if it processes, for example,
; '((-1 + (-1 * %b1)) umax {(-1 + (-1 * %yStart)),+,-1}<%.preheader>)'.
;
; In this case, the SCoP is invalid. However, SCoP detection failed when
; running over it.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@vertPlane = external global ptr, align 8

define fastcc void @Maze2Mech(i64 %i, i64 %b1, i64 %yStart) {
.split:
  br i1 undef, label %DrawSegment.exit, label %DrawSegment.exit34

DrawSegment.exit34:                               ; preds = %.split
  %tmp = icmp ugt i64 %yStart, %b1
  %tmp1 = select i1 %tmp, i64 %b1, i64 %yStart
  %tmp2 = load ptr, ptr @vertPlane, align 8
  %y.04.i21 = add i64 %tmp1, 1
  br label %.lr.ph.i24

.lr.ph.i24:                                       ; preds = %.lr.ph.i24, %DrawSegment.exit34
  %y.05.i22 = phi i64 [ %y.0.i23, %.lr.ph.i24 ], [ %y.04.i21, %DrawSegment.exit34 ]
  %tmp3 = mul i64 %y.05.i22, undef
  %tmp4 = add i64 %tmp3, %i
  %tmp5 = getelementptr inbounds i8, ptr %tmp2, i64 %tmp4
  %tmp6 = load i8, ptr %tmp5, align 1
  %y.0.i23 = add nuw i64 %y.05.i22, 1
  br i1 false, label %bb, label %.lr.ph.i24

bb:                                               ; preds = %.lr.ph.i24
  unreachable

DrawSegment.exit:                                 ; preds = %.split
  ret void
}
