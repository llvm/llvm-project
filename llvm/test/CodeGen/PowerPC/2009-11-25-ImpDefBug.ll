; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu.5 -mcpu=g5
; rdar://7422268

%struct..0EdgeT = type { i32, i32, float, float, i32, i32, i32, float, i32, i32 }

define void @smooth_color_z_triangle(i32 %v0, i32 %v1, i32 %v2, i32 %pv) nounwind {
entry:
  br i1 undef, label %return, label %bb14

bb14:                                             ; preds = %entry
  br i1 undef, label %bb15, label %return

bb15:                                             ; preds = %bb14
  br i1 undef, label %bb16, label %bb17

bb16:                                             ; preds = %bb15
  br label %bb17

bb17:                                             ; preds = %bb16, %bb15
  %0 = fcmp olt float undef, 0.000000e+00         ; <i1> [#uses=2]
  %eTop.eMaj = select i1 %0, ptr undef, ptr null ; <ptr> [#uses=1]
  br label %bb69

bb24:                                             ; preds = %bb69
  br i1 undef, label %bb25, label %bb28

bb25:                                             ; preds = %bb24
  br label %bb33

bb28:                                             ; preds = %bb24
  br i1 undef, label %return, label %bb32

bb32:                                             ; preds = %bb28
  br i1 %0, label %bb38, label %bb33

bb33:                                             ; preds = %bb32, %bb25
  br i1 undef, label %bb34, label %bb38

bb34:                                             ; preds = %bb33
  br label %bb38

bb38:                                             ; preds = %bb34, %bb33, %bb32
  %eRight.08 = phi ptr [ %eTop.eMaj, %bb32 ], [ undef, %bb34 ], [ undef, %bb33 ] ; <ptr> [#uses=0]
  %fdgOuter.0 = phi i32 [ %fdgOuter.1, %bb32 ], [ undef, %bb34 ], [ %fdgOuter.1, %bb33 ] ; <i32> [#uses=1]
  %fz.3 = phi i32 [ %fz.2, %bb32 ], [ 2147483647, %bb34 ], [ %fz.2, %bb33 ] ; <i32> [#uses=1]
  %1 = add i32 undef, 1                           ; <i32> [#uses=0]
  br label %bb69

bb69:                                             ; preds = %bb38, %bb17
  %fdgOuter.1 = phi i32 [ undef, %bb17 ], [ %fdgOuter.0, %bb38 ] ; <i32> [#uses=2]
  %fz.2 = phi i32 [ undef, %bb17 ], [ %fz.3, %bb38 ] ; <i32> [#uses=2]
  br i1 undef, label %bb24, label %return

return:                                           ; preds = %bb69, %bb28, %bb14, %entry
  ret void
}
