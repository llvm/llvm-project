; RUN: llc -march=hexagon -machine-sink-split < %s
; REQUIRES: asserts
; MachineSink should not sink an MI which is used in a non-phi instruction
; in an MBB with multiple predecessors.

target triple = "hexagon-unknown--elf"

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  unreachable

b2:                                               ; preds = %b0
  %v0 = load ptr, ptr undef, align 4
  %v1 = getelementptr inbounds i8, ptr %v0, i32 1
  %v2 = load i8, ptr %v0, align 1, !tbaa !0
  %v3 = zext i8 %v2 to i32
  %v4 = shl nuw nsw i32 %v3, 8
  br i1 undef, label %b3, label %b5

b3:                                               ; preds = %b2
  br i1 undef, label %b15, label %b4

b4:                                               ; preds = %b3
  br label %b5

b5:                                               ; preds = %b4, %b2
  %v5 = phi ptr [ undef, %b4 ], [ %v1, %b2 ]
  %v6 = load i8, ptr %v5, align 1, !tbaa !0
  %v7 = zext i8 %v6 to i32
  %v8 = add nsw i32 %v7, %v4
  %v9 = add nsw i32 %v8, -2
  br label %b6

b6:                                               ; preds = %b8, %b5
  br i1 false, label %b7, label %b8

b7:                                               ; preds = %b6
  unreachable

b8:                                               ; preds = %b6
  br i1 undef, label %b6, label %b9

b9:                                               ; preds = %b8
  br i1 undef, label %b10, label %b14

b10:                                              ; preds = %b9
  br i1 undef, label %b11, label %b13

b11:                                              ; preds = %b10
  br i1 undef, label %b12, label %b13

b12:                                              ; preds = %b11
  unreachable

b13:                                              ; preds = %b11, %b10
  store i32 %v9, ptr undef, align 4, !tbaa !3
  unreachable

b14:                                              ; preds = %b9
  unreachable

b15:                                              ; preds = %b3
  ret void
}

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !1, i64 0}
