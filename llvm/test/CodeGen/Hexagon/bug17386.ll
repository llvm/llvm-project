; RUN: llc -march=hexagon -O2 < %s
; REQUIRES: asserts

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(i8* %a0, ...) #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  unreachable

b2:                                               ; preds = %b2, %b0
  br i1 undef, label %b2, label %b3

b3:                                               ; preds = %b2
  br i1 undef, label %b4, label %b5

b4:                                               ; preds = %b3
  br label %b5

b5:                                               ; preds = %b4, %b3
  br label %b6

b6:                                               ; preds = %b12, %b5
  br i1 undef, label %b9, label %b7

b7:                                               ; preds = %b6
  %v0 = load i8, i8* undef, align 1, !tbaa !0
  %v1 = zext i8 %v0 to i32
  br i1 undef, label %b9, label %b8

b8:                                               ; preds = %b7
  br i1 undef, label %b9, label %b10

b9:                                               ; preds = %b8, %b7, %b6
  br label %b10

b10:                                              ; preds = %b9, %b8
  %v2 = phi i32 [ undef, %b9 ], [ %v1, %b8 ]
  %v3 = icmp eq i32 %v2, 37
  %v4 = sext i1 %v3 to i32
  %v5 = icmp slt i32 0, 1
  br i1 %v5, label %b12, label %b11

b11:                                              ; preds = %b10
  unreachable

b12:                                              ; preds = %b10
  br i1 undef, label %b13, label %b6

b13:                                              ; preds = %b12
  br label %b14

b14:                                              ; preds = %b15, %b13
  br i1 undef, label %b16, label %b15

b15:                                              ; preds = %b14
  br i1 undef, label %b14, label %b16

b16:                                              ; preds = %b15, %b14
  br label %b17

b17:                                              ; preds = %b18, %b16
  %v6 = phi i8* [ undef, %b16 ], [ %v7, %b18 ]
  %v7 = getelementptr inbounds i8, i8* %v6, i32 1
  %v8 = load i8, i8* %v7, align 1, !tbaa !0
  br label %b18

b18:                                              ; preds = %b19, %b17
  %v9 = phi i32 [ 5, %b17 ], [ %v11, %b19 ]
  %v10 = icmp eq i8 undef, %v8
  br i1 %v10, label %b17, label %b19

b19:                                              ; preds = %b18
  %v11 = add i32 %v9, -1
  %v12 = icmp eq i32 %v11, 0
  br i1 %v12, label %b20, label %b18

b20:                                              ; preds = %b19
  unreachable
}

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}
