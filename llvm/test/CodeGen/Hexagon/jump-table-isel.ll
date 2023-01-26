; RUN: llc -march=hexagon -hexagon-emit-jump-tables=0 < %s
; REQUIRES: asserts
; Check for successful compilation.

target triple = "hexagon"

%s.0 = type opaque
%s.1 = type { i32, i32, i32 }

@g0 = external global %s.0
@g1 = external global %s.0
@g2 = external global %s.0
@g3 = external global %s.0
@g4 = external global %s.0

; Function Attrs: nounwind optsize
define zeroext i8 @f0(ptr %a0, ptr nocapture %a1) #0 {
b0:
  store ptr null, ptr %a1, align 4, !tbaa !0
  %v0 = getelementptr inbounds %s.1, ptr %a0, i32 0, i32 1
  %v1 = load i32, ptr %v0, align 4, !tbaa !4
  %v2 = icmp eq i32 %v1, 0
  br i1 %v2, label %b1, label %b8

b1:                                               ; preds = %b0
  %v4 = load i32, ptr %a0, align 4, !tbaa !7
  switch i32 %v4, label %b8 [
    i32 0, label %b2
    i32 1, label %b4
    i32 4, label %b5
    i32 5, label %b6
    i32 2, label %b7
  ]

b2:                                               ; preds = %b1
  %v5 = getelementptr inbounds %s.1, ptr %a0, i32 0, i32 2
  %v6 = load i32, ptr %v5, align 4, !tbaa !8
  switch i32 %v6, label %b8 [
    i32 27, label %b3
    i32 44, label %b3
  ]

b3:                                               ; preds = %b7, %b7, %b7, %b6, %b6, %b5, %b5, %b4, %b4, %b2, %b2
  %v7 = phi ptr [ @g0, %b2 ], [ @g0, %b2 ], [ @g1, %b4 ], [ @g1, %b4 ], [ @g2, %b5 ], [ @g2, %b5 ], [ @g3, %b6 ], [ @g3, %b6 ], [ @g4, %b7 ], [ @g4, %b7 ], [ @g4, %b7 ]
  store ptr %v7, ptr %a1, align 4, !tbaa !0
  br label %b8

b4:                                               ; preds = %b1
  %v8 = getelementptr inbounds %s.1, ptr %a0, i32 0, i32 2
  %v9 = load i32, ptr %v8, align 4, !tbaa !8
  switch i32 %v9, label %b8 [
    i32 27, label %b3
    i32 44, label %b3
  ]

b5:                                               ; preds = %b1
  %v10 = getelementptr inbounds %s.1, ptr %a0, i32 0, i32 2
  %v11 = load i32, ptr %v10, align 4, !tbaa !8
  switch i32 %v11, label %b8 [
    i32 27, label %b3
    i32 44, label %b3
  ]

b6:                                               ; preds = %b1
  %v12 = getelementptr inbounds %s.1, ptr %a0, i32 0, i32 2
  %v13 = load i32, ptr %v12, align 4, !tbaa !8
  switch i32 %v13, label %b8 [
    i32 27, label %b3
    i32 44, label %b3
  ]

b7:                                               ; preds = %b1
  %v14 = getelementptr inbounds %s.1, ptr %a0, i32 0, i32 2
  %v15 = load i32, ptr %v14, align 4, !tbaa !8
  switch i32 %v15, label %b8 [
    i32 40, label %b3
    i32 46, label %b3
    i32 47, label %b3
  ]

b8:                                               ; preds = %b7, %b6, %b5, %b4, %b3, %b2, %b1, %b0
  %v16 = phi i8 [ 1, %b3 ], [ 0, %b0 ], [ 0, %b2 ], [ 0, %b4 ], [ 0, %b5 ], [ 0, %b6 ], [ 0, %b1 ], [ 0, %b7 ]
  ret i8 %v16
}

attributes #0 = { nounwind optsize }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !6, i64 4}
!5 = !{!"_ZTS3bar", !6, i64 0, !6, i64 4, !6, i64 8}
!6 = !{!"int", !2, i64 0}
!7 = !{!5, !6, i64 0}
!8 = !{!5, !6, i64 8}
