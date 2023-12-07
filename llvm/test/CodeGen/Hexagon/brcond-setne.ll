; RUN: llc -O3 -march=hexagon < %s | FileCheck %s
; CHECK: cmpb.eq

target triple = "hexagon"

; Function Attrs: nounwind
define zeroext i8 @f0(ptr nocapture %a0, ptr nocapture %a1) #0 {
b0:
  %v0 = load ptr, ptr %a0, align 4, !tbaa !0
  %v1 = load i8, ptr %v0, align 1, !tbaa !4
  %v2 = icmp eq i8 %v1, 0
  br i1 %v2, label %b11, label %b1

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b9, %b1
  %v3 = phi i8 [ %v20, %b9 ], [ %v1, %b1 ]
  %v4 = phi i8 [ %v17, %b9 ], [ 0, %b1 ]
  %v5 = phi ptr [ %v18, %b9 ], [ %v0, %b1 ]
  %v6 = icmp eq i8 %v3, 44
  br i1 %v6, label %b3, label %b4

b3:                                               ; preds = %b2
  %v7 = phi ptr [ %v5, %b2 ]
  %v8 = phi i8 [ %v4, %b2 ]
  %v9 = getelementptr inbounds i8, ptr %v7, i32 1
  br label %b11

b4:                                               ; preds = %b2
  %v10 = icmp eq i8 %v4, 0
  br i1 %v10, label %b5, label %b9

b5:                                               ; preds = %b4
  %v11 = tail call zeroext i8 @f1(i8 zeroext %v3) #0
  %v12 = icmp eq i8 %v11, 0
  br i1 %v12, label %b6, label %b8

b6:                                               ; preds = %b5
  %v13 = icmp eq i8 %v3, 45
  br i1 %v13, label %b7, label %b8

b7:                                               ; preds = %b6
  br label %b8

b8:                                               ; preds = %b7, %b6, %b5
  %v14 = phi i8 [ 2, %b7 ], [ 0, %b6 ], [ 4, %b5 ]
  %v15 = load i32, ptr %a1, align 4, !tbaa !5
  %v16 = add i32 %v15, 1
  store i32 %v16, ptr %a1, align 4, !tbaa !5
  br label %b9

b9:                                               ; preds = %b8, %b4
  %v17 = phi i8 [ %v14, %b8 ], [ %v4, %b4 ]
  %v18 = getelementptr inbounds i8, ptr %v5, i32 1
  %v19 = getelementptr i8, ptr %v5, i32 1
  %v20 = load i8, ptr %v19, align 1, !tbaa !4
  %v21 = icmp ne i8 %v20, 0
  %v22 = icmp ne i8 %v17, 1
  %v23 = and i1 %v21, %v22
  br i1 %v23, label %b2, label %b10

b10:                                              ; preds = %b9
  %v24 = phi ptr [ %v18, %b9 ]
  %v25 = phi i8 [ %v17, %b9 ]
  br label %b11

b11:                                              ; preds = %b10, %b3, %b0
  %v26 = phi i8 [ %v8, %b3 ], [ 0, %b0 ], [ %v25, %b10 ]
  %v27 = phi ptr [ %v9, %b3 ], [ %v0, %b0 ], [ %v24, %b10 ]
  store ptr %v27, ptr %a0, align 4, !tbaa !0
  ret i8 %v26
}

declare zeroext i8 @f1(i8 zeroext)

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!2, !2, i64 0}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !2}
