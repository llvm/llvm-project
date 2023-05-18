; RUN: llc -O2 -march=hexagon < %s | FileCheck %s
; CHECK-NOT: memw
; CHECK: memub

target triple = "hexagon"

%s.0 = type <{ i8, %s.1 }>
%s.1 = type { [16 x i32] }

; Function Attrs: nounwind
define i32 @f0(i32 %a0, ptr nocapture %a1) #0 {
b0:
  %v0 = icmp sgt i32 %a0, 0
  br i1 %v0, label %b1, label %b10

b1:                                               ; preds = %b0
  br label %b4

b2:                                               ; preds = %b4
  br i1 %v0, label %b3, label %b10

b3:                                               ; preds = %b2
  br label %b7

b4:                                               ; preds = %b4, %b1
  %v1 = phi i32 [ %v6, %b4 ], [ 0, %b1 ]
  %v2 = phi i32 [ %v5, %b4 ], [ 0, %b1 ]
  %v3 = getelementptr inbounds %s.0, ptr %a1, i32 0, i32 1, i32 0, i32 %v1
  %v4 = load i32, ptr %v3, align 1, !tbaa !0
  %v5 = add nsw i32 %v4, %v2
  %v6 = add nsw i32 %v1, 1
  %v7 = icmp eq i32 %v6, %a0
  br i1 %v7, label %b2, label %b4

b5:                                               ; preds = %b7
  br i1 %v0, label %b6, label %b10

b6:                                               ; preds = %b5
  br label %b8

b7:                                               ; preds = %b7, %b3
  %v8 = phi i32 [ %v13, %b7 ], [ 0, %b3 ]
  %v9 = phi i32 [ %v12, %b7 ], [ %v5, %b3 ]
  %v10 = getelementptr inbounds %s.0, ptr %a1, i32 0, i32 1, i32 0, i32 %v8
  %v11 = load i32, ptr %v10, align 1, !tbaa !0
  %v12 = add nsw i32 %v11, %v9
  %v13 = add nsw i32 %v8, 1
  %v14 = icmp eq i32 %v13, %a0
  br i1 %v14, label %b5, label %b7

b8:                                               ; preds = %b8, %b6
  %v15 = phi i32 [ %v17, %b8 ], [ 0, %b6 ]
  %v16 = getelementptr inbounds %s.0, ptr %a1, i32 0, i32 1, i32 0, i32 %v15
  store i32 %a0, ptr %v16, align 1, !tbaa !0
  %v17 = add nsw i32 %v15, 1
  %v18 = icmp eq i32 %v17, %a0
  br i1 %v18, label %b9, label %b8

b9:                                               ; preds = %b8
  br label %b10

b10:                                              ; preds = %b9, %b5, %b2, %b0
  %v19 = phi i32 [ %v12, %b5 ], [ %v5, %b2 ], [ 0, %b0 ], [ %v12, %b9 ]
  ret i32 %v19
}

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
