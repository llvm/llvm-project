; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: swiz

target triple = "hexagon"

%s.0 = type { [4 x i32], [2 x i32], [64 x i8] }

; Function Attrs: nounwind
define void @f0(%s.0* nocapture %a0, i8* nocapture %a1, i32 %a2) #0 {
b0:
  %v0 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 2, i32 0
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v1 = phi i16 [ 16, %b0 ], [ %v22, %b1 ]
  %v2 = phi i8* [ %v0, %b0 ], [ %v21, %b1 ]
  %v3 = getelementptr inbounds i8, i8* %v2, i32 3
  %v4 = load i8, i8* %v3, align 1, !tbaa !0
  %v5 = zext i8 %v4 to i32
  %v6 = shl nuw nsw i32 %v5, 8
  %v7 = getelementptr inbounds i8, i8* %v2, i32 2
  %v8 = load i8, i8* %v7, align 1, !tbaa !0
  %v9 = zext i8 %v8 to i32
  %v10 = or i32 %v6, %v9
  %v11 = shl nuw i32 %v10, 16
  %v12 = getelementptr inbounds i8, i8* %v2, i32 1
  %v13 = load i8, i8* %v12, align 1, !tbaa !0
  %v14 = zext i8 %v13 to i32
  %v15 = shl nuw nsw i32 %v14, 8
  %v16 = load i8, i8* %v2, align 1, !tbaa !0
  %v17 = zext i8 %v16 to i32
  %v18 = or i32 %v11, %v15
  %v19 = or i32 %v18, %v17
  %v20 = bitcast i8* %v2 to i32*
  store i32 %v19, i32* %v20, align 4, !tbaa !3
  %v21 = getelementptr inbounds i8, i8* %v2, i32 4
  %v22 = add i16 %v1, -1
  %v23 = icmp eq i16 %v22, 0
  br i1 %v23, label %b2, label %b1

b2:                                               ; preds = %b1
  ret void
}

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !1}
