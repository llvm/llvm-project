; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: loop0

target triple = "hexagon"

; Function Attrs: nounwind readonly
define i32 @f0(ptr nocapture %a0, ptr nocapture %a1, i32 %a2) #0 {
b0:
  %v0 = icmp eq i32 %a2, 0
  br i1 %v0, label %b6, label %b1

b1:                                               ; preds = %b0
  br label %b2

b2:                                               ; preds = %b4, %b1
  %v1 = phi ptr [ %v10, %b4 ], [ %a1, %b1 ]
  %v2 = phi ptr [ %v9, %b4 ], [ %a0, %b1 ]
  %v3 = phi i32 [ %v11, %b4 ], [ %a2, %b1 ]
  %v4 = load i8, ptr %v2, align 1, !tbaa !0
  %v5 = load i8, ptr %v1, align 1, !tbaa !0
  %v6 = icmp eq i8 %v4, %v5
  br i1 %v6, label %b4, label %b3

b3:                                               ; preds = %b2
  %v7 = icmp ult i8 %v4, %v5
  %v8 = select i1 %v7, i32 -1, i32 1
  br label %b6

b4:                                               ; preds = %b2
  %v9 = getelementptr inbounds i8, ptr %v2, i32 1
  %v10 = getelementptr inbounds i8, ptr %v1, i32 1
  %v11 = add i32 %v3, -1
  %v12 = icmp eq i32 %v11, 0
  br i1 %v12, label %b5, label %b2

b5:                                               ; preds = %b4
  br label %b6

b6:                                               ; preds = %b5, %b3, %b0
  %v13 = phi i32 [ %v8, %b3 ], [ 0, %b0 ], [ 0, %b5 ]
  ret i32 %v13
}

attributes #0 = { nounwind readonly "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
