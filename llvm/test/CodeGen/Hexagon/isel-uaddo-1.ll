; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that a hardware loop is generated.
; CHECK: loop0

target triple = "hexagon"

; Function Attrs: norecurse nounwind
define dso_local void @f0(i32* nocapture readonly %a0, i32* nocapture %a1) local_unnamed_addr #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ %v3, %b1 ], [ 100, %b0 ]
  %v1 = phi i32* [ %v6, %b1 ], [ %a1, %b0 ]
  %v2 = phi i32* [ %v4, %b1 ], [ %a0, %b0 ]
  %v3 = add nsw i32 %v0, -1
  %v4 = getelementptr inbounds i32, i32* %v2, i32 1
  %v5 = load i32, i32* %v2, align 4, !tbaa !1
  %v6 = getelementptr inbounds i32, i32* %v1, i32 1
  store i32 %v5, i32* %v1, align 4, !tbaa !1
  %v7 = icmp eq i32 %v3, 0
  br i1 %v7, label %b2, label %b1

b2:                                               ; preds = %b1
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv62" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
