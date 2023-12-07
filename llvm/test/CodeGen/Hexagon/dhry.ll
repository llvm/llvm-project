; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: combine(#11,#10)

; Function Attrs: nounwind
define void @f0(ptr nocapture %a0, ptr nocapture %a1) #0 {
b0:
  br label %b2

b1:                                               ; preds = %b4
  br label %b5

b2:                                               ; preds = %b0
  %v0 = getelementptr inbounds i32, ptr %a0, i32 2
  %v1 = getelementptr inbounds i32, ptr %a0, i32 3
  br label %b3

b3:                                               ; preds = %b2
  br label %b4

b4:                                               ; preds = %b4, %b3
  %v2 = load i32, ptr %v0, align 4, !tbaa !0
  %v3 = load i32, ptr %v1, align 4, !tbaa !0
  %v4 = tail call i32 @f1(i32 %v2, i32 %v3) #0
  %v5 = icmp eq i32 %v4, 0
  br i1 %v5, label %b4, label %b1

b5:                                               ; preds = %b1
  %v6 = tail call i32 @f1(i32 10, i32 11) #0
  ret void
}

declare i32 @f1(i32, i32)

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
