; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Test that the DAG combiner doesn't assert because it attempts to replace
; the chain of a post-increment store based upon alias information. The code
; in DAGCombiner is unable to convert indexed stores.

; Function Attrs: nounwind
define void @f0(i32 %a0, ptr %a1, ptr %a2) #0 {
b0:
  switch i32 %a0, label %b5 [
    i32 67830273, label %b1
    i32 67502595, label %b3
  ]

b1:                                               ; preds = %b0
  br i1 undef, label %b2, label %b5

b2:                                               ; preds = %b1
  br label %b5

b3:                                               ; preds = %b0
  br i1 undef, label %b4, label %b5

b4:                                               ; preds = %b3
  store i32 0, ptr %a2, align 1, !tbaa !0
  %v1 = getelementptr inbounds i8, ptr %a1, i32 4
  %v3 = load i32, ptr %v1, align 4, !tbaa !5
  %v4 = getelementptr inbounds i8, ptr %a2, i32 4
  store i32 %v3, ptr %v4, align 1, !tbaa !5
  br label %b5

b5:                                               ; preds = %b4, %b3, %b2, %b1, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !2, i64 0}
!1 = !{!"", !2, i64 0, !2, i64 4}
!2 = !{!"long", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!1, !2, i64 4}
