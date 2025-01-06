; RUN: llc -mtriple=hexagon -enable-pipeliner < %s
; REQUIRES: asserts

; Test that the pipeliner doesn't ICE in the ScheduleDAG code because
; the latency values are not updated properly. The pipeliner should
; not change the latency of chain edges.

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  %v0 = alloca [10 x i16], align 8
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v1 = phi i32 [ %v7, %b1 ], [ undef, %b0 ]
  %v2 = add i32 %v1, -1
  %v3 = getelementptr inbounds [10 x i16], ptr %v0, i32 0, i32 %v2
  %v4 = add i32 %v1, -2
  %v5 = getelementptr inbounds [10 x i16], ptr %v0, i32 0, i32 %v4
  %v6 = load i16, ptr %v5, align 2, !tbaa !0
  store i16 %v6, ptr %v3, align 2, !tbaa !0
  %v7 = add i32 %v1, -4
  %v8 = icmp sgt i32 %v7, 3
  br i1 %v8, label %b1, label %b2

b2:                                               ; preds = %b2, %b1
  br label %b2
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
