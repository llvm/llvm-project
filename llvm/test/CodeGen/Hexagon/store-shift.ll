; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-DAG: r[[BASE:[0-9]+]] = add(r1,#1000)
; CHECK-DAG: r[[IDX0:[0-9]+]] = add(r2,#5)
; CHECK-DAG: r[[IDX1:[0-9]+]] = add(r2,#6)
; CHECK-DAG: memw(r0+r[[IDX0]]<<#2) = r3
; CHECK-DAG: memw(r0+r[[IDX1]]<<#2) = r3
; CHECK-DAG: memw(r[[BASE]]+r[[IDX0]]<<#2) = r[[IDX0]]
; CHECK-DAG: memw(r[[BASE]]+r[[IDX1]]<<#2) = r[[IDX0]]

target triple = "hexagon"

@G = external global i32, align 4

; Function Attrs: norecurse nounwind
define void @fred(ptr nocapture %A, ptr nocapture %B, i32 %N, i32 %M) #0 {
entry:
  %add = add nsw i32 %N, 5
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %add
  store i32 %M, ptr %arrayidx, align 4, !tbaa !1
  %add2 = add nsw i32 %N, 6
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i32 %add2
  store i32 %M, ptr %arrayidx3, align 4, !tbaa !1
  %add4 = add nsw i32 %N, 35
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i32 %add4
  store i32 %add, ptr %arrayidx5, align 4, !tbaa !1
  %arrayidx8 = getelementptr inbounds [50 x i32], ptr %B, i32 %add, i32 %add
  store i32 %add, ptr %arrayidx8, align 4, !tbaa !1
  %inc = add nsw i32 %N, 6
  %arrayidx8.1 = getelementptr inbounds [50 x i32], ptr %B, i32 %add, i32 %inc
  store i32 %add, ptr %arrayidx8.1, align 4, !tbaa !1
  %sub = add nsw i32 %N, 4
  %arrayidx10 = getelementptr inbounds [50 x i32], ptr %B, i32 %add, i32 %sub
  %0 = load i32, ptr %arrayidx10, align 4, !tbaa !1
  %add11 = add nsw i32 %0, 1
  store i32 %add11, ptr %arrayidx10, align 4, !tbaa !1
  %1 = load i32, ptr %arrayidx, align 4, !tbaa !1
  %add13 = add nsw i32 %N, 25
  %arrayidx15 = getelementptr inbounds [50 x i32], ptr %B, i32 %add13, i32 %add
  store i32 %1, ptr %arrayidx15, align 4, !tbaa !1
  store i32 5, ptr @G, align 4, !tbaa !1
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }

!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
