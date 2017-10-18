; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-DAG: r[[BASE:[0-9]+]] += add
; CHECK-DAG: r[[IDX0:[0-9]+]] = add(r2,#5)
; CHECK-DAG: r[[IDX1:[0-9]+]] = add(r2,#6)
; CHECK-DAG: memw(r0+r[[IDX0]]<<#2) = r3
; CHECK-DAG: memw(r0+r[[IDX1]]<<#2) = r3
; CHECK-DAG: memw(r[[BASE]]+r[[IDX0]]<<#2) = r[[IDX0]]
; CHECK-DAG: memw(r[[BASE]]+r[[IDX1]]<<#2) = r[[IDX0]]

target triple = "hexagon"

@G = external global i32, align 4

; Function Attrs: norecurse nounwind
define void @fred(i32* nocapture %A, [50 x i32]* nocapture %B, i32 %N, i32 %M) #0 {
entry:
  %add = add nsw i32 %N, 5
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %add
  store i32 %M, i32* %arrayidx, align 4, !tbaa !1
  %add2 = add nsw i32 %N, 6
  %arrayidx3 = getelementptr inbounds i32, i32* %A, i32 %add2
  store i32 %M, i32* %arrayidx3, align 4, !tbaa !1
  %add4 = add nsw i32 %N, 35
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i32 %add4
  store i32 %add, i32* %arrayidx5, align 4, !tbaa !1
  %arrayidx8 = getelementptr inbounds [50 x i32], [50 x i32]* %B, i32 %add, i32 %add
  store i32 %add, i32* %arrayidx8, align 4, !tbaa !1
  %inc = add nsw i32 %N, 6
  %arrayidx8.1 = getelementptr inbounds [50 x i32], [50 x i32]* %B, i32 %add, i32 %inc
  store i32 %add, i32* %arrayidx8.1, align 4, !tbaa !1
  %sub = add nsw i32 %N, 4
  %arrayidx10 = getelementptr inbounds [50 x i32], [50 x i32]* %B, i32 %add, i32 %sub
  %0 = load i32, i32* %arrayidx10, align 4, !tbaa !1
  %add11 = add nsw i32 %0, 1
  store i32 %add11, i32* %arrayidx10, align 4, !tbaa !1
  %1 = load i32, i32* %arrayidx, align 4, !tbaa !1
  %add13 = add nsw i32 %N, 25
  %arrayidx15 = getelementptr inbounds [50 x i32], [50 x i32]* %B, i32 %add13, i32 %add
  store i32 %1, i32* %arrayidx15, align 4, !tbaa !1
  store i32 5, i32* @G, align 4, !tbaa !1
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }

!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
