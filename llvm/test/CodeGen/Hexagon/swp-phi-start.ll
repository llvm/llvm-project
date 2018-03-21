; RUN: llc -march=hexagon -enable-pipeliner -pipeliner-max-stages=2 -disable-packetizer < %s | FileCheck %s

; Test that the early start and late start values are computed correctly
; when a Phi depends on another Phi. In this case, they should occur in
; the same stage.

; CHECK-DAG: [[REG3:(r[0-9]+)]] = add([[REG1:(r[0-9]+)]],#-1)
; CHECK-DAG: [[REG2:(r[0-9]+)]] = add([[REG1]],#-1)
; CHECK-DAG: loop0(.LBB0_[[LOOP:.]],[[REG3]])
; CHECK-NOT: = [[REG2]]
; CHECK: .LBB0_[[LOOP]]:
; CHECK: }{{[ \t]*}}:endloop

; Function Attrs: nounwind
define void @f0(i32 %a0, i16* nocapture %a1) #0 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  %v0 = add nsw i32 undef, -8
  br i1 undef, label %b3, label %b2

b2:                                               ; preds = %b2, %b1, %b0
  %v1 = phi i32 [ %v7, %b2 ], [ undef, %b0 ], [ %v0, %b1 ]
  %v2 = phi i32 [ %v1, %b2 ], [ %a0, %b0 ], [ undef, %b1 ]
  %v3 = add nsw i32 %v2, -2
  %v4 = getelementptr inbounds i16, i16* %a1, i32 %v3
  %v5 = load i16, i16* %v4, align 2, !tbaa !0
  %v6 = getelementptr inbounds i16, i16* %a1, i32 %v1
  store i16 %v5, i16* %v6, align 2, !tbaa !0
  %v7 = add nsw i32 %v1, -1
  %v8 = icmp sgt i32 %v7, 0
  br i1 %v8, label %b2, label %b3

b3:                                               ; preds = %b2, %b1
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"short", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
