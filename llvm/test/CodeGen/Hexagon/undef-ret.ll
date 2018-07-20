; RUN: llc -march=hexagon -O3 < %s | FileCheck %s
; Check for direct use of r0 in addadd.
; CHECK: = add(r0,add(r1,#2))

; Function Attrs: nounwind
define i32 @f0(i32 %a0, i32 %a1, i32* nocapture %a2) #0 {
b0:
  %v0 = add nsw i32 %a0, 2
  %v1 = add nsw i32 %v0, %a1
  store i32 %v1, i32* %a2, align 4, !tbaa !0
  ret i32 undef
}

attributes #0 = { nounwind "target-cpu"="hexagonv55" }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
