; RUN: opt -Os -march=hexagon -S < %s | FileCheck %s
; Don't reset the alignment on the struct to 1.
; CHECK: align 4

target triple = "hexagon"

%s.0 = type <{ i32, [2 x i8], [2 x i8] }>

; Function Attrs: nounwind optsize
define i32 @f0(i32 %a0) #0 {
b0:
  %v0 = inttoptr i32 %a0 to %s.0*
  %v1 = getelementptr inbounds %s.0, %s.0* %v0, i32 0, i32 0
  %v2 = load volatile i32, i32* %v1, align 4, !tbaa !0
  ret i32 %v2
}

attributes #0 = { nounwind optsize }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
