; RUN: llc -march=hexagon < %s | FileCheck %s

target triple = "hexagon"

; CHECK-LABEL: f0:
; CHECK: memw({{.*}}) = clrbit(#18)
define void @f0(ptr nocapture %a0) #0 {
b0:
  %v0 = load i32, ptr %a0, align 4, !tbaa !0
  %v1 = and i32 %v0, -262145
  store i32 %v1, ptr %a0, align 4, !tbaa !0
  ret void
}

attributes #0 = { norecurse nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
