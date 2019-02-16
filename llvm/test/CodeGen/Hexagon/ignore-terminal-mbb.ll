; RUN: llc -march=hexagon -hexagon-initial-cfg-cleanup=0 < %s | FileCheck %s

; CHECK-NOT: if{{.*}}jump{{.*}}-1
; CHECK: memw

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  br i1 undef, label %b2, label %b1

b1:                                               ; preds = %b0
  store i32 0, i32* undef, align 4, !tbaa !0
  unreachable

b2:                                               ; preds = %b0
  br i1 undef, label %b4, label %b3

b3:                                               ; preds = %b2
  %v0 = or i32 undef, 2048
  br label %b4

b4:                                               ; preds = %b3, %b2
  ret void
}

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
