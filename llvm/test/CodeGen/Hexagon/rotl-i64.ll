; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: rol

; Function Attrs: nounwind
define fastcc void @f0(i64* %a0) #0 {
b0:                                               ; preds = %b3, %b2
  %v0 = load i64, i64* %a0, align 8, !tbaa !0
  %v1 = lshr i64 %v0, 8
  %v2 = shl i64 %v0, 56
  %v3 = or i64 %v2, %v1
  %v4 = xor i64 %v3, 0
  %v5 = xor i64 %v4, 0
  %v6 = add i64 0, %v5
  store i64 %v6, i64* %a0, align 8, !tbaa !0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }

!0 = !{!1, !1, i64 0}
!1 = !{!"long long", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
