; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: asl

; Function Attrs: nounwind
define fastcc void @f0() #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  br i1 undef, label %b2, label %b1

b2:                                               ; preds = %b1
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v0 = load i64, i64* undef, align 8, !tbaa !0
  %v1 = lshr i64 %v0, 8
  %v2 = shl i64 %v0, 56
  %v3 = or i64 %v2, %v1
  %v4 = xor i64 %v3, 0
  %v5 = xor i64 %v4, 0
  %v6 = add i64 0, %v5
  store i64 %v6, i64* undef, align 8, !tbaa !0
  br label %b3
}

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"long long", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
