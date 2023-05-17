; RUN: llc -march=hexagon < %s | FileCheck %s

; This test checks to see if, after lowering the two loads below, we set up the
; memrefs of the resulting load MIs correctly, so that they are packetized
; together.

; CHECK: {
; CHECK:       r{{[0-9]*}} = memw(r1{{[678]}}+#0)
; CHECK-NEXT:  r{{[0-9]*}} = memw(r1{{[678]}}+#0)

; Function Attrs: nounwind
define i64 @f0(ptr nocapture %a0, ptr nocapture %a1, ptr nocapture %a2) #0 {
b0:
  %v0 = tail call i32 @f1() #0
  store i32 %v0, ptr %a2, align 4, !tbaa !0
  %v1 = load i32, ptr %a0, align 4, !tbaa !0
  %v2 = sext i32 %v1 to i64
  %v3 = load i32, ptr %a1, align 4, !tbaa !0
  %v4 = sext i32 %v3 to i64
  %v5 = mul nsw i64 %v4, %v2
  ret i64 %v5
}

declare i32 @f1(...)

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
