; RUN: llvm-link -o - -S %s | FileCheck %s
; Crasher for PR22456: MapMetadata() should resolve all cycles.

; CHECK: !named = !{!0}
!named = !{!0}

; CHECK: !0 = distinct !{!1}
!0 = distinct !{!1}

; CHECK-NEXT: !1 = !{!2}
; CHECK-NEXT: !2 = !{!1}
!1 = !{!2}
!2 = !{!1}
