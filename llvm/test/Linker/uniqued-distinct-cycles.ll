; RUN: llvm-link -o - %s | llvm-dis | FileCheck %s

; CHECK: !named = !{!0, !2}
!named = !{!0, !2}

; CHECK:      !0 = !{!1}
; CHECK-NEXT: !1 = distinct !{!0}
!0 = !{!1}
!1 = distinct !{!0}

; CHECK-NEXT: !2 = distinct !{!3}
; CHECK-NEXT: !3 = !{!2}
!2 = distinct !{!3}
!3 = !{!2}
