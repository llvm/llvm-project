; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10}

!0 = !{}
!1 = !{}   ; This should merge with !0.
!2 = !{!0}
!3 = !{!0} ; This should merge with !2.
!4 = distinct !{}
!5 = distinct !{}
!6 = distinct !{!0}
!7 = distinct !{!0}
!8 = distinct !{!8}
!9 = distinct !{!9}
!10 = !{!10} ; This should become distinct.

; CHECK: !named = !{!0, !0, !1, !1, !2, !3, !4, !5, !6, !7, !8}
; CHECK:      !0 = !{}
; CHECK-NEXT: !1 = !{!0}
; CHECK-NEXT: !2 = distinct !{}
; CHECK-NEXT: !3 = distinct !{}
; CHECK-NEXT: !4 = distinct !{!0}
; CHECK-NEXT: !5 = distinct !{!0}
; CHECK-NEXT: !6 = distinct !{!6}
; CHECK-NEXT: !7 = distinct !{!7}
; CHECK-NEXT: !8 = distinct !{!8}
; CHECK-NOT:  !
