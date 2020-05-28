; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: !named = !{!0, !0, !1, !2}
!named = !{!0, !1, !2, !3}

; CHECK:      !0 = !DISubrange(count: -1, lowerBound: 0)
; CHECK-NEXT: !1 = !DISubrange(count: -1, lowerBound: 4)
; CHECK-NEXT: !2 = !DISubrange(count: -1, lowerBound: -5)
!0 = !DISubrange(count: -1, lowerBound: 0)
!1 = !DISubrange(count: -1, lowerBound: 0)

!2 = !DISubrange(count: -1, lowerBound: 4)
!3 = !DISubrange(count: -1, lowerBound: -5)
