; RUN: llvm-link %s %S/Inputs/mdlocation.ll -o - -S | FileCheck %s

; Test that DILocations are remapped properly.

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11}
!named = !{!0, !1, !2, !3, !4, !5}

; CHECK:      !0 = distinct !DISubprogram(
; CHECK-NEXT: !1 = !DILocation(line: 3, column: 7, scope: !0)
; CHECK-NEXT: !2 = !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !1)
; CHECK-NEXT: !3 = !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !2)
; CHECK-NEXT: !4 = distinct !DILocation(line: 3, column: 7, scope: !0)
; CHECK-NEXT: !5 = distinct !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !4)
; CHECK-NEXT: !6 = distinct !DISubprogram(
; CHECK-NEXT: !7 = !DILocation(line: 3, column: 7, scope: !6)
; CHECK-NEXT: !8 = !DILocation(line: 3, column: 7, scope: !6, inlinedAt: !7)
; CHECK-NEXT: !9 = !DILocation(line: 3, column: 7, scope: !6, inlinedAt: !8)
; CHECK-NEXT: !10 = distinct !DILocation(line: 3, column: 7, scope: !6)
; CHECK-NEXT: !11 = distinct !DILocation(line: 3, column: 7, scope: !6, inlinedAt: !10)
!0 = distinct !DISubprogram() ; Use this as a scope.
!1 = !DILocation(line: 3, column: 7, scope: !0)
!2 = !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !1)
!3 = !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !2)
; Test distinct nodes.
!4 = distinct !DILocation(line: 3, column: 7, scope: !0)
!5 = distinct !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !4)
