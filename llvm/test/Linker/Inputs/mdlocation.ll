!named = !{!0, !1, !2, !3, !4, !5}

!0 = distinct !DISubprogram() ; Use this as a scope.
!1 = !DILocation(line: 3, column: 7, scope: !0)
!2 = !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !1)
!3 = !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !2)
; Test distinct nodes.
!4 = distinct !DILocation(line: 3, column: 7, scope: !0)
!5 = distinct !DILocation(line: 3, column: 7, scope: !0, inlinedAt: !4)
