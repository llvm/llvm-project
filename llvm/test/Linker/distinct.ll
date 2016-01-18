; RUN: llvm-link %s %S/Inputs/distinct.ll -o - -S | FileCheck %s

; Test that distinct nodes from other modules remain distinct.  The @global
; cases are the most interesting, since the operands actually need to be
; remapped.

; CHECK: @global = linkonce global i32 0
@global = linkonce global i32 0

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !0, !1, !2, !9, !10, !11, !12, !13, !14}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}

; CHECK:      !0 = !{}
; CHECK-NEXT: !1 = !{!0}
; CHECK-NEXT: !2 = !{i32* @global}
; CHECK-NEXT: !3 = distinct !{}
; CHECK-NEXT: !4 = distinct !{!0}
; CHECK-NEXT: !5 = distinct !{i32* @global}
; CHECK-NEXT: !6 = !{!3}
; CHECK-NEXT: !7 = !{!4}
; CHECK-NEXT: !8 = !{!5}
; CHECK-NEXT: !9 = distinct !{}
; CHECK-NEXT: !10 = distinct !{!0}
; CHECK-NEXT: !11 = distinct !{i32* @global}
; CHECK-NEXT: !12 = !{!9}
; CHECK-NEXT: !13 = !{!10}
; CHECK-NEXT: !14 = !{!11}
; CHECK-NOT:  !
!0 = !{}
!1 = !{!0}
!2 = !{i32* @global}
!3 = distinct !{}
!4 = distinct !{!0}
!5 = distinct !{i32* @global}
!6 = !{!3}
!7 = !{!4}
!8 = !{!5}
