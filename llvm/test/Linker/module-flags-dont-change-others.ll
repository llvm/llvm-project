; RUN: llvm-link %s %S/Inputs/module-flags-dont-change-others.ll -S -o - | FileCheck %s

; Test that module-flag linking doesn't change other metadata.  In particular,
; !named should still point at the unmodified tuples (!3, !4, and !5) that
; happen to also serve as module flags.

; CHECK: !named = !{!0, !1, !2, !3, !4, !5}
; CHECK: !llvm.module.flags = !{!6, !7, !9}
!named = !{!0, !1, !2, !3, !4, !5}
!llvm.module.flags = !{!3, !4, !5}

; CHECK: !0 = !{}
; CHECK: !1 = !{!0}
; CHECK: !2 = !{!0, !1}
; CHECK: !3 = !{i32 1, !"foo", i32 927}
; CHECK: !4 = !{i32 5, !"bar", !0}
; CHECK: !5 = !{i32 6, !"baz", !1}
; CHECK: !6 = !{i32 4, !"foo", i32 37}
; CHECK: !7 = distinct !{i32 5, !"bar", !8}
; CHECK: !8 = distinct !{!0}
; CHECK: !9 = distinct !{i32 6, !"baz", !10}
; CHECK: !10 = distinct !{!0, !1}
!0 = !{}
!1 = !{!0}
!2 = !{!0, !1}
!3 = !{i32 1, !"foo", i32 927}
!4 = !{i32 5, !"bar", !0}
!5 = !{i32 6, !"baz", !1}
