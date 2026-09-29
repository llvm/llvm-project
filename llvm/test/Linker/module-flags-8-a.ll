; RUN: llvm-link %s %p/module-flags-8-b.ll -S -o - | sort | FileCheck %s

; Test append-type module flags.

; CHECK: !0 = distinct !{i32 5, !"flag-0", !1}
; CHECK: !1 = distinct !{i32 0, i32 0, i32 1}
; CHECK: !2 = distinct !{i32 6, !"flag-1", !3}
; CHECK: !3 = distinct !{i32 0, i32 1, i32 2}
; CHECK: !llvm.module.flags = !{!0, !2}

!0 = !{ i32 5, !"flag-0", !{ i32 0 } }
!1 = !{ i32 6, !"flag-1", !{ i32 0, i32 1 } }

!llvm.module.flags = !{ !0, !1 }
