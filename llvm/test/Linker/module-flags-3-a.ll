; RUN: llvm-link %s %p/module-flags-3-b.ll -S -o - | sort | FileCheck %s

; Test 'require' behavior.

; CHECK: !0 = !{i32 1, !"foo", i32 37}
; CHECK: !1 = !{i32 1, !"bar", i32 42}
; CHECK: !2 = !{i32 3, !"foo", !3}
; CHECK: !3 = !{!"bar", i32 42}
; CHECK: !llvm.module.flags = !{!0, !1, !2}

!0 = !{ i32 1, !"foo", i32 37 }
!1 = !{ i32 1, !"bar", i32 42 }

!llvm.module.flags = !{ !0, !1 }
