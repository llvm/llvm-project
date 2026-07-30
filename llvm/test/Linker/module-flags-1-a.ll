; RUN: llvm-link %s %p/module-flags-1-b.ll -S -o - | sort | FileCheck %s

; Test basic functionality of module flags.

; CHECK: !0 = !{i32 1, !"foo", i32 37}
; CHECK: !1 = !{i32 2, !"bar", i32 42}
; CHECK: !2 = !{i32 1, !"mux", !3}
; CHECK: !3 = !{!"hello world", i32 927}
; CHECK: !4 = !{i32 1, !"qux", i32 42}
; CHECK: !llvm.module.flags = !{!0, !1, !2, !4}

!0 = !{ i32 1, !"foo", i32 37 }
!1 = !{ i32 2, !"bar", i32 42 }
!2 = !{ i32 1, !"mux", !{ !"hello world", i32 927 } }

!llvm.module.flags = !{ !0, !1, !2 }
