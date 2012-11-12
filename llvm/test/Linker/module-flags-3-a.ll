; RUN: llvm-link %s %p/module-flags-3-b.ll -S -o - | sort | FileCheck %s

; Test 'require' behavior.

; CHECK: !0 = metadata !{i32 1, metadata !"foo", i32 37}
; CHECK: !1 = metadata !{i32 3, metadata !"foo", metadata !2}
; CHECK: !2 = metadata !{metadata !"bar", i32 42}
; CHECK: !3 = metadata !{i32 1, metadata !"bar", i32 42}
; CHECK: !llvm.module.flags = !{!0, !1, !3}

!0 = metadata !{ i32 1, metadata !"foo", i32 37 }
!1 = metadata !{ i32 1, metadata !"bar", i32 42 }

!llvm.module.flags = !{ !0, !1 }
