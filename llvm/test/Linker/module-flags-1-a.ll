; RUN: llvm-link %s %p/module-flags-1-b.ll -S -o - | sort | FileCheck %s

; Test basic functionality of module flags.

; CHECK: !0 = metadata !{i32 1, metadata !"foo", i32 37}
; CHECK: !1 = metadata !{i32 1, metadata !"qux", i32 42}
; CHECK: !2 = metadata !{i32 1, metadata !"mux", metadata !3}
; CHECK: !3 = metadata !{metadata !"hello world", i32 927}
; CHECK: !4 = metadata !{i32 2, metadata !"bar", i32 42}
; CHECK: !llvm.module.flags = !{!0, !1, !2, !4}

!0 = metadata !{ i32 1, metadata !"foo", i32 37 }
!1 = metadata !{ i32 2, metadata !"bar", i32 42 }
!2 = metadata !{ i32 1, metadata !"mux", metadata !{ metadata !"hello world", i32 927 } }

!llvm.module.flags = !{ !0, !1, !2 }
