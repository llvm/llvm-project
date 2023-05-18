; This file is used with module-flags-3-a.ll
; RUN: true

!0 = !{i32 3, !"foo", !{!"bar", i32 42}}
!1 = !{i32 1, !"bar", i32 42}

!llvm.module.flags = !{ !0, !1 }
