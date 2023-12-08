; This file is used with module-flags-6-a.ll
; RUN: true

!0 = !{ i32 5, !"flag-0", !{ i32 0, i32 1 } }
!1 = !{ i32 6, !"flag-1", !{ i32 1, i32 2 } }

!llvm.module.flags = !{ !0, !1 }
