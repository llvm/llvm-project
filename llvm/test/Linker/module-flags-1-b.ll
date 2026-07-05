; This file is used with module-flags-1-a.ll
; RUN: true

!0 = !{ i32 1, !"foo", i32 37 }
!1 = !{ i32 1, !"qux", i32 42 }
!2 = !{ i32 1, !"mux", !{ !"hello world", i32 927 } }

!llvm.module.flags = !{ !0, !1, !2 }
