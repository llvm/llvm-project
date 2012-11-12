; This file is used with module-flags-1-a.ll
; RUN: true

!0 = metadata !{ i32 1, metadata !"foo", i32 37 }
!1 = metadata !{ i32 1, metadata !"qux", i32 42 }
!2 = metadata !{ i32 1, metadata !"mux", metadata !{ metadata !"hello world", i32 927 } }

!llvm.module.flags = !{ !0, !1, !2 }
