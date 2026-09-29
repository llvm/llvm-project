; This file is used with module-flags-5-a.ll
; RUN: true

!0 = !{ i32 4, !"foo", i32 37 } ; Override the "foo" value.

!llvm.module.flags = !{ !0 }
