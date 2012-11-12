; This file is used with module-flags-3-a.ll
; RUN: true

!0 = metadata !{ i32 3, metadata !"foo",
  metadata !{ metadata !"bar", i32 42 }
}

!llvm.module.flags = !{ !0 }
