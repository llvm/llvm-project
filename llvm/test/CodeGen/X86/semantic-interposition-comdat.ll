; RUN: llc -mtriple=x86_64 -relocation-model=pic < %s | FileCheck %s

$comdat_func = comdat any

; CHECK-LABEL: func2:
; CHECK-NOT: .Lfunc2$local

declare void @func()

define hidden void @func2() {
entry:
  call void @func()
  ret void
}

; CHECK: comdat_func:
; CHECK-NOT: .Lcomdat_func$local

define hidden void @comdat_func() comdat {
entry:
  call void @func()
  ret void
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"SemanticInterposition", i32 0}
!1 = !{i32 7, !"PIC Level", i32 2}
