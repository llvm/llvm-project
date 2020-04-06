; RUN: llc -mtriple x86_64-unknown-linux-gnu %s -o - | FileCheck %s

$comdat_func = comdat any

; CHECK-LABEL: func2:
; CHECK-NEXT: .Lfunc2$local

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
