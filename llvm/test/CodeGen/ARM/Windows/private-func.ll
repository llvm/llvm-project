; RUN: llc -mtriple thumbv7-windows -filetype asm -o - %s | FileCheck %s

define dso_local void @func1() {
entry:
  call void @func2()
  ret void
}

define private void @func2() {
entry:
  ret void
}

; CHECK:      .def    .Lfunc2;
; CHECK-NEXT: .scl    3;
; CHECK-NEXT: .type   32;
; CHECK-NEXT: .endef
