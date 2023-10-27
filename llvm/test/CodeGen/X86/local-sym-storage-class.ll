; RUN: llc -mtriple x86_64-unknown-windows-msvc %s -o - | FileCheck %s

define internal void @internal() {
  ret void
}

define private void @private() {
  ret void
}

; Check that the internal and private linkage symbols have IMAGE_SYM_CLASS_STATIC (3).
; CHECK: .def    internal;
; CHECK: .scl    3;
; CHECK: .def    .Lprivate;
; CHECK: .scl    3;
