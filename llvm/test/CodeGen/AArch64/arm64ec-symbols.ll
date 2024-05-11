; RUN: llc -mtriple=arm64ec-pc-windows-msvc < %s | FileCheck %s

declare void @func() nounwind;

define void @caller() nounwind {
  call void @func()
  ret void
}

; CHECK:      .def    caller;
; CHECK-NEXT: .type   32;
; CHECK-NEXT: .endef
; CHECK-NEXT: .weak_anti_dep  caller
; CHECK-NEXT: .set caller, "#caller"@WEAKREF

; CHECK:      .def    func;
; CHECK-NEXT: .type   32;
; CHECK-NEXT: .endef
; CHECK-NEXT: .weak_anti_dep  func
; CHECK-NEXT: .set func, "#func"@WEAKREF
; CHECK-NEXT: .def    "#func";
; CHECK-NEXT: .type   32;
; CHECK-NEXT: .endef
; CHECK-NEXT: .weak_anti_dep  "#func"
; CHECK-NEXT: .set "#func", "#func$exit_thunk"@WEAKREF
