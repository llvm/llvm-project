; RUN: llc -mtriple=x86_64-windows-gnu -o - %s | FileCheck %s

%struct.MyStruct = type { i8 }

@_ZN8MyStructC1Ev = dso_local alias void (%struct.MyStruct*), void (%struct.MyStruct*)* @_ZN8MyStructC2Ev

define dso_local void @_ZN8MyStructC2Ev(%struct.MyStruct* %this) {
entry:
  ret void
}

; CHECK:      .def     _ZN8MyStructC2Ev
; CHECK-NEXT: .scl     2
; CHECK-NEXT: .type    32
; CHECK-NEXT: .endef
; CHECK-NEXT: .globl   _ZN8MyStructC2Ev
; CHECK:      {{^}}_ZN8MyStructC2Ev:

; CHECK:      .globl   _ZN8MyStructC1Ev
; CHECK-NEXT: .def     _ZN8MyStructC1Ev
; CHECK-NEXT: .scl     2
; CHECK-NEXT: .type    32
; CHECK-NEXT: .endef
; CHECK-NEXT: .set     _ZN8MyStructC1Ev, _ZN8MyStructC2Ev
