; RUN: llvm-as < %s > %t.opaque.bc
; RUN: llvm-ar cr %t.a %S/Inputs/typed.bc %t.opaque.bc
; RUN: llvm-nm --just-symbol-name %t.a | FileCheck %s

; CHECK-LABEL: typed.bc:
; CHECK: test
; CHECK-LABEL: opaque.bc:
; CHECK: test

define void @test() {
  ret void
}
