; RUN: llvm-as -opaque-pointers=0 < %s > %t.typed.bc
; RUN: llvm-as -opaque-pointers=1 < %s > %t.opaque.bc
; RUN: llvm-ar cr %t.a %t.typed.bc %t.opaque.bc
; RUN: llvm-nm --just-symbol-name %t.a | FileCheck %s

; CHECK-LABEL: typed.bc:
; CHECK: test
; CHECK-LABEL: opaque.bc:
; CHECK: test

define void @test() {
  ret void
}
