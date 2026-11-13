; RUN: llc < %s -mtriple=x86_64-pc-linux-gnu | FileCheck %s

; Test for vector floating point extension in inline assembly.
; gh184180

define <2 x double> @test(double %0) {
L.entry:
; CHECK-LABEL: test:
; CHECK:       # %bb.0:
; CHECK-NEXT:    pushq %rax
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    callq __extenddftf2@PLT
; CHECK-NEXT:    #APP
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:    popq %rax
; CHECK-NEXT:    .cfi_def_cfa_offset 8
; CHECK-NEXT:    retq
  %1 = call <2 x double> asm sideeffect "", "=x,0"(double %0)
  ret <2 x double> %1
}
