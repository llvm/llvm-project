; RUN: llc -relocation-model=pic -verify-machineinstrs -mcpu=pwr7 -O0 -code-model=medium < %s | FileCheck %s
; RUN: llc -relocation-model=pic -verify-machineinstrs -mcpu=pwr7 -O0 -code-model=large < %s | FileCheck %s

; Test correct code generation for medium and large code model
; for loading and storing a tentatively defined variable.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@ti = common global i32 0, align 4

define signext i32 @test_tentative() nounwind {
entry:
  %0 = load i32, ptr @ti, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @ti, align 4
  ret i32 %0
}

; CHECK-LABEL: test_tentative:
; CHECK: addis [[REG1:[0-9]+]], 2, .LC[[TOCNUM:[0-9]+]]@toc@ha
; CHECK: ld [[REG2:[0-9]+]], .LC[[TOCNUM]]@toc@l([[REG1]])
; CHECK: lwz {{[0-9]+}}, 0([[REG2]])
; CHECK: addis [[REG3:[0-9]+]], 2, .LC[[TOCNUM]]@toc@ha
; CHECK: ld [[REG4:[0-9]+]], .LC[[TOCNUM]]@toc@l([[REG3]])
; CHECK: stw {{[0-9]+}}, 0([[REG4]])
; CHECK: .comm ti,4,4
; CHECK:      .section .toc,"aw",@progbits
; CHECK-NEXT: .LC[[TOCNUM]]:
; CHECK-NEXT: .tc ti[TC],ti
