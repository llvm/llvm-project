; RUN: llc -mtriple=aarch64-apple-ios7.0 -disable-block-placement -aarch64-tbz-offset-bits=4 -o - %s | FileCheck %s
define i32 @test_asm_length(i32 %in) {
; CHECK-LABEL: test_asm_length:

  ; It would be more natural to use just one "tbnz %false" here, but if the
  ; number of instructions in the asm is counted reasonably, that block is out
  ; of the limited range we gave tbz. So branch relaxation has to invert the
  ; condition.
; CHECK:     tbz w0, #0, [[TRUE:LBB[0-9]+_[0-9]+]]
; CHECK:     b [[FALSE:LBB[0-9]+_[0-9]+]]

; CHECK: [[TRUE]]:
; CHECK:     mov w0, #4
; CHECK:     nop
; CHECK:     nop
; CHECK:     nop
; CHECK:     nop
; CHECK:     nop
; CHECK:     nop
; CHECK:     ret

; CHECK: [[FALSE]]:
; CHECK:     ret

  %val = and i32 %in, 1
  %tst = icmp eq i32 %val, 0
  br i1 %tst, label %true, label %false

true:
  call void asm sideeffect "nop\0A\09nop\0A\09nop\0A\09nop\0A\09nop\0A\09nop", ""()
  ret i32 4

false:
  ret i32 0
}
