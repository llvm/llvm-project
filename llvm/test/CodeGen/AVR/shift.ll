; RUN: llc < %s -march=avr | FileCheck %s

; CHECK-LABEL: shift_i64_i64
define i64 @shift_i64_i64(i64 %a, i64 %b) {
  ; CHECK: call    __ashldi3
  %result = shl i64 %a, %b
  ret i64 %result
}
