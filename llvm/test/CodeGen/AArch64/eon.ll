; RUN: llc -mtriple=aarch64-none-linux-gnu < %s | FileCheck %s
; RUN: llc %s -pass-remarks-missed=gisel* -mtriple=aarch64-none-linux-gnu -global-isel -o - 2>&1 | FileCheck %s

; CHECK-NOT: remark

; Check that the eon instruction is generated instead of eor,movn
define i64 @test1(i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: test1:
; CHECK: eon
; CHECK: ret
entry:
  %shl = shl i64 %b, 4
  %neg = xor i64 %a, -1
  %xor = xor i64 %shl, %neg
  ret i64 %xor
}

; Same check with multiple uses of %neg
define i64 @test2(i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: test2:
; CHECK: eon
; CHECK: eon
; CHECK: lsl
; CHECK: ret
entry:
  %shl = shl i64 %b, 4
  %neg = xor i64 %shl, -1
  %xor = xor i64 %neg, %a
  %xor1 = xor i64 %c, %neg
  %shl2 = shl i64 %xor, %xor1
  ret i64 %shl2
}
