; RUN: llc -mtriple=i686-- < %s | FileCheck %s

define i32 @test1(i32 %x) {
  %xor = xor i32 %x, 31
  %sub = sub i32 32, %xor
  ret i32 %sub
; CHECK-LABEL: test1:
; CHECK:      xorl $-32
; CHECK-NEXT: addl $33
; CHECK-NEXT: ret
}
