; RUN: llc < %s -march=x86-64 | FileCheck %s
; Make sure none of these crash, and that the power-of-two transformations
; trigger correctly.

define i128 @test1(i128 %x) {
  ; CHECK: test1:
  ; CHECK-NOT: call
  %tmp = sdiv i128 %x, 73786976294838206464
  ret i128 %tmp
}

define i128 @test2(i128 %x) {
  ; CHECK: test2:
  ; CHECK-NOT: call
  %tmp = sdiv i128 %x, -73786976294838206464
  ret i128 %tmp
}

define i128 @test3(i128 %x) {
  ; CHECK: test3:
  ; CHECK: call
  %tmp = sdiv i128 %x, -73786976294838206467
  ret i128 %tmp
}
