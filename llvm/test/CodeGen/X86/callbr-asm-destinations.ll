; RUN: not llc -mtriple=i686-- < %s 2> %t
; RUN: FileCheck %s < %t

; CHECK: Duplicate callbr destination

; A test for asm-goto duplicate labels limitation

define i32 @test(i32 %a) {
entry:
  %0 = add i32 %a, 4
  callbr void asm "xorl $0, $0; jmp ${1:l}", "r,i,~{dirflag},~{fpsr},~{flags}"(i32 %0, ptr blockaddress(@test, %fail)) to label %fail [label %fail]

fail:
  ret i32 1
}
