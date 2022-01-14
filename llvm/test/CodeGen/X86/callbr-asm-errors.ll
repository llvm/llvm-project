; RUN: not llc -mtriple=i686-- < %s 2> %t
; RUN: FileCheck %s < %t

; CHECK: Duplicate callbr destination

; A test for asm-goto duplicate labels limitation

define i32 @test(i32 %a) {
entry:
  %0 = add i32 %a, 4
  callbr void asm "xorl $0, $0; jmp ${1:l}", "r,i,i,~{dirflag},~{fpsr},~{flags}"(i32 %0, i8* blockaddress(@test, %fail), i8* blockaddress(@test, %fail)) to label %normal [label %fail, label %fail]

normal:
  ret i32 %0

fail:
  ret i32 1
}
