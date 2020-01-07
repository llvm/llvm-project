; RUN: not llc -mtriple=i686-- < %s 2> %t
; RUN: FileCheck %s < %t

; CHECK: error: asm-goto outputs not supported

; A test for asm-goto output prohibition

define i32 @test(i32 %a) {
entry:
  %0 = add i32 %a, 4
  %1 = callbr i32 asm "xorl $1, $1; jmp ${1:l}", "=&r,r,X,~{dirflag},~{fpsr},~{flags}"(i32 %0, i8* blockaddress(@test, %fail)) to label %normal [label %fail]

normal:
  ret i32 %1

fail:
  ret i32 1
}
