; RUN: llvm-as < %s | opt -instcombine | llvm-dis > %t
; RUN: grep urem %t | count 3
; RUN: grep srem %t | count 1
; RUN: grep sub %t | count 2
; RUN: grep add %t | count 1
; PR3103

define i8 @test1(i8 %x, i8 %y) {
  %A = udiv i8 %x, %y
  %B = mul i8 %A, %y
  %C = sub i8 %x, %B
  ret i8 %C
}

define i8 @test2(i8 %x, i8 %y) {
  %A = sdiv i8 %x, %y
  %B = mul i8 %A, %y
  %C = sub i8 %x, %B
  ret i8 %C
}

define i8 @test3(i8 %x, i8 %y) {
  %A = udiv i8 %x, %y
  %B = mul i8 %A, %y
  %C = sub i8 %B, %x
  ret i8 %C
}

define i8 @test4(i8 %x) {
  %A = udiv i8 %x, 3
  %B = mul i8 %A, -3
  %C = sub i8 %x, %B
  ret i8 %C
}
