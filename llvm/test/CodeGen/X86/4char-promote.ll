; A test for checking PR 9623
; RUN: llc -march=x86-64 -mcpu=corei7 < %s | FileCheck %s

target triple = "x86_64-apple-darwin"

; CHECK:  pmulld
; CHECK:  paddd
; CHECK-NOT:  movdqa
; CHECK:  ret

define <4 x i8> @foo(<4 x i8> %x, <4 x i8> %y) {
entry:
 %binop = mul <4 x i8> %x, %y
 %binop6 = add <4 x i8> %binop, %x
 ret <4 x i8> %binop6
}


