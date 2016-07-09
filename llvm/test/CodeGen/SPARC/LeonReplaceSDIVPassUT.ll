; RUN: llc %s -O0 -march=sparc -mcpu=at697e -o - | FileCheck %s

; CHECK: sdivcc %o0, %o1, %o0

define i32 @lbr59(i32 %a, i32 %b)
{
  %r = sdiv i32 %a, %b
  ret i32 %r 
}
