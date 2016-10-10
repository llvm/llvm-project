; RUN: llc %s -O0 -march=sparc -o - | FileCheck %s -check-prefix=NO_REPLACE_SDIV
; RUN: llc %s -O0 -march=sparc -mcpu=at697e -o - | FileCheck %s -check-prefix=REPLACE_SDIV

; REPLACE_SDIV: sdivcc %o0, %o1, %o0
; NO_REPLACE_SDIV: sdiv %o0, %o1, %o0

define i32 @lbr59(i32 %a, i32 %b)
{
  %r = sdiv i32 %a, %b
  ret i32 %r 
}
