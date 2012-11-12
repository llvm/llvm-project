; RUN: llc --march=cellspu %s -o - | FileCheck %s

; signed division rounds towards zero, rotma don't.
define i32 @sdivide (i32 %val )
{
; CHECK: rotmai
; CHECK: rotmi
; CHECK: a
; CHECK: rotmai
; CHECK: bi $lr
   %rv = sdiv i32 %val, 4
   ret i32 %rv
}

define i32 @udivide (i32 %val )
{
; CHECK: rotmi
; CHECK: bi $lr
   %rv = udiv i32 %val, 4
   ret i32 %rv
}

