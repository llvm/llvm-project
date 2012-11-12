; RUN: llc < %s -march=cellspu > %t1.s

; ModuleID = 'i8ops.bc'
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define i8 @add_i8(i8 %a, i8 %b) nounwind {
  %1 = add i8 %a, %b
  ret i8 %1
}

define i8 @add_i8_imm(i8 %a, i8 %b) nounwind {
  %1 = add i8 %a, 15 
  ret i8 %1
}

define i8 @sub_i8(i8 %a, i8 %b) nounwind {
  %1 = sub i8 %a, %b
  ret i8 %1
}

define i8 @sub_i8_imm(i8 %a, i8 %b) nounwind {
  %1 = sub i8 %a, 15 
  ret i8 %1
}
