; RUN: llc < %s -march=cellspu > %t1.s
; RUN: grep xswd	     %t1.s | count 3
; RUN: grep xsbh	     %t1.s | count 1
; RUN: grep xshw	     %t1.s | count 2
; RUN: grep shufb        %t1.s | count 7
; RUN: grep cg           %t1.s | count 4
; RUN: grep addx         %t1.s | count 4
; RUN: grep fsmbi        %t1.s | count 3
; RUN: grep il           %t1.s | count 2
; RUN: grep mpy          %t1.s | count 10
; RUN: grep mpyh         %t1.s | count 6
; RUN: grep mpyhhu       %t1.s | count 2
; RUN: grep mpyu         %t1.s | count 4

; ModuleID = 'stores.bc'
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define i64 @sext_i64_i8(i8 %a) nounwind {
  %1 = sext i8 %a to i64
  ret i64 %1
}

define i64 @sext_i64_i16(i16 %a) nounwind {
  %1 = sext i16 %a to i64
  ret i64 %1
}

define i64 @sext_i64_i32(i32 %a) nounwind {
  %1 = sext i32 %a to i64
  ret i64 %1
}

define i64 @zext_i64_i8(i8 %a) nounwind {
  %1 = zext i8 %a to i64
  ret i64 %1
}

define i64 @zext_i64_i16(i16 %a) nounwind {
  %1 = zext i16 %a to i64
  ret i64 %1
}

define i64 @zext_i64_i32(i32 %a) nounwind {
  %1 = zext i32 %a to i64
  ret i64 %1
}

define i64 @add_i64(i64 %a, i64 %b) nounwind {
  %1 = add i64 %a, %b
  ret i64 %1
}

define i64 @mul_i64(i64 %a, i64 %b) nounwind {
  %1 = mul i64 %a, %b
  ret i64 %1
}
