; RUN: llc -mcpu=cortex-a8 -mtriple=thumbv7-eabi -float-abi=hard < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

define <4 x i64> @f_4_i64(<4 x i64> %a, <4 x i64> %b) nounwind {
; CHECK: vadd.i64
 %y = add <4 x i64> %a, %b
 ret <4 x i64> %y
}
