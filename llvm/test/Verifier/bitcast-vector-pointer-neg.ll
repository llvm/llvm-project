; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n8:16:32"

; CHECK: error: invalid cast opcode for cast from '<2 x i32*>' to 'i64*'

define i64* @illegal_bitcast_vector_of_pointers_to_pointer(<2 x i32*> %a) {
  %b = bitcast <2 x i32*> %a to i64*
  ret i64* %b
}
