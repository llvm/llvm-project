; RUN: llvm-as -disable-output %s

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n8:16:32"

define <1 x ptr> @vector_legal_bitcast_pointer_to_vector_of_pointers(ptr %a) {
  %b = bitcast ptr %a to <1 x ptr>
  ret <1 x ptr> %b
}

define ptr @vector_legal_bitcast_vector_of_pointers_to_pointer(<1 x ptr> %a) {
  %b = bitcast <1 x ptr> %a to ptr
  ret ptr %b
}

