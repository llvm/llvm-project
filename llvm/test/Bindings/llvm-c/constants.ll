; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo
;
; Test constant value coverage for echo command.

source_filename = "constants.ll"

; Test constant vectors
@const_vec_i32 = global <4 x i32> <i32 1, i32 2, i32 3, i32 4>
@const_vec_i8 = global <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>

; Test constant data vector (should be recognized differently)
@const_data_vec = global <8 x i16> <i16 100, i16 200, i16 300, i16 400, i16 500, i16 600, i16 700, i16 800>

; Test constant GEP (bitcast doesn't work as const expr in opaque pointers)
@int_array = global [4 x i32] [i32 1, i32 2, i32 3, i32 4]
@ptr_to_second = global ptr getelementptr ([4 x i32], ptr @int_array, i64 0, i64 1)

; Test using constant vectors in function
define <4 x i32> @test_const_vector() {
  ret <4 x i32> <i32 10, i32 20, i32 30, i32 40>
}

; Test constant vector operations
define <4 x i32> @test_vector_ops() {
  %1 = load <4 x i32>, ptr @const_vec_i32
  %2 = add <4 x i32> %1, <i32 1, i32 1, i32 1, i32 1>
  ret <4 x i32> %2
}
