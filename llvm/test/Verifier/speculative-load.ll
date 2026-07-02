; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare b128 @llvm.speculative.load.b128.p0(ptr, i1, ...)
declare i32 @llvm.speculative.load.i32.p0(ptr, i1, ...)
declare float @llvm.speculative.load.f32.p0(ptr, i1, ...)
declare b12 @llvm.speculative.load.b12.p0(ptr, i1, ...)
declare b1 @llvm.speculative.load.b1.p0(ptr, i1, ...)
declare b9 @llvm.speculative.load.b9.p0(ptr, i1, ...)
declare b24 @llvm.speculative.load.b24.p0(ptr, i1, ...)
declare <3 x i32> @llvm.speculative.load.v3i32.p0(ptr, i1, ...)
declare {i32, i32} @llvm.speculative.load.sl_i32i32s.p0(ptr, i1, ...)
declare [4 x i32] @llvm.speculative.load.a4i32.p0(ptr, i1, ...)
declare <4 x b3> @llvm.speculative.load.v4b3.p0(ptr, i1, ...)

declare i32 @bad_oracle_ret(ptr, i64) memory(argmem: read)
declare i64 @good_oracle(ptr, i64) memory(argmem: read)
declare i64 @oracle_i32_param(i32) memory(argmem: read)
declare i64 @side_effecting_oracle(ptr, i64)

define i32 @test_non_byte_non_vector_int(ptr %ptr) {
; CHECK: llvm.speculative.load return type must be a byte type or a vector type
; CHECK-NEXT: %res = call i32 (ptr, i1, ...) @llvm.speculative.load.i32.p0(ptr %ptr, i1 false, i64 0)
  %res = call i32 (ptr, i1, ...) @llvm.speculative.load.i32.p0(ptr %ptr, i1 false, i64 0)
  ret i32 %res
}

define float @test_non_byte_non_vector_float(ptr %ptr) {
; CHECK: llvm.speculative.load return type must be a byte type or a vector type
; CHECK-NEXT: %res = call float (ptr, i1, ...) @llvm.speculative.load.f32.p0(ptr %ptr, i1 false, i64 0)
  %res = call float (ptr, i1, ...) @llvm.speculative.load.f32.p0(ptr %ptr, i1 false, i64 0)
  ret float %res
}

define {i32, i32} @test_struct(ptr %ptr) {
; CHECK: llvm.speculative.load return type must be a byte type or a vector type
; CHECK-NEXT: %res = call { i32, i32 } (ptr, i1, ...) @llvm.speculative.load.sl_i32i32s.p0(ptr %ptr, i1 false, i64 0)
  %res = call {i32, i32} (ptr, i1, ...) @llvm.speculative.load.sl_i32i32s.p0(ptr %ptr, i1 false, i64 0)
  ret {i32, i32} %res
}

define [4 x i32] @test_array(ptr %ptr) {
; CHECK: llvm.speculative.load return type must be a byte type or a vector type
; CHECK-NEXT: %res = call [4 x i32] (ptr, i1, ...) @llvm.speculative.load.a4i32.p0(ptr %ptr, i1 false, i64 0)
  %res = call [4 x i32] (ptr, i1, ...) @llvm.speculative.load.a4i32.p0(ptr %ptr, i1 false, i64 0)
  ret [4 x i32] %res
}

define b1 @test_byte_b1(ptr %ptr) {
; CHECK: llvm.speculative.load byte type must have a bit width that is a multiple of 8
; CHECK-NEXT: %res = call b1 (ptr, i1, ...) @llvm.speculative.load.b1.p0(ptr %ptr, i1 false, i64 0)
  %res = call b1 (ptr, i1, ...) @llvm.speculative.load.b1.p0(ptr %ptr, i1 false, i64 0)
  ret b1 %res
}

define b9 @test_byte_b9(ptr %ptr) {
; CHECK: llvm.speculative.load byte type must have a bit width that is a multiple of 8
; CHECK-NEXT: %res = call b9 (ptr, i1, ...) @llvm.speculative.load.b9.p0(ptr %ptr, i1 false, i64 0)
  %res = call b9 (ptr, i1, ...) @llvm.speculative.load.b9.p0(ptr %ptr, i1 false, i64 0)
  ret b9 %res
}

define b12 @test_byte_not_multiple_of_8(ptr %ptr) {
; CHECK: llvm.speculative.load byte type must have a bit width that is a multiple of 8
; CHECK-NEXT: %res = call b12 (ptr, i1, ...) @llvm.speculative.load.b12.p0(ptr %ptr, i1 false, i64 0)
  %res = call b12 (ptr, i1, ...) @llvm.speculative.load.b12.p0(ptr %ptr, i1 false, i64 0)
  ret b12 %res
}

define <4 x b3> @test_byte_vector_not_multiple_of_8(ptr %ptr) {
; CHECK: llvm.speculative.load byte type must have a bit width that is a multiple of 8
; CHECK-NEXT: %res = call <4 x b3> (ptr, i1, ...) @llvm.speculative.load.v4b3.p0(ptr %ptr, i1 false, i64 0)
  %res = call <4 x b3> (ptr, i1, ...) @llvm.speculative.load.v4b3.p0(ptr %ptr, i1 false, i64 0)
  ret <4 x b3> %res
}

define b128 @test_too_few_args(ptr %ptr) {
; CHECK: llvm.speculative.load requires at least 3 arguments
; CHECK-NEXT: call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 true)
  %res = call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 true)
  ret b128 %res
}

define b128 @test_direct_form_extra_args(ptr %ptr) {
; CHECK: llvm.speculative.load direct form has too many arguments
; CHECK-NEXT: call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 false, i64 8, i64 4)
  %res = call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 false, i64 8, i64 4)
  ret b128 %res
}

define b128 @test_oracle_wrong_return_type(ptr %ptr, i64 %n) {
; CHECK: llvm.speculative.load oracle function must return i64
; CHECK-NEXT: call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 false, ptr @bad_oracle_ret, ptr %ptr, i64 %n)
  %res = call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 false, ptr @bad_oracle_ret, ptr %ptr, i64 %n)
  ret b128 %res
}

define b128 @test_oracle_too_few_args(ptr %ptr) {
; CHECK: llvm.speculative.load oracle function argument count mismatch
; CHECK-NEXT: call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 false, ptr @good_oracle)
  %res = call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 false, ptr @good_oracle)
  ret b128 %res
}

define b128 @test_oracle_too_many_args(ptr %ptr, i64 %n) {
; CHECK: llvm.speculative.load oracle function argument count mismatch
; CHECK-NEXT: call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 false, ptr @good_oracle, ptr %ptr, i64 %n, i64 %n)
  %res = call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 false, ptr @good_oracle, ptr %ptr, i64 %n, i64 %n)
  ret b128 %res
}

define b128 @test_oracle_arg_type_mismatch(ptr %ptr) {
; CHECK: llvm.speculative.load oracle function argument type mismatch
; CHECK-NEXT: call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 false, ptr @oracle_i32_param, i64 42)
  %res = call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 false, ptr @oracle_i32_param, i64 42)
  ret b128 %res
}

define b128 @test_non_function_oracle(ptr %ptr, ptr %not_fn) {
; CHECK: llvm.speculative.load third argument must be i64 or a direct reference to an oracle function
; CHECK-NEXT: call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 false, ptr %not_fn)
  %res = call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 false, ptr %not_fn)
  ret b128 %res
}

define b128 @test_oracle_side_effects(ptr %ptr, i64 %n) {
; CHECK: llvm.speculative.load oracle function must not have side effects and may only read memory through its arguments
; CHECK-NEXT: call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 false, ptr @side_effecting_oracle, ptr %ptr, i64 %n)
  %res = call b128 (ptr, i1, ...) @llvm.speculative.load.b128.p0(ptr %ptr, i1 false, ptr @side_effecting_oracle, ptr %ptr, i64 %n)
  ret b128 %res
}

define b24 @test_byte_size_not_pow2(ptr %ptr) {
; CHECK: llvm.speculative.load return type size in bytes must be a positive power of 2
; CHECK-NEXT: %res = call b24 (ptr, i1, ...) @llvm.speculative.load.b24.p0(ptr %ptr, i1 false, i64 0)
  %res = call b24 (ptr, i1, ...) @llvm.speculative.load.b24.p0(ptr %ptr, i1 false, i64 0)
  ret b24 %res
}

define <3 x i32> @test_vector_byte_size_not_pow2(ptr %ptr) {
; CHECK: llvm.speculative.load return type size in bytes must be a positive power of 2
; CHECK-NEXT: %res = call <3 x i32> (ptr, i1, ...) @llvm.speculative.load.v3i32.p0(ptr %ptr, i1 false, i64 0)
  %res = call <3 x i32> (ptr, i1, ...) @llvm.speculative.load.v3i32.p0(ptr %ptr, i1 false, i64 0)
  ret <3 x i32> %res
}
