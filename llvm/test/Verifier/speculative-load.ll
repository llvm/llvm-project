; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare <4 x i32> @llvm.speculative.load.v4i32.p0(ptr, i1, ...)
declare <3 x i32> @llvm.speculative.load.v3i32.p0(ptr, i1, ...)
declare <vscale x 3 x i32> @llvm.speculative.load.nxv3i32.p0(ptr, i1, ...)

declare i32 @bad_oracle_ret(ptr, i64) memory(argmem: read)
declare i64 @good_oracle(ptr, i64) memory(argmem: read)
declare i64 @oracle_i32_param(i32) memory(argmem: read)
declare i64 @side_effecting_oracle(ptr, i64)

define <3 x i32> @test_non_power_of_2_fixed(ptr %ptr) {
; CHECK: llvm.speculative.load type must have a power-of-2 size
; CHECK-NEXT: %res = call <3 x i32> (ptr, i1, ...) @llvm.speculative.load.v3i32.p0(ptr %ptr, i1 false, i64 0)
  %res = call <3 x i32> (ptr, i1, ...) @llvm.speculative.load.v3i32.p0(ptr %ptr, i1 false, i64 0)
  ret <3 x i32> %res
}

define <vscale x 3 x i32> @test_non_power_of_2_scalable(ptr %ptr) {
; CHECK: llvm.speculative.load type must have a power-of-2 size
; CHECK-NEXT: %res = call <vscale x 3 x i32> (ptr, i1, ...) @llvm.speculative.load.nxv3i32.p0(ptr %ptr, i1 false, i64 0)
  %res = call <vscale x 3 x i32> (ptr, i1, ...) @llvm.speculative.load.nxv3i32.p0(ptr %ptr, i1 false, i64 0)
  ret <vscale x 3 x i32> %res
}

define <4 x i32> @test_too_few_args(ptr %ptr) {
; CHECK: llvm.speculative.load requires at least 3 arguments
; CHECK-NEXT: call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 true)
  %res = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 true)
  ret <4 x i32> %res
}

define <4 x i32> @test_direct_form_extra_args(ptr %ptr) {
; CHECK: llvm.speculative.load direct form has too many arguments
; CHECK-NEXT: call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, i64 8, i64 4)
  %res = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, i64 8, i64 4)
  ret <4 x i32> %res
}

define <4 x i32> @test_oracle_wrong_return_type(ptr %ptr, i64 %n) {
; CHECK: llvm.speculative.load oracle function must return i64
; CHECK-NEXT: call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, ptr @bad_oracle_ret, ptr %ptr, i64 %n)
  %res = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, ptr @bad_oracle_ret, ptr %ptr, i64 %n)
  ret <4 x i32> %res
}

define <4 x i32> @test_oracle_too_few_args(ptr %ptr) {
; CHECK: llvm.speculative.load oracle function argument count mismatch
; CHECK-NEXT: call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, ptr @good_oracle)
  %res = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, ptr @good_oracle)
  ret <4 x i32> %res
}

define <4 x i32> @test_oracle_too_many_args(ptr %ptr, i64 %n) {
; CHECK: llvm.speculative.load oracle function argument count mismatch
; CHECK-NEXT: call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, ptr @good_oracle, ptr %ptr, i64 %n, i64 %n)
  %res = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, ptr @good_oracle, ptr %ptr, i64 %n, i64 %n)
  ret <4 x i32> %res
}

define <4 x i32> @test_oracle_arg_type_mismatch(ptr %ptr) {
; CHECK: llvm.speculative.load oracle function argument type mismatch
; CHECK-NEXT: call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, ptr @oracle_i32_param, i64 42)
  %res = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, ptr @oracle_i32_param, i64 42)
  ret <4 x i32> %res
}

define <4 x i32> @test_non_function_oracle(ptr %ptr, ptr %not_fn) {
; CHECK: llvm.speculative.load third argument must be i64 or a direct reference to an oracle function
; CHECK-NEXT: call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, ptr %not_fn)
  %res = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, ptr %not_fn)
  ret <4 x i32> %res
}

define <4 x i32> @test_oracle_side_effects(ptr %ptr, i64 %n) {
; CHECK: llvm.speculative.load oracle function must not have side effects and may only read memory through its arguments
; CHECK-NEXT: call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, ptr @side_effecting_oracle, ptr %ptr, i64 %n)
  %res = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, ptr @side_effecting_oracle, ptr %ptr, i64 %n)
  ret <4 x i32> %res
}

define <4 x i32> @test_num_accessible_bytes_exceeds_size(ptr %ptr) {
  %res = call <4 x i32> (ptr, i1, ...) @llvm.speculative.load.v4i32.p0(ptr %ptr, i1 false, i64 32)
  ret <4 x i32> %res
}
