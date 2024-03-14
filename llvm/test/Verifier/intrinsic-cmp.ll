; RUN: not opt -S -passes=verify 2>&1 < %s | FileCheck %s


define void @matching_vector_lens(<4 x i32> %arg1, <4 x i32> %arg2) {
  ; CHECK-LABEL: cmp_vector_lens_match
  ; CHECK: return type and first arg type must have the same number of elements
  %res = call <8 x i32> @llvm.scmp.v8i32.v4i32(<4 x i32> %arg1, <4 x i32> %arg2)
  ret void
}

define void @result_len_is_at_least_2(i32 %arg1, i32 %arg2) {
  ; CHECK-LABEL: cmp_result_len_is_at_least_2bits_wide
  ; CHECK: result type must be at least 2 bits wide
  %res2 = call i1 @llvm.scmp.i1.i32(i32 %arg1, i32 %arg2)
  ret void
}

define void @both_args_are_vecs_or_neither(<4 x i32> %arg1, i32 %arg2) {
  ; CHECK-LABEL: cmp_sources_and_dest_types_ranks_match
  ; CHECK: [us]cmp source and destination must both be a vector or neither
  @res3 = call i2 @llvm.scmp.i2.v4i32(<4 x i32> %arg1, i32 %arg2)
  ; CHECK: [us]cmp source and destination must both be a vector or neither
  @res4 = call <4 x i32> @llvm.scmp.v4i32.i32(<4 x i32> %arg2, i32 %arg2)
  ret void
}

