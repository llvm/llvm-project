; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

define <4 x i8> @test_out_bound_values(<4 x i8> %a, <4 x i8> %b) {
  ; CHECK: immarg value 12 for arg 2 out of range [0,12)
  %d = call <4 x i8> @llvm.nvvm.compare.and.set.v4i8(<4 x i8> %a, <4 x i8> %b, i32 12)
  ret <4 x i8> %d
}

define <2 x i16> @test_negative_value(<2 x i16> %a, <2 x i16> %b) {
  ; CHECK: immarg value -1 for arg 2 out of range [0,12)
  %d = call <2 x i16> @llvm.nvvm.compare.and.set.v2i16(<2 x i16> %a, <2 x i16> %b, i32 -1)
  ret <2 x i16> %d
}

define <8 x i8> @test_unsupported_v8i8(<8 x i8> %a, <8 x i8> %b) {
  ; CHECK: llvm.nvvm.compare.and.set only supports <4 x i8> and <2 x i16> operand types, got <8 x i8>
  %d = call <8 x i8> @llvm.nvvm.compare.and.set.v8i8(<8 x i8> %a, <8 x i8> %b, i32 0)
  ret <8 x i8> %d
}

define <4 x i16> @test_unsupported_v4i16(<4 x i16> %a, <4 x i16> %b) {
  ; CHECK: llvm.nvvm.compare.and.set only supports <4 x i8> and <2 x i16> operand types, got <4 x i16>
  %d = call <4 x i16> @llvm.nvvm.compare.and.set.v4i16(<4 x i16> %a, <4 x i16> %b, i32 0)
  ret <4 x i16> %d
}

define <2 x i32> @test_unsupported_v2i32(<2 x i32> %a, <2 x i32> %b) {
  ; CHECK: llvm.nvvm.compare.and.set only supports <4 x i8> and <2 x i16> operand types, got <2 x i32>
  %d = call <2 x i32> @llvm.nvvm.compare.and.set.v2i32(<2 x i32> %a, <2 x i32> %b, i32 0)
  ret <2 x i32> %d
}
