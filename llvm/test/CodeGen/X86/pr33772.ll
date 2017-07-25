; RUN: not llc < %s -mcpu=skylake-avx512 2>&1 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; make sure we don't crash if scale for gather isn't constant.

; CHECK: LLVM ERROR: Cannot select: intrinsic %llvm.x86.avx512.gather.dpi.512
declare <16 x i32> @llvm.x86.avx512.gather.dpi.512(<16 x i32>, i8*, <16 x i32>, i16, i32)

define internal <16 x i32> @__gather_base_offsets32_i32(i8* readonly %ptr, i32 %offset_scale, <16 x i32> %offsets, <16 x i8> %vecmask) {
  %mask_vec_i1.i.i = icmp ne <16 x i8> %vecmask, zeroinitializer
  %mask_i16.i = bitcast <16 x i1> %mask_vec_i1.i.i to i16
  %res = tail call <16 x i32> @llvm.x86.avx512.gather.dpi.512(<16 x i32> undef, i8* %ptr, <16 x i32> %offsets, i16 %mask_i16.i, i32 %offset_scale)
  ret <16 x i32> %res
}
