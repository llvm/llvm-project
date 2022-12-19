; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define void @test_vext_s8() nounwind ssp {
  ; CHECK-LABEL: test_vext_s8:
  ; CHECK: {{ext.8.*#1}}
  %xS8x8 = alloca <8 x i8>, align 8
  %__a = alloca <8 x i8>, align 8
  %__b = alloca <8 x i8>, align 8
  %tmp = load <8 x i8>, ptr %xS8x8, align 8
  store <8 x i8> %tmp, ptr %__a, align 8
  %tmp1 = load <8 x i8>, ptr %xS8x8, align 8
  store <8 x i8> %tmp1, ptr %__b, align 8
  %tmp2 = load <8 x i8>, ptr %__a, align 8
  %tmp3 = load <8 x i8>, ptr %__b, align 8
  %vext = shufflevector <8 x i8> %tmp2, <8 x i8> %tmp3, <8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>
  store <8 x i8> %vext, ptr %xS8x8, align 8
  ret void
}

define void @test_vext_u8() nounwind ssp {
  ; CHECK-LABEL: test_vext_u8:
  ; CHECK: {{ext.8.*#2}}
  %xU8x8 = alloca <8 x i8>, align 8
  %__a = alloca <8 x i8>, align 8
  %__b = alloca <8 x i8>, align 8
  %tmp = load <8 x i8>, ptr %xU8x8, align 8
  store <8 x i8> %tmp, ptr %__a, align 8
  %tmp1 = load <8 x i8>, ptr %xU8x8, align 8
  store <8 x i8> %tmp1, ptr %__b, align 8
  %tmp2 = load <8 x i8>, ptr %__a, align 8
  %tmp3 = load <8 x i8>, ptr %__b, align 8
  %vext = shufflevector <8 x i8> %tmp2, <8 x i8> %tmp3, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9>
  store <8 x i8> %vext, ptr %xU8x8, align 8
  ret void
}

define void @test_vext_p8() nounwind ssp {
  ; CHECK-LABEL: test_vext_p8:
  ; CHECK: {{ext.8.*#3}}
  %xP8x8 = alloca <8 x i8>, align 8
  %__a = alloca <8 x i8>, align 8
  %__b = alloca <8 x i8>, align 8
  %tmp = load <8 x i8>, ptr %xP8x8, align 8
  store <8 x i8> %tmp, ptr %__a, align 8
  %tmp1 = load <8 x i8>, ptr %xP8x8, align 8
  store <8 x i8> %tmp1, ptr %__b, align 8
  %tmp2 = load <8 x i8>, ptr %__a, align 8
  %tmp3 = load <8 x i8>, ptr %__b, align 8
  %vext = shufflevector <8 x i8> %tmp2, <8 x i8> %tmp3, <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
  store <8 x i8> %vext, ptr %xP8x8, align 8
  ret void
}

define void @test_vext_s16() nounwind ssp {
  ; CHECK-LABEL: test_vext_s16:
  ; CHECK: {{ext.8.*#2}}
  %xS16x4 = alloca <4 x i16>, align 8
  %__a = alloca <4 x i16>, align 8
  %__b = alloca <4 x i16>, align 8
  %tmp = load <4 x i16>, ptr %xS16x4, align 8
  store <4 x i16> %tmp, ptr %__a, align 8
  %tmp1 = load <4 x i16>, ptr %xS16x4, align 8
  store <4 x i16> %tmp1, ptr %__b, align 8
  %tmp2 = load <4 x i16>, ptr %__a, align 8
  %tmp3 = bitcast <4 x i16> %tmp2 to <8 x i8>
  %tmp4 = load <4 x i16>, ptr %__b, align 8
  %tmp5 = bitcast <4 x i16> %tmp4 to <8 x i8>
  %tmp6 = bitcast <8 x i8> %tmp3 to <4 x i16>
  %tmp7 = bitcast <8 x i8> %tmp5 to <4 x i16>
  %vext = shufflevector <4 x i16> %tmp6, <4 x i16> %tmp7, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  store <4 x i16> %vext, ptr %xS16x4, align 8
  ret void
}

define void @test_vext_u16() nounwind ssp {
  ; CHECK-LABEL: test_vext_u16:
  ; CHECK: {{ext.8.*#4}}
  %xU16x4 = alloca <4 x i16>, align 8
  %__a = alloca <4 x i16>, align 8
  %__b = alloca <4 x i16>, align 8
  %tmp = load <4 x i16>, ptr %xU16x4, align 8
  store <4 x i16> %tmp, ptr %__a, align 8
  %tmp1 = load <4 x i16>, ptr %xU16x4, align 8
  store <4 x i16> %tmp1, ptr %__b, align 8
  %tmp2 = load <4 x i16>, ptr %__a, align 8
  %tmp3 = bitcast <4 x i16> %tmp2 to <8 x i8>
  %tmp4 = load <4 x i16>, ptr %__b, align 8
  %tmp5 = bitcast <4 x i16> %tmp4 to <8 x i8>
  %tmp6 = bitcast <8 x i8> %tmp3 to <4 x i16>
  %tmp7 = bitcast <8 x i8> %tmp5 to <4 x i16>
  %vext = shufflevector <4 x i16> %tmp6, <4 x i16> %tmp7, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  store <4 x i16> %vext, ptr %xU16x4, align 8
  ret void
}

define void @test_vext_p16() nounwind ssp {
  ; CHECK-LABEL: test_vext_p16:
  ; CHECK: {{ext.8.*#6}}
  %xP16x4 = alloca <4 x i16>, align 8
  %__a = alloca <4 x i16>, align 8
  %__b = alloca <4 x i16>, align 8
  %tmp = load <4 x i16>, ptr %xP16x4, align 8
  store <4 x i16> %tmp, ptr %__a, align 8
  %tmp1 = load <4 x i16>, ptr %xP16x4, align 8
  store <4 x i16> %tmp1, ptr %__b, align 8
  %tmp2 = load <4 x i16>, ptr %__a, align 8
  %tmp3 = bitcast <4 x i16> %tmp2 to <8 x i8>
  %tmp4 = load <4 x i16>, ptr %__b, align 8
  %tmp5 = bitcast <4 x i16> %tmp4 to <8 x i8>
  %tmp6 = bitcast <8 x i8> %tmp3 to <4 x i16>
  %tmp7 = bitcast <8 x i8> %tmp5 to <4 x i16>
  %vext = shufflevector <4 x i16> %tmp6, <4 x i16> %tmp7, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  store <4 x i16> %vext, ptr %xP16x4, align 8
  ret void
}

define void @test_vext_s32() nounwind ssp {
  ; CHECK-LABEL: test_vext_s32:
  ; CHECK: {{rev64.2s.*}}
  %xS32x2 = alloca <2 x i32>, align 8
  %__a = alloca <2 x i32>, align 8
  %__b = alloca <2 x i32>, align 8
  %tmp = load <2 x i32>, ptr %xS32x2, align 8
  store <2 x i32> %tmp, ptr %__a, align 8
  %tmp1 = load <2 x i32>, ptr %xS32x2, align 8
  store <2 x i32> %tmp1, ptr %__b, align 8
  %tmp2 = load <2 x i32>, ptr %__a, align 8
  %tmp3 = bitcast <2 x i32> %tmp2 to <8 x i8>
  %tmp4 = load <2 x i32>, ptr %__b, align 8
  %tmp5 = bitcast <2 x i32> %tmp4 to <8 x i8>
  %tmp6 = bitcast <8 x i8> %tmp3 to <2 x i32>
  %tmp7 = bitcast <8 x i8> %tmp5 to <2 x i32>
  %vext = shufflevector <2 x i32> %tmp6, <2 x i32> %tmp7, <2 x i32> <i32 1, i32 2>
  store <2 x i32> %vext, ptr %xS32x2, align 8
  ret void
}

define void @test_vext_u32() nounwind ssp {
  ; CHECK-LABEL: test_vext_u32:
  ; CHECK: {{rev64.2s.*}}
  %xU32x2 = alloca <2 x i32>, align 8
  %__a = alloca <2 x i32>, align 8
  %__b = alloca <2 x i32>, align 8
  %tmp = load <2 x i32>, ptr %xU32x2, align 8
  store <2 x i32> %tmp, ptr %__a, align 8
  %tmp1 = load <2 x i32>, ptr %xU32x2, align 8
  store <2 x i32> %tmp1, ptr %__b, align 8
  %tmp2 = load <2 x i32>, ptr %__a, align 8
  %tmp3 = bitcast <2 x i32> %tmp2 to <8 x i8>
  %tmp4 = load <2 x i32>, ptr %__b, align 8
  %tmp5 = bitcast <2 x i32> %tmp4 to <8 x i8>
  %tmp6 = bitcast <8 x i8> %tmp3 to <2 x i32>
  %tmp7 = bitcast <8 x i8> %tmp5 to <2 x i32>
  %vext = shufflevector <2 x i32> %tmp6, <2 x i32> %tmp7, <2 x i32> <i32 1, i32 2>
  store <2 x i32> %vext, ptr %xU32x2, align 8
  ret void
}

define void @test_vext_f32() nounwind ssp {
  ; CHECK-LABEL: test_vext_f32:
  ; CHECK: {{rev64.2s.*}}
  %xF32x2 = alloca <2 x float>, align 8
  %__a = alloca <2 x float>, align 8
  %__b = alloca <2 x float>, align 8
  %tmp = load <2 x float>, ptr %xF32x2, align 8
  store <2 x float> %tmp, ptr %__a, align 8
  %tmp1 = load <2 x float>, ptr %xF32x2, align 8
  store <2 x float> %tmp1, ptr %__b, align 8
  %tmp2 = load <2 x float>, ptr %__a, align 8
  %tmp3 = bitcast <2 x float> %tmp2 to <8 x i8>
  %tmp4 = load <2 x float>, ptr %__b, align 8
  %tmp5 = bitcast <2 x float> %tmp4 to <8 x i8>
  %tmp6 = bitcast <8 x i8> %tmp3 to <2 x float>
  %tmp7 = bitcast <8 x i8> %tmp5 to <2 x float>
  %vext = shufflevector <2 x float> %tmp6, <2 x float> %tmp7, <2 x i32> <i32 1, i32 2>
  store <2 x float> %vext, ptr %xF32x2, align 8
  ret void
}

define void @test_vext_s64() nounwind ssp {
  ; CHECK-LABEL: test_vext_s64:
  ; CHECK_FIXME: {{rev64.2s.*}}
  ; this just turns into a load of the second element
  %xS64x1 = alloca <1 x i64>, align 8
  %__a = alloca <1 x i64>, align 8
  %__b = alloca <1 x i64>, align 8
  %tmp = load <1 x i64>, ptr %xS64x1, align 8
  store <1 x i64> %tmp, ptr %__a, align 8
  %tmp1 = load <1 x i64>, ptr %xS64x1, align 8
  store <1 x i64> %tmp1, ptr %__b, align 8
  %tmp2 = load <1 x i64>, ptr %__a, align 8
  %tmp3 = bitcast <1 x i64> %tmp2 to <8 x i8>
  %tmp4 = load <1 x i64>, ptr %__b, align 8
  %tmp5 = bitcast <1 x i64> %tmp4 to <8 x i8>
  %tmp6 = bitcast <8 x i8> %tmp3 to <1 x i64>
  %tmp7 = bitcast <8 x i8> %tmp5 to <1 x i64>
  %vext = shufflevector <1 x i64> %tmp6, <1 x i64> %tmp7, <1 x i32> <i32 1>
  store <1 x i64> %vext, ptr %xS64x1, align 8
  ret void
}

define void @test_vext_u64() nounwind ssp {
  ; CHECK-LABEL: test_vext_u64:
  ; CHECK_FIXME: {{ext.8.*#1}}
  ; this is turned into a simple load of the 2nd element
  %xU64x1 = alloca <1 x i64>, align 8
  %__a = alloca <1 x i64>, align 8
  %__b = alloca <1 x i64>, align 8
  %tmp = load <1 x i64>, ptr %xU64x1, align 8
  store <1 x i64> %tmp, ptr %__a, align 8
  %tmp1 = load <1 x i64>, ptr %xU64x1, align 8
  store <1 x i64> %tmp1, ptr %__b, align 8
  %tmp2 = load <1 x i64>, ptr %__a, align 8
  %tmp3 = bitcast <1 x i64> %tmp2 to <8 x i8>
  %tmp4 = load <1 x i64>, ptr %__b, align 8
  %tmp5 = bitcast <1 x i64> %tmp4 to <8 x i8>
  %tmp6 = bitcast <8 x i8> %tmp3 to <1 x i64>
  %tmp7 = bitcast <8 x i8> %tmp5 to <1 x i64>
  %vext = shufflevector <1 x i64> %tmp6, <1 x i64> %tmp7, <1 x i32> <i32 1>
  store <1 x i64> %vext, ptr %xU64x1, align 8
  ret void
}

define void @test_vextq_s8() nounwind ssp {
  ; CHECK-LABEL: test_vextq_s8:
  ; CHECK: {{ext.16.*#4}}
  %xS8x16 = alloca <16 x i8>, align 16
  %__a = alloca <16 x i8>, align 16
  %__b = alloca <16 x i8>, align 16
  %tmp = load <16 x i8>, ptr %xS8x16, align 16
  store <16 x i8> %tmp, ptr %__a, align 16
  %tmp1 = load <16 x i8>, ptr %xS8x16, align 16
  store <16 x i8> %tmp1, ptr %__b, align 16
  %tmp2 = load <16 x i8>, ptr %__a, align 16
  %tmp3 = load <16 x i8>, ptr %__b, align 16
  %vext = shufflevector <16 x i8> %tmp2, <16 x i8> %tmp3, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
  store <16 x i8> %vext, ptr %xS8x16, align 16
  ret void
}

define void @test_vextq_u8() nounwind ssp {
  ; CHECK-LABEL: test_vextq_u8:
  ; CHECK: {{ext.16.*#5}}
  %xU8x16 = alloca <16 x i8>, align 16
  %__a = alloca <16 x i8>, align 16
  %__b = alloca <16 x i8>, align 16
  %tmp = load <16 x i8>, ptr %xU8x16, align 16
  store <16 x i8> %tmp, ptr %__a, align 16
  %tmp1 = load <16 x i8>, ptr %xU8x16, align 16
  store <16 x i8> %tmp1, ptr %__b, align 16
  %tmp2 = load <16 x i8>, ptr %__a, align 16
  %tmp3 = load <16 x i8>, ptr %__b, align 16
  %vext = shufflevector <16 x i8> %tmp2, <16 x i8> %tmp3, <16 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20>
  store <16 x i8> %vext, ptr %xU8x16, align 16
  ret void
}

define void @test_vextq_p8() nounwind ssp {
  ; CHECK-LABEL: test_vextq_p8:
  ; CHECK: {{ext.16.*#6}}
  %xP8x16 = alloca <16 x i8>, align 16
  %__a = alloca <16 x i8>, align 16
  %__b = alloca <16 x i8>, align 16
  %tmp = load <16 x i8>, ptr %xP8x16, align 16
  store <16 x i8> %tmp, ptr %__a, align 16
  %tmp1 = load <16 x i8>, ptr %xP8x16, align 16
  store <16 x i8> %tmp1, ptr %__b, align 16
  %tmp2 = load <16 x i8>, ptr %__a, align 16
  %tmp3 = load <16 x i8>, ptr %__b, align 16
  %vext = shufflevector <16 x i8> %tmp2, <16 x i8> %tmp3, <16 x i32> <i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21>
  store <16 x i8> %vext, ptr %xP8x16, align 16
  ret void
}

define void @test_vextq_s16() nounwind ssp {
  ; CHECK-LABEL: test_vextq_s16:
  ; CHECK: {{ext.16.*#14}}
  %xS16x8 = alloca <8 x i16>, align 16
  %__a = alloca <8 x i16>, align 16
  %__b = alloca <8 x i16>, align 16
  %tmp = load <8 x i16>, ptr %xS16x8, align 16
  store <8 x i16> %tmp, ptr %__a, align 16
  %tmp1 = load <8 x i16>, ptr %xS16x8, align 16
  store <8 x i16> %tmp1, ptr %__b, align 16
  %tmp2 = load <8 x i16>, ptr %__a, align 16
  %tmp3 = bitcast <8 x i16> %tmp2 to <16 x i8>
  %tmp4 = load <8 x i16>, ptr %__b, align 16
  %tmp5 = bitcast <8 x i16> %tmp4 to <16 x i8>
  %tmp6 = bitcast <16 x i8> %tmp3 to <8 x i16>
  %tmp7 = bitcast <16 x i8> %tmp5 to <8 x i16>
  %vext = shufflevector <8 x i16> %tmp6, <8 x i16> %tmp7, <8 x i32> <i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  store <8 x i16> %vext, ptr %xS16x8, align 16
  ret void
}

define void @test_vextq_u16() nounwind ssp {
  ; CHECK-LABEL: test_vextq_u16:
  ; CHECK: {{ext.16.*#8}}
  %xU16x8 = alloca <8 x i16>, align 16
  %__a = alloca <8 x i16>, align 16
  %__b = alloca <8 x i16>, align 16
  %tmp = load <8 x i16>, ptr %xU16x8, align 16
  store <8 x i16> %tmp, ptr %__a, align 16
  %tmp1 = load <8 x i16>, ptr %xU16x8, align 16
  store <8 x i16> %tmp1, ptr %__b, align 16
  %tmp2 = load <8 x i16>, ptr %__a, align 16
  %tmp3 = bitcast <8 x i16> %tmp2 to <16 x i8>
  %tmp4 = load <8 x i16>, ptr %__b, align 16
  %tmp5 = bitcast <8 x i16> %tmp4 to <16 x i8>
  %tmp6 = bitcast <16 x i8> %tmp3 to <8 x i16>
  %tmp7 = bitcast <16 x i8> %tmp5 to <8 x i16>
  %vext = shufflevector <8 x i16> %tmp6, <8 x i16> %tmp7, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
  store <8 x i16> %vext, ptr %xU16x8, align 16
  ret void
}

define void @test_vextq_p16() nounwind ssp {
  ; CHECK-LABEL: test_vextq_p16:
  ; CHECK: {{ext.16.*#10}}
  %xP16x8 = alloca <8 x i16>, align 16
  %__a = alloca <8 x i16>, align 16
  %__b = alloca <8 x i16>, align 16
  %tmp = load <8 x i16>, ptr %xP16x8, align 16
  store <8 x i16> %tmp, ptr %__a, align 16
  %tmp1 = load <8 x i16>, ptr %xP16x8, align 16
  store <8 x i16> %tmp1, ptr %__b, align 16
  %tmp2 = load <8 x i16>, ptr %__a, align 16
  %tmp3 = bitcast <8 x i16> %tmp2 to <16 x i8>
  %tmp4 = load <8 x i16>, ptr %__b, align 16
  %tmp5 = bitcast <8 x i16> %tmp4 to <16 x i8>
  %tmp6 = bitcast <16 x i8> %tmp3 to <8 x i16>
  %tmp7 = bitcast <16 x i8> %tmp5 to <8 x i16>
  %vext = shufflevector <8 x i16> %tmp6, <8 x i16> %tmp7, <8 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12>
  store <8 x i16> %vext, ptr %xP16x8, align 16
  ret void
}

define void @test_vextq_s32() nounwind ssp {
  ; CHECK-LABEL: test_vextq_s32:
  ; CHECK: {{ext.16.*#4}}
  %xS32x4 = alloca <4 x i32>, align 16
  %__a = alloca <4 x i32>, align 16
  %__b = alloca <4 x i32>, align 16
  %tmp = load <4 x i32>, ptr %xS32x4, align 16
  store <4 x i32> %tmp, ptr %__a, align 16
  %tmp1 = load <4 x i32>, ptr %xS32x4, align 16
  store <4 x i32> %tmp1, ptr %__b, align 16
  %tmp2 = load <4 x i32>, ptr %__a, align 16
  %tmp3 = bitcast <4 x i32> %tmp2 to <16 x i8>
  %tmp4 = load <4 x i32>, ptr %__b, align 16
  %tmp5 = bitcast <4 x i32> %tmp4 to <16 x i8>
  %tmp6 = bitcast <16 x i8> %tmp3 to <4 x i32>
  %tmp7 = bitcast <16 x i8> %tmp5 to <4 x i32>
  %vext = shufflevector <4 x i32> %tmp6, <4 x i32> %tmp7, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  store <4 x i32> %vext, ptr %xS32x4, align 16
  ret void
}

define void @test_vextq_u32() nounwind ssp {
  ; CHECK-LABEL: test_vextq_u32:
  ; CHECK: {{ext.16.*#8}}
  %xU32x4 = alloca <4 x i32>, align 16
  %__a = alloca <4 x i32>, align 16
  %__b = alloca <4 x i32>, align 16
  %tmp = load <4 x i32>, ptr %xU32x4, align 16
  store <4 x i32> %tmp, ptr %__a, align 16
  %tmp1 = load <4 x i32>, ptr %xU32x4, align 16
  store <4 x i32> %tmp1, ptr %__b, align 16
  %tmp2 = load <4 x i32>, ptr %__a, align 16
  %tmp3 = bitcast <4 x i32> %tmp2 to <16 x i8>
  %tmp4 = load <4 x i32>, ptr %__b, align 16
  %tmp5 = bitcast <4 x i32> %tmp4 to <16 x i8>
  %tmp6 = bitcast <16 x i8> %tmp3 to <4 x i32>
  %tmp7 = bitcast <16 x i8> %tmp5 to <4 x i32>
  %vext = shufflevector <4 x i32> %tmp6, <4 x i32> %tmp7, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  store <4 x i32> %vext, ptr %xU32x4, align 16
  ret void
}

define void @test_vextq_f32() nounwind ssp {
  ; CHECK-LABEL: test_vextq_f32:
  ; CHECK: {{ext.16.*#12}}
  %xF32x4 = alloca <4 x float>, align 16
  %__a = alloca <4 x float>, align 16
  %__b = alloca <4 x float>, align 16
  %tmp = load <4 x float>, ptr %xF32x4, align 16
  store <4 x float> %tmp, ptr %__a, align 16
  %tmp1 = load <4 x float>, ptr %xF32x4, align 16
  store <4 x float> %tmp1, ptr %__b, align 16
  %tmp2 = load <4 x float>, ptr %__a, align 16
  %tmp3 = bitcast <4 x float> %tmp2 to <16 x i8>
  %tmp4 = load <4 x float>, ptr %__b, align 16
  %tmp5 = bitcast <4 x float> %tmp4 to <16 x i8>
  %tmp6 = bitcast <16 x i8> %tmp3 to <4 x float>
  %tmp7 = bitcast <16 x i8> %tmp5 to <4 x float>
  %vext = shufflevector <4 x float> %tmp6, <4 x float> %tmp7, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  store <4 x float> %vext, ptr %xF32x4, align 16
  ret void
}

define void @test_vextq_s64() nounwind ssp {
  ; CHECK-LABEL: test_vextq_s64:
  ; CHECK: {{ext.16.*#8}}
  %xS64x2 = alloca <2 x i64>, align 16
  %__a = alloca <2 x i64>, align 16
  %__b = alloca <2 x i64>, align 16
  %tmp = load <2 x i64>, ptr %xS64x2, align 16
  store <2 x i64> %tmp, ptr %__a, align 16
  %tmp1 = load <2 x i64>, ptr %xS64x2, align 16
  store <2 x i64> %tmp1, ptr %__b, align 16
  %tmp2 = load <2 x i64>, ptr %__a, align 16
  %tmp3 = bitcast <2 x i64> %tmp2 to <16 x i8>
  %tmp4 = load <2 x i64>, ptr %__b, align 16
  %tmp5 = bitcast <2 x i64> %tmp4 to <16 x i8>
  %tmp6 = bitcast <16 x i8> %tmp3 to <2 x i64>
  %tmp7 = bitcast <16 x i8> %tmp5 to <2 x i64>
  %vext = shufflevector <2 x i64> %tmp6, <2 x i64> %tmp7, <2 x i32> <i32 1, i32 2>
  store <2 x i64> %vext, ptr %xS64x2, align 16
  ret void
}

define void @test_vextq_u64() nounwind ssp {
  ; CHECK-LABEL: test_vextq_u64:
  ; CHECK: {{ext.16.*#8}}
  %xU64x2 = alloca <2 x i64>, align 16
  %__a = alloca <2 x i64>, align 16
  %__b = alloca <2 x i64>, align 16
  %tmp = load <2 x i64>, ptr %xU64x2, align 16
  store <2 x i64> %tmp, ptr %__a, align 16
  %tmp1 = load <2 x i64>, ptr %xU64x2, align 16
  store <2 x i64> %tmp1, ptr %__b, align 16
  %tmp2 = load <2 x i64>, ptr %__a, align 16
  %tmp3 = bitcast <2 x i64> %tmp2 to <16 x i8>
  %tmp4 = load <2 x i64>, ptr %__b, align 16
  %tmp5 = bitcast <2 x i64> %tmp4 to <16 x i8>
  %tmp6 = bitcast <16 x i8> %tmp3 to <2 x i64>
  %tmp7 = bitcast <16 x i8> %tmp5 to <2 x i64>
  %vext = shufflevector <2 x i64> %tmp6, <2 x i64> %tmp7, <2 x i32> <i32 1, i32 2>
  store <2 x i64> %vext, ptr %xU64x2, align 16
  ret void
}

; shuffles with an undef second operand can use an EXT also so long as the
; indices wrap and stay sequential.
; rdar://12051674
define <16 x i8> @vext1(<16 x i8> %_a) nounwind {
; CHECK-LABEL: vext1:
; CHECK: ext.16b  v0, v0, v0, #8
  %vext = shufflevector <16 x i8> %_a, <16 x i8> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <16 x i8> %vext
}

; <rdar://problem/12212062>
define <2 x i64> @vext2(<2 x i64> %p0, <2 x i64> %p1) nounwind readnone ssp {
entry:
; CHECK-LABEL: vext2:
; CHECK:       add.2d  v0, v0, v1
; CHECK-NEXT:  ext.16b v0, v0, v0, #8
; CHECK-NEXT:  ret
  %t0 = shufflevector <2 x i64> %p1, <2 x i64> undef, <2 x i32> <i32 1, i32 0>
  %t1 = shufflevector <2 x i64> %p0, <2 x i64> undef, <2 x i32> <i32 1, i32 0>
  %t2 = add <2 x i64> %t1, %t0
  ret <2 x i64> %t2
}
