; RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon -fp-contract=fast | FileCheck %s


define <8 x i8> @test_vrev16_s8(<8 x i8> %a) #0 {
; CHECK: rev16 v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  ret <8 x i8> %shuffle.i
}

define <16 x i8> @test_vrev16q_s8(<16 x i8> %a) #0 {
; CHECK: rev16 v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> undef, <16 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6, i32 9, i32 8, i32 11, i32 10, i32 13, i32 12, i32 15, i32 14>
  ret <16 x i8> %shuffle.i
}

define <8 x i8> @test_vrev32_s8(<8 x i8> %a) #0 {
; CHECK: rev32 v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x i8> %shuffle.i
}

define <4 x i16> @test_vrev32_s16(<4 x i16> %a) #0 {
; CHECK: rev32 v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  ret <4 x i16> %shuffle.i
}

define <16 x i8> @test_vrev32q_s8(<16 x i8> %a) #0 {
; CHECK: rev32 v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> undef, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4, i32 11, i32 10, i32 9, i32 8, i32 15, i32 14, i32 13, i32 12>
  ret <16 x i8> %shuffle.i
}

define <8 x i16> @test_vrev32q_s16(<8 x i16> %a) #0 {
; CHECK: rev32 v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  ret <8 x i16> %shuffle.i
}

define <8 x i8> @test_vrev64_s8(<8 x i8> %a) #0 {
; CHECK: rev64 v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i8> %shuffle.i
}

define <4 x i16> @test_vrev64_s16(<4 x i16> %a) #0 {
; CHECK: rev64 v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i16> %shuffle.i
}

define <2 x i32> @test_vrev64_s32(<2 x i32> %a) #0 {
; CHECK: rev64 v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> undef, <2 x i32> <i32 1, i32 0>
  ret <2 x i32> %shuffle.i
}

define <2 x float> @test_vrev64_f32(<2 x float> %a) #0 {
; CHECK: rev64 v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %shuffle.i = shufflevector <2 x float> %a, <2 x float> undef, <2 x i32> <i32 1, i32 0>
  ret <2 x float> %shuffle.i
}

define <16 x i8> @test_vrev64q_s8(<16 x i8> %a) #0 {
; CHECK: rev64 v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> undef, <16 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8>
  ret <16 x i8> %shuffle.i
}

define <8 x i16> @test_vrev64q_s16(<8 x i16> %a) #0 {
; CHECK: rev64 v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x i16> %shuffle.i
}

define <4 x i32> @test_vrev64q_s32(<4 x i32> %a) #0 {
; CHECK: rev64 v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  ret <4 x i32> %shuffle.i
}

define <4 x float> @test_vrev64q_f32(<4 x float> %a) #0 {
; CHECK: rev64 v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %shuffle.i = shufflevector <4 x float> %a, <4 x float> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  ret <4 x float> %shuffle.i
}

define <4 x i16> @test_vpaddl_s8(<8 x i8> %a) #0 {
; CHECK: saddlp v{{[0-9]+}}.4h, v{{[0-9]+}}.8b
  %vpaddl.i = tail call <4 x i16> @llvm.arm.neon.vpaddls.v4i16.v8i8(<8 x i8> %a) #4
  ret <4 x i16> %vpaddl.i
}

define <2 x i32> @test_vpaddl_s16(<4 x i16> %a) #0 {
; CHECK: saddlp v{{[0-9]+}}.2s, v{{[0-9]+}}.4h
  %vpaddl1.i = tail call <2 x i32> @llvm.arm.neon.vpaddls.v2i32.v4i16(<4 x i16> %a) #4
  ret <2 x i32> %vpaddl1.i
}

define <1 x i64> @test_vpaddl_s32(<2 x i32> %a) #0 {
; CHECK: saddlp v{{[0-9]+}}.1d, v{{[0-9]+}}.2s
  %vpaddl1.i = tail call <1 x i64> @llvm.arm.neon.vpaddls.v1i64.v2i32(<2 x i32> %a) #4
  ret <1 x i64> %vpaddl1.i
}

define <4 x i16> @test_vpaddl_u8(<8 x i8> %a) #0 {
; CHECK: uaddlp v{{[0-9]+}}.4h, v{{[0-9]+}}.8b
  %vpaddl.i = tail call <4 x i16> @llvm.arm.neon.vpaddlu.v4i16.v8i8(<8 x i8> %a) #4
  ret <4 x i16> %vpaddl.i
}

define <2 x i32> @test_vpaddl_u16(<4 x i16> %a) #0 {
; CHECK: uaddlp v{{[0-9]+}}.2s, v{{[0-9]+}}.4h
  %vpaddl1.i = tail call <2 x i32> @llvm.arm.neon.vpaddlu.v2i32.v4i16(<4 x i16> %a) #4
  ret <2 x i32> %vpaddl1.i
}

define <1 x i64> @test_vpaddl_u32(<2 x i32> %a) #0 {
; CHECK: uaddlp v{{[0-9]+}}.1d, v{{[0-9]+}}.2s
  %vpaddl1.i = tail call <1 x i64> @llvm.arm.neon.vpaddlu.v1i64.v2i32(<2 x i32> %a) #4
  ret <1 x i64> %vpaddl1.i
}

define <8 x i16> @test_vpaddlq_s8(<16 x i8> %a) #0 {
; CHECK: saddlp v{{[0-9]+}}.8h, v{{[0-9]+}}.16b
  %vpaddl.i = tail call <8 x i16> @llvm.arm.neon.vpaddls.v8i16.v16i8(<16 x i8> %a) #4
  ret <8 x i16> %vpaddl.i
}

define <4 x i32> @test_vpaddlq_s16(<8 x i16> %a) #0 {
; CHECK: saddlp v{{[0-9]+}}.4s, v{{[0-9]+}}.8h
  %vpaddl1.i = tail call <4 x i32> @llvm.arm.neon.vpaddls.v4i32.v8i16(<8 x i16> %a) #4
  ret <4 x i32> %vpaddl1.i
}

define <2 x i64> @test_vpaddlq_s32(<4 x i32> %a) #0 {
; CHECK: saddlp v{{[0-9]+}}.2d, v{{[0-9]+}}.4s
  %vpaddl1.i = tail call <2 x i64> @llvm.arm.neon.vpaddls.v2i64.v4i32(<4 x i32> %a) #4
  ret <2 x i64> %vpaddl1.i
}

define <8 x i16> @test_vpaddlq_u8(<16 x i8> %a) #0 {
; CHECK: uaddlp v{{[0-9]+}}.8h, v{{[0-9]+}}.16b
  %vpaddl.i = tail call <8 x i16> @llvm.arm.neon.vpaddlu.v8i16.v16i8(<16 x i8> %a) #4
  ret <8 x i16> %vpaddl.i
}

define <4 x i32> @test_vpaddlq_u16(<8 x i16> %a) #0 {
; CHECK: uaddlp v{{[0-9]+}}.4s, v{{[0-9]+}}.8h
  %vpaddl1.i = tail call <4 x i32> @llvm.arm.neon.vpaddlu.v4i32.v8i16(<8 x i16> %a) #4
  ret <4 x i32> %vpaddl1.i
}

define <2 x i64> @test_vpaddlq_u32(<4 x i32> %a) #0 {
; CHECK: uaddlp v{{[0-9]+}}.2d, v{{[0-9]+}}.4s
  %vpaddl1.i = tail call <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32> %a) #4
  ret <2 x i64> %vpaddl1.i
}

define <4 x i16> @test_vpadal_s8(<4 x i16> %a, <8 x i8> %b) #0 {
; CHECK: sadalp v{{[0-9]+}}.4h, v{{[0-9]+}}.8b
  %vpadal1.i = tail call <4 x i16> @llvm.arm.neon.vpadals.v4i16.v8i8(<4 x i16> %a, <8 x i8> %b) #4
  ret <4 x i16> %vpadal1.i
}

define <2 x i32> @test_vpadal_s16(<2 x i32> %a, <4 x i16> %b) #0 {
; CHECK: sadalp v{{[0-9]+}}.2s, v{{[0-9]+}}.4h
  %vpadal2.i = tail call <2 x i32> @llvm.arm.neon.vpadals.v2i32.v4i16(<2 x i32> %a, <4 x i16> %b) #4
  ret <2 x i32> %vpadal2.i
}

define <1 x i64> @test_vpadal_s32(<1 x i64> %a, <2 x i32> %b) #0 {
; CHECK: sadalp v{{[0-9]+}}.1d, v{{[0-9]+}}.2s
  %vpadal2.i = tail call <1 x i64> @llvm.arm.neon.vpadals.v1i64.v2i32(<1 x i64> %a, <2 x i32> %b) #4
  ret <1 x i64> %vpadal2.i
}

define <4 x i16> @test_vpadal_u8(<4 x i16> %a, <8 x i8> %b) #0 {
; CHECK: uadalp v{{[0-9]+}}.4h, v{{[0-9]+}}.8b
  %vpadal1.i = tail call <4 x i16> @llvm.arm.neon.vpadalu.v4i16.v8i8(<4 x i16> %a, <8 x i8> %b) #4
  ret <4 x i16> %vpadal1.i
}

define <2 x i32> @test_vpadal_u16(<2 x i32> %a, <4 x i16> %b) #0 {
; CHECK: uadalp v{{[0-9]+}}.2s, v{{[0-9]+}}.4h
  %vpadal2.i = tail call <2 x i32> @llvm.arm.neon.vpadalu.v2i32.v4i16(<2 x i32> %a, <4 x i16> %b) #4
  ret <2 x i32> %vpadal2.i
}

define <1 x i64> @test_vpadal_u32(<1 x i64> %a, <2 x i32> %b) #0 {
; CHECK: uadalp v{{[0-9]+}}.1d, v{{[0-9]+}}.2s
  %vpadal2.i = tail call <1 x i64> @llvm.arm.neon.vpadalu.v1i64.v2i32(<1 x i64> %a, <2 x i32> %b) #4
  ret <1 x i64> %vpadal2.i
}

define <8 x i16> @test_vpadalq_s8(<8 x i16> %a, <16 x i8> %b) #0 {
; CHECK: sadalp v{{[0-9]+}}.8h, v{{[0-9]+}}.16b
  %vpadal1.i = tail call <8 x i16> @llvm.arm.neon.vpadals.v8i16.v16i8(<8 x i16> %a, <16 x i8> %b) #4
  ret <8 x i16> %vpadal1.i
}

define <4 x i32> @test_vpadalq_s16(<4 x i32> %a, <8 x i16> %b) #0 {
; CHECK: sadalp v{{[0-9]+}}.4s, v{{[0-9]+}}.8h
  %vpadal2.i = tail call <4 x i32> @llvm.arm.neon.vpadals.v4i32.v8i16(<4 x i32> %a, <8 x i16> %b) #4
  ret <4 x i32> %vpadal2.i
}

define <2 x i64> @test_vpadalq_s32(<2 x i64> %a, <4 x i32> %b) #0 {
; CHECK: sadalp v{{[0-9]+}}.2d, v{{[0-9]+}}.4s
  %vpadal2.i = tail call <2 x i64> @llvm.arm.neon.vpadals.v2i64.v4i32(<2 x i64> %a, <4 x i32> %b) #4
  ret <2 x i64> %vpadal2.i
}

define <8 x i16> @test_vpadalq_u8(<8 x i16> %a, <16 x i8> %b) #0 {
; CHECK: uadalp v{{[0-9]+}}.8h, v{{[0-9]+}}.16b
  %vpadal1.i = tail call <8 x i16> @llvm.arm.neon.vpadalu.v8i16.v16i8(<8 x i16> %a, <16 x i8> %b) #4
  ret <8 x i16> %vpadal1.i
}

define <4 x i32> @test_vpadalq_u16(<4 x i32> %a, <8 x i16> %b) #0 {
; CHECK: uadalp v{{[0-9]+}}.4s, v{{[0-9]+}}.8h
  %vpadal2.i = tail call <4 x i32> @llvm.arm.neon.vpadalu.v4i32.v8i16(<4 x i32> %a, <8 x i16> %b) #4
  ret <4 x i32> %vpadal2.i
}

define <2 x i64> @test_vpadalq_u32(<2 x i64> %a, <4 x i32> %b) #0 {
; CHECK: uadalp v{{[0-9]+}}.2d, v{{[0-9]+}}.4s
  %vpadal2.i = tail call <2 x i64> @llvm.arm.neon.vpadalu.v2i64.v4i32(<2 x i64> %a, <4 x i32> %b) #4
  ret <2 x i64> %vpadal2.i
}

define <8 x i8> @test_vqabs_s8(<8 x i8> %a) #0 {
; CHECK: sqabs v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %vqabs.i = tail call <8 x i8> @llvm.arm.neon.vqabs.v8i8(<8 x i8> %a) #4
  ret <8 x i8> %vqabs.i
}

define <16 x i8> @test_vqabsq_s8(<16 x i8> %a) #0 {
; CHECK: sqabs v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %vqabs.i = tail call <16 x i8> @llvm.arm.neon.vqabs.v16i8(<16 x i8> %a) #4
  ret <16 x i8> %vqabs.i
}

define <4 x i16> @test_vqabs_s16(<4 x i16> %a) #0 {
; CHECK: sqabs v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
  %vqabs1.i = tail call <4 x i16> @llvm.arm.neon.vqabs.v4i16(<4 x i16> %a) #4
  ret <4 x i16> %vqabs1.i
}

define <8 x i16> @test_vqabsq_s16(<8 x i16> %a) #0 {
; CHECK: sqabs v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
  %vqabs1.i = tail call <8 x i16> @llvm.arm.neon.vqabs.v8i16(<8 x i16> %a) #4
  ret <8 x i16> %vqabs1.i
}

define <2 x i32> @test_vqabs_s32(<2 x i32> %a) #0 {
; CHECK: sqabs v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vqabs1.i = tail call <2 x i32> @llvm.arm.neon.vqabs.v2i32(<2 x i32> %a) #4
  ret <2 x i32> %vqabs1.i
}

define <4 x i32> @test_vqabsq_s32(<4 x i32> %a) #0 {
; CHECK: sqabs v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vqabs1.i = tail call <4 x i32> @llvm.arm.neon.vqabs.v4i32(<4 x i32> %a) #4
  ret <4 x i32> %vqabs1.i
}

define <2 x i64> @test_vqabsq_s64(<2 x i64> %a) #0 {
; CHECK: sqabs v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vqabs1.i = tail call <2 x i64> @llvm.arm.neon.vqabs.v2i64(<2 x i64> %a) #4
  ret <2 x i64> %vqabs1.i
}

define <8 x i8> @test_vqneg_s8(<8 x i8> %a) #0 {
; CHECK: sqneg v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %vqneg.i = tail call <8 x i8> @llvm.arm.neon.vqneg.v8i8(<8 x i8> %a) #4
  ret <8 x i8> %vqneg.i
}

define <16 x i8> @test_vqnegq_s8(<16 x i8> %a) #0 {
; CHECK: sqneg v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %vqneg.i = tail call <16 x i8> @llvm.arm.neon.vqneg.v16i8(<16 x i8> %a) #4
  ret <16 x i8> %vqneg.i
}

define <4 x i16> @test_vqneg_s16(<4 x i16> %a) #0 {
; CHECK: sqneg v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
  %vqneg1.i = tail call <4 x i16> @llvm.arm.neon.vqneg.v4i16(<4 x i16> %a) #4
  ret <4 x i16> %vqneg1.i
}

define <8 x i16> @test_vqnegq_s16(<8 x i16> %a) #0 {
; CHECK: sqneg v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
  %vqneg1.i = tail call <8 x i16> @llvm.arm.neon.vqneg.v8i16(<8 x i16> %a) #4
  ret <8 x i16> %vqneg1.i
}

define <2 x i32> @test_vqneg_s32(<2 x i32> %a) #0 {
; CHECK: sqneg v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vqneg1.i = tail call <2 x i32> @llvm.arm.neon.vqneg.v2i32(<2 x i32> %a) #4
  ret <2 x i32> %vqneg1.i
}

define <4 x i32> @test_vqnegq_s32(<4 x i32> %a) #0 {
; CHECK: sqneg v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vqneg1.i = tail call <4 x i32> @llvm.arm.neon.vqneg.v4i32(<4 x i32> %a) #4
  ret <4 x i32> %vqneg1.i
}

define <2 x i64> @test_vqnegq_s64(<2 x i64> %a) #0 {
; CHECK: sqneg v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vqneg1.i = tail call <2 x i64> @llvm.arm.neon.vqneg.v2i64(<2 x i64> %a) #4
  ret <2 x i64> %vqneg1.i
}

define <8 x i8> @test_vneg_s8(<8 x i8> %a) #0 {
; CHECK: neg v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %sub.i = sub <8 x i8> zeroinitializer, %a
  ret <8 x i8> %sub.i
}

define <16 x i8> @test_vnegq_s8(<16 x i8> %a) #0 {
; CHECK: neg v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %sub.i = sub <16 x i8> zeroinitializer, %a
  ret <16 x i8> %sub.i
}

define <4 x i16> @test_vneg_s16(<4 x i16> %a) #0 {
; CHECK: neg v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
  %sub.i = sub <4 x i16> zeroinitializer, %a
  ret <4 x i16> %sub.i
}

define <8 x i16> @test_vnegq_s16(<8 x i16> %a) #0 {
; CHECK: neg v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
  %sub.i = sub <8 x i16> zeroinitializer, %a
  ret <8 x i16> %sub.i
}

define <2 x i32> @test_vneg_s32(<2 x i32> %a) #0 {
; CHECK: neg v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %sub.i = sub <2 x i32> zeroinitializer, %a
  ret <2 x i32> %sub.i
}

define <4 x i32> @test_vnegq_s32(<4 x i32> %a) #0 {
; CHECK: neg v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %sub.i = sub <4 x i32> zeroinitializer, %a
  ret <4 x i32> %sub.i
}

define <2 x i64> @test_vnegq_s64(<2 x i64> %a) #0 {
; CHECK: neg v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %sub.i = sub <2 x i64> zeroinitializer, %a
  ret <2 x i64> %sub.i
}

define <2 x float> @test_vneg_f32(<2 x float> %a) #0 {
; CHECK: fneg v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %sub.i = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %a
  ret <2 x float> %sub.i
}

define <4 x float> @test_vnegq_f32(<4 x float> %a) #0 {
; CHECK: fneg v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %sub.i = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %a
  ret <4 x float> %sub.i
}

define <2 x double> @test_vnegq_f64(<2 x double> %a) #0 {
; CHECK: fneg v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %sub.i = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %a
  ret <2 x double> %sub.i
}

define <8 x i8> @test_vabs_s8(<8 x i8> %a) #0 {
; CHECK: abs v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %vabs.i = tail call <8 x i8> @llvm.arm.neon.vabs.v8i8(<8 x i8> %a) #4
  ret <8 x i8> %vabs.i
}

define <16 x i8> @test_vabsq_s8(<16 x i8> %a) #0 {
; CHECK: abs v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %vabs.i = tail call <16 x i8> @llvm.arm.neon.vabs.v16i8(<16 x i8> %a) #4
  ret <16 x i8> %vabs.i
}

define <4 x i16> @test_vabs_s16(<4 x i16> %a) #0 {
; CHECK: abs v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
  %vabs1.i = tail call <4 x i16> @llvm.arm.neon.vabs.v4i16(<4 x i16> %a) #4
  ret <4 x i16> %vabs1.i
}

define <8 x i16> @test_vabsq_s16(<8 x i16> %a) #0 {
; CHECK: abs v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
  %vabs1.i = tail call <8 x i16> @llvm.arm.neon.vabs.v8i16(<8 x i16> %a) #4
  ret <8 x i16> %vabs1.i
}

define <2 x i32> @test_vabs_s32(<2 x i32> %a) #0 {
; CHECK: abs v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vabs1.i = tail call <2 x i32> @llvm.arm.neon.vabs.v2i32(<2 x i32> %a) #4
  ret <2 x i32> %vabs1.i
}

define <4 x i32> @test_vabsq_s32(<4 x i32> %a) #0 {
; CHECK: abs v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vabs1.i = tail call <4 x i32> @llvm.arm.neon.vabs.v4i32(<4 x i32> %a) #4
  ret <4 x i32> %vabs1.i
}

define <2 x i64> @test_vabsq_s64(<2 x i64> %a) #0 {
; CHECK: abs v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vabs1.i = tail call <2 x i64> @llvm.arm.neon.vabs.v2i64(<2 x i64> %a) #4
  ret <2 x i64> %vabs1.i
}

define <2 x float> @test_vabs_f32(<2 x float> %a) #1 {
; CHECK: fabs v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vabs1.i = tail call <2 x float> @llvm.fabs.v2f32(<2 x float> %a) #4
  ret <2 x float> %vabs1.i
}

define <4 x float> @test_vabsq_f32(<4 x float> %a) #1 {
; CHECK: fabs v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vabs1.i = tail call <4 x float> @llvm.fabs.v4f32(<4 x float> %a) #4
  ret <4 x float> %vabs1.i
}

define <2 x double> @test_vabsq_f64(<2 x double> %a) #1 {
; CHECK: fabs v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vabs1.i = tail call <2 x double> @llvm.fabs.v2f64(<2 x double> %a) #4
  ret <2 x double> %vabs1.i
}

define <8 x i8> @test_vuqadd_s8(<8 x i8> %a, <8 x i8> %b) #0 {
; CHECK: suqadd v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %vuqadd.i = tail call <8 x i8> @llvm.aarch64.neon.suqadd.v8i8(<8 x i8> %a, <8 x i8> %b) #4
  ret <8 x i8> %vuqadd.i
}

define <16 x i8> @test_vuqaddq_s8(<16 x i8> %a, <16 x i8> %b) #0 {
; CHECK: suqadd v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %vuqadd.i = tail call <16 x i8> @llvm.aarch64.neon.suqadd.v16i8(<16 x i8> %a, <16 x i8> %b) #4
  ret <16 x i8> %vuqadd.i
}

define <4 x i16> @test_vuqadd_s16(<4 x i16> %a, <4 x i16> %b) #0 {
; CHECK: suqadd v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
  %vuqadd2.i = tail call <4 x i16> @llvm.aarch64.neon.suqadd.v4i16(<4 x i16> %a, <4 x i16> %b) #4
  ret <4 x i16> %vuqadd2.i
}

define <8 x i16> @test_vuqaddq_s16(<8 x i16> %a, <8 x i16> %b) #0 {
; CHECK: suqadd v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
  %vuqadd2.i = tail call <8 x i16> @llvm.aarch64.neon.suqadd.v8i16(<8 x i16> %a, <8 x i16> %b) #4
  ret <8 x i16> %vuqadd2.i
}

define <2 x i32> @test_vuqadd_s32(<2 x i32> %a, <2 x i32> %b) #0 {
; CHECK: suqadd v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vuqadd2.i = tail call <2 x i32> @llvm.aarch64.neon.suqadd.v2i32(<2 x i32> %a, <2 x i32> %b) #4
  ret <2 x i32> %vuqadd2.i
}

define <4 x i32> @test_vuqaddq_s32(<4 x i32> %a, <4 x i32> %b) #0 {
; CHECK: suqadd v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vuqadd2.i = tail call <4 x i32> @llvm.aarch64.neon.suqadd.v4i32(<4 x i32> %a, <4 x i32> %b) #4
  ret <4 x i32> %vuqadd2.i
}

define <2 x i64> @test_vuqaddq_s64(<2 x i64> %a, <2 x i64> %b) #0 {
; CHECK: suqadd v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vuqadd2.i = tail call <2 x i64> @llvm.aarch64.neon.suqadd.v2i64(<2 x i64> %a, <2 x i64> %b) #4
  ret <2 x i64> %vuqadd2.i
}

define <8 x i8> @test_vcls_s8(<8 x i8> %a) #0 {
; CHECK: cls v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %vcls.i = tail call <8 x i8> @llvm.arm.neon.vcls.v8i8(<8 x i8> %a) #4
  ret <8 x i8> %vcls.i
}

define <16 x i8> @test_vclsq_s8(<16 x i8> %a) #0 {
; CHECK: cls v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %vcls.i = tail call <16 x i8> @llvm.arm.neon.vcls.v16i8(<16 x i8> %a) #4
  ret <16 x i8> %vcls.i
}

define <4 x i16> @test_vcls_s16(<4 x i16> %a) #0 {
; CHECK: cls v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
  %vcls1.i = tail call <4 x i16> @llvm.arm.neon.vcls.v4i16(<4 x i16> %a) #4
  ret <4 x i16> %vcls1.i
}

define <8 x i16> @test_vclsq_s16(<8 x i16> %a) #0 {
; CHECK: cls v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
  %vcls1.i = tail call <8 x i16> @llvm.arm.neon.vcls.v8i16(<8 x i16> %a) #4
  ret <8 x i16> %vcls1.i
}

define <2 x i32> @test_vcls_s32(<2 x i32> %a) #0 {
; CHECK: cls v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vcls1.i = tail call <2 x i32> @llvm.arm.neon.vcls.v2i32(<2 x i32> %a) #4
  ret <2 x i32> %vcls1.i
}

define <4 x i32> @test_vclsq_s32(<4 x i32> %a) #0 {
; CHECK: cls v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vcls1.i = tail call <4 x i32> @llvm.arm.neon.vcls.v4i32(<4 x i32> %a) #4
  ret <4 x i32> %vcls1.i
}

define <8 x i8> @test_vclz_s8(<8 x i8> %a) #0 {
; CHECK: clz v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %vclz.i = tail call <8 x i8> @llvm.ctlz.v8i8(<8 x i8> %a, i1 false) #4
  ret <8 x i8> %vclz.i
}

define <16 x i8> @test_vclzq_s8(<16 x i8> %a) #0 {
; CHECK: clz v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %vclz.i = tail call <16 x i8> @llvm.ctlz.v16i8(<16 x i8> %a, i1 false) #4
  ret <16 x i8> %vclz.i
}

define <4 x i16> @test_vclz_s16(<4 x i16> %a) #0 {
; CHECK: clz v{{[0-9]+}}.4h, v{{[0-9]+}}.4h
  %vclz1.i = tail call <4 x i16> @llvm.ctlz.v4i16(<4 x i16> %a, i1 false) #4
  ret <4 x i16> %vclz1.i
}

define <8 x i16> @test_vclzq_s16(<8 x i16> %a) #0 {
; CHECK: clz v{{[0-9]+}}.8h, v{{[0-9]+}}.8h
  %vclz1.i = tail call <8 x i16> @llvm.ctlz.v8i16(<8 x i16> %a, i1 false) #4
  ret <8 x i16> %vclz1.i
}

define <2 x i32> @test_vclz_s32(<2 x i32> %a) #0 {
; CHECK: clz v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vclz1.i = tail call <2 x i32> @llvm.ctlz.v2i32(<2 x i32> %a, i1 false) #4
  ret <2 x i32> %vclz1.i
}

define <4 x i32> @test_vclzq_s32(<4 x i32> %a) #0 {
; CHECK: clz v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vclz1.i = tail call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %a, i1 false) #4
  ret <4 x i32> %vclz1.i
}

define <8 x i8> @test_vcnt_s8(<8 x i8> %a) #0 {
; CHECK: cnt v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %vctpop.i = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %a) #4
  ret <8 x i8> %vctpop.i
}

define <16 x i8> @test_vcntq_s8(<16 x i8> %a) #0 {
; CHECK: cnt v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %vctpop.i = tail call <16 x i8> @llvm.ctpop.v16i8(<16 x i8> %a) #4
  ret <16 x i8> %vctpop.i
}

define <8 x i8> @test_vmvn_s8(<8 x i8> %a) #0 {
; CHECK: not v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %neg.i = xor <8 x i8> %a, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  ret <8 x i8> %neg.i
}

define <16 x i8> @test_vmvnq_s8(<16 x i8> %a) #0 {
; CHECK: not v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %neg.i = xor <16 x i8> %a, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  ret <16 x i8> %neg.i
}

define <4 x i16> @test_vmvn_s16(<4 x i16> %a) #0 {
; CHECK: not v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %neg.i = xor <4 x i16> %a, <i16 -1, i16 -1, i16 -1, i16 -1>
  ret <4 x i16> %neg.i
}

define <8 x i16> @test_vmvnq_s16(<8 x i16> %a) #0 {
; CHECK: not v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %neg.i = xor <8 x i16> %a, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  ret <8 x i16> %neg.i
}

define <2 x i32> @test_vmvn_s32(<2 x i32> %a) #0 {
; CHECK: not v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %neg.i = xor <2 x i32> %a, <i32 -1, i32 -1>
  ret <2 x i32> %neg.i
}

define <4 x i32> @test_vmvnq_s32(<4 x i32> %a) #0 {
; CHECK: not v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %neg.i = xor <4 x i32> %a, <i32 -1, i32 -1, i32 -1, i32 -1>
  ret <4 x i32> %neg.i
}

define <8 x i8> @test_vrbit_s8(<8 x i8> %a) #0 {
; CHECK: rbit v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %vrbit.i = tail call <8 x i8> @llvm.aarch64.neon.rbit.v8i8(<8 x i8> %a) #4
  ret <8 x i8> %vrbit.i
}

define <16 x i8> @test_vrbitq_s8(<16 x i8> %a) #0 {
; CHECK: rbit v{{[0-9]+}}.16b, v{{[0-9]+}}.16b
  %vrbit.i = tail call <16 x i8> @llvm.aarch64.neon.rbit.v16i8(<16 x i8> %a) #4
  ret <16 x i8> %vrbit.i
}

define <8 x i8> @test_vmovn_s16(<8 x i16> %a) #0 {
; CHECK: xtn v{{[0-9]+}}.8b, v{{[0-9]+}}.8h
  %vmovn.i = trunc <8 x i16> %a to <8 x i8>
  ret <8 x i8> %vmovn.i
}

define <4 x i16> @test_vmovn_s32(<4 x i32> %a) #0 {
; CHECK: xtn v{{[0-9]+}}.4h, v{{[0-9]+}}.4s
  %vmovn.i = trunc <4 x i32> %a to <4 x i16>
  ret <4 x i16> %vmovn.i
}

define <2 x i32> @test_vmovn_s64(<2 x i64> %a) #0 {
; CHECK: xtn v{{[0-9]+}}.2s, v{{[0-9]+}}.2d
  %vmovn.i = trunc <2 x i64> %a to <2 x i32>
  ret <2 x i32> %vmovn.i
}

define <16 x i8> @test_vmovn_high_s16(<8 x i8> %a, <8 x i16> %b) #0 {
; CHECK: xtn2 v{{[0-9]+}}.16b, v{{[0-9]+}}.8h
  %vmovn.i.i = trunc <8 x i16> %b to <8 x i8>
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %vmovn.i.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %shuffle.i
}

define <8 x i16> @test_vmovn_high_s32(<4 x i16> %a, <4 x i32> %b) #0 {
; CHECK: xtn2 v{{[0-9]+}}.8h, v{{[0-9]+}}.4s
  %vmovn.i.i = trunc <4 x i32> %b to <4 x i16>
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %vmovn.i.i, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %shuffle.i
}

define <4 x i32> @test_vmovn_high_s64(<2 x i32> %a, <2 x i64> %b) #0 {
; CHECK: xtn2 v{{[0-9]+}}.4s, v{{[0-9]+}}.2d
  %vmovn.i.i = trunc <2 x i64> %b to <2 x i32>
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %vmovn.i.i, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %shuffle.i
}

define <8 x i8> @test_vqmovun_s16(<8 x i16> %a) #0 {
; CHECK: sqxtun v{{[0-9]+}}.8b, v{{[0-9]+}}.8h
  %vqdmull1.i = tail call <8 x i8> @llvm.arm.neon.vqmovnsu.v8i8(<8 x i16> %a) #4
  ret <8 x i8> %vqdmull1.i
}

define <4 x i16> @test_vqmovun_s32(<4 x i32> %a) #0 {
; CHECK: sqxtun v{{[0-9]+}}.4h, v{{[0-9]+}}.4s
  %vqdmull1.i = tail call <4 x i16> @llvm.arm.neon.vqmovnsu.v4i16(<4 x i32> %a) #4
  ret <4 x i16> %vqdmull1.i
}

define <2 x i32> @test_vqmovun_s64(<2 x i64> %a) #0 {
; CHECK: sqxtun v{{[0-9]+}}.2s, v{{[0-9]+}}.2d
  %vqdmull1.i = tail call <2 x i32> @llvm.arm.neon.vqmovnsu.v2i32(<2 x i64> %a) #4
  ret <2 x i32> %vqdmull1.i
}

define <16 x i8> @test_vqmovun_high_s16(<8 x i8> %a, <8 x i16> %b) #0 {
; CHECK: sqxtun2 v{{[0-9]+}}.16b, v{{[0-9]+}}.8h
  %vqdmull1.i.i = tail call <8 x i8> @llvm.arm.neon.vqmovnsu.v8i8(<8 x i16> %b) #4
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %vqdmull1.i.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %shuffle.i
}

define <8 x i16> @test_vqmovun_high_s32(<4 x i16> %a, <4 x i32> %b) #0 {
; CHECK: sqxtun2 v{{[0-9]+}}.8h, v{{[0-9]+}}.4s
  %vqdmull1.i.i = tail call <4 x i16> @llvm.arm.neon.vqmovnsu.v4i16(<4 x i32> %b) #4
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %vqdmull1.i.i, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %shuffle.i
}

define <4 x i32> @test_vqmovun_high_s64(<2 x i32> %a, <2 x i64> %b) #0 {
; CHECK: sqxtun2 v{{[0-9]+}}.4s, v{{[0-9]+}}.2d
  %vqdmull1.i.i = tail call <2 x i32> @llvm.arm.neon.vqmovnsu.v2i32(<2 x i64> %b) #4
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %vqdmull1.i.i, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %shuffle.i
}

define <8 x i8> @test_vqmovn_s16(<8 x i16> %a) #0 {
; CHECK: sqxtn v{{[0-9]+}}.8b, v{{[0-9]+}}.8h
  %vqmovn1.i = tail call <8 x i8> @llvm.arm.neon.vqmovns.v8i8(<8 x i16> %a) #4
  ret <8 x i8> %vqmovn1.i
}

define <4 x i16> @test_vqmovn_s32(<4 x i32> %a) #0 {
; CHECK: sqxtn v{{[0-9]+}}.4h, v{{[0-9]+}}.4s
  %vqmovn1.i = tail call <4 x i16> @llvm.arm.neon.vqmovns.v4i16(<4 x i32> %a) #4
  ret <4 x i16> %vqmovn1.i
}

define <2 x i32> @test_vqmovn_s64(<2 x i64> %a) #0 {
; CHECK: sqxtn v{{[0-9]+}}.2s, v{{[0-9]+}}.2d
  %vqmovn1.i = tail call <2 x i32> @llvm.arm.neon.vqmovns.v2i32(<2 x i64> %a) #4
  ret <2 x i32> %vqmovn1.i
}

define <16 x i8> @test_vqmovn_high_s16(<8 x i8> %a, <8 x i16> %b) #0 {
; CHECK: sqxtn2 v{{[0-9]+}}.16b, v{{[0-9]+}}.8h
  %vqmovn1.i.i = tail call <8 x i8> @llvm.arm.neon.vqmovns.v8i8(<8 x i16> %b) #4
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %vqmovn1.i.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %shuffle.i
}

define <8 x i16> @test_vqmovn_high_s32(<4 x i16> %a, <4 x i32> %b) #0 {
; CHECK: test_vqmovn_high_s32
  %vqmovn1.i.i = tail call <4 x i16> @llvm.arm.neon.vqmovns.v4i16(<4 x i32> %b) #4
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %vqmovn1.i.i, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %shuffle.i
}

define <4 x i32> @test_vqmovn_high_s64(<2 x i32> %a, <2 x i64> %b) #0 {
; CHECK: test_vqmovn_high_s64
  %vqmovn1.i.i = tail call <2 x i32> @llvm.arm.neon.vqmovns.v2i32(<2 x i64> %b) #4
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %vqmovn1.i.i, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %shuffle.i
}

define <8 x i8> @test_vqmovn_u16(<8 x i16> %a) #0 {
; CHECK: uqxtn v{{[0-9]+}}.8b, v{{[0-9]+}}.8h
  %vqmovn1.i = tail call <8 x i8> @llvm.arm.neon.vqmovnu.v8i8(<8 x i16> %a) #4
  ret <8 x i8> %vqmovn1.i
}

define <4 x i16> @test_vqmovn_u32(<4 x i32> %a) #0 {
; CHECK: uqxtn v{{[0-9]+}}.4h, v{{[0-9]+}}.4s
  %vqmovn1.i = tail call <4 x i16> @llvm.arm.neon.vqmovnu.v4i16(<4 x i32> %a) #4
  ret <4 x i16> %vqmovn1.i
}

define <2 x i32> @test_vqmovn_u64(<2 x i64> %a) #0 {
; CHECK: uqxtn v{{[0-9]+}}.2s, v{{[0-9]+}}.2d
  %vqmovn1.i = tail call <2 x i32> @llvm.arm.neon.vqmovnu.v2i32(<2 x i64> %a) #4
  ret <2 x i32> %vqmovn1.i
}

define <16 x i8> @test_vqmovn_high_u16(<8 x i8> %a, <8 x i16> %b) #0 {
; CHECK: uqxtn2 v{{[0-9]+}}.16b, v{{[0-9]+}}.8h
  %vqmovn1.i.i = tail call <8 x i8> @llvm.arm.neon.vqmovnu.v8i8(<8 x i16> %b) #4
  %shuffle.i = shufflevector <8 x i8> %a, <8 x i8> %vqmovn1.i.i, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %shuffle.i
}

define <8 x i16> @test_vqmovn_high_u32(<4 x i16> %a, <4 x i32> %b) #0 {
; CHECK: uqxtn2 v{{[0-9]+}}.8h, v{{[0-9]+}}.4s
  %vqmovn1.i.i = tail call <4 x i16> @llvm.arm.neon.vqmovnu.v4i16(<4 x i32> %b) #4
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %vqmovn1.i.i, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %shuffle.i
}

define <4 x i32> @test_vqmovn_high_u64(<2 x i32> %a, <2 x i64> %b) #0 {
; CHECK: uqxtn2 v{{[0-9]+}}.4s, v{{[0-9]+}}.2d
  %vqmovn1.i.i = tail call <2 x i32> @llvm.arm.neon.vqmovnu.v2i32(<2 x i64> %b) #4
  %shuffle.i = shufflevector <2 x i32> %a, <2 x i32> %vqmovn1.i.i, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %shuffle.i
}

define <8 x i16> @test_vshll_n_s8(<8 x i8> %a) #0 {
; CHECK: shll {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, #8
  %1 = sext <8 x i8> %a to <8 x i16>
  %vshll_n = shl <8 x i16> %1, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  ret <8 x i16> %vshll_n
}

define <4 x i32> @test_vshll_n_s16(<4 x i16> %a) #0 {
; CHECK: shll {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, #16
  %1 = sext <4 x i16> %a to <4 x i32>
  %vshll_n = shl <4 x i32> %1, <i32 16, i32 16, i32 16, i32 16>
  ret <4 x i32> %vshll_n
}

define <2 x i64> @test_vshll_n_s32(<2 x i32> %a) #0 {
; CHECK: shll {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, #32
  %1 = sext <2 x i32> %a to <2 x i64>
  %vshll_n = shl <2 x i64> %1, <i64 32, i64 32>
  ret <2 x i64> %vshll_n
}

define <8 x i16> @test_vshll_n_u8(<8 x i8> %a) #0 {
; CHECK: shll {{v[0-9]+}}.8h, {{v[0-9]+}}.8b, #8
  %1 = zext <8 x i8> %a to <8 x i16>
  %vshll_n = shl <8 x i16> %1, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  ret <8 x i16> %vshll_n
}

define <4 x i32> @test_vshll_n_u16(<4 x i16> %a) #0 {
; CHECK: shll {{v[0-9]+}}.4s, {{v[0-9]+}}.4h, #16
  %1 = zext <4 x i16> %a to <4 x i32>
  %vshll_n = shl <4 x i32> %1, <i32 16, i32 16, i32 16, i32 16>
  ret <4 x i32> %vshll_n
}

define <2 x i64> @test_vshll_n_u32(<2 x i32> %a) #0 {
; CHECK: shll {{v[0-9]+}}.2d, {{v[0-9]+}}.2s, #32
  %1 = zext <2 x i32> %a to <2 x i64>
  %vshll_n = shl <2 x i64> %1, <i64 32, i64 32>
  ret <2 x i64> %vshll_n
}

define <8 x i16> @test_vshll_high_n_s8(<16 x i8> %a) #0 {
; CHECK: shll2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, #8
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1 = sext <8 x i8> %shuffle.i to <8 x i16>
  %vshll_n = shl <8 x i16> %1, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  ret <8 x i16> %vshll_n
}

define <4 x i32> @test_vshll_high_n_s16(<8 x i16> %a) #0 {
; CHECK: shll2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, #16
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1 = sext <4 x i16> %shuffle.i to <4 x i32>
  %vshll_n = shl <4 x i32> %1, <i32 16, i32 16, i32 16, i32 16>
  ret <4 x i32> %vshll_n
}

define <2 x i64> @test_vshll_high_n_s32(<4 x i32> %a) #0 {
; CHECK: shll2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, #32
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %1 = sext <2 x i32> %shuffle.i to <2 x i64>
  %vshll_n = shl <2 x i64> %1, <i64 32, i64 32>
  ret <2 x i64> %vshll_n
}

define <8 x i16> @test_vshll_high_n_u8(<16 x i8> %a) #0 {
; CHECK: shll2 {{v[0-9]+}}.8h, {{v[0-9]+}}.16b, #8
  %shuffle.i = shufflevector <16 x i8> %a, <16 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %1 = zext <8 x i8> %shuffle.i to <8 x i16>
  %vshll_n = shl <8 x i16> %1, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  ret <8 x i16> %vshll_n
}

define <4 x i32> @test_vshll_high_n_u16(<8 x i16> %a) #0 {
; CHECK: shll2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, #16
  %shuffle.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %1 = zext <4 x i16> %shuffle.i to <4 x i32>
  %vshll_n = shl <4 x i32> %1, <i32 16, i32 16, i32 16, i32 16>
  ret <4 x i32> %vshll_n
}

define <2 x i64> @test_vshll_high_n_u32(<4 x i32> %a) #0 {
; CHECK: shll2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, #32
  %shuffle.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %1 = zext <2 x i32> %shuffle.i to <2 x i64>
  %vshll_n = shl <2 x i64> %1, <i64 32, i64 32>
  ret <2 x i64> %vshll_n
}

define <4 x i16> @test_vcvt_f16_f32(<4 x float> %a) #0 {
; CHECK: fcvtn v{{[0-9]+}}.4h, v{{[0-9]+}}.4s
  %vcvt1.i = tail call <4 x i16> @llvm.arm.neon.vcvtfp2hf(<4 x float> %a) #4
  ret <4 x i16> %vcvt1.i
}

define <8 x i16> @test_vcvt_high_f16_f32(<4 x i16> %a, <4 x float> %b) #0 {
; CHECK: fcvtn2 v{{[0-9]+}}.8h, v{{[0-9]+}}.4s
  %vcvt1.i.i = tail call <4 x i16> @llvm.arm.neon.vcvtfp2hf(<4 x float> %b) #4
  %shuffle.i = shufflevector <4 x i16> %a, <4 x i16> %vcvt1.i.i, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %shuffle.i
}

define <4 x float> @test_vcvt_f32_f16(<4 x i16> %a) #0 {
; CHECK: fcvtl v{{[0-9]+}}.4s, v{{[0-9]+}}.4h
  %vcvt1.i = tail call <4 x float> @llvm.arm.neon.vcvthf2fp(<4 x i16> %a) #4
  ret <4 x float> %vcvt1.i
}

define <4 x float> @test_vcvt_high_f32_f16(<8 x i16> %a) #0 {
; CHECK: fcvtl2 v{{[0-9]+}}.4s, v{{[0-9]+}}.8h
  %shuffle.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vcvt1.i.i = tail call <4 x float> @llvm.arm.neon.vcvthf2fp(<4 x i16> %shuffle.i.i) #4
  ret <4 x float> %vcvt1.i.i
}

define <2 x float> @test_vcvt_f32_f64(<2 x double> %a) #0 {
; CHECK: fcvtn v{{[0-9]+}}.2s, v{{[0-9]+}}.2d
  %vcvt.i = fptrunc <2 x double> %a to <2 x float>
  ret <2 x float> %vcvt.i
}

define <4 x float> @test_vcvt_high_f32_f64(<2 x float> %a, <2 x double> %b) #0 {
; CHECK: fcvtn2 v{{[0-9]+}}.4s, v{{[0-9]+}}.2d
  %vcvt.i.i = fptrunc <2 x double> %b to <2 x float>
  %shuffle.i = shufflevector <2 x float> %a, <2 x float> %vcvt.i.i, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x float> %shuffle.i
}

define <2 x float> @test_vcvtx_f32_f64(<2 x double> %a) #0 {
; CHECK: fcvtxn v{{[0-9]+}}.2s, v{{[0-9]+}}.2d
  %vcvtx_f32_f641.i = tail call <2 x float> @llvm.aarch64.neon.fcvtxn.v2f32.v2f64(<2 x double> %a) #4
  ret <2 x float> %vcvtx_f32_f641.i
}

define <4 x float> @test_vcvtx_high_f32_f64(<2 x float> %a, <2 x double> %b) #0 {
; CHECK: fcvtxn2 v{{[0-9]+}}.4s, v{{[0-9]+}}.2d
  %vcvtx_f32_f641.i.i = tail call <2 x float> @llvm.aarch64.neon.fcvtxn.v2f32.v2f64(<2 x double> %b) #4
  %shuffle.i = shufflevector <2 x float> %a, <2 x float> %vcvtx_f32_f641.i.i, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x float> %shuffle.i
}

define <2 x double> @test_vcvt_f64_f32(<2 x float> %a) #0 {
; CHECK: fcvtl v{{[0-9]+}}.2d, v{{[0-9]+}}.2s
  %vcvt.i = fpext <2 x float> %a to <2 x double>
  ret <2 x double> %vcvt.i
}

define <2 x double> @test_vcvt_high_f64_f32(<4 x float> %a) #0 {
; CHECK: fcvtl2 v{{[0-9]+}}.2d, v{{[0-9]+}}.4s
  %shuffle.i.i = shufflevector <4 x float> %a, <4 x float> undef, <2 x i32> <i32 2, i32 3>
  %vcvt.i.i = fpext <2 x float> %shuffle.i.i to <2 x double>
  ret <2 x double> %vcvt.i.i
}

define <2 x float> @test_vrndn_f32(<2 x float> %a) #0 {
; CHECK: frintn v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vrndn1.i = tail call <2 x float> @llvm.aarch64.neon.frintn.v2f32(<2 x float> %a) #4
  ret <2 x float> %vrndn1.i
}

define <4 x float> @test_vrndnq_f32(<4 x float> %a) #0 {
; CHECK: frintn v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vrndn1.i = tail call <4 x float> @llvm.aarch64.neon.frintn.v4f32(<4 x float> %a) #4
  ret <4 x float> %vrndn1.i
}

define <2 x double> @test_vrndnq_f64(<2 x double> %a) #0 {
; CHECK: frintn v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vrndn1.i = tail call <2 x double> @llvm.aarch64.neon.frintn.v2f64(<2 x double> %a) #4
  ret <2 x double> %vrndn1.i
}

define <2 x float> @test_vrnda_f32(<2 x float> %a) #0 {
; CHECK: frinta v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vrnda1.i = tail call <2 x float> @llvm.round.v2f32(<2 x float> %a) #4
  ret <2 x float> %vrnda1.i
}

define <4 x float> @test_vrndaq_f32(<4 x float> %a) #0 {
; CHECK: frinta v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
   %vrnda1.i = tail call <4 x float> @llvm.round.v4f32(<4 x float> %a) #4
  ret <4 x float> %vrnda1.i
}

define <2 x double> @test_vrndaq_f64(<2 x double> %a) #0 {
; CHECK: frinta v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vrnda1.i = tail call <2 x double> @llvm.round.v2f64(<2 x double> %a) #4
  ret <2 x double> %vrnda1.i
}

define <2 x float> @test_vrndp_f32(<2 x float> %a) #0 {
; CHECK: frintp v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vrndp1.i = tail call <2 x float> @llvm.ceil.v2f32(<2 x float> %a) #4
  ret <2 x float> %vrndp1.i
}

define <4 x float> @test_vrndpq_f32(<4 x float> %a) #0 {
; CHECK: frintp v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
 %vrndp1.i = tail call <4 x float> @llvm.ceil.v4f32(<4 x float> %a) #4
  ret <4 x float> %vrndp1.i
}

define <2 x double> @test_vrndpq_f64(<2 x double> %a) #0 {
; CHECK: frintp v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vrndp1.i = tail call <2 x double> @llvm.ceil.v2f64(<2 x double> %a) #4
  ret <2 x double> %vrndp1.i
}

define <2 x float> @test_vrndm_f32(<2 x float> %a) #0 {
; CHECK: frintm v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vrndm1.i = tail call <2 x float> @llvm.floor.v2f32(<2 x float> %a) #4
  ret <2 x float> %vrndm1.i
}

define <4 x float> @test_vrndmq_f32(<4 x float> %a) #0 {
; CHECK: frintm v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vrndm1.i = tail call <4 x float> @llvm.floor.v4f32(<4 x float> %a) #4
  ret <4 x float> %vrndm1.i
}

define <2 x double> @test_vrndmq_f64(<2 x double> %a) #0 {
; CHECK: frintm v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
   %vrndm1.i = tail call <2 x double> @llvm.floor.v2f64(<2 x double> %a) #4
  ret <2 x double> %vrndm1.i
}

define <2 x float> @test_vrndx_f32(<2 x float> %a) #0 {
; CHECK: frintx v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vrndx1.i = tail call <2 x float> @llvm.rint.v2f32(<2 x float> %a) #4
  ret <2 x float> %vrndx1.i
}

define <4 x float> @test_vrndxq_f32(<4 x float> %a) #0 {
; CHECK: frintx v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vrndx1.i = tail call <4 x float> @llvm.rint.v4f32(<4 x float> %a) #4
  ret <4 x float> %vrndx1.i
}

define <2 x double> @test_vrndxq_f64(<2 x double> %a) #0 {
; CHECK: frintx v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vrndx1.i = tail call <2 x double> @llvm.rint.v2f64(<2 x double> %a) #4
  ret <2 x double> %vrndx1.i
}

define <2 x float> @test_vrnd_f32(<2 x float> %a) #0 {
; CHECK: frintz v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
   %vrnd1.i = tail call <2 x float> @llvm.trunc.v2f32(<2 x float> %a) #4
  ret <2 x float> %vrnd1.i
}

define <4 x float> @test_vrndq_f32(<4 x float> %a) #0 {
; CHECK: frintz v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vrnd1.i = tail call <4 x float> @llvm.trunc.v4f32(<4 x float> %a) #4
  ret <4 x float> %vrnd1.i
}

define <2 x double> @test_vrndq_f64(<2 x double> %a) #0 {
; CHECK: frintz v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vrnd1.i = tail call <2 x double> @llvm.trunc.v2f64(<2 x double> %a) #4
  ret <2 x double> %vrnd1.i
}

define <2 x float> @test_vrndi_f32(<2 x float> %a) #0 {
; CHECK: frinti v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vrndi1.i = tail call <2 x float> @llvm.nearbyint.v2f32(<2 x float> %a) #4
  ret <2 x float> %vrndi1.i
}

define <4 x float> @test_vrndiq_f32(<4 x float> %a) #0 {
; CHECK: frinti v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vrndi1.i = tail call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %a) #4
  ret <4 x float> %vrndi1.i
}

define <2 x double> @test_vrndiq_f64(<2 x double> %a) #0 {
; CHECK: frinti v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vrndi1.i = tail call <2 x double> @llvm.nearbyint.v2f64(<2 x double> %a) #4
  ret <2 x double> %vrndi1.i
}

define <2 x i32> @test_vcvt_s32_f32(<2 x float> %a) #0 {
; CHECK: fcvtzs v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vcvt.i = fptosi <2 x float> %a to <2 x i32>
  ret <2 x i32> %vcvt.i
}

define <4 x i32> @test_vcvtq_s32_f32(<4 x float> %a) #0 {
; CHECK: fcvtzs v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vcvt.i = fptosi <4 x float> %a to <4 x i32>
  ret <4 x i32> %vcvt.i
}

define <2 x i64> @test_vcvtq_s64_f64(<2 x double> %a) #0 {
; CHECK: fcvtzs v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vcvt.i = fptosi <2 x double> %a to <2 x i64>
  ret <2 x i64> %vcvt.i
}

define <2 x i32> @test_vcvt_u32_f32(<2 x float> %a) #0 {
; CHECK: fcvtzu v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vcvt.i = fptoui <2 x float> %a to <2 x i32>
  ret <2 x i32> %vcvt.i
}

define <4 x i32> @test_vcvtq_u32_f32(<4 x float> %a) #0 {
; CHECK: fcvtzu v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vcvt.i = fptoui <4 x float> %a to <4 x i32>
  ret <4 x i32> %vcvt.i
}

define <2 x i64> @test_vcvtq_u64_f64(<2 x double> %a) #0 {
; CHECK: fcvtzu v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vcvt.i = fptoui <2 x double> %a to <2 x i64>
  ret <2 x i64> %vcvt.i
}

define <2 x i32> @test_vcvtn_s32_f32(<2 x float> %a) #0 {
; CHECK: fcvtns v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vcvtns_f321.i = tail call <2 x i32> @llvm.aarch64.neon.fcvtns.v2i32.v2f32(<2 x float> %a) #4
  ret <2 x i32> %vcvtns_f321.i
}

define <4 x i32> @test_vcvtnq_s32_f32(<4 x float> %a) #0 {
; CHECK: fcvtns v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vcvtns_f321.i = tail call <4 x i32> @llvm.aarch64.neon.fcvtns.v4i32.v4f32(<4 x float> %a) #4
  ret <4 x i32> %vcvtns_f321.i
}

define <2 x i64> @test_vcvtnq_s64_f64(<2 x double> %a) #0 {
; CHECK: fcvtns v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vcvtns_f641.i = tail call <2 x i64> @llvm.aarch64.neon.fcvtns.v2i64.v2f64(<2 x double> %a) #4
  ret <2 x i64> %vcvtns_f641.i
}

define <2 x i32> @test_vcvtn_u32_f32(<2 x float> %a) #0 {
; CHECK: fcvtnu v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vcvtnu_f321.i = tail call <2 x i32> @llvm.aarch64.neon.fcvtnu.v2i32.v2f32(<2 x float> %a) #4
  ret <2 x i32> %vcvtnu_f321.i
}

define <4 x i32> @test_vcvtnq_u32_f32(<4 x float> %a) #0 {
; CHECK: fcvtnu v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vcvtnu_f321.i = tail call <4 x i32> @llvm.aarch64.neon.fcvtnu.v4i32.v4f32(<4 x float> %a) #4
  ret <4 x i32> %vcvtnu_f321.i
}

define <2 x i64> @test_vcvtnq_u64_f64(<2 x double> %a) #0 {
; CHECK: fcvtnu v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vcvtnu_f641.i = tail call <2 x i64> @llvm.aarch64.neon.fcvtnu.v2i64.v2f64(<2 x double> %a) #4
  ret <2 x i64> %vcvtnu_f641.i
}

define <2 x i32> @test_vcvtp_s32_f32(<2 x float> %a) #0 {
; CHECK: fcvtps v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vcvtps_f321.i = tail call <2 x i32> @llvm.aarch64.neon.fcvtps.v2i32.v2f32(<2 x float> %a) #4
  ret <2 x i32> %vcvtps_f321.i
}

define <4 x i32> @test_vcvtpq_s32_f32(<4 x float> %a) #0 {
; CHECK: fcvtps v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vcvtps_f321.i = tail call <4 x i32> @llvm.aarch64.neon.fcvtps.v4i32.v4f32(<4 x float> %a) #4
  ret <4 x i32> %vcvtps_f321.i
}

define <2 x i64> @test_vcvtpq_s64_f64(<2 x double> %a) #0 {
; CHECK: fcvtps v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vcvtps_f641.i = tail call <2 x i64> @llvm.aarch64.neon.fcvtps.v2i64.v2f64(<2 x double> %a) #4
  ret <2 x i64> %vcvtps_f641.i
}

define <2 x i32> @test_vcvtp_u32_f32(<2 x float> %a) #0 {
; CHECK: fcvtpu v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vcvtpu_f321.i = tail call <2 x i32> @llvm.aarch64.neon.fcvtpu.v2i32.v2f32(<2 x float> %a) #4
  ret <2 x i32> %vcvtpu_f321.i
}

define <4 x i32> @test_vcvtpq_u32_f32(<4 x float> %a) #0 {
; CHECK: fcvtpu v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vcvtpu_f321.i = tail call <4 x i32> @llvm.aarch64.neon.fcvtpu.v4i32.v4f32(<4 x float> %a) #4
  ret <4 x i32> %vcvtpu_f321.i
}

define <2 x i64> @test_vcvtpq_u64_f64(<2 x double> %a) #0 {
; CHECK: fcvtpu v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vcvtpu_f641.i = tail call <2 x i64> @llvm.aarch64.neon.fcvtpu.v2i64.v2f64(<2 x double> %a) #4
  ret <2 x i64> %vcvtpu_f641.i
}

define <2 x i32> @test_vcvtm_s32_f32(<2 x float> %a) #0 {
; CHECK: fcvtms v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vcvtms_f321.i = tail call <2 x i32> @llvm.aarch64.neon.fcvtms.v2i32.v2f32(<2 x float> %a) #4
  ret <2 x i32> %vcvtms_f321.i
}

define <4 x i32> @test_vcvtmq_s32_f32(<4 x float> %a) #0 {
; CHECK: fcvtms v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vcvtms_f321.i = tail call <4 x i32> @llvm.aarch64.neon.fcvtms.v4i32.v4f32(<4 x float> %a) #4
  ret <4 x i32> %vcvtms_f321.i
}

define <2 x i64> @test_vcvtmq_s64_f64(<2 x double> %a) #0 {
; CHECK: fcvtms v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vcvtms_f641.i = tail call <2 x i64> @llvm.aarch64.neon.fcvtms.v2i64.v2f64(<2 x double> %a) #4
  ret <2 x i64> %vcvtms_f641.i
}

define <2 x i32> @test_vcvtm_u32_f32(<2 x float> %a) #0 {
; CHECK: fcvtmu v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vcvtmu_f321.i = tail call <2 x i32> @llvm.aarch64.neon.fcvtmu.v2i32.v2f32(<2 x float> %a) #4
  ret <2 x i32> %vcvtmu_f321.i
}

define <4 x i32> @test_vcvtmq_u32_f32(<4 x float> %a) #0 {
; CHECK: fcvtmu v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vcvtmu_f321.i = tail call <4 x i32> @llvm.aarch64.neon.fcvtmu.v4i32.v4f32(<4 x float> %a) #4
  ret <4 x i32> %vcvtmu_f321.i
}

define <2 x i64> @test_vcvtmq_u64_f64(<2 x double> %a) #0 {
; CHECK: fcvtmu v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vcvtmu_f641.i = tail call <2 x i64> @llvm.aarch64.neon.fcvtmu.v2i64.v2f64(<2 x double> %a) #4
  ret <2 x i64> %vcvtmu_f641.i
}

define <2 x i32> @test_vcvta_s32_f32(<2 x float> %a) #0 {
; CHECK: fcvtas v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vcvtas_f321.i = tail call <2 x i32> @llvm.aarch64.neon.fcvtas.v2i32.v2f32(<2 x float> %a) #4
  ret <2 x i32> %vcvtas_f321.i
}

define <4 x i32> @test_vcvtaq_s32_f32(<4 x float> %a) #0 {
; CHECK: fcvtas v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vcvtas_f321.i = tail call <4 x i32> @llvm.aarch64.neon.fcvtas.v4i32.v4f32(<4 x float> %a) #4
  ret <4 x i32> %vcvtas_f321.i
}

define <2 x i64> @test_vcvtaq_s64_f64(<2 x double> %a) #0 {
; CHECK: fcvtas v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vcvtas_f641.i = tail call <2 x i64> @llvm.aarch64.neon.fcvtas.v2i64.v2f64(<2 x double> %a) #4
  ret <2 x i64> %vcvtas_f641.i
}

define <2 x i32> @test_vcvta_u32_f32(<2 x float> %a) #0 {
; CHECK: fcvtau v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vcvtau_f321.i = tail call <2 x i32> @llvm.aarch64.neon.fcvtau.v2i32.v2f32(<2 x float> %a) #4
  ret <2 x i32> %vcvtau_f321.i
}

define <4 x i32> @test_vcvtaq_u32_f32(<4 x float> %a) #0 {
; CHECK: fcvtau v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vcvtau_f321.i = tail call <4 x i32> @llvm.aarch64.neon.fcvtau.v4i32.v4f32(<4 x float> %a) #4
  ret <4 x i32> %vcvtau_f321.i
}

define <2 x i64> @test_vcvtaq_u64_f64(<2 x double> %a) #0 {
; CHECK: fcvtau v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vcvtau_f641.i = tail call <2 x i64> @llvm.aarch64.neon.fcvtau.v2i64.v2f64(<2 x double> %a) #4
  ret <2 x i64> %vcvtau_f641.i
}

define <2 x float> @test_vrsqrte_f32(<2 x float> %a) #0 {
; CHECK: frsqrte v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vrsqrte1.i = tail call <2 x float> @llvm.arm.neon.vrsqrte.v2f32(<2 x float> %a) #4
  ret <2 x float> %vrsqrte1.i
}

define <4 x float> @test_vrsqrteq_f32(<4 x float> %a) #0 {
; CHECK: frsqrte v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vrsqrte1.i = tail call <4 x float> @llvm.arm.neon.vrsqrte.v4f32(<4 x float> %a) #4
  ret <4 x float> %vrsqrte1.i
}

define <2 x double> @test_vrsqrteq_f64(<2 x double> %a) #0 {
; CHECK: frsqrte v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vrsqrte1.i = tail call <2 x double> @llvm.arm.neon.vrsqrte.v2f64(<2 x double> %a) #4
  ret <2 x double> %vrsqrte1.i
}

define <2 x float> @test_vrecpe_f32(<2 x float> %a) #0 {
; CHECK: frecpe v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vrecpe1.i = tail call <2 x float> @llvm.arm.neon.vrecpe.v2f32(<2 x float> %a) #4
  ret <2 x float> %vrecpe1.i
}

define <4 x float> @test_vrecpeq_f32(<4 x float> %a) #0 {
; CHECK: frecpe v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vrecpe1.i = tail call <4 x float> @llvm.arm.neon.vrecpe.v4f32(<4 x float> %a) #4
  ret <4 x float> %vrecpe1.i
}

define <2 x double> @test_vrecpeq_f64(<2 x double> %a) #0 {
; CHECK: frecpe v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vrecpe1.i = tail call <2 x double> @llvm.arm.neon.vrecpe.v2f64(<2 x double> %a) #4
  ret <2 x double> %vrecpe1.i
}

define <2 x i32> @test_vrecpe_u32(<2 x i32> %a) #0 {
; CHECK: urecpe v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vrecpe1.i = tail call <2 x i32> @llvm.arm.neon.vrecpe.v2i32(<2 x i32> %a) #4
  ret <2 x i32> %vrecpe1.i
}

define <4 x i32> @test_vrecpeq_u32(<4 x i32> %a) #0 {
; CHECK: urecpe v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vrecpe1.i = tail call <4 x i32> @llvm.arm.neon.vrecpe.v4i32(<4 x i32> %a) #4
  ret <4 x i32> %vrecpe1.i
}

define <2 x float> @test_vsqrt_f32(<2 x float> %a) #0 {
; CHECK: fsqrt v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vsqrt1.i = tail call <2 x float> @llvm.sqrt.v2f32(<2 x float> %a) #4
  ret <2 x float> %vsqrt1.i
}

define <4 x float> @test_vsqrtq_f32(<4 x float> %a) #0 {
; CHECK: fsqrt v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vsqrt1.i = tail call <4 x float> @llvm.sqrt.v4f32(<4 x float> %a) #4
  ret <4 x float> %vsqrt1.i
}

define <2 x double> @test_vsqrtq_f64(<2 x double> %a) #0 {
; CHECK: fsqrt v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vsqrt1.i = tail call <2 x double> @llvm.sqrt.v2f64(<2 x double> %a) #4
  ret <2 x double> %vsqrt1.i
}

define <2 x float> @test_vcvt_f32_s32(<2 x i32> %a) #0 {
; CHECK: scvtf v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vcvt.i = sitofp <2 x i32> %a to <2 x float>
  ret <2 x float> %vcvt.i
}

define <2 x float> @test_vcvt_f32_u32(<2 x i32> %a) #0 {
; CHECK: ucvtf v{{[0-9]+}}.2s, v{{[0-9]+}}.2s
  %vcvt.i = uitofp <2 x i32> %a to <2 x float>
  ret <2 x float> %vcvt.i
}

define <4 x float> @test_vcvtq_f32_s32(<4 x i32> %a) #0 {
; CHECK: scvtf v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vcvt.i = sitofp <4 x i32> %a to <4 x float>
  ret <4 x float> %vcvt.i
}

define <4 x float> @test_vcvtq_f32_u32(<4 x i32> %a) #0 {
; CHECK: ucvtf v{{[0-9]+}}.4s, v{{[0-9]+}}.4s
  %vcvt.i = uitofp <4 x i32> %a to <4 x float>
  ret <4 x float> %vcvt.i
}

define <2 x double> @test_vcvtq_f64_s64(<2 x i64> %a) #0 {
; CHECK: scvtf v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vcvt.i = sitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %vcvt.i
}

define <2 x double> @test_vcvtq_f64_u64(<2 x i64> %a) #0 {
; CHECK: ucvtf v{{[0-9]+}}.2d, v{{[0-9]+}}.2d
  %vcvt.i = uitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %vcvt.i
}

declare <2 x double> @llvm.sqrt.v2f64(<2 x double>) #2

declare <4 x float> @llvm.sqrt.v4f32(<4 x float>) #2

declare <2 x float> @llvm.sqrt.v2f32(<2 x float>) #2

declare <4 x i32> @llvm.arm.neon.vrecpe.v4i32(<4 x i32>) #2

declare <2 x i32> @llvm.arm.neon.vrecpe.v2i32(<2 x i32>) #2

declare <2 x double> @llvm.arm.neon.vrecpe.v2f64(<2 x double>) #2

declare <4 x float> @llvm.arm.neon.vrecpe.v4f32(<4 x float>) #2

declare <2 x float> @llvm.arm.neon.vrecpe.v2f32(<2 x float>) #2

declare <2 x double> @llvm.arm.neon.vrsqrte.v2f64(<2 x double>) #2

declare <4 x float> @llvm.arm.neon.vrsqrte.v4f32(<4 x float>) #2

declare <2 x float> @llvm.arm.neon.vrsqrte.v2f32(<2 x float>) #2

declare <2 x i64> @llvm.aarch64.neon.fcvtau.v2i64.v2f64(<2 x double>) #2

declare <4 x i32> @llvm.aarch64.neon.fcvtau.v4i32.v4f32(<4 x float>) #2

declare <2 x i32> @llvm.aarch64.neon.fcvtau.v2i32.v2f32(<2 x float>) #2

declare <2 x i64> @llvm.aarch64.neon.fcvtas.v2i64.v2f64(<2 x double>) #2

declare <4 x i32> @llvm.aarch64.neon.fcvtas.v4i32.v4f32(<4 x float>) #2

declare <2 x i32> @llvm.aarch64.neon.fcvtas.v2i32.v2f32(<2 x float>) #2

declare <2 x i64> @llvm.aarch64.neon.fcvtmu.v2i64.v2f64(<2 x double>) #2

declare <4 x i32> @llvm.aarch64.neon.fcvtmu.v4i32.v4f32(<4 x float>) #2

declare <2 x i32> @llvm.aarch64.neon.fcvtmu.v2i32.v2f32(<2 x float>) #2

declare <2 x i64> @llvm.aarch64.neon.fcvtms.v2i64.v2f64(<2 x double>) #2

declare <4 x i32> @llvm.aarch64.neon.fcvtms.v4i32.v4f32(<4 x float>) #2

declare <2 x i32> @llvm.aarch64.neon.fcvtms.v2i32.v2f32(<2 x float>) #2

declare <2 x i64> @llvm.aarch64.neon.fcvtpu.v2i64.v2f64(<2 x double>) #2

declare <4 x i32> @llvm.aarch64.neon.fcvtpu.v4i32.v4f32(<4 x float>) #2

declare <2 x i32> @llvm.aarch64.neon.fcvtpu.v2i32.v2f32(<2 x float>) #2

declare <2 x i64> @llvm.aarch64.neon.fcvtps.v2i64.v2f64(<2 x double>) #2

declare <4 x i32> @llvm.aarch64.neon.fcvtps.v4i32.v4f32(<4 x float>) #2

declare <2 x i32> @llvm.aarch64.neon.fcvtps.v2i32.v2f32(<2 x float>) #2

declare <2 x i64> @llvm.aarch64.neon.fcvtnu.v2i64.v2f64(<2 x double>) #2

declare <4 x i32> @llvm.aarch64.neon.fcvtnu.v4i32.v4f32(<4 x float>) #2

declare <2 x i32> @llvm.aarch64.neon.fcvtnu.v2i32.v2f32(<2 x float>) #2

declare <2 x i64> @llvm.aarch64.neon.fcvtns.v2i64.v2f64(<2 x double>) #2

declare <4 x i32> @llvm.aarch64.neon.fcvtns.v4i32.v4f32(<4 x float>) #2

declare <2 x i32> @llvm.aarch64.neon.fcvtns.v2i32.v2f32(<2 x float>) #2

declare <2 x double> @llvm.nearbyint.v2f64(<2 x double>) #3

declare <4 x float> @llvm.nearbyint.v4f32(<4 x float>) #3

declare <2 x float> @llvm.nearbyint.v2f32(<2 x float>) #3

declare <2 x double> @llvm.trunc.v2f64(<2 x double>) #3

declare <4 x float> @llvm.trunc.v4f32(<4 x float>) #3

declare <2 x float> @llvm.trunc.v2f32(<2 x float>) #3

declare <2 x double> @llvm.rint.v2f64(<2 x double>) #3

declare <4 x float> @llvm.rint.v4f32(<4 x float>) #3

declare <2 x float> @llvm.rint.v2f32(<2 x float>) #3

declare <2 x double> @llvm.floor.v2f64(<2 x double>) #3

declare <4 x float> @llvm.floor.v4f32(<4 x float>) #3

declare <2 x float> @llvm.floor.v2f32(<2 x float>) #3

declare <2 x double> @llvm.ceil.v2f64(<2 x double>) #3

declare <4 x float> @llvm.ceil.v4f32(<4 x float>) #3

declare <2 x float> @llvm.ceil.v2f32(<2 x float>) #3

declare <2 x double> @llvm.round.v2f64(<2 x double>) #3

declare <4 x float> @llvm.round.v4f32(<4 x float>) #3

declare <2 x float> @llvm.round.v2f32(<2 x float>) #3

declare <2 x double> @llvm.aarch64.neon.frintn.v2f64(<2 x double>) #2

declare <4 x float> @llvm.aarch64.neon.frintn.v4f32(<4 x float>) #2

declare <2 x float> @llvm.aarch64.neon.frintn.v2f32(<2 x float>) #2

declare <2 x float> @llvm.aarch64.neon.fcvtxn.v2f32.v2f64(<2 x double>) #2

declare <2 x float> @llvm.aarch64.neon.fcvtn.v2f32.v2f64(<2 x double>) #2

declare <2 x i32> @llvm.arm.neon.vqmovnu.v2i32(<2 x i64>) #2

declare <4 x i16> @llvm.arm.neon.vqmovnu.v4i16(<4 x i32>) #2

declare <8 x i8> @llvm.arm.neon.vqmovnu.v8i8(<8 x i16>) #2

declare <2 x i32> @llvm.arm.neon.vqmovns.v2i32(<2 x i64>) #2

declare <4 x i16> @llvm.arm.neon.vqmovns.v4i16(<4 x i32>) #2

declare <8 x i8> @llvm.arm.neon.vqmovns.v8i8(<8 x i16>) #2

declare <2 x i32> @llvm.arm.neon.vqmovnsu.v2i32(<2 x i64>) #2

declare <4 x i16> @llvm.arm.neon.vqmovnsu.v4i16(<4 x i32>) #2

declare <8 x i8> @llvm.arm.neon.vqmovnsu.v8i8(<8 x i16>) #2

declare <16 x i8> @llvm.aarch64.neon.rbit.v16i8(<16 x i8>) #2

declare <8 x i8> @llvm.aarch64.neon.rbit.v8i8(<8 x i8>) #2

declare <16 x i8> @llvm.ctpop.v16i8(<16 x i8>) #2

declare <8 x i8> @llvm.ctpop.v8i8(<8 x i8>) #2

declare <4 x i32> @llvm.ctlz.v4i32(<4 x i32>, i1) #2

declare <2 x i32> @llvm.ctlz.v2i32(<2 x i32>, i1) #2

declare <8 x i16> @llvm.ctlz.v8i16(<8 x i16>, i1) #2

declare <4 x i16> @llvm.ctlz.v4i16(<4 x i16>, i1) #2

declare <16 x i8> @llvm.ctlz.v16i8(<16 x i8>, i1) #2

declare <8 x i8> @llvm.ctlz.v8i8(<8 x i8>, i1) #2

declare <4 x i32> @llvm.arm.neon.vcls.v4i32(<4 x i32>) #2

declare <2 x i32> @llvm.arm.neon.vcls.v2i32(<2 x i32>) #2

declare <8 x i16> @llvm.arm.neon.vcls.v8i16(<8 x i16>) #2

declare <4 x i16> @llvm.arm.neon.vcls.v4i16(<4 x i16>) #2

declare <16 x i8> @llvm.arm.neon.vcls.v16i8(<16 x i8>) #2

declare <8 x i8> @llvm.arm.neon.vcls.v8i8(<8 x i8>) #2

declare <2 x i64> @llvm.aarch64.neon.suqadd.v2i64(<2 x i64>, <2 x i64>) #2

declare <4 x i32> @llvm.aarch64.neon.suqadd.v4i32(<4 x i32>, <4 x i32>) #2

declare <2 x i32> @llvm.aarch64.neon.suqadd.v2i32(<2 x i32>, <2 x i32>) #2

declare <8 x i16> @llvm.aarch64.neon.suqadd.v8i16(<8 x i16>, <8 x i16>) #2

declare <4 x i16> @llvm.aarch64.neon.suqadd.v4i16(<4 x i16>, <4 x i16>) #2

declare <16 x i8> @llvm.aarch64.neon.suqadd.v16i8(<16 x i8>, <16 x i8>) #2

declare <8 x i8> @llvm.aarch64.neon.suqadd.v8i8(<8 x i8>, <8 x i8>) #2

declare <2 x double> @llvm.fabs.v2f64(<2 x double>) #3

declare <4 x float> @llvm.fabs.v4f32(<4 x float>) #3

declare <2 x float> @llvm.fabs.v2f32(<2 x float>) #3

declare <2 x i64> @llvm.arm.neon.vabs.v2i64(<2 x i64>) #2

declare <4 x i32> @llvm.arm.neon.vabs.v4i32(<4 x i32>) #2

declare <2 x i32> @llvm.arm.neon.vabs.v2i32(<2 x i32>) #2

declare <8 x i16> @llvm.arm.neon.vabs.v8i16(<8 x i16>) #2

declare <4 x i16> @llvm.arm.neon.vabs.v4i16(<4 x i16>) #2

declare <16 x i8> @llvm.arm.neon.vabs.v16i8(<16 x i8>) #2

declare <8 x i8> @llvm.arm.neon.vabs.v8i8(<8 x i8>) #2

declare <2 x i64> @llvm.arm.neon.vqneg.v2i64(<2 x i64>) #2

declare <4 x i32> @llvm.arm.neon.vqneg.v4i32(<4 x i32>) #2

declare <2 x i32> @llvm.arm.neon.vqneg.v2i32(<2 x i32>) #2

declare <8 x i16> @llvm.arm.neon.vqneg.v8i16(<8 x i16>) #2

declare <4 x i16> @llvm.arm.neon.vqneg.v4i16(<4 x i16>) #2

declare <16 x i8> @llvm.arm.neon.vqneg.v16i8(<16 x i8>) #2

declare <8 x i8> @llvm.arm.neon.vqneg.v8i8(<8 x i8>) #2

declare <2 x i64> @llvm.arm.neon.vqabs.v2i64(<2 x i64>) #2

declare <4 x i32> @llvm.arm.neon.vqabs.v4i32(<4 x i32>) #2

declare <2 x i32> @llvm.arm.neon.vqabs.v2i32(<2 x i32>) #2

declare <8 x i16> @llvm.arm.neon.vqabs.v8i16(<8 x i16>) #2

declare <4 x i16> @llvm.arm.neon.vqabs.v4i16(<4 x i16>) #2

declare <16 x i8> @llvm.arm.neon.vqabs.v16i8(<16 x i8>) #2

declare <8 x i8> @llvm.arm.neon.vqabs.v8i8(<8 x i8>) #2

declare <2 x i64> @llvm.arm.neon.vpadalu.v2i64.v4i32(<2 x i64>, <4 x i32>) #2

declare <4 x i32> @llvm.arm.neon.vpadalu.v4i32.v8i16(<4 x i32>, <8 x i16>) #2

declare <8 x i16> @llvm.arm.neon.vpadalu.v8i16.v16i8(<8 x i16>, <16 x i8>) #2

declare <2 x i64> @llvm.arm.neon.vpadals.v2i64.v4i32(<2 x i64>, <4 x i32>) #2

declare <4 x i32> @llvm.arm.neon.vpadals.v4i32.v8i16(<4 x i32>, <8 x i16>) #2

declare <8 x i16> @llvm.arm.neon.vpadals.v8i16.v16i8(<8 x i16>, <16 x i8>) #2

declare <1 x i64> @llvm.arm.neon.vpadalu.v1i64.v2i32(<1 x i64>, <2 x i32>) #2

declare <2 x i32> @llvm.arm.neon.vpadalu.v2i32.v4i16(<2 x i32>, <4 x i16>) #2

declare <4 x i16> @llvm.arm.neon.vpadalu.v4i16.v8i8(<4 x i16>, <8 x i8>) #2

declare <1 x i64> @llvm.arm.neon.vpadals.v1i64.v2i32(<1 x i64>, <2 x i32>) #2

declare <2 x i32> @llvm.arm.neon.vpadals.v2i32.v4i16(<2 x i32>, <4 x i16>) #2

declare <4 x i16> @llvm.arm.neon.vpadals.v4i16.v8i8(<4 x i16>, <8 x i8>) #2

declare <2 x i64> @llvm.arm.neon.vpaddlu.v2i64.v4i32(<4 x i32>) #2

declare <4 x i32> @llvm.arm.neon.vpaddlu.v4i32.v8i16(<8 x i16>) #2

declare <8 x i16> @llvm.arm.neon.vpaddlu.v8i16.v16i8(<16 x i8>) #2

declare <2 x i64> @llvm.arm.neon.vpaddls.v2i64.v4i32(<4 x i32>) #2

declare <4 x i32> @llvm.arm.neon.vpaddls.v4i32.v8i16(<8 x i16>) #2

declare <8 x i16> @llvm.arm.neon.vpaddls.v8i16.v16i8(<16 x i8>) #2

declare <1 x i64> @llvm.arm.neon.vpaddlu.v1i64.v2i32(<2 x i32>) #2

declare <2 x i32> @llvm.arm.neon.vpaddlu.v2i32.v4i16(<4 x i16>) #2

declare <4 x i16> @llvm.arm.neon.vpaddlu.v4i16.v8i8(<8 x i8>) #2

declare <1 x i64> @llvm.arm.neon.vpaddls.v1i64.v2i32(<2 x i32>) #2

declare <2 x i32> @llvm.arm.neon.vpaddls.v2i32.v4i16(<4 x i16>) #2

declare <4 x i16> @llvm.arm.neon.vpaddls.v4i16.v8i8(<8 x i8>) #2

declare <4 x float> @llvm.arm.neon.vcvthf2fp(<4 x i16>) #2

declare <4 x i16> @llvm.arm.neon.vcvtfp2hf(<4 x float>) #2


define <1 x i64> @test_vcvt_s64_f64(<1 x double> %a) {
; CHECK-LABEL: test_vcvt_s64_f64
; CHECK: fcvtzs d{{[0-9]+}}, d{{[0-9]+}}
  %1 = fptosi <1 x double> %a to <1 x i64>
  ret <1 x i64> %1
}

define <1 x i64> @test_vcvt_u64_f64(<1 x double> %a) {
; CHECK-LABEL: test_vcvt_u64_f64
; CHECK: fcvtzu d{{[0-9]+}}, d{{[0-9]+}}
  %1 = fptoui <1 x double> %a to <1 x i64>
  ret <1 x i64> %1
}

define <1 x i64> @test_vcvtn_s64_f64(<1 x double> %a) {
; CHECK-LABEL: test_vcvtn_s64_f64
; CHECK: fcvtns d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x i64> @llvm.aarch64.neon.fcvtns.v1i64.v1f64(<1 x double> %a)
  ret <1 x i64> %1
}

define <1 x i64> @test_vcvtn_u64_f64(<1 x double> %a) {
; CHECK-LABEL: test_vcvtn_u64_f64
; CHECK: fcvtnu d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x i64> @llvm.aarch64.neon.fcvtnu.v1i64.v1f64(<1 x double> %a)
  ret <1 x i64> %1
}

define <1 x i64> @test_vcvtp_s64_f64(<1 x double> %a) {
; CHECK-LABEL: test_vcvtp_s64_f64
; CHECK: fcvtps d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x i64> @llvm.aarch64.neon.fcvtps.v1i64.v1f64(<1 x double> %a)
  ret <1 x i64> %1
}

define <1 x i64> @test_vcvtp_u64_f64(<1 x double> %a) {
; CHECK-LABEL: test_vcvtp_u64_f64
; CHECK: fcvtpu d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x i64> @llvm.aarch64.neon.fcvtpu.v1i64.v1f64(<1 x double> %a)
  ret <1 x i64> %1
}

define <1 x i64> @test_vcvtm_s64_f64(<1 x double> %a) {
; CHECK-LABEL: test_vcvtm_s64_f64
; CHECK: fcvtms d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x i64> @llvm.aarch64.neon.fcvtms.v1i64.v1f64(<1 x double> %a)
  ret <1 x i64> %1
}

define <1 x i64> @test_vcvtm_u64_f64(<1 x double> %a) {
; CHECK-LABEL: test_vcvtm_u64_f64
; CHECK: fcvtmu d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x i64> @llvm.aarch64.neon.fcvtmu.v1i64.v1f64(<1 x double> %a)
  ret <1 x i64> %1
}

define <1 x i64> @test_vcvta_s64_f64(<1 x double> %a) {
; CHECK-LABEL: test_vcvta_s64_f64
; CHECK: fcvtas d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x i64> @llvm.aarch64.neon.fcvtas.v1i64.v1f64(<1 x double> %a)
  ret <1 x i64> %1
}

define <1 x i64> @test_vcvta_u64_f64(<1 x double> %a) {
; CHECK-LABEL: test_vcvta_u64_f64
; CHECK: fcvtau d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x i64> @llvm.aarch64.neon.fcvtau.v1i64.v1f64(<1 x double> %a)
  ret <1 x i64> %1
}

define <1 x double> @test_vcvt_f64_s64(<1 x i64> %a) {
; CHECK-LABEL: test_vcvt_f64_s64
; CHECK: scvtf d{{[0-9]+}}, d{{[0-9]+}}
  %1 = sitofp <1 x i64> %a to <1 x double>
  ret <1 x double> %1
}

define <1 x double> @test_vcvt_f64_u64(<1 x i64> %a) {
; CHECK-LABEL: test_vcvt_f64_u64
; CHECK: ucvtf d{{[0-9]+}}, d{{[0-9]+}}
  %1 = uitofp <1 x i64> %a to <1 x double>
  ret <1 x double> %1
}

declare <1 x i64> @llvm.aarch64.neon.fcvtau.v1i64.v1f64(<1 x double>)
declare <1 x i64> @llvm.aarch64.neon.fcvtas.v1i64.v1f64(<1 x double>)
declare <1 x i64> @llvm.aarch64.neon.fcvtmu.v1i64.v1f64(<1 x double>)
declare <1 x i64> @llvm.aarch64.neon.fcvtms.v1i64.v1f64(<1 x double>)
declare <1 x i64> @llvm.aarch64.neon.fcvtpu.v1i64.v1f64(<1 x double>)
declare <1 x i64> @llvm.aarch64.neon.fcvtps.v1i64.v1f64(<1 x double>)
declare <1 x i64> @llvm.aarch64.neon.fcvtnu.v1i64.v1f64(<1 x double>)
declare <1 x i64> @llvm.aarch64.neon.fcvtns.v1i64.v1f64(<1 x double>)

define <1 x double> @test_vrndn_f64(<1 x double> %a) {
; CHECK-LABEL: test_vrndn_f64
; CHECK: frintn d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.aarch64.neon.frintn.v1f64(<1 x double> %a)
  ret <1 x double> %1
}

define <1 x double> @test_vrnda_f64(<1 x double> %a) {
; CHECK-LABEL: test_vrnda_f64
; CHECK: frinta d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.round.v1f64(<1 x double> %a)
  ret <1 x double> %1
}

define <1 x double> @test_vrndp_f64(<1 x double> %a) {
; CHECK-LABEL: test_vrndp_f64
; CHECK: frintp d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.ceil.v1f64(<1 x double> %a)
  ret <1 x double> %1
}

define <1 x double> @test_vrndm_f64(<1 x double> %a) {
; CHECK-LABEL: test_vrndm_f64
; CHECK: frintm d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.floor.v1f64(<1 x double> %a)
  ret <1 x double> %1
}

define <1 x double> @test_vrndx_f64(<1 x double> %a) {
; CHECK-LABEL: test_vrndx_f64
; CHECK: frintx d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.rint.v1f64(<1 x double> %a)
  ret <1 x double> %1
}

define <1 x double> @test_vrnd_f64(<1 x double> %a) {
; CHECK-LABEL: test_vrnd_f64
; CHECK: frintz d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.trunc.v1f64(<1 x double> %a)
  ret <1 x double> %1
}

define <1 x double> @test_vrndi_f64(<1 x double> %a) {
; CHECK-LABEL: test_vrndi_f64
; CHECK: frinti d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.nearbyint.v1f64(<1 x double> %a)
  ret <1 x double> %1
}

declare <1 x double> @llvm.nearbyint.v1f64(<1 x double>)
declare <1 x double> @llvm.trunc.v1f64(<1 x double>)
declare <1 x double> @llvm.rint.v1f64(<1 x double>)
declare <1 x double> @llvm.floor.v1f64(<1 x double>)
declare <1 x double> @llvm.ceil.v1f64(<1 x double>)
declare <1 x double> @llvm.round.v1f64(<1 x double>)
declare <1 x double> @llvm.aarch64.neon.frintn.v1f64(<1 x double>)

define <1 x double> @test_vrsqrte_f64(<1 x double> %a) {
; CHECK-LABEL: test_vrsqrte_f64
; CHECK: frsqrte d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.arm.neon.vrsqrte.v1f64(<1 x double> %a)
  ret <1 x double> %1
}

define <1 x double> @test_vrecpe_f64(<1 x double> %a) {
; CHECK-LABEL: test_vrecpe_f64
; CHECK: frecpe d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.arm.neon.vrecpe.v1f64(<1 x double> %a)
  ret <1 x double> %1
}

define <1 x double> @test_vsqrt_f64(<1 x double> %a) {
; CHECK-LABEL: test_vsqrt_f64
; CHECK: fsqrt d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.sqrt.v1f64(<1 x double> %a)
  ret <1 x double> %1
}

define <1 x double> @test_vrecps_f64(<1 x double> %a, <1 x double> %b) {
; CHECK-LABEL: test_vrecps_f64
; CHECK: frecps d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.arm.neon.vrecps.v1f64(<1 x double> %a, <1 x double> %b)
  ret <1 x double> %1
}

define <1 x double> @test_vrsqrts_f64(<1 x double> %a, <1 x double> %b) {
; CHECK-LABEL: test_vrsqrts_f64
; CHECK: frsqrts d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.arm.neon.vrsqrts.v1f64(<1 x double> %a, <1 x double> %b)
  ret <1 x double> %1
}

declare <1 x double> @llvm.arm.neon.vrsqrts.v1f64(<1 x double>, <1 x double>)
declare <1 x double> @llvm.arm.neon.vrecps.v1f64(<1 x double>, <1 x double>)
declare <1 x double> @llvm.sqrt.v1f64(<1 x double>)
declare <1 x double> @llvm.arm.neon.vrecpe.v1f64(<1 x double>)
declare <1 x double> @llvm.arm.neon.vrsqrte.v1f64(<1 x double>)