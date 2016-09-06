; RUN: llc -mtriple=arm64-none-linux-gnu -mattr=+neon -fp-contract=fast \
; RUN:     < %s -verify-machineinstrs -asm-verbose=false | FileCheck %s

define <4 x i32> @test_vmull_high_n_s16(<8 x i16> %a, i16 %b) #0 {
; CHECK-LABEL: test_vmull_high_n_s16:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].8h, w0
; CHECK-NEXT: smull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %b, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %b, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %b, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %b, i32 3
  %vmull15.i.i = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  ret <4 x i32> %vmull15.i.i
}

define <4 x i32> @test_vmull_high_n_s16_imm(<8 x i16> %a) #0 {
; CHECK-LABEL: test_vmull_high_n_s16_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].8h, #29
; CHECK-NEXT: smull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vmull15.i.i = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> <i16 29, i16 29, i16 29, i16 29>)
  ret <4 x i32> %vmull15.i.i
}

define <2 x i64> @test_vmull_high_n_s32(<4 x i32> %a, i32 %b) #0 {
; CHECK-LABEL: test_vmull_high_n_s32:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].4s, w0
; CHECK-NEXT: smull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %b, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %b, i32 1
  %vmull9.i.i = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  ret <2 x i64> %vmull9.i.i
}

define <2 x i64> @test_vmull_high_n_s32_imm(<4 x i32> %a) #0 {
; CHECK-LABEL: test_vmull_high_n_s32_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].4s, #1, msl #8
; CHECK-NEXT: smull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vmull9.i.i = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> <i32 511, i32 511>)
  ret <2 x i64> %vmull9.i.i
}

define <4 x i32> @test_vmull_high_n_u16(<8 x i16> %a, i16 %b) #0 {
; CHECK-LABEL: test_vmull_high_n_u16:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].8h, w0
; CHECK-NEXT: umull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %b, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %b, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %b, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %b, i32 3
  %vmull15.i.i = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  ret <4 x i32> %vmull15.i.i
}

define <4 x i32> @test_vmull_high_n_u16_imm(<8 x i16> %a) #0 {
; CHECK-LABEL: test_vmull_high_n_u16_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].8h, #17, lsl #8
; CHECK-NEXT: umull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vmull15.i.i = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> <i16 4352, i16 4352, i16 4352, i16 4352>)
  ret <4 x i32> %vmull15.i.i
}

define <2 x i64> @test_vmull_high_n_u32(<4 x i32> %a, i32 %b) #0 {
; CHECK-LABEL: test_vmull_high_n_u32:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].4s, w0
; CHECK-NEXT: umull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %b, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %b, i32 1
  %vmull9.i.i = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  ret <2 x i64> %vmull9.i.i
}

define <2 x i64> @test_vmull_high_n_u32_imm(<4 x i32> %a) #0 {
; CHECK-LABEL: test_vmull_high_n_u32_imm:
; CHECK-NEXT: mvni [[REPLICATE:v[0-9]+]].4s, #1, msl #8
; CHECK-NEXT: umull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vmull9.i.i = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> <i32 4294966784, i32 4294966784>)
  ret <2 x i64> %vmull9.i.i
}

define <4 x i32> @test_vqdmull_high_n_s16(<8 x i16> %a, i16 %b) #0 {
; CHECK-LABEL: test_vqdmull_high_n_s16:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].8h, w0
; CHECK-NEXT: sqdmull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %b, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %b, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %b, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %b, i32 3
  %vqdmull15.i.i = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  ret <4 x i32> %vqdmull15.i.i
}

define <4 x i32> @test_vqdmull_high_n_s16_imm(<8 x i16> %a) #0 {
; CHECK-LABEL: test_vqdmull_high_n_s16_imm:
; CHECK-NEXT: mvni [[REPLICATE:v[0-9]+]].8h, #17, lsl #8
; CHECK-NEXT: sqdmull2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %a, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vqdmull15.i.i = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> <i16 61183, i16 61183, i16 61183, i16 61183>)
  ret <4 x i32> %vqdmull15.i.i
}

define <2 x i64> @test_vqdmull_high_n_s32(<4 x i32> %a, i32 %b) #0 {
; CHECK-LABEL: test_vqdmull_high_n_s32:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].4s, w0
; CHECK-NEXT: sqdmull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %b, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %b, i32 1
  %vqdmull9.i.i = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  ret <2 x i64> %vqdmull9.i.i
}

define <2 x i64> @test_vqdmull_high_n_s32_imm(<4 x i32> %a) #0 {
; CHECK-LABEL: test_vqdmull_high_n_s32_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].4s, #29
; CHECK-NEXT: sqdmull2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %a, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vqdmull9.i.i = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> <i32 29, i32 29>)
  ret <2 x i64> %vqdmull9.i.i
}

define <4 x i32> @test_vmlal_high_n_s16(<4 x i32> %a, <8 x i16> %b, i16 %c) #0 {
; CHECK-LABEL: test_vmlal_high_n_s16:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].8h, w0
; CHECK-NEXT: smlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %c, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %c, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %c, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %c, i32 3
  %vmull2.i.i.i = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  %add.i.i = add <4 x i32> %vmull2.i.i.i, %a
  ret <4 x i32> %add.i.i
}

define <4 x i32> @test_vmlal_high_n_s16_imm(<4 x i32> %a, <8 x i16> %b) #0 {
; CHECK-LABEL: test_vmlal_high_n_s16_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].8h, #29
; CHECK-NEXT: smlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vmull2.i.i.i = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> <i16 29, i16 29, i16 29, i16 29>)
  %add.i.i = add <4 x i32> %vmull2.i.i.i, %a
  ret <4 x i32> %add.i.i
}

define <2 x i64> @test_vmlal_high_n_s32(<2 x i64> %a, <4 x i32> %b, i32 %c) #0 {
; CHECK-LABEL: test_vmlal_high_n_s32:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].4s, w0
; CHECK-NEXT: smlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %c, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %c, i32 1
  %vmull2.i.i.i = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  %add.i.i = add <2 x i64> %vmull2.i.i.i, %a
  ret <2 x i64> %add.i.i
}

define <2 x i64> @test_vmlal_high_n_s32_imm(<2 x i64> %a, <4 x i32> %b) #0 {
; CHECK-LABEL: test_vmlal_high_n_s32_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].4s, #29
; CHECK-NEXT: smlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vmull2.i.i.i = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> <i32 29, i32 29>)
  %add.i.i = add <2 x i64> %vmull2.i.i.i, %a
  ret <2 x i64> %add.i.i
}

define <4 x i32> @test_vmlal_high_n_u16(<4 x i32> %a, <8 x i16> %b, i16 %c) #0 {
; CHECK-LABEL: test_vmlal_high_n_u16:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].8h, w0
; CHECK-NEXT: umlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %c, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %c, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %c, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %c, i32 3
  %vmull2.i.i.i = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  %add.i.i = add <4 x i32> %vmull2.i.i.i, %a
  ret <4 x i32> %add.i.i
}

define <4 x i32> @test_vmlal_high_n_u16_imm(<4 x i32> %a, <8 x i16> %b) #0 {
; CHECK-LABEL: test_vmlal_high_n_u16_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].8h, #29
; CHECK-NEXT: umlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vmull2.i.i.i = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> <i16 29, i16 29, i16 29, i16 29>)
  %add.i.i = add <4 x i32> %vmull2.i.i.i, %a
  ret <4 x i32> %add.i.i
}

define <2 x i64> @test_vmlal_high_n_u32(<2 x i64> %a, <4 x i32> %b, i32 %c) #0 {
; CHECK-LABEL: test_vmlal_high_n_u32:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].4s, w0
; CHECK-NEXT: umlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %c, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %c, i32 1
  %vmull2.i.i.i = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  %add.i.i = add <2 x i64> %vmull2.i.i.i, %a
  ret <2 x i64> %add.i.i
}

define <2 x i64> @test_vmlal_high_n_u32_imm(<2 x i64> %a, <4 x i32> %b) #0 {
; CHECK-LABEL: test_vmlal_high_n_u32_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].4s, #29
; CHECK-NEXT: umlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vmull2.i.i.i = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> <i32 29, i32 29>)
  %add.i.i = add <2 x i64> %vmull2.i.i.i, %a
  ret <2 x i64> %add.i.i
}

define <4 x i32> @test_vqdmlal_high_n_s16(<4 x i32> %a, <8 x i16> %b, i16 %c) #0 {
; CHECK-LABEL: test_vqdmlal_high_n_s16:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].8h, w0
; CHECK-NEXT: sqdmlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %c, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %c, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %c, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %c, i32 3
  %vqdmlal15.i.i = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  %vqdmlal17.i.i = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> %vqdmlal15.i.i)
  ret <4 x i32> %vqdmlal17.i.i
}

define <4 x i32> @test_vqdmlal_high_n_s16_imm(<4 x i32> %a, <8 x i16> %b) #0 {
; CHECK-LABEL: test_vqdmlal_high_n_s16_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].8h, #29
; CHECK-NEXT: sqdmlal2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vqdmlal15.i.i = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> <i16 29, i16 29, i16 29, i16 29>)
  %vqdmlal17.i.i = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> %vqdmlal15.i.i)
  ret <4 x i32> %vqdmlal17.i.i
}

define <2 x i64> @test_vqdmlal_high_n_s32(<2 x i64> %a, <4 x i32> %b, i32 %c) #0 {
; CHECK-LABEL: test_vqdmlal_high_n_s32:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].4s, w0
; CHECK-NEXT: sqdmlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %c, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %c, i32 1
  %vqdmlal9.i.i = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  %vqdmlal11.i.i = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> %vqdmlal9.i.i)
  ret <2 x i64> %vqdmlal11.i.i
}

define <2 x i64> @test_vqdmlal_high_n_s32_imm(<2 x i64> %a, <4 x i32> %b) #0 {
; CHECK-LABEL: test_vqdmlal_high_n_s32_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].4s, #29
; CHECK-NEXT: sqdmlal2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vqdmlal9.i.i = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> <i32 29, i32 29>)
  %vqdmlal11.i.i = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> %vqdmlal9.i.i)
  ret <2 x i64> %vqdmlal11.i.i
}

define <4 x i32> @test_vmlsl_high_n_s16(<4 x i32> %a, <8 x i16> %b, i16 %c) #0 {
; CHECK-LABEL: test_vmlsl_high_n_s16:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].8h, w0
; CHECK-NEXT: smlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %c, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %c, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %c, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %c, i32 3
  %vmull2.i.i.i = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  %sub.i.i = sub <4 x i32> %a, %vmull2.i.i.i
  ret <4 x i32> %sub.i.i
}

define <4 x i32> @test_vmlsl_high_n_s16_imm(<4 x i32> %a, <8 x i16> %b) #0 {
; CHECK-LABEL: test_vmlsl_high_n_s16_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].8h, #29
; CHECK-NEXT: smlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vmull2.i.i.i = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> <i16 29, i16 29, i16 29, i16 29>)
  %sub.i.i = sub <4 x i32> %a, %vmull2.i.i.i
  ret <4 x i32> %sub.i.i
}

define <2 x i64> @test_vmlsl_high_n_s32(<2 x i64> %a, <4 x i32> %b, i32 %c) #0 {
; CHECK-LABEL: test_vmlsl_high_n_s32:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].4s, w0
; CHECK-NEXT: smlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %c, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %c, i32 1
  %vmull2.i.i.i = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  %sub.i.i = sub <2 x i64> %a, %vmull2.i.i.i
  ret <2 x i64> %sub.i.i
}

define <2 x i64> @test_vmlsl_high_n_s32_imm(<2 x i64> %a, <4 x i32> %b) #0 {
; CHECK-LABEL: test_vmlsl_high_n_s32_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].4s, #29
; CHECK-NEXT: smlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vmull2.i.i.i = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> <i32 29, i32 29>)
  %sub.i.i = sub <2 x i64> %a, %vmull2.i.i.i
  ret <2 x i64> %sub.i.i
}

define <4 x i32> @test_vmlsl_high_n_u16(<4 x i32> %a, <8 x i16> %b, i16 %c) #0 {
; CHECK-LABEL: test_vmlsl_high_n_u16:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].8h, w0
; CHECK-NEXT: umlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %c, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %c, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %c, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %c, i32 3
  %vmull2.i.i.i = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  %sub.i.i = sub <4 x i32> %a, %vmull2.i.i.i
  ret <4 x i32> %sub.i.i
}

define <4 x i32> @test_vmlsl_high_n_u16_imm(<4 x i32> %a, <8 x i16> %b) #0 {
; CHECK-LABEL: test_vmlsl_high_n_u16_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].8h, #29
; CHECK-NEXT: umlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vmull2.i.i.i = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> <i16 29, i16 29, i16 29, i16 29>)
  %sub.i.i = sub <4 x i32> %a, %vmull2.i.i.i
  ret <4 x i32> %sub.i.i
}

define <2 x i64> @test_vmlsl_high_n_u32(<2 x i64> %a, <4 x i32> %b, i32 %c) #0 {
; CHECK-LABEL: test_vmlsl_high_n_u32:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].4s, w0
; CHECK-NEXT: umlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %c, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %c, i32 1
  %vmull2.i.i.i = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  %sub.i.i = sub <2 x i64> %a, %vmull2.i.i.i
  ret <2 x i64> %sub.i.i
}

define <2 x i64> @test_vmlsl_high_n_u32_imm(<2 x i64> %a, <4 x i32> %b) #0 {
; CHECK-LABEL: test_vmlsl_high_n_u32_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].4s, #29
; CHECK-NEXT: umlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vmull2.i.i.i = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> <i32 29, i32 29>)
  %sub.i.i = sub <2 x i64> %a, %vmull2.i.i.i
  ret <2 x i64> %sub.i.i
}

define <4 x i32> @test_vqdmlsl_high_n_s16(<4 x i32> %a, <8 x i16> %b, i16 %c) #0 {
; CHECK-LABEL: test_vqdmlsl_high_n_s16:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].8h, w0
; CHECK-NEXT: sqdmlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vecinit.i.i = insertelement <4 x i16> undef, i16 %c, i32 0
  %vecinit1.i.i = insertelement <4 x i16> %vecinit.i.i, i16 %c, i32 1
  %vecinit2.i.i = insertelement <4 x i16> %vecinit1.i.i, i16 %c, i32 2
  %vecinit3.i.i = insertelement <4 x i16> %vecinit2.i.i, i16 %c, i32 3
  %vqdmlsl15.i.i = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> %vecinit3.i.i)
  %vqdmlsl17.i.i = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> %vqdmlsl15.i.i)
  ret <4 x i32> %vqdmlsl17.i.i
}

define <4 x i32> @test_vqdmlsl_high_n_s16_imm(<4 x i32> %a, <8 x i16> %b) #0 {
; CHECK-LABEL: test_vqdmlsl_high_n_s16_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].8h, #29
; CHECK-NEXT: sqdmlsl2 {{v[0-9]+}}.4s, {{v[0-9]+}}.8h, [[REPLICATE]].8h
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <8 x i16> %b, <8 x i16> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %vqdmlsl15.i.i = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %shuffle.i.i, <4 x i16> <i16 29, i16 29, i16 29, i16 29>)
  %vqdmlsl17.i.i = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> %vqdmlsl15.i.i)
  ret <4 x i32> %vqdmlsl17.i.i
}

define <2 x i64> @test_vqdmlsl_high_n_s32(<2 x i64> %a, <4 x i32> %b, i32 %c) #0 {
; CHECK-LABEL: test_vqdmlsl_high_n_s32:
; CHECK-NEXT: dup [[REPLICATE:v[0-9]+]].4s, w0
; CHECK-NEXT: sqdmlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vecinit.i.i = insertelement <2 x i32> undef, i32 %c, i32 0
  %vecinit1.i.i = insertelement <2 x i32> %vecinit.i.i, i32 %c, i32 1
  %vqdmlsl9.i.i = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> %vecinit1.i.i)
  %vqdmlsl11.i.i = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> %vqdmlsl9.i.i)
  ret <2 x i64> %vqdmlsl11.i.i
}

define <2 x i64> @test_vqdmlsl_high_n_s32_imm(<2 x i64> %a, <4 x i32> %b) #0 {
; CHECK-LABEL: test_vqdmlsl_high_n_s32_imm:
; CHECK-NEXT: movi [[REPLICATE:v[0-9]+]].4s, #29
; CHECK-NEXT: sqdmlsl2 {{v[0-9]+}}.2d, {{v[0-9]+}}.4s, [[REPLICATE]].4s
; CHECK-NEXT: ret
entry:
  %shuffle.i.i = shufflevector <4 x i32> %b, <4 x i32> undef, <2 x i32> <i32 2, i32 3>
  %vqdmlsl9.i.i = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %shuffle.i.i, <2 x i32> <i32 29, i32 29>)
  %vqdmlsl11.i.i = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> %vqdmlsl9.i.i)
  ret <2 x i64> %vqdmlsl11.i.i
}

define <2 x float> @test_vmul_n_f32(<2 x float> %a, float %b) #0 {
; CHECK-LABEL: test_vmul_n_f32:
; CHECK-NEXT: fmul {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[0]
; CHECK-NEXT: ret
entry:
  %vecinit.i = insertelement <2 x float> undef, float %b, i32 0
  %vecinit1.i = insertelement <2 x float> %vecinit.i, float %b, i32 1
  %mul.i = fmul <2 x float> %vecinit1.i, %a
  ret <2 x float> %mul.i
}

define <4 x float> @test_vmulq_n_f32(<4 x float> %a, float %b) #0 {
; CHECK-LABEL: test_vmulq_n_f32:
; CHECK-NEXT: fmul {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[0]
; CHECK-NEXT: ret
entry:
  %vecinit.i = insertelement <4 x float> undef, float %b, i32 0
  %vecinit1.i = insertelement <4 x float> %vecinit.i, float %b, i32 1
  %vecinit2.i = insertelement <4 x float> %vecinit1.i, float %b, i32 2
  %vecinit3.i = insertelement <4 x float> %vecinit2.i, float %b, i32 3
  %mul.i = fmul <4 x float> %vecinit3.i, %a
  ret <4 x float> %mul.i
}

define <2 x double> @test_vmulq_n_f64(<2 x double> %a, double %b) #0 {
; CHECK-LABEL: test_vmulq_n_f64:
; CHECK-NEXT: fmul {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.d[0]
; CHECK-NEXT: ret
entry:
  %vecinit.i = insertelement <2 x double> undef, double %b, i32 0
  %vecinit1.i = insertelement <2 x double> %vecinit.i, double %b, i32 1
  %mul.i = fmul <2 x double> %vecinit1.i, %a
  ret <2 x double> %mul.i
}

define <2 x float> @test_vfma_n_f32(<2 x float> %a, <2 x float> %b, float %n) #0 {
; CHECK-LABEL: test_vfma_n_f32:
; CHECK-NEXT: fmla {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[{{[0-9]+}}]
; CHECK-NEXT: ret
entry:
  %vecinit.i = insertelement <2 x float> undef, float %n, i32 0
  %vecinit1.i = insertelement <2 x float> %vecinit.i, float %n, i32 1
  %0 = call <2 x float> @llvm.fma.v2f32(<2 x float> %b, <2 x float> %vecinit1.i, <2 x float> %a)
  ret <2 x float> %0
}

define <4 x float> @test_vfmaq_n_f32(<4 x float> %a, <4 x float> %b, float %n) #0 {
; CHECK-LABEL: test_vfmaq_n_f32:
; CHECK-NEXT: fmla {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[{{[0-9]+}}]
; CHECK-NEXT: ret
entry:
  %vecinit.i = insertelement <4 x float> undef, float %n, i32 0
  %vecinit1.i = insertelement <4 x float> %vecinit.i, float %n, i32 1
  %vecinit2.i = insertelement <4 x float> %vecinit1.i, float %n, i32 2
  %vecinit3.i = insertelement <4 x float> %vecinit2.i, float %n, i32 3
  %0 = call <4 x float> @llvm.fma.v4f32(<4 x float> %b, <4 x float> %vecinit3.i, <4 x float> %a)
  ret <4 x float> %0
}

define <2 x float> @test_vfms_n_f32(<2 x float> %a, <2 x float> %b, float %n) #0 {
; CHECK-LABEL: test_vfms_n_f32:
; CHECK-NEXT: fmls {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.s[{{[0-9]+}}]
; CHECK-NEXT: ret
entry:
  %vecinit.i = insertelement <2 x float> undef, float %n, i32 0
  %vecinit1.i = insertelement <2 x float> %vecinit.i, float %n, i32 1
  %0 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %b
  %1 = call <2 x float> @llvm.fma.v2f32(<2 x float> %0, <2 x float> %vecinit1.i, <2 x float> %a)
  ret <2 x float> %1
}

define <4 x float> @test_vfmsq_n_f32(<4 x float> %a, <4 x float> %b, float %n) #0 {
; CHECK-LABEL: test_vfmsq_n_f32:
; CHECK-NEXT: fmls {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.s[{{[0-9]+}}]
; CHECK-NEXT: ret
entry:
  %vecinit.i = insertelement <4 x float> undef, float %n, i32 0
  %vecinit1.i = insertelement <4 x float> %vecinit.i, float %n, i32 1
  %vecinit2.i = insertelement <4 x float> %vecinit1.i, float %n, i32 2
  %vecinit3.i = insertelement <4 x float> %vecinit2.i, float %n, i32 3
  %0 = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %b
  %1 = call <4 x float> @llvm.fma.v4f32(<4 x float> %0, <4 x float> %vecinit3.i, <4 x float> %a)
  ret <4 x float> %1
}

attributes #0 = { nounwind }

declare <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16>, <4 x i16>)
declare <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16>, <4 x i16>)
declare <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16>, <4 x i16>)
declare <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32>, <2 x i32>)
declare <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32>, <4 x i32>)
declare <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64>, <2 x i64>)
declare <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32>, <4 x i32>)
declare <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64>, <2 x i64>)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <2 x float> @llvm.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)
