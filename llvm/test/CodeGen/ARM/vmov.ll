; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @v_movi8() nounwind {
;CHECK: v_movi8:
;CHECK: vmov.i8 d{{.*}}, #0x8
	ret <8 x i8> < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
}

define <4 x i16> @v_movi16a() nounwind {
;CHECK: v_movi16a:
;CHECK: vmov.i16 d{{.*}}, #0x10
	ret <4 x i16> < i16 16, i16 16, i16 16, i16 16 >
}

define <4 x i16> @v_movi16b() nounwind {
;CHECK: v_movi16b:
;CHECK: vmov.i16 d{{.*}}, #0x1000
	ret <4 x i16> < i16 4096, i16 4096, i16 4096, i16 4096 >
}

define <4 x i16> @v_mvni16a() nounwind {
;CHECK: v_mvni16a:
;CHECK: vmvn.i16 d{{.*}}, #0x10
	ret <4 x i16> < i16 65519, i16 65519, i16 65519, i16 65519 >
}

define <4 x i16> @v_mvni16b() nounwind {
;CHECK: v_mvni16b:
;CHECK: vmvn.i16 d{{.*}}, #0x1000
	ret <4 x i16> < i16 61439, i16 61439, i16 61439, i16 61439 >
}

define <2 x i32> @v_movi32a() nounwind {
;CHECK: v_movi32a:
;CHECK: vmov.i32 d{{.*}}, #0x20
	ret <2 x i32> < i32 32, i32 32 >
}

define <2 x i32> @v_movi32b() nounwind {
;CHECK: v_movi32b:
;CHECK: vmov.i32 d{{.*}}, #0x2000
	ret <2 x i32> < i32 8192, i32 8192 >
}

define <2 x i32> @v_movi32c() nounwind {
;CHECK: v_movi32c:
;CHECK: vmov.i32 d{{.*}}, #0x200000
	ret <2 x i32> < i32 2097152, i32 2097152 >
}

define <2 x i32> @v_movi32d() nounwind {
;CHECK: v_movi32d:
;CHECK: vmov.i32 d{{.*}}, #0x20000000
	ret <2 x i32> < i32 536870912, i32 536870912 >
}

define <2 x i32> @v_movi32e() nounwind {
;CHECK: v_movi32e:
;CHECK: vmov.i32 d{{.*}}, #0x20ff
	ret <2 x i32> < i32 8447, i32 8447 >
}

define <2 x i32> @v_movi32f() nounwind {
;CHECK: v_movi32f:
;CHECK: vmov.i32 d{{.*}}, #0x20ffff
	ret <2 x i32> < i32 2162687, i32 2162687 >
}

define <2 x i32> @v_mvni32a() nounwind {
;CHECK: v_mvni32a:
;CHECK: vmvn.i32 d{{.*}}, #0x20
	ret <2 x i32> < i32 4294967263, i32 4294967263 >
}

define <2 x i32> @v_mvni32b() nounwind {
;CHECK: v_mvni32b:
;CHECK: vmvn.i32 d{{.*}}, #0x2000
	ret <2 x i32> < i32 4294959103, i32 4294959103 >
}

define <2 x i32> @v_mvni32c() nounwind {
;CHECK: v_mvni32c:
;CHECK: vmvn.i32 d{{.*}}, #0x200000
	ret <2 x i32> < i32 4292870143, i32 4292870143 >
}

define <2 x i32> @v_mvni32d() nounwind {
;CHECK: v_mvni32d:
;CHECK: vmvn.i32 d{{.*}}, #0x20000000
	ret <2 x i32> < i32 3758096383, i32 3758096383 >
}

define <2 x i32> @v_mvni32e() nounwind {
;CHECK: v_mvni32e:
;CHECK: vmvn.i32 d{{.*}}, #0x20ff
	ret <2 x i32> < i32 4294958848, i32 4294958848 >
}

define <2 x i32> @v_mvni32f() nounwind {
;CHECK: v_mvni32f:
;CHECK: vmvn.i32 d{{.*}}, #0x20ffff
	ret <2 x i32> < i32 4292804608, i32 4292804608 >
}

define <1 x i64> @v_movi64() nounwind {
;CHECK: v_movi64:
;CHECK: vmov.i64 d{{.*}}, #0xff0000ff0000ffff
	ret <1 x i64> < i64 18374687574888349695 >
}

define <16 x i8> @v_movQi8() nounwind {
;CHECK: v_movQi8:
;CHECK: vmov.i8 q{{.*}}, #0x8
	ret <16 x i8> < i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8 >
}

define <8 x i16> @v_movQi16a() nounwind {
;CHECK: v_movQi16a:
;CHECK: vmov.i16 q{{.*}}, #0x10
	ret <8 x i16> < i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16, i16 16 >
}

define <8 x i16> @v_movQi16b() nounwind {
;CHECK: v_movQi16b:
;CHECK: vmov.i16 q{{.*}}, #0x1000
	ret <8 x i16> < i16 4096, i16 4096, i16 4096, i16 4096, i16 4096, i16 4096, i16 4096, i16 4096 >
}

define <4 x i32> @v_movQi32a() nounwind {
;CHECK: v_movQi32a:
;CHECK: vmov.i32 q{{.*}}, #0x20
	ret <4 x i32> < i32 32, i32 32, i32 32, i32 32 >
}

define <4 x i32> @v_movQi32b() nounwind {
;CHECK: v_movQi32b:
;CHECK: vmov.i32 q{{.*}}, #0x2000
	ret <4 x i32> < i32 8192, i32 8192, i32 8192, i32 8192 >
}

define <4 x i32> @v_movQi32c() nounwind {
;CHECK: v_movQi32c:
;CHECK: vmov.i32 q{{.*}}, #0x200000
	ret <4 x i32> < i32 2097152, i32 2097152, i32 2097152, i32 2097152 >
}

define <4 x i32> @v_movQi32d() nounwind {
;CHECK: v_movQi32d:
;CHECK: vmov.i32 q{{.*}}, #0x20000000
	ret <4 x i32> < i32 536870912, i32 536870912, i32 536870912, i32 536870912 >
}

define <4 x i32> @v_movQi32e() nounwind {
;CHECK: v_movQi32e:
;CHECK: vmov.i32 q{{.*}}, #0x20ff
	ret <4 x i32> < i32 8447, i32 8447, i32 8447, i32 8447 >
}

define <4 x i32> @v_movQi32f() nounwind {
;CHECK: v_movQi32f:
;CHECK: vmov.i32 q{{.*}}, #0x20ffff
	ret <4 x i32> < i32 2162687, i32 2162687, i32 2162687, i32 2162687 >
}

define <2 x i64> @v_movQi64() nounwind {
;CHECK: v_movQi64:
;CHECK: vmov.i64 q{{.*}}, #0xff0000ff0000ffff
	ret <2 x i64> < i64 18374687574888349695, i64 18374687574888349695 >
}

; Check for correct assembler printing for immediate values.
%struct.int8x8_t = type { <8 x i8> }
define void @vdupn128(%struct.int8x8_t* noalias nocapture sret %agg.result) nounwind {
entry:
;CHECK: vdupn128:
;CHECK: vmov.i8 d{{.*}}, #0x80
  %0 = getelementptr inbounds %struct.int8x8_t* %agg.result, i32 0, i32 0 ; <<8 x i8>*> [#uses=1]
  store <8 x i8> <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>, <8 x i8>* %0, align 8
  ret void
}

define void @vdupnneg75(%struct.int8x8_t* noalias nocapture sret %agg.result) nounwind {
entry:
;CHECK: vdupnneg75:
;CHECK: vmov.i8 d{{.*}}, #0xb5
  %0 = getelementptr inbounds %struct.int8x8_t* %agg.result, i32 0, i32 0 ; <<8 x i8>*> [#uses=1]
  store <8 x i8> <i8 -75, i8 -75, i8 -75, i8 -75, i8 -75, i8 -75, i8 -75, i8 -75>, <8 x i8>* %0, align 8
  ret void
}

define <8 x i16> @vmovls8(<8 x i8>* %A) nounwind {
;CHECK: vmovls8:
;CHECK: vmovl.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = sext <8 x i8> %tmp1 to <8 x i16>
	ret <8 x i16> %tmp2
}

define <4 x i32> @vmovls16(<4 x i16>* %A) nounwind {
;CHECK: vmovls16:
;CHECK: vmovl.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = sext <4 x i16> %tmp1 to <4 x i32>
	ret <4 x i32> %tmp2
}

define <2 x i64> @vmovls32(<2 x i32>* %A) nounwind {
;CHECK: vmovls32:
;CHECK: vmovl.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = sext <2 x i32> %tmp1 to <2 x i64>
	ret <2 x i64> %tmp2
}

define <8 x i16> @vmovlu8(<8 x i8>* %A) nounwind {
;CHECK: vmovlu8:
;CHECK: vmovl.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = zext <8 x i8> %tmp1 to <8 x i16>
	ret <8 x i16> %tmp2
}

define <4 x i32> @vmovlu16(<4 x i16>* %A) nounwind {
;CHECK: vmovlu16:
;CHECK: vmovl.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = zext <4 x i16> %tmp1 to <4 x i32>
	ret <4 x i32> %tmp2
}

define <2 x i64> @vmovlu32(<2 x i32>* %A) nounwind {
;CHECK: vmovlu32:
;CHECK: vmovl.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = zext <2 x i32> %tmp1 to <2 x i64>
	ret <2 x i64> %tmp2
}

define <8 x i8> @vmovni16(<8 x i16>* %A) nounwind {
;CHECK: vmovni16:
;CHECK: vmovn.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = trunc <8 x i16> %tmp1 to <8 x i8>
	ret <8 x i8> %tmp2
}

define <4 x i16> @vmovni32(<4 x i32>* %A) nounwind {
;CHECK: vmovni32:
;CHECK: vmovn.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = trunc <4 x i32> %tmp1 to <4 x i16>
	ret <4 x i16> %tmp2
}

define <2 x i32> @vmovni64(<2 x i64>* %A) nounwind {
;CHECK: vmovni64:
;CHECK: vmovn.i64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = trunc <2 x i64> %tmp1 to <2 x i32>
	ret <2 x i32> %tmp2
}

define <8 x i8> @vqmovns16(<8 x i16>* %A) nounwind {
;CHECK: vqmovns16:
;CHECK: vqmovn.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqmovns.v8i8(<8 x i16> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqmovns32(<4 x i32>* %A) nounwind {
;CHECK: vqmovns32:
;CHECK: vqmovn.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqmovns.v4i16(<4 x i32> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqmovns64(<2 x i64>* %A) nounwind {
;CHECK: vqmovns64:
;CHECK: vqmovn.s64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqmovns.v2i32(<2 x i64> %tmp1)
	ret <2 x i32> %tmp2
}

define <8 x i8> @vqmovnu16(<8 x i16>* %A) nounwind {
;CHECK: vqmovnu16:
;CHECK: vqmovn.u16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqmovnu.v8i8(<8 x i16> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqmovnu32(<4 x i32>* %A) nounwind {
;CHECK: vqmovnu32:
;CHECK: vqmovn.u32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqmovnu.v4i16(<4 x i32> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqmovnu64(<2 x i64>* %A) nounwind {
;CHECK: vqmovnu64:
;CHECK: vqmovn.u64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqmovnu.v2i32(<2 x i64> %tmp1)
	ret <2 x i32> %tmp2
}

define <8 x i8> @vqmovuns16(<8 x i16>* %A) nounwind {
;CHECK: vqmovuns16:
;CHECK: vqmovun.s16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = call <8 x i8> @llvm.arm.neon.vqmovnsu.v8i8(<8 x i16> %tmp1)
	ret <8 x i8> %tmp2
}

define <4 x i16> @vqmovuns32(<4 x i32>* %A) nounwind {
;CHECK: vqmovuns32:
;CHECK: vqmovun.s32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = call <4 x i16> @llvm.arm.neon.vqmovnsu.v4i16(<4 x i32> %tmp1)
	ret <4 x i16> %tmp2
}

define <2 x i32> @vqmovuns64(<2 x i64>* %A) nounwind {
;CHECK: vqmovuns64:
;CHECK: vqmovun.s64
	%tmp1 = load <2 x i64>* %A
	%tmp2 = call <2 x i32> @llvm.arm.neon.vqmovnsu.v2i32(<2 x i64> %tmp1)
	ret <2 x i32> %tmp2
}

declare <8 x i8>  @llvm.arm.neon.vqmovns.v8i8(<8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqmovns.v4i16(<4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqmovns.v2i32(<2 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqmovnu.v8i8(<8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqmovnu.v4i16(<4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqmovnu.v2i32(<2 x i64>) nounwind readnone

declare <8 x i8>  @llvm.arm.neon.vqmovnsu.v8i8(<8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vqmovnsu.v4i16(<4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqmovnsu.v2i32(<2 x i64>) nounwind readnone

; Truncating vector stores are not supported.  The following should not crash.
; Radar 8598391.
define void @noTruncStore(<4 x i32>* %a, <4 x i16>* %b) nounwind {
;CHECK: vmovn
  %tmp1 = load <4 x i32>* %a, align 16
  %tmp2 = trunc <4 x i32> %tmp1 to <4 x i16>
  store <4 x i16> %tmp2, <4 x i16>* %b, align 8
  ret void
}

; Use vmov.f32 to materialize f32 immediate splats
; rdar://10437054
define void @v_mov_v2f32(<2 x float>* nocapture %p) nounwind {
entry:
;CHECK: v_mov_v2f32:
;CHECK: vmov.f32 d{{.*}}, #-1.600000e+01
  store <2 x float> <float -1.600000e+01, float -1.600000e+01>, <2 x float>* %p, align 4
  ret void
}

define void @v_mov_v4f32(<4 x float>* nocapture %p) nounwind {
entry:
;CHECK: v_mov_v4f32:
;CHECK: vmov.f32 q{{.*}}, #3.100000e+01
  store <4 x float> <float 3.100000e+01, float 3.100000e+01, float 3.100000e+01, float 3.100000e+01>, <4 x float>* %p, align 4
  ret void
}

define void @v_mov_v4f32_undef(<4 x float> * nocapture %p) nounwind {
entry:
;CHECK: v_mov_v4f32_undef:
;CHECK: vmov.f32 q{{.*}}, #1.000000e+00
  %a = load <4 x float> *%p
  %b = fadd <4 x float> %a, <float undef, float 1.0, float 1.0, float 1.0>
  store <4 x float> %b, <4 x float> *%p
  ret void
}

; Vector any_extends must be selected as either vmovl.u or vmovl.s.
; rdar://10723651
define void @any_extend(<4 x i1> %x, <4 x i32> %y) nounwind ssp {
entry:
;CHECK: any_extend
;CHECK: vmovl
  %and.i186 = zext <4 x i1> %x to <4 x i32>
  %add.i185 = sub <4 x i32> %and.i186, %y
  %sub.i = sub <4 x i32> %add.i185, zeroinitializer
  %add.i = add <4 x i32> %sub.i, zeroinitializer
  %vmovn.i = trunc <4 x i32> %add.i to <4 x i16>
  tail call void @llvm.arm.neon.vst1.v4i16(i8* undef, <4 x i16> %vmovn.i, i32 2)
  unreachable
}

declare void @llvm.arm.neon.vst1.v4i16(i8*, <4 x i16>, i32) nounwind
