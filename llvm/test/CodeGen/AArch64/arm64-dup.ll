; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple -asm-verbose=false | FileCheck %s

define <8 x i8> @v_dup8(i8 %A) nounwind {
;CHECK-LABEL: v_dup8:
;CHECK: dup.8b
	%tmp1 = insertelement <8 x i8> zeroinitializer, i8 %A, i32 0
	%tmp2 = insertelement <8 x i8> %tmp1, i8 %A, i32 1
	%tmp3 = insertelement <8 x i8> %tmp2, i8 %A, i32 2
	%tmp4 = insertelement <8 x i8> %tmp3, i8 %A, i32 3
	%tmp5 = insertelement <8 x i8> %tmp4, i8 %A, i32 4
	%tmp6 = insertelement <8 x i8> %tmp5, i8 %A, i32 5
	%tmp7 = insertelement <8 x i8> %tmp6, i8 %A, i32 6
	%tmp8 = insertelement <8 x i8> %tmp7, i8 %A, i32 7
	ret <8 x i8> %tmp8
}

define <4 x i16> @v_dup16(i16 %A) nounwind {
;CHECK-LABEL: v_dup16:
;CHECK: dup.4h
	%tmp1 = insertelement <4 x i16> zeroinitializer, i16 %A, i32 0
	%tmp2 = insertelement <4 x i16> %tmp1, i16 %A, i32 1
	%tmp3 = insertelement <4 x i16> %tmp2, i16 %A, i32 2
	%tmp4 = insertelement <4 x i16> %tmp3, i16 %A, i32 3
	ret <4 x i16> %tmp4
}

define <2 x i32> @v_dup32(i32 %A) nounwind {
;CHECK-LABEL: v_dup32:
;CHECK: dup.2s
	%tmp1 = insertelement <2 x i32> zeroinitializer, i32 %A, i32 0
	%tmp2 = insertelement <2 x i32> %tmp1, i32 %A, i32 1
	ret <2 x i32> %tmp2
}

define <2 x float> @v_dupfloat(float %A) nounwind {
;CHECK-LABEL: v_dupfloat:
;CHECK: dup.2s
	%tmp1 = insertelement <2 x float> zeroinitializer, float %A, i32 0
	%tmp2 = insertelement <2 x float> %tmp1, float %A, i32 1
	ret <2 x float> %tmp2
}

define <16 x i8> @v_dupQ8(i8 %A) nounwind {
;CHECK-LABEL: v_dupQ8:
;CHECK: dup.16b
	%tmp1 = insertelement <16 x i8> zeroinitializer, i8 %A, i32 0
	%tmp2 = insertelement <16 x i8> %tmp1, i8 %A, i32 1
	%tmp3 = insertelement <16 x i8> %tmp2, i8 %A, i32 2
	%tmp4 = insertelement <16 x i8> %tmp3, i8 %A, i32 3
	%tmp5 = insertelement <16 x i8> %tmp4, i8 %A, i32 4
	%tmp6 = insertelement <16 x i8> %tmp5, i8 %A, i32 5
	%tmp7 = insertelement <16 x i8> %tmp6, i8 %A, i32 6
	%tmp8 = insertelement <16 x i8> %tmp7, i8 %A, i32 7
	%tmp9 = insertelement <16 x i8> %tmp8, i8 %A, i32 8
	%tmp10 = insertelement <16 x i8> %tmp9, i8 %A, i32 9
	%tmp11 = insertelement <16 x i8> %tmp10, i8 %A, i32 10
	%tmp12 = insertelement <16 x i8> %tmp11, i8 %A, i32 11
	%tmp13 = insertelement <16 x i8> %tmp12, i8 %A, i32 12
	%tmp14 = insertelement <16 x i8> %tmp13, i8 %A, i32 13
	%tmp15 = insertelement <16 x i8> %tmp14, i8 %A, i32 14
	%tmp16 = insertelement <16 x i8> %tmp15, i8 %A, i32 15
	ret <16 x i8> %tmp16
}

define <8 x i16> @v_dupQ16(i16 %A) nounwind {
;CHECK-LABEL: v_dupQ16:
;CHECK: dup.8h
	%tmp1 = insertelement <8 x i16> zeroinitializer, i16 %A, i32 0
	%tmp2 = insertelement <8 x i16> %tmp1, i16 %A, i32 1
	%tmp3 = insertelement <8 x i16> %tmp2, i16 %A, i32 2
	%tmp4 = insertelement <8 x i16> %tmp3, i16 %A, i32 3
	%tmp5 = insertelement <8 x i16> %tmp4, i16 %A, i32 4
	%tmp6 = insertelement <8 x i16> %tmp5, i16 %A, i32 5
	%tmp7 = insertelement <8 x i16> %tmp6, i16 %A, i32 6
	%tmp8 = insertelement <8 x i16> %tmp7, i16 %A, i32 7
	ret <8 x i16> %tmp8
}

define <4 x i32> @v_dupQ32(i32 %A) nounwind {
;CHECK-LABEL: v_dupQ32:
;CHECK: dup.4s
	%tmp1 = insertelement <4 x i32> zeroinitializer, i32 %A, i32 0
	%tmp2 = insertelement <4 x i32> %tmp1, i32 %A, i32 1
	%tmp3 = insertelement <4 x i32> %tmp2, i32 %A, i32 2
	%tmp4 = insertelement <4 x i32> %tmp3, i32 %A, i32 3
	ret <4 x i32> %tmp4
}

define <4 x float> @v_dupQfloat(float %A) nounwind {
;CHECK-LABEL: v_dupQfloat:
;CHECK: dup.4s
	%tmp1 = insertelement <4 x float> zeroinitializer, float %A, i32 0
	%tmp2 = insertelement <4 x float> %tmp1, float %A, i32 1
	%tmp3 = insertelement <4 x float> %tmp2, float %A, i32 2
	%tmp4 = insertelement <4 x float> %tmp3, float %A, i32 3
	ret <4 x float> %tmp4
}

; Check to make sure it works with shuffles, too.

define <8 x i8> @v_shuffledup8(i8 %A) nounwind {
;CHECK-LABEL: v_shuffledup8:
;CHECK: dup.8b
	%tmp1 = insertelement <8 x i8> undef, i8 %A, i32 0
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> zeroinitializer
	ret <8 x i8> %tmp2
}

define <4 x i16> @v_shuffledup16(i16 %A) nounwind {
;CHECK-LABEL: v_shuffledup16:
;CHECK: dup.4h
	%tmp1 = insertelement <4 x i16> undef, i16 %A, i32 0
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> zeroinitializer
	ret <4 x i16> %tmp2
}

define <2 x i32> @v_shuffledup32(i32 %A) nounwind {
;CHECK-LABEL: v_shuffledup32:
;CHECK: dup.2s
	%tmp1 = insertelement <2 x i32> undef, i32 %A, i32 0
	%tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <2 x i32> zeroinitializer
	ret <2 x i32> %tmp2
}

define <2 x float> @v_shuffledupfloat(float %A) nounwind {
;CHECK-LABEL: v_shuffledupfloat:
;CHECK: dup.2s
	%tmp1 = insertelement <2 x float> undef, float %A, i32 0
	%tmp2 = shufflevector <2 x float> %tmp1, <2 x float> undef, <2 x i32> zeroinitializer
	ret <2 x float> %tmp2
}

define <16 x i8> @v_shuffledupQ8(i8 %A) nounwind {
;CHECK-LABEL: v_shuffledupQ8:
;CHECK: dup.16b
	%tmp1 = insertelement <16 x i8> undef, i8 %A, i32 0
	%tmp2 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <16 x i32> zeroinitializer
	ret <16 x i8> %tmp2
}

define <8 x i16> @v_shuffledupQ16(i16 %A) nounwind {
;CHECK-LABEL: v_shuffledupQ16:
;CHECK: dup.8h
	%tmp1 = insertelement <8 x i16> undef, i16 %A, i32 0
	%tmp2 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <8 x i32> zeroinitializer
	ret <8 x i16> %tmp2
}

define <4 x i32> @v_shuffledupQ32(i32 %A) nounwind {
;CHECK-LABEL: v_shuffledupQ32:
;CHECK: dup.4s
	%tmp1 = insertelement <4 x i32> undef, i32 %A, i32 0
	%tmp2 = shufflevector <4 x i32> %tmp1, <4 x i32> undef, <4 x i32> zeroinitializer
	ret <4 x i32> %tmp2
}

define <4 x float> @v_shuffledupQfloat(float %A) nounwind {
;CHECK-LABEL: v_shuffledupQfloat:
;CHECK: dup.4s
	%tmp1 = insertelement <4 x float> undef, float %A, i32 0
	%tmp2 = shufflevector <4 x float> %tmp1, <4 x float> undef, <4 x i32> zeroinitializer
	ret <4 x float> %tmp2
}

define <8 x i8> @vduplane8(<8 x i8>* %A) nounwind {
;CHECK-LABEL: vduplane8:
;CHECK: dup.8b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> < i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1 >
	ret <8 x i8> %tmp2
}

define <4 x i16> @vduplane16(<4 x i16>* %A) nounwind {
;CHECK-LABEL: vduplane16:
;CHECK: dup.4h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >
	ret <4 x i16> %tmp2
}

define <2 x i32> @vduplane32(<2 x i32>* %A) nounwind {
;CHECK-LABEL: vduplane32:
;CHECK: dup.2s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <2 x i32> < i32 1, i32 1 >
	ret <2 x i32> %tmp2
}

define <2 x float> @vduplanefloat(<2 x float>* %A) nounwind {
;CHECK-LABEL: vduplanefloat:
;CHECK: dup.2s
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = shufflevector <2 x float> %tmp1, <2 x float> undef, <2 x i32> < i32 1, i32 1 >
	ret <2 x float> %tmp2
}

define <16 x i8> @vduplaneQ8(<8 x i8>* %A) nounwind {
;CHECK-LABEL: vduplaneQ8:
;CHECK: dup.16b
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <16 x i32> < i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1 >
	ret <16 x i8> %tmp2
}

define <8 x i16> @vduplaneQ16(<4 x i16>* %A) nounwind {
;CHECK-LABEL: vduplaneQ16:
;CHECK: dup.8h
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <8 x i32> < i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1 >
	ret <8 x i16> %tmp2
}

define <4 x i32> @vduplaneQ32(<2 x i32>* %A) nounwind {
;CHECK-LABEL: vduplaneQ32:
;CHECK: dup.4s
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >
	ret <4 x i32> %tmp2
}

define <4 x float> @vduplaneQfloat(<2 x float>* %A) nounwind {
;CHECK-LABEL: vduplaneQfloat:
;CHECK: dup.4s
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = shufflevector <2 x float> %tmp1, <2 x float> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >
	ret <4 x float> %tmp2
}

define <2 x i64> @foo(<2 x i64> %arg0_int64x1_t) nounwind readnone {
;CHECK-LABEL: foo:
;CHECK: dup.2d
entry:
  %0 = shufflevector <2 x i64> %arg0_int64x1_t, <2 x i64> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x i64> %0
}

define <2 x i64> @bar(<2 x i64> %arg0_int64x1_t) nounwind readnone {
;CHECK-LABEL: bar:
;CHECK: dup.2d
entry:
  %0 = shufflevector <2 x i64> %arg0_int64x1_t, <2 x i64> undef, <2 x i32> <i32 0, i32 0>
  ret <2 x i64> %0
}

define <2 x double> @baz(<2 x double> %arg0_int64x1_t) nounwind readnone {
;CHECK-LABEL: baz:
;CHECK: dup.2d
entry:
  %0 = shufflevector <2 x double> %arg0_int64x1_t, <2 x double> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x double> %0
}

define <2 x double> @qux(<2 x double> %arg0_int64x1_t) nounwind readnone {
;CHECK-LABEL: qux:
;CHECK: dup.2d
entry:
  %0 = shufflevector <2 x double> %arg0_int64x1_t, <2 x double> undef, <2 x i32> <i32 0, i32 0>
  ret <2 x double> %0
}

define <2 x i32> @f(i32 %a, i32 %b) nounwind readnone  {
; CHECK-LABEL: f:
; CHECK-NEXT: fmov s0, w0
; CHECK-NEXT: ins.s v0[1], w1
; CHECK-NEXT: ret
  %vecinit = insertelement <2 x i32> undef, i32 %a, i32 0
  %vecinit1 = insertelement <2 x i32> %vecinit, i32 %b, i32 1
  ret <2 x i32> %vecinit1
}

define <4 x i32> @g(i32 %a, i32 %b) nounwind readnone  {
; CHECK-LABEL: g:
; CHECK-NEXT: fmov s0, w0
; CHECK-NEXT: ins.s v0[1], w1
; CHECK-NEXT: ins.s v0[2], w1
; CHECK-NEXT: ins.s v0[3], w0
; CHECK-NEXT: ret
  %vecinit = insertelement <4 x i32> undef, i32 %a, i32 0
  %vecinit1 = insertelement <4 x i32> %vecinit, i32 %b, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit1, i32 %b, i32 2
  %vecinit3 = insertelement <4 x i32> %vecinit2, i32 %a, i32 3
  ret <4 x i32> %vecinit3
}

define <2 x i64> @h(i64 %a, i64 %b) nounwind readnone  {
; CHECK-LABEL: h:
; CHECK-NEXT: fmov d0, x0
; CHECK-NEXT: ins.d v0[1], x1
; CHECK-NEXT: ret
  %vecinit = insertelement <2 x i64> undef, i64 %a, i32 0
  %vecinit1 = insertelement <2 x i64> %vecinit, i64 %b, i32 1
  ret <2 x i64> %vecinit1
}

; We used to spot this as a BUILD_VECTOR implementable by dup, but assume that
; the single value needed was of the same type as the vector. This is false if
; the scalar corresponding to the vector type is illegal (e.g. a <4 x i16>
; BUILD_VECTOR will have an i32 as its source). In that case, the operation is
; not a simple "dup vD.4h, vN.h[idx]" after all, and we crashed.
;
; *However*, it is a dup vD.4h, vN.h[2*idx].
define <4 x i16> @test_build_illegal(<4 x i32> %in) {
; CHECK-LABEL: test_build_illegal:
; CHECK: dup.4h v0, v0[6]
  %val = extractelement <4 x i32> %in, i32 3
  %smallval = trunc i32 %val to i16
  %vec = insertelement <4x i16> undef, i16 %smallval, i32 3

  ret <4 x i16> %vec
}

; We used to inherit an already extract_subvectored v4i16 from
; SelectionDAGBuilder here. We then added a DUPLANE on top of that, preventing
; the formation of an indexed-by-7 MLS.
define <4 x i16> @test_high_splat(<4 x i16> %a, <4 x i16> %b, <8 x i16> %v) #0 {
; CHECK-LABEL: test_high_splat:
; CHECK: mls.4h v0, v1, v2[7]
entry:
  %shuffle = shufflevector <8 x i16> %v, <8 x i16> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mul = mul <4 x i16> %shuffle, %b
  %sub = sub <4 x i16> %a, %mul
  ret <4 x i16> %sub
}

; Also test the DUP path in the PerfectShuffle generator.

; CHECK-LABEL: test_perfectshuffle_dupext_v4i16:
; CHECK-NEXT: dup.4h v0, v0[0]
; CHECK-NEXT: ext.8b v0, v0, v1, #4
define <4 x i16> @test_perfectshuffle_dupext_v4i16(<4 x i16> %a, <4 x i16> %b) nounwind {
  %r = shufflevector <4 x i16> %a, <4 x i16> %b, <4 x i32> <i32 0, i32 0, i32 4, i32 5>
  ret <4 x i16> %r
}

; CHECK-LABEL: test_perfectshuffle_dupext_v4f16:
; CHECK-NEXT: dup.4h v0, v0[0]
; CHECK-NEXT: ext.8b v0, v0, v1, #4
; CHECK-NEXT: ret
define <4 x half> @test_perfectshuffle_dupext_v4f16(<4 x half> %a, <4 x half> %b) nounwind {
  %r = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 0, i32 0, i32 4, i32 5>
  ret <4 x half> %r
}

; CHECK-LABEL: test_perfectshuffle_dupext_v4i32:
; CHECK-NEXT: dup.4s v0, v0[0]
; CHECK-NEXT: ext.16b v0, v0, v1, #8
; CHECK-NEXT: ret
define <4 x i32> @test_perfectshuffle_dupext_v4i32(<4 x i32> %a, <4 x i32> %b) nounwind {
  %r = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 0, i32 4, i32 5>
  ret <4 x i32> %r
}

; CHECK-LABEL: test_perfectshuffle_dupext_v4f32:
; CHECK-NEXT: dup.4s v0, v0[0]
; CHECK-NEXT: ext.16b v0, v0, v1, #8
; CHECK-NEXT: ret
define <4 x float> @test_perfectshuffle_dupext_v4f32(<4 x float> %a, <4 x float> %b) nounwind {
  %r = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 0, i32 4, i32 5>
  ret <4 x float> %r
}
