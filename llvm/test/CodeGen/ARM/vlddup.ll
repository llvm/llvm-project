; RUN: llc -mtriple=arm-eabi -float-abi=soft -mattr=+neon %s -o - | FileCheck %s

define <8 x i8> @vld1dupi8(i8* %A) nounwind {
;CHECK-LABEL: vld1dupi8:
;Check the (default) alignment value.
;CHECK: vld1.8 {d16[]}, [{{r[0-9]+|lr}}]
	%tmp1 = load i8, i8* %A, align 8
	%tmp2 = insertelement <8 x i8> undef, i8 %tmp1, i32 0
	%tmp3 = shufflevector <8 x i8> %tmp2, <8 x i8> undef, <8 x i32> zeroinitializer
        ret <8 x i8> %tmp3
}

define <8 x i8> @vld1dupi8_preinc(i8** noalias nocapture %a, i32 %b) nounwind {
entry:
;CHECK-LABEL: vld1dupi8_preinc:
;CHECK: vld1.8 {d16[]}, [{{r[0-9]+|lr}}]
  %0 = load i8*, i8** %a, align 4
  %add.ptr = getelementptr inbounds i8, i8* %0, i32 %b
  %1 = load i8, i8* %add.ptr, align 1
  %2 = insertelement <8 x i8> undef, i8 %1, i32 0
  %lane = shufflevector <8 x i8> %2, <8 x i8> undef, <8 x i32> zeroinitializer
  store i8* %add.ptr, i8** %a, align 4
  ret <8 x i8> %lane
}

define <8 x i8> @vld1dupi8_postinc_fixed(i8** noalias nocapture %a) nounwind {
entry:
;CHECK-LABEL: vld1dupi8_postinc_fixed:
;CHECK: vld1.8 {d16[]}, [{{r[0-9]+|lr}}]!
  %0 = load i8*, i8** %a, align 4
  %1 = load i8, i8* %0, align 1
  %2 = insertelement <8 x i8> undef, i8 %1, i32 0
  %lane = shufflevector <8 x i8> %2, <8 x i8> undef, <8 x i32> zeroinitializer
  %add.ptr = getelementptr inbounds i8, i8* %0, i32 1
  store i8* %add.ptr, i8** %a, align 4
  ret <8 x i8> %lane
}

define <8 x i8> @vld1dupi8_postinc_register(i8** noalias nocapture %a, i32 %n) nounwind {
entry:
;CHECK-LABEL: vld1dupi8_postinc_register:
;CHECK: vld1.8 {d16[]}, [{{r[0-9]+|lr}}], r1
  %0 = load i8*, i8** %a, align 4
  %1 = load i8, i8* %0, align 1
  %2 = insertelement <8 x i8> undef, i8 %1, i32 0
  %lane = shufflevector <8 x i8> %2, <8 x i8> undef, <8 x i32> zeroinitializer
  %add.ptr = getelementptr inbounds i8, i8* %0, i32 %n
  store i8* %add.ptr, i8** %a, align 4
  ret <8 x i8> %lane
}

define <16 x i8> @vld1dupqi8_preinc(i8** noalias nocapture %a, i32 %b) nounwind {
entry:
;CHECK-LABEL: vld1dupqi8_preinc:
;CHECK: vld1.8 {d16[], d17[]}, [{{r[0-9]+|lr}}]
  %0 = load i8*, i8** %a, align 4
  %add.ptr = getelementptr inbounds i8, i8* %0, i32 %b
  %1 = load i8, i8* %add.ptr, align 1
  %2 = insertelement <16 x i8> undef, i8 %1, i32 0
  %lane = shufflevector <16 x i8> %2, <16 x i8> undef, <16 x i32> zeroinitializer
  store i8* %add.ptr, i8** %a, align 4
  ret <16 x i8> %lane
}

define <16 x i8> @vld1dupqi8_postinc_fixed(i8** noalias nocapture %a) nounwind {
entry:
;CHECK-LABEL: vld1dupqi8_postinc_fixed:
;CHECK: vld1.8 {d16[], d17[]}, [{{r[0-9]+|lr}}]!
  %0 = load i8*, i8** %a, align 4
  %1 = load i8, i8* %0, align 1
  %2 = insertelement <16 x i8> undef, i8 %1, i32 0
  %lane = shufflevector <16 x i8> %2, <16 x i8> undef, <16 x i32> zeroinitializer
  %add.ptr = getelementptr inbounds i8, i8* %0, i32 1
  store i8* %add.ptr, i8** %a, align 4
  ret <16 x i8> %lane
}

define <16 x i8> @vld1dupqi8_postinc_register(i8** noalias nocapture %a, i32 %n) nounwind {
entry:
;CHECK-LABEL: vld1dupqi8_postinc_register:
;CHECK: vld1.8 {d16[], d17[]}, [{{r[0-9]+|lr}}], r1
  %0 = load i8*, i8** %a, align 4
  %1 = load i8, i8* %0, align 1
  %2 = insertelement <16 x i8> undef, i8 %1, i32 0
  %lane = shufflevector <16 x i8> %2, <16 x i8> undef, <16 x i32> zeroinitializer
  %add.ptr = getelementptr inbounds i8, i8* %0, i32 %n
  store i8* %add.ptr, i8** %a, align 4
  ret <16 x i8> %lane
}

define <4 x i16> @vld1dupi16(i16* %A) nounwind {
;CHECK-LABEL: vld1dupi16:
;Check the alignment value.  Max for this instruction is 16 bits:
;CHECK: vld1.16 {d16[]}, [{{r[0-9]+|lr}}:16]
	%tmp1 = load i16, i16* %A, align 8
	%tmp2 = insertelement <4 x i16> undef, i16 %tmp1, i32 0
	%tmp3 = shufflevector <4 x i16> %tmp2, <4 x i16> undef, <4 x i32> zeroinitializer
        ret <4 x i16> %tmp3
}

define <4 x i16> @vld1dupi16_misaligned(i16* %A) nounwind {
;CHECK-LABEL: vld1dupi16_misaligned:
;CHECK: vld1.16 {d16[]}, [{{r[0-9]+|lr}}]
	%tmp1 = load i16, i16* %A, align 1
	%tmp2 = insertelement <4 x i16> undef, i16 %tmp1, i32 0
	%tmp3 = shufflevector <4 x i16> %tmp2, <4 x i16> undef, <4 x i32> zeroinitializer
        ret <4 x i16> %tmp3
}

; This sort of looks like a vld1dup, but there's an extension in the way.
define <4 x i16> @load_i16_dup_zext(i8* %A) nounwind {
;CHECK-LABEL: load_i16_dup_zext:
;CHECK: ldrb    r0, [{{r[0-9]+|lr}}]
;CHECK-NEXT: vdup.16 d16, r0
        %tmp1 = load i8, i8* %A, align 1
        %tmp2 = zext i8 %tmp1 to i16
        %tmp3 = insertelement <4 x i16> undef, i16 %tmp2, i32 0
        %tmp4 = shufflevector <4 x i16> %tmp3, <4 x i16> undef, <4 x i32> zeroinitializer
        ret <4 x i16> %tmp4
}

; This sort of looks like a vld1dup, but there's an extension in the way.
define <4 x i16> @load_i16_dup_sext(i8* %A) nounwind {
;CHECK-LABEL: load_i16_dup_sext:
;CHECK: ldrsb    r0, [{{r[0-9]+|lr}}]
;CHECK-NEXT: vdup.16 d16, r0
        %tmp1 = load i8, i8* %A, align 1
        %tmp2 = sext i8 %tmp1 to i16
        %tmp3 = insertelement <4 x i16> undef, i16 %tmp2, i32 0
        %tmp4 = shufflevector <4 x i16> %tmp3, <4 x i16> undef, <4 x i32> zeroinitializer
        ret <4 x i16> %tmp4
}

; This sort of looks like a vld1dup, but there's an extension in the way.
define <8 x i16> @load_i16_dupq_zext(i8* %A) nounwind {
;CHECK-LABEL: load_i16_dupq_zext:
;CHECK: ldrb    r0, [{{r[0-9]+|lr}}]
;CHECK-NEXT: vdup.16 q8, r0
        %tmp1 = load i8, i8* %A, align 1
        %tmp2 = zext i8 %tmp1 to i16
        %tmp3 = insertelement <8 x i16> undef, i16 %tmp2, i32 0
        %tmp4 = shufflevector <8 x i16> %tmp3, <8 x i16> undef, <8 x i32> zeroinitializer
        ret <8 x i16> %tmp4
}

define <2 x i32> @vld1dupi32(i32* %A) nounwind {
;CHECK-LABEL: vld1dupi32:
;Check the alignment value.  Max for this instruction is 32 bits:
;CHECK: vld1.32 {d16[]}, [{{r[0-9]+|lr}}:32]
	%tmp1 = load i32, i32* %A, align 8
	%tmp2 = insertelement <2 x i32> undef, i32 %tmp1, i32 0
	%tmp3 = shufflevector <2 x i32> %tmp2, <2 x i32> undef, <2 x i32> zeroinitializer
        ret <2 x i32> %tmp3
}

; This sort of looks like a vld1dup, but there's an extension in the way.
define <4 x i32> @load_i32_dup_zext(i8* %A) nounwind {
;CHECK-LABEL: load_i32_dup_zext:
;CHECK: ldrb    r0, [{{r[0-9]+|lr}}]
;CHECK-NEXT: vdup.32 q8, r0
        %tmp1 = load i8, i8* %A, align 1
        %tmp2 = zext i8 %tmp1 to i32
        %tmp3 = insertelement <4 x i32> undef, i32 %tmp2, i32 0
        %tmp4 = shufflevector <4 x i32> %tmp3, <4 x i32> undef, <4 x i32> zeroinitializer
        ret <4 x i32> %tmp4
}

; This sort of looks like a vld1dup, but there's an extension in the way.
define <4 x i32> @load_i32_dup_sext(i8* %A) nounwind {
;CHECK-LABEL: load_i32_dup_sext:
;CHECK: ldrsb    r0, [{{r[0-9]+|lr}}]
;CHECK-NEXT: vdup.32 q8, r0
        %tmp1 = load i8, i8* %A, align 1
        %tmp2 = sext i8 %tmp1 to i32
        %tmp3 = insertelement <4 x i32> undef, i32 %tmp2, i32 0
        %tmp4 = shufflevector <4 x i32> %tmp3, <4 x i32> undef, <4 x i32> zeroinitializer
        ret <4 x i32> %tmp4
}

define <2 x float> @vld1dupf(float* %A) nounwind {
;CHECK-LABEL: vld1dupf:
;CHECK: vld1.32 {d16[]}, [{{r[0-9]+|lr}}:32]
	%tmp0 = load float, float* %A
        %tmp1 = insertelement <2 x float> undef, float %tmp0, i32 0
        %tmp2 = shufflevector <2 x float> %tmp1, <2 x float> undef, <2 x i32> zeroinitializer
        ret <2 x float> %tmp2
}

define <16 x i8> @vld1dupQi8(i8* %A) nounwind {
;CHECK-LABEL: vld1dupQi8:
;Check the (default) alignment value.
;CHECK: vld1.8 {d16[], d17[]}, [{{r[0-9]+|lr}}]
	%tmp1 = load i8, i8* %A, align 8
	%tmp2 = insertelement <16 x i8> undef, i8 %tmp1, i32 0
	%tmp3 = shufflevector <16 x i8> %tmp2, <16 x i8> undef, <16 x i32> zeroinitializer
        ret <16 x i8> %tmp3
}

define <4 x float> @vld1dupQf(float* %A) nounwind {
;CHECK-LABEL: vld1dupQf:
;CHECK: vld1.32 {d16[], d17[]}, [{{r[0-9]+|lr}}:32]
        %tmp0 = load float, float* %A
        %tmp1 = insertelement <4 x float> undef, float %tmp0, i32 0
        %tmp2 = shufflevector <4 x float> %tmp1, <4 x float> undef, <4 x i32> zeroinitializer
        ret <4 x float> %tmp2
}

%struct.__neon_int8x8x2_t = type { <8 x i8>, <8 x i8> }
%struct.__neon_int4x16x2_t = type { <4 x i16>, <4 x i16> }
%struct.__neon_int2x32x2_t = type { <2 x i32>, <2 x i32> }

define <8 x i8> @vld2dupi8(i8* %A) nounwind {
;CHECK-LABEL: vld2dupi8:
;Check the (default) alignment value.
;CHECK: vld2.8 {d16[0], d17[0]}, [{{r[0-9]+|lr}}]
	%tmp0 = tail call %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8.p0i8(i8* %A, <8 x i8> undef, <8 x i8> undef, i32 0, i32 1)
	%tmp1 = extractvalue %struct.__neon_int8x8x2_t %tmp0, 0
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> zeroinitializer
	%tmp3 = extractvalue %struct.__neon_int8x8x2_t %tmp0, 1
	%tmp4 = shufflevector <8 x i8> %tmp3, <8 x i8> undef, <8 x i32> zeroinitializer
        %tmp5 = add <8 x i8> %tmp2, %tmp4
        ret <8 x i8> %tmp5
}

define void @vld2dupi8_preinc(%struct.__neon_int8x8x2_t* noalias nocapture sret(%struct.__neon_int8x8x2_t) %agg.result, i8** noalias nocapture %a, i32 %b) nounwind {
;CHECK-LABEL: vld2dupi8_preinc:
;CHECK: vld2.8 {d16[], d17[]}, [r2]
entry:
  %0 = load i8*, i8** %a, align 4
  %add.ptr = getelementptr inbounds i8, i8* %0, i32 %b
  %vld_dup = tail call %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8.p0i8(i8* %add.ptr, <8 x i8> undef, <8 x i8> undef, i32 0, i32 1)
  %1 = extractvalue %struct.__neon_int8x8x2_t %vld_dup, 0
  %lane = shufflevector <8 x i8> %1, <8 x i8> undef, <8 x i32> zeroinitializer
  %2 = extractvalue %struct.__neon_int8x8x2_t %vld_dup, 1
  %lane1 = shufflevector <8 x i8> %2, <8 x i8> undef, <8 x i32> zeroinitializer
  store i8* %add.ptr, i8** %a, align 4
  %r8 = getelementptr inbounds %struct.__neon_int8x8x2_t, %struct.__neon_int8x8x2_t* %agg.result, i32 0, i32 0
  store <8 x i8> %lane, <8 x i8>* %r8, align 8
  %r11 = getelementptr inbounds %struct.__neon_int8x8x2_t, %struct.__neon_int8x8x2_t* %agg.result, i32 0, i32 1
  store <8 x i8> %lane1, <8 x i8>* %r11, align 8
  ret void
}

define void @vld2dupi8_postinc_fixed(%struct.__neon_int8x8x2_t* noalias nocapture sret(%struct.__neon_int8x8x2_t) %agg.result, i8** noalias nocapture %a) nounwind {
entry:
;CHECK-LABEL: vld2dupi8_postinc_fixed:
;CHECK: vld2.8 {d16[], d17[]}, [r2]!
  %0 = load i8*, i8** %a, align 4
  %vld_dup = tail call %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8.p0i8(i8* %0, <8 x i8> undef, <8 x i8> undef, i32 0, i32 1)
  %1 = extractvalue %struct.__neon_int8x8x2_t %vld_dup, 0
  %lane = shufflevector <8 x i8> %1, <8 x i8> undef, <8 x i32> zeroinitializer
  %2 = extractvalue %struct.__neon_int8x8x2_t %vld_dup, 1
  %lane1 = shufflevector <8 x i8> %2, <8 x i8> undef, <8 x i32> zeroinitializer
  %add.ptr = getelementptr inbounds i8, i8* %0, i32 2
  store i8* %add.ptr, i8** %a, align 4
  %r7 = getelementptr inbounds %struct.__neon_int8x8x2_t, %struct.__neon_int8x8x2_t* %agg.result, i32 0, i32 0
  store <8 x i8> %lane, <8 x i8>* %r7, align 8
  %r10 = getelementptr inbounds %struct.__neon_int8x8x2_t, %struct.__neon_int8x8x2_t* %agg.result, i32 0, i32 1
  store <8 x i8> %lane1, <8 x i8>* %r10, align 8
  ret void
}

define void @vld2dupi8_postinc_variable(%struct.__neon_int8x8x2_t* noalias nocapture sret(%struct.__neon_int8x8x2_t) %agg.result, i8** noalias nocapture %a, i32 %n) nounwind {
entry:
;CHECK-LABEL: vld2dupi8_postinc_variable:
;CHECK: vld2.8 {d16[], d17[]}, [r3], r2
  %0 = load i8*, i8** %a, align 4
  %vld_dup = tail call %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8.p0i8(i8* %0, <8 x i8> undef, <8 x i8> undef, i32 0, i32 1)
  %1 = extractvalue %struct.__neon_int8x8x2_t %vld_dup, 0
  %lane = shufflevector <8 x i8> %1, <8 x i8> undef, <8 x i32> zeroinitializer
  %2 = extractvalue %struct.__neon_int8x8x2_t %vld_dup, 1
  %lane1 = shufflevector <8 x i8> %2, <8 x i8> undef, <8 x i32> zeroinitializer
  %add.ptr = getelementptr inbounds i8, i8* %0, i32 %n
  store i8* %add.ptr, i8** %a, align 4
  %r7 = getelementptr inbounds %struct.__neon_int8x8x2_t, %struct.__neon_int8x8x2_t* %agg.result, i32 0, i32 0
  store <8 x i8> %lane, <8 x i8>* %r7, align 8
  %r10 = getelementptr inbounds %struct.__neon_int8x8x2_t, %struct.__neon_int8x8x2_t* %agg.result, i32 0, i32 1
  store <8 x i8> %lane1, <8 x i8>* %r10, align 8
  ret void
}

define <4 x i16> @vld2dupi16(i8* %A) nounwind {
;CHECK-LABEL: vld2dupi16:
;Check that a power-of-two alignment smaller than the total size of the memory
;being loaded is ignored.
;CHECK: vld2.16 {d16[0], d17[0]}, [{{r[0-9]+|lr}}]
	%tmp0 = tail call %struct.__neon_int4x16x2_t @llvm.arm.neon.vld2lane.v4i16.p0i8(i8* %A, <4 x i16> undef, <4 x i16> undef, i32 0, i32 2)
	%tmp1 = extractvalue %struct.__neon_int4x16x2_t %tmp0, 0
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> zeroinitializer
	%tmp3 = extractvalue %struct.__neon_int4x16x2_t %tmp0, 1
	%tmp4 = shufflevector <4 x i16> %tmp3, <4 x i16> undef, <4 x i32> zeroinitializer
        %tmp5 = add <4 x i16> %tmp2, %tmp4
        ret <4 x i16> %tmp5
}

;Check for a post-increment updating load. 
define <4 x i16> @vld2dupi16_update(i16** %ptr) nounwind {
;CHECK-LABEL: vld2dupi16_update:
;CHECK: vld2.16 {d16[0], d17[0]}, [{{r[0-9]+|lr}}]!
	%A = load i16*, i16** %ptr
        %A2 = bitcast i16* %A to i8*
	%tmp0 = tail call %struct.__neon_int4x16x2_t @llvm.arm.neon.vld2lane.v4i16.p0i8(i8* %A2, <4 x i16> undef, <4 x i16> undef, i32 0, i32 2)
	%tmp1 = extractvalue %struct.__neon_int4x16x2_t %tmp0, 0
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> zeroinitializer
	%tmp3 = extractvalue %struct.__neon_int4x16x2_t %tmp0, 1
	%tmp4 = shufflevector <4 x i16> %tmp3, <4 x i16> undef, <4 x i32> zeroinitializer
	%tmp5 = add <4 x i16> %tmp2, %tmp4
	%tmp6 = getelementptr i16, i16* %A, i32 2
	store i16* %tmp6, i16** %ptr
	ret <4 x i16> %tmp5
}

define <4 x i16> @vld2dupi16_odd_update(i16** %ptr) nounwind {
;CHECK-LABEL: vld2dupi16_odd_update:
;CHECK: mov [[INC:r[0-9]+]], #6
;CHECK: vld2.16 {d16[0], d17[0]}, [{{r[0-9]+|lr}}], [[INC]]
	%A = load i16*, i16** %ptr
        %A2 = bitcast i16* %A to i8*
	%tmp0 = tail call %struct.__neon_int4x16x2_t @llvm.arm.neon.vld2lane.v4i16.p0i8(i8* %A2, <4 x i16> undef, <4 x i16> undef, i32 0, i32 2)
	%tmp1 = extractvalue %struct.__neon_int4x16x2_t %tmp0, 0
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> zeroinitializer
	%tmp3 = extractvalue %struct.__neon_int4x16x2_t %tmp0, 1
	%tmp4 = shufflevector <4 x i16> %tmp3, <4 x i16> undef, <4 x i32> zeroinitializer
	%tmp5 = add <4 x i16> %tmp2, %tmp4
	%tmp6 = getelementptr i16, i16* %A, i32 3
	store i16* %tmp6, i16** %ptr
	ret <4 x i16> %tmp5
}

define <2 x i32> @vld2dupi32(i8* %A) nounwind {
;CHECK-LABEL: vld2dupi32:
;Check the alignment value.  Max for this instruction is 64 bits:
;CHECK: vld2.32 {d16[0], d17[0]}, [{{r[0-9]+|lr}}:64]
	%tmp0 = tail call %struct.__neon_int2x32x2_t @llvm.arm.neon.vld2lane.v2i32.p0i8(i8* %A, <2 x i32> undef, <2 x i32> undef, i32 0, i32 16)
	%tmp1 = extractvalue %struct.__neon_int2x32x2_t %tmp0, 0
	%tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <2 x i32> zeroinitializer
	%tmp3 = extractvalue %struct.__neon_int2x32x2_t %tmp0, 1
	%tmp4 = shufflevector <2 x i32> %tmp3, <2 x i32> undef, <2 x i32> zeroinitializer
        %tmp5 = add <2 x i32> %tmp2, %tmp4
        ret <2 x i32> %tmp5
}

declare %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8.p0i8(i8*, <8 x i8>, <8 x i8>, i32, i32) nounwind readonly
declare %struct.__neon_int4x16x2_t @llvm.arm.neon.vld2lane.v4i16.p0i8(i8*, <4 x i16>, <4 x i16>, i32, i32) nounwind readonly
declare %struct.__neon_int2x32x2_t @llvm.arm.neon.vld2lane.v2i32.p0i8(i8*, <2 x i32>, <2 x i32>, i32, i32) nounwind readonly

%struct.__neon_int8x8x3_t = type { <8 x i8>, <8 x i8>, <8 x i8> }
%struct.__neon_int16x4x3_t = type { <4 x i16>, <4 x i16>, <4 x i16> }

;Check for a post-increment updating load with register increment.
define <8 x i8> @vld3dupi8_update(i8** %ptr, i32 %inc) nounwind {
;CHECK-LABEL: vld3dupi8_update:
;CHECK: vld3.8 {d16[0], d17[0], d18[0]}, [{{r[0-9]+|lr}}], r1
	%A = load i8*, i8** %ptr
	%tmp0 = tail call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3lane.v8i8.p0i8(i8* %A, <8 x i8> undef, <8 x i8> undef, <8 x i8> undef, i32 0, i32 8)
	%tmp1 = extractvalue %struct.__neon_int8x8x3_t %tmp0, 0
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> zeroinitializer
	%tmp3 = extractvalue %struct.__neon_int8x8x3_t %tmp0, 1
	%tmp4 = shufflevector <8 x i8> %tmp3, <8 x i8> undef, <8 x i32> zeroinitializer
	%tmp5 = extractvalue %struct.__neon_int8x8x3_t %tmp0, 2
	%tmp6 = shufflevector <8 x i8> %tmp5, <8 x i8> undef, <8 x i32> zeroinitializer
	%tmp7 = add <8 x i8> %tmp2, %tmp4
	%tmp8 = add <8 x i8> %tmp7, %tmp6
	%tmp9 = getelementptr i8, i8* %A, i32 %inc
	store i8* %tmp9, i8** %ptr
	ret <8 x i8> %tmp8
}

define <4 x i16> @vld3dupi16(i8* %A) nounwind {
;CHECK-LABEL: vld3dupi16:
;Check the (default) alignment value. VLD3 does not support alignment.
;CHECK: vld3.16 {d16[0], d17[0], d18[0]}, [{{r[0-9]+|lr}}]
	%tmp0 = tail call %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3lane.v4i16.p0i8(i8* %A, <4 x i16> undef, <4 x i16> undef, <4 x i16> undef, i32 0, i32 8)
	%tmp1 = extractvalue %struct.__neon_int16x4x3_t %tmp0, 0
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> zeroinitializer
	%tmp3 = extractvalue %struct.__neon_int16x4x3_t %tmp0, 1
	%tmp4 = shufflevector <4 x i16> %tmp3, <4 x i16> undef, <4 x i32> zeroinitializer
	%tmp5 = extractvalue %struct.__neon_int16x4x3_t %tmp0, 2
	%tmp6 = shufflevector <4 x i16> %tmp5, <4 x i16> undef, <4 x i32> zeroinitializer
        %tmp7 = add <4 x i16> %tmp2, %tmp4
        %tmp8 = add <4 x i16> %tmp7, %tmp6
        ret <4 x i16> %tmp8
}

declare %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3lane.v8i8.p0i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, i32, i32) nounwind readonly
declare %struct.__neon_int16x4x3_t @llvm.arm.neon.vld3lane.v4i16.p0i8(i8*, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32) nounwind readonly

%struct.__neon_int16x4x4_t = type { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> }
%struct.__neon_int32x2x4_t = type { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> }

;Check for a post-increment updating load.
define <4 x i16> @vld4dupi16_update(i16** %ptr) nounwind {
;CHECK-LABEL: vld4dupi16_update:
;CHECK: vld4.16 {d16[0], d17[0], d18[0], d19[0]}, [{{r[0-9]+|lr}}]!
	%A = load i16*, i16** %ptr
        %A2 = bitcast i16* %A to i8*
	%tmp0 = tail call %struct.__neon_int16x4x4_t @llvm.arm.neon.vld4lane.v4i16.p0i8(i8* %A2, <4 x i16> undef, <4 x i16> undef, <4 x i16> undef, <4 x i16> undef, i32 0, i32 1)
	%tmp1 = extractvalue %struct.__neon_int16x4x4_t %tmp0, 0
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> zeroinitializer
	%tmp3 = extractvalue %struct.__neon_int16x4x4_t %tmp0, 1
	%tmp4 = shufflevector <4 x i16> %tmp3, <4 x i16> undef, <4 x i32> zeroinitializer
	%tmp5 = extractvalue %struct.__neon_int16x4x4_t %tmp0, 2
	%tmp6 = shufflevector <4 x i16> %tmp5, <4 x i16> undef, <4 x i32> zeroinitializer
	%tmp7 = extractvalue %struct.__neon_int16x4x4_t %tmp0, 3
	%tmp8 = shufflevector <4 x i16> %tmp7, <4 x i16> undef, <4 x i32> zeroinitializer
	%tmp9 = add <4 x i16> %tmp2, %tmp4
	%tmp10 = add <4 x i16> %tmp6, %tmp8
	%tmp11 = add <4 x i16> %tmp9, %tmp10
	%tmp12 = getelementptr i16, i16* %A, i32 4
	store i16* %tmp12, i16** %ptr
	ret <4 x i16> %tmp11
}

define <2 x i32> @vld4dupi32(i8* %A) nounwind {
;CHECK-LABEL: vld4dupi32:
;Check the alignment value.  An 8-byte alignment is allowed here even though
;it is smaller than the total size of the memory being loaded.
;CHECK: vld4.32 {d16[0], d17[0], d18[0], d19[0]}, [{{r[0-9]+|lr}}:64]
	%tmp0 = tail call %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4lane.v2i32.p0i8(i8* %A, <2 x i32> undef, <2 x i32> undef, <2 x i32> undef, <2 x i32> undef, i32 0, i32 8)
	%tmp1 = extractvalue %struct.__neon_int32x2x4_t %tmp0, 0
	%tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <2 x i32> zeroinitializer
	%tmp3 = extractvalue %struct.__neon_int32x2x4_t %tmp0, 1
	%tmp4 = shufflevector <2 x i32> %tmp3, <2 x i32> undef, <2 x i32> zeroinitializer
	%tmp5 = extractvalue %struct.__neon_int32x2x4_t %tmp0, 2
	%tmp6 = shufflevector <2 x i32> %tmp5, <2 x i32> undef, <2 x i32> zeroinitializer
	%tmp7 = extractvalue %struct.__neon_int32x2x4_t %tmp0, 3
	%tmp8 = shufflevector <2 x i32> %tmp7, <2 x i32> undef, <2 x i32> zeroinitializer
        %tmp9 = add <2 x i32> %tmp2, %tmp4
        %tmp10 = add <2 x i32> %tmp6, %tmp8
        %tmp11 = add <2 x i32> %tmp9, %tmp10
        ret <2 x i32> %tmp11
}

declare %struct.__neon_int16x4x4_t @llvm.arm.neon.vld4lane.v4i16.p0i8(i8*, <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16>, i32, i32) nounwind readonly
declare %struct.__neon_int32x2x4_t @llvm.arm.neon.vld4lane.v2i32.p0i8(i8*, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, i32, i32) nounwind readonly
