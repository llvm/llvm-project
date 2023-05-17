; RUN: llc < %s -mattr=+neon | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32"
target triple = "thumbv7-elf"

define i32 @vget_lanes8(ptr %A) nounwind {
;CHECK-LABEL: vget_lanes8:
;CHECK: vmov.s8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = extractelement <8 x i8> %tmp1, i32 1
	%tmp3 = sext i8 %tmp2 to i32
	ret i32 %tmp3
}

define i32 @vget_lanes16(ptr %A) nounwind {
;CHECK-LABEL: vget_lanes16:
;CHECK: vmov.s16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = extractelement <4 x i16> %tmp1, i32 1
	%tmp3 = sext i16 %tmp2 to i32
	ret i32 %tmp3
}

define i32 @vget_laneu8(ptr %A) nounwind {
;CHECK-LABEL: vget_laneu8:
;CHECK: vmov.u8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = extractelement <8 x i8> %tmp1, i32 1
	%tmp3 = zext i8 %tmp2 to i32
	ret i32 %tmp3
}

define i32 @vget_laneu16(ptr %A) nounwind {
;CHECK-LABEL: vget_laneu16:
;CHECK: vmov.u16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = extractelement <4 x i16> %tmp1, i32 1
	%tmp3 = zext i16 %tmp2 to i32
	ret i32 %tmp3
}

; Do a vector add to keep the extraction from being done directly from memory.
define i32 @vget_lanei32(ptr %A) nounwind {
;CHECK-LABEL: vget_lanei32:
;CHECK: vmov.32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = add <2 x i32> %tmp1, %tmp1
	%tmp3 = extractelement <2 x i32> %tmp2, i32 1
	ret i32 %tmp3
}

define i32 @vgetQ_lanes8(ptr %A) nounwind {
;CHECK-LABEL: vgetQ_lanes8:
;CHECK: vmov.s8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = extractelement <16 x i8> %tmp1, i32 1
	%tmp3 = sext i8 %tmp2 to i32
	ret i32 %tmp3
}

define i32 @vgetQ_lanes16(ptr %A) nounwind {
;CHECK-LABEL: vgetQ_lanes16:
;CHECK: vmov.s16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = extractelement <8 x i16> %tmp1, i32 1
	%tmp3 = sext i16 %tmp2 to i32
	ret i32 %tmp3
}

define i32 @vgetQ_laneu8(ptr %A) nounwind {
;CHECK-LABEL: vgetQ_laneu8:
;CHECK: vmov.u8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = extractelement <16 x i8> %tmp1, i32 1
	%tmp3 = zext i8 %tmp2 to i32
	ret i32 %tmp3
}

define i32 @vgetQ_laneu16(ptr %A) nounwind {
;CHECK-LABEL: vgetQ_laneu16:
;CHECK: vmov.u16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = extractelement <8 x i16> %tmp1, i32 1
	%tmp3 = zext i16 %tmp2 to i32
	ret i32 %tmp3
}

; Do a vector add to keep the extraction from being done directly from memory.
define i32 @vgetQ_lanei32(ptr %A) nounwind {
;CHECK-LABEL: vgetQ_lanei32:
;CHECK: vmov.32
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = add <4 x i32> %tmp1, %tmp1
	%tmp3 = extractelement <4 x i32> %tmp2, i32 1
	ret i32 %tmp3
}

define arm_aapcs_vfpcc void @test_vget_laneu16() nounwind {
entry:
; CHECK: vmov.u16 r0, d{{.*}}[1]
  %arg0_uint16x4_t = alloca <4 x i16>             ; <ptr> [#uses=1]
  %out_uint16_t = alloca i16                      ; <ptr> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %0 = load <4 x i16>, ptr %arg0_uint16x4_t, align 8  ; <<4 x i16>> [#uses=1]
  %1 = extractelement <4 x i16> %0, i32 1         ; <i16> [#uses=1]
  %2 = add i16 %1, %1
  store i16 %2, ptr %out_uint16_t, align 2
  br label %return

return:                                           ; preds = %entry
  ret void
}

define arm_aapcs_vfpcc void @test_vget_laneu8() nounwind {
entry:
; CHECK: vmov.u8 r0, d{{.*}}[1]
  %arg0_uint8x8_t = alloca <8 x i8>               ; <ptr> [#uses=1]
  %out_uint8_t = alloca i8                        ; <ptr> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %0 = load <8 x i8>, ptr %arg0_uint8x8_t, align 8    ; <<8 x i8>> [#uses=1]
  %1 = extractelement <8 x i8> %0, i32 1          ; <i8> [#uses=1]
  %2 = add i8 %1, %1
  store i8 %2, ptr %out_uint8_t, align 1
  br label %return

return:                                           ; preds = %entry
  ret void
}

define arm_aapcs_vfpcc void @test_vgetQ_laneu16() nounwind {
entry:
; CHECK: vmov.u16 r0, d{{.*}}[1]
  %arg0_uint16x8_t = alloca <8 x i16>             ; <ptr> [#uses=1]
  %out_uint16_t = alloca i16                      ; <ptr> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %0 = load <8 x i16>, ptr %arg0_uint16x8_t, align 16 ; <<8 x i16>> [#uses=1]
  %1 = extractelement <8 x i16> %0, i32 1         ; <i16> [#uses=1]
  %2 = add i16 %1, %1
  store i16 %2, ptr %out_uint16_t, align 2
  br label %return

return:                                           ; preds = %entry
  ret void
}

define arm_aapcs_vfpcc void @test_vgetQ_laneu8() nounwind {
entry:
; CHECK: vmov.u8 r0, d{{.*}}[1]
  %arg0_uint8x16_t = alloca <16 x i8>             ; <ptr> [#uses=1]
  %out_uint8_t = alloca i8                        ; <ptr> [#uses=1]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %0 = load <16 x i8>, ptr %arg0_uint8x16_t, align 16 ; <<16 x i8>> [#uses=1]
  %1 = extractelement <16 x i8> %0, i32 1         ; <i8> [#uses=1]
  %2 = add i8 %1, %1
  store i8 %2, ptr %out_uint8_t, align 1
  br label %return

return:                                           ; preds = %entry
  ret void
}

define <8 x i8> @vset_lane8(ptr %A, i8 %B) nounwind {
;CHECK-LABEL: vset_lane8:
;CHECK: vmov.8
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = insertelement <8 x i8> %tmp1, i8 %B, i32 1
	ret <8 x i8> %tmp2
}

define <4 x i16> @vset_lane16(ptr %A, i16 %B) nounwind {
;CHECK-LABEL: vset_lane16:
;CHECK: vmov.16
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = insertelement <4 x i16> %tmp1, i16 %B, i32 1
	ret <4 x i16> %tmp2
}

define <2 x i32> @vset_lane32(ptr %A, i32 %B) nounwind {
;CHECK-LABEL: vset_lane32:
;CHECK: vmov.32
	%tmp1 = load <2 x i32>, ptr %A
	%tmp2 = insertelement <2 x i32> %tmp1, i32 %B, i32 1
	ret <2 x i32> %tmp2
}

define <16 x i8> @vsetQ_lane8(ptr %A, i8 %B) nounwind {
;CHECK-LABEL: vsetQ_lane8:
;CHECK: vmov.8
	%tmp1 = load <16 x i8>, ptr %A
	%tmp2 = insertelement <16 x i8> %tmp1, i8 %B, i32 1
	ret <16 x i8> %tmp2
}

define <8 x i16> @vsetQ_lane16(ptr %A, i16 %B) nounwind {
;CHECK-LABEL: vsetQ_lane16:
;CHECK: vmov.16
	%tmp1 = load <8 x i16>, ptr %A
	%tmp2 = insertelement <8 x i16> %tmp1, i16 %B, i32 1
	ret <8 x i16> %tmp2
}

define <4 x i32> @vsetQ_lane32(ptr %A, i32 %B) nounwind {
;CHECK-LABEL: vsetQ_lane32:
;CHECK: vmov.32 d{{.*}}[1], r1
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = insertelement <4 x i32> %tmp1, i32 %B, i32 1
	ret <4 x i32> %tmp2
}

define arm_aapcs_vfpcc <2 x float> @test_vset_lanef32(float %arg0_float32_t, <2 x float> %arg1_float32x2_t) nounwind {
;CHECK-LABEL: test_vset_lanef32:
;CHECK: vmov.f32 s3, s0
;CHECK: vmov.f64 d0, d1
entry:
  %0 = insertelement <2 x float> %arg1_float32x2_t, float %arg0_float32_t, i32 1 ; <<2 x float>> [#uses=1]
  ret <2 x float> %0
}

; The llvm extractelement instruction does not require that the lane number
; be an immediate constant.  Make sure a variable lane number is handled.

define i32 @vget_variable_lanes8(ptr %A, i32 %B) nounwind {
	%tmp1 = load <8 x i8>, ptr %A
	%tmp2 = extractelement <8 x i8> %tmp1, i32 %B
	%tmp3 = sext i8 %tmp2 to i32
	ret i32 %tmp3
}

define i32 @vgetQ_variable_lanei32(ptr %A, i32 %B) nounwind {
	%tmp1 = load <4 x i32>, ptr %A
	%tmp2 = add <4 x i32> %tmp1, %tmp1
	%tmp3 = extractelement <4 x i32> %tmp2, i32 %B
	ret i32 %tmp3
}
