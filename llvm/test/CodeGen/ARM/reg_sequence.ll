; RUN: llc < %s -mtriple=arm-apple-ios -mcpu=cortex-a8 -arm-atomic-cfg-tidy=0 | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: llc < %s -mtriple=arm-apple-ios -mcpu=cortex-a8 -arm-atomic-cfg-tidy=0 -regalloc=basic | FileCheck %s --check-prefixes=CHECK,BASIC
; Implementing vld / vst as REG_SEQUENCE eliminates the extra vmov's.

%struct.int16x8_t = type { <8 x i16> }
%struct.int32x4_t = type { <4 x i32> }
%struct.__neon_int8x8x2_t = type { <8 x i8>, <8 x i8> }
%struct.__neon_int8x8x3_t = type { <8 x i8>,  <8 x i8>,  <8 x i8> }
%struct.__neon_int16x8x2_t = type { <8 x i16>, <8 x i16> }
%struct.__neon_int32x4x2_t = type { <4 x i32>, <4 x i32> }

define void @t1(i16* %i_ptr, i16* %o_ptr, %struct.int32x4_t* nocapture %vT0ptr, %struct.int32x4_t* nocapture %vT1ptr) nounwind {
; DEFAULT-LABEL: t1:
; DEFAULT:       @ %bb.0: @ %entry
; DEFAULT-NEXT:    vld1.16 {d16, d17}, [r0]
; DEFAULT-NEXT:    vmovl.s16 q9, d17
; DEFAULT-NEXT:    vld1.64 {d20, d21}, [r3:128]
; DEFAULT-NEXT:    vmovl.s16 q8, d16
; DEFAULT-NEXT:    vld1.64 {d22, d23}, [r2:128]
; DEFAULT-NEXT:    vmul.i32 q9, q10, q9
; DEFAULT-NEXT:    vmul.i32 q8, q11, q8
; DEFAULT-NEXT:    vshrn.i32 d19, q9, #12
; DEFAULT-NEXT:    vshrn.i32 d18, q8, #12
; DEFAULT-NEXT:    vst1.16 {d18, d19}, [r1]
; DEFAULT-NEXT:    bx lr
;
; BASIC-LABEL: t1:
; BASIC:       @ %bb.0: @ %entry
; BASIC-NEXT:    vld1.16 {d16, d17}, [r0]
; BASIC-NEXT:    vmovl.s16 q12, d17
; BASIC-NEXT:    vld1.64 {d22, d23}, [r3:128]
; BASIC-NEXT:    vmovl.s16 q10, d16
; BASIC-NEXT:    vld1.64 {d18, d19}, [r2:128]
; BASIC-NEXT:    vmul.i32 q8, q11, q12
; BASIC-NEXT:    vmul.i32 q9, q9, q10
; BASIC-NEXT:    vshrn.i32 d17, q8, #12
; BASIC-NEXT:    vshrn.i32 d16, q9, #12
; BASIC-NEXT:    vst1.16 {d16, d17}, [r1]
; BASIC-NEXT:    bx lr
entry:
  %0 = getelementptr inbounds %struct.int32x4_t, %struct.int32x4_t* %vT0ptr, i32 0, i32 0 ; <<4 x i32>*> [#uses=1]
  %1 = load <4 x i32>, <4 x i32>* %0, align 16               ; <<4 x i32>> [#uses=1]
  %2 = getelementptr inbounds %struct.int32x4_t, %struct.int32x4_t* %vT1ptr, i32 0, i32 0 ; <<4 x i32>*> [#uses=1]
  %3 = load <4 x i32>, <4 x i32>* %2, align 16               ; <<4 x i32>> [#uses=1]
  %4 = bitcast i16* %i_ptr to i8*                 ; <i8*> [#uses=1]
  %5 = tail call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %4, i32 1) ; <<8 x i16>> [#uses=1]
  %6 = bitcast <8 x i16> %5 to <2 x double>       ; <<2 x double>> [#uses=2]
  %7 = extractelement <2 x double> %6, i32 0      ; <double> [#uses=1]
  %8 = bitcast double %7 to <4 x i16>             ; <<4 x i16>> [#uses=1]
  %9 = sext <4 x i16> %8 to <4 x i32>             ; <<4 x i32>> [#uses=1]
  %10 = extractelement <2 x double> %6, i32 1     ; <double> [#uses=1]
  %11 = bitcast double %10 to <4 x i16>           ; <<4 x i16>> [#uses=1]
  %12 = sext <4 x i16> %11 to <4 x i32>           ; <<4 x i32>> [#uses=1]
  %13 = mul <4 x i32> %1, %9                      ; <<4 x i32>> [#uses=1]
  %14 = mul <4 x i32> %3, %12                     ; <<4 x i32>> [#uses=1]
  %15 = lshr <4 x i32> %13, <i32 12, i32 12, i32 12, i32 12>
  %trunc_15 = trunc <4 x i32> %15 to <4 x i16>
  %16 = lshr <4 x i32> %14, <i32 12, i32 12, i32 12, i32 12>
  %trunc_16 = trunc <4 x i32> %16 to <4 x i16>
  %17 = shufflevector <4 x i16> %trunc_15, <4 x i16> %trunc_16, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7> ; <<8 x i16>> [#uses=1]
  %18 = bitcast i16* %o_ptr to i8*                ; <i8*> [#uses=1]
  tail call void @llvm.arm.neon.vst1.p0i8.v8i16(i8* %18, <8 x i16> %17, i32 1)
  ret void
}

define void @t2(i16* %i_ptr, i16* %o_ptr, %struct.int16x8_t* nocapture %vT0ptr, %struct.int16x8_t* nocapture %vT1ptr) nounwind {
; DEFAULT-LABEL: t2:
; DEFAULT:       @ %bb.0: @ %entry
; DEFAULT-NEXT:    vld1.16 {d16, d17}, [r0]!
; DEFAULT-NEXT:    vld1.64 {d18, d19}, [r2:128]
; DEFAULT-NEXT:    vmul.i16 q8, q9, q8
; DEFAULT-NEXT:    vld1.64 {d18, d19}, [r3:128]
; DEFAULT-NEXT:    vld1.16 {d20, d21}, [r0]
; DEFAULT-NEXT:    vmul.i16 q9, q9, q10
; DEFAULT-NEXT:    vst1.16 {d16, d17}, [r1]!
; DEFAULT-NEXT:    vst1.16 {d18, d19}, [r1]
; DEFAULT-NEXT:    bx lr
;
; BASIC-LABEL: t2:
; BASIC:       @ %bb.0: @ %entry
; BASIC-NEXT:    vld1.16 {d18, d19}, [r0]!
; BASIC-NEXT:    vld1.64 {d16, d17}, [r2:128]
; BASIC-NEXT:    vmul.i16 q10, q8, q9
; BASIC-NEXT:    vld1.64 {d18, d19}, [r3:128]
; BASIC-NEXT:    vld1.16 {d16, d17}, [r0]
; BASIC-NEXT:    vmul.i16 q8, q9, q8
; BASIC-NEXT:    vst1.16 {d20, d21}, [r1]!
; BASIC-NEXT:    vst1.16 {d16, d17}, [r1]
; BASIC-NEXT:    bx lr
entry:
  %0 = getelementptr inbounds %struct.int16x8_t, %struct.int16x8_t* %vT0ptr, i32 0, i32 0 ; <<8 x i16>*> [#uses=1]
  %1 = load <8 x i16>, <8 x i16>* %0, align 16               ; <<8 x i16>> [#uses=1]
  %2 = getelementptr inbounds %struct.int16x8_t, %struct.int16x8_t* %vT1ptr, i32 0, i32 0 ; <<8 x i16>*> [#uses=1]
  %3 = load <8 x i16>, <8 x i16>* %2, align 16               ; <<8 x i16>> [#uses=1]
  %4 = bitcast i16* %i_ptr to i8*                 ; <i8*> [#uses=1]
  %5 = tail call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %4, i32 1) ; <<8 x i16>> [#uses=1]
  %6 = getelementptr inbounds i16, i16* %i_ptr, i32 8  ; <i16*> [#uses=1]
  %7 = bitcast i16* %6 to i8*                     ; <i8*> [#uses=1]
  %8 = tail call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %7, i32 1) ; <<8 x i16>> [#uses=1]
  %9 = mul <8 x i16> %1, %5                       ; <<8 x i16>> [#uses=1]
  %10 = mul <8 x i16> %3, %8                      ; <<8 x i16>> [#uses=1]
  %11 = bitcast i16* %o_ptr to i8*                ; <i8*> [#uses=1]
  tail call void @llvm.arm.neon.vst1.p0i8.v8i16(i8* %11, <8 x i16> %9, i32 1)
  %12 = getelementptr inbounds i16, i16* %o_ptr, i32 8 ; <i16*> [#uses=1]
  %13 = bitcast i16* %12 to i8*                   ; <i8*> [#uses=1]
  tail call void @llvm.arm.neon.vst1.p0i8.v8i16(i8* %13, <8 x i16> %10, i32 1)
  ret void
}

define <8 x i8> @t3(i8* %A, i8* %B) nounwind {
; DEFAULT-LABEL: t3:
; DEFAULT:       @ %bb.0:
; DEFAULT-NEXT:    vld3.8 {d16, d17, d18}, [r0]
; DEFAULT-NEXT:    vmul.i8 d22, d17, d16
; DEFAULT-NEXT:    vmov r0, r2, d17
; DEFAULT-NEXT:    vadd.i8 d21, d16, d18
; DEFAULT-NEXT:    vsub.i8 d20, d18, d17
; DEFAULT-NEXT:    vst3.8 {d20, d21, d22}, [r1]
; DEFAULT-NEXT:    mov r1, r2
; DEFAULT-NEXT:    bx lr
;
; BASIC-LABEL: t3:
; BASIC:       @ %bb.0:
; BASIC-NEXT:    vld3.8 {d20, d21, d22}, [r0]
; BASIC-NEXT:    mov r2, r1
; BASIC-NEXT:    vmul.i8 d18, d21, d20
; BASIC-NEXT:    vmov r0, r1, d21
; BASIC-NEXT:    vadd.i8 d17, d20, d22
; BASIC-NEXT:    vsub.i8 d16, d22, d21
; BASIC-NEXT:    vst3.8 {d16, d17, d18}, [r2]
; BASIC-NEXT:    bx lr
  %tmp1 = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8.p0i8(i8* %A, i32 1) ; <%struct.__neon_int8x8x3_t> [#uses=2]
  %tmp2 = extractvalue %struct.__neon_int8x8x3_t %tmp1, 0 ; <<8 x i8>> [#uses=1]
  %tmp3 = extractvalue %struct.__neon_int8x8x3_t %tmp1, 2 ; <<8 x i8>> [#uses=1]
  %tmp4 = extractvalue %struct.__neon_int8x8x3_t %tmp1, 1 ; <<8 x i8>> [#uses=1]
  %tmp5 = sub <8 x i8> %tmp3, %tmp4
  %tmp6 = add <8 x i8> %tmp2, %tmp3               ; <<8 x i8>> [#uses=1]
  %tmp7 = mul <8 x i8> %tmp4, %tmp2
  tail call void @llvm.arm.neon.vst3.p0i8.v8i8(i8* %B, <8 x i8> %tmp5, <8 x i8> %tmp6, <8 x i8> %tmp7, i32 1)
  ret <8 x i8> %tmp4
}

define void @t4(i32* %in, i32* %out) nounwind {
; DEFAULT-LABEL: t4:
; DEFAULT:       @ %bb.0: @ %entry
; DEFAULT-NEXT:    vld2.32 {d20, d21, d22, d23}, [r0]!
; DEFAULT-NEXT:    vld2.32 {d16, d17, d18, d19}, [r0]
; DEFAULT-NEXT:    mov r0, #0
; DEFAULT-NEXT:    cmp r0, #0
; DEFAULT-NEXT:    bne LBB3_2
; DEFAULT-NEXT:  @ %bb.1: @ %return1
; DEFAULT-NEXT:    vadd.i32 q12, q10, q8
; DEFAULT-NEXT:    vadd.i32 q13, q11, q9
; DEFAULT-NEXT:    vst2.32 {d24, d25, d26, d27}, [r1]
; DEFAULT-NEXT:    bx lr
; DEFAULT-NEXT:  LBB3_2: @ %return2
; DEFAULT-NEXT:    vadd.i32 q8, q10, q9
; DEFAULT-NEXT:    vst2.32 {d16, d17, d18, d19}, [r1]
; DEFAULT-NEXT:    trap
;
; BASIC-LABEL: t4:
; BASIC:       @ %bb.0: @ %entry
; BASIC-NEXT:    vld2.32 {d20, d21, d22, d23}, [r0]!
; BASIC-NEXT:    vld2.32 {d24, d25, d26, d27}, [r0]
; BASIC-NEXT:    mov r0, #0
; BASIC-NEXT:    cmp r0, #0
; BASIC-NEXT:    bne LBB3_2
; BASIC-NEXT:  @ %bb.1: @ %return1
; BASIC-NEXT:    vadd.i32 q8, q10, q12
; BASIC-NEXT:    vadd.i32 q9, q11, q13
; BASIC-NEXT:    vst2.32 {d16, d17, d18, d19}, [r1]
; BASIC-NEXT:    bx lr
; BASIC-NEXT:  LBB3_2: @ %return2
; BASIC-NEXT:    vadd.i32 q12, q10, q13
; BASIC-NEXT:    vst2.32 {d24, d25, d26, d27}, [r1]
; BASIC-NEXT:    trap
entry:
  %tmp1 = bitcast i32* %in to i8*                 ; <i8*> [#uses=1]
  %tmp2 = tail call %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2.v4i32.p0i8(i8* %tmp1, i32 1) ; <%struct.__neon_int32x4x2_t> [#uses=2]
  %tmp3 = getelementptr inbounds i32, i32* %in, i32 8  ; <i32*> [#uses=1]
  %tmp4 = bitcast i32* %tmp3 to i8*               ; <i8*> [#uses=1]
  %tmp5 = tail call %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2.v4i32.p0i8(i8* %tmp4, i32 1) ; <%struct.__neon_int32x4x2_t> [#uses=2]
  %tmp8 = bitcast i32* %out to i8*                ; <i8*> [#uses=1]
  br i1 undef, label %return1, label %return2

return1:
  %tmp52 = extractvalue %struct.__neon_int32x4x2_t %tmp2, 0 ; <<4 x i32>> [#uses=1]
  %tmp57 = extractvalue %struct.__neon_int32x4x2_t %tmp2, 1 ; <<4 x i32>> [#uses=1]
  %tmp = extractvalue %struct.__neon_int32x4x2_t %tmp5, 0 ; <<4 x i32>> [#uses=1]
  %tmp39 = extractvalue %struct.__neon_int32x4x2_t %tmp5, 1 ; <<4 x i32>> [#uses=1]
  %tmp6 = add <4 x i32> %tmp52, %tmp              ; <<4 x i32>> [#uses=1]
  %tmp7 = add <4 x i32> %tmp57, %tmp39            ; <<4 x i32>> [#uses=1]
  tail call void @llvm.arm.neon.vst2.p0i8.v4i32(i8* %tmp8, <4 x i32> %tmp6, <4 x i32> %tmp7, i32 1)
  ret void

return2:
  %tmp100 = extractvalue %struct.__neon_int32x4x2_t %tmp2, 0 ; <<4 x i32>> [#uses=1]
  %tmp101 = extractvalue %struct.__neon_int32x4x2_t %tmp5, 1 ; <<4 x i32>> [#uses=1]
  %tmp102 = add <4 x i32> %tmp100, %tmp101              ; <<4 x i32>> [#uses=1]
  tail call void @llvm.arm.neon.vst2.p0i8.v4i32(i8* %tmp8, <4 x i32> %tmp102, <4 x i32> %tmp101, i32 1)
  call void @llvm.trap()
  unreachable
}

define <8 x i16> @t5(i16* %A, <8 x i16>* %B) nounwind {
; CHECK-LABEL: t5:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.32 {d16, d17}, [r1]
; CHECK-NEXT:    vorr q9, q8, q8
; CHECK-NEXT:    vld2.16 {d16[1], d18[1]}, [r0]
; CHECK-NEXT:    vadd.i16 q8, q8, q9
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    bx lr
  %tmp0 = bitcast i16* %A to i8*                  ; <i8*> [#uses=1]
  %tmp1 = load <8 x i16>, <8 x i16>* %B                      ; <<8 x i16>> [#uses=2]
  %tmp2 = call %struct.__neon_int16x8x2_t @llvm.arm.neon.vld2lane.v8i16.p0i8(i8* %tmp0, <8 x i16> %tmp1, <8 x i16> %tmp1, i32 1, i32 1) ; <%struct.__neon_int16x8x2_t> [#uses=2]
  %tmp3 = extractvalue %struct.__neon_int16x8x2_t %tmp2, 0 ; <<8 x i16>> [#uses=1]
  %tmp4 = extractvalue %struct.__neon_int16x8x2_t %tmp2, 1 ; <<8 x i16>> [#uses=1]
  %tmp5 = add <8 x i16> %tmp3, %tmp4              ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %tmp5
}

define <8 x i8> @t6(i8* %A, <8 x i8>* %B) nounwind {
; CHECK-LABEL: t6:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vorr d17, d16, d16
; CHECK-NEXT:    vld2.8 {d16[1], d17[1]}, [r0]
; CHECK-NEXT:    vadd.i8 d16, d16, d17
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    bx lr
  %tmp1 = load <8 x i8>, <8 x i8>* %B                       ; <<8 x i8>> [#uses=2]
  %tmp2 = call %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8.p0i8(i8* %A, <8 x i8> %tmp1, <8 x i8> %tmp1, i32 1, i32 1) ; <%struct.__neon_int8x8x2_t> [#uses=2]
  %tmp3 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 0 ; <<8 x i8>> [#uses=1]
  %tmp4 = extractvalue %struct.__neon_int8x8x2_t %tmp2, 1 ; <<8 x i8>> [#uses=1]
  %tmp5 = add <8 x i8> %tmp3, %tmp4               ; <<8 x i8>> [#uses=1]
  ret <8 x i8> %tmp5
}

define void @t7(i32* %iptr, i32* %optr) nounwind {
; DEFAULT-LABEL: t7:
; DEFAULT:       @ %bb.0: @ %entry
; DEFAULT-NEXT:    vld2.32 {d16, d17, d18, d19}, [r0]
; DEFAULT-NEXT:    vst2.32 {d16, d17, d18, d19}, [r1]
; DEFAULT-NEXT:    vld1.32 {d16, d17}, [r0]
; DEFAULT-NEXT:    vorr q9, q8, q8
; DEFAULT-NEXT:    vuzp.32 q8, q9
; DEFAULT-NEXT:    vst1.32 {d16, d17}, [r1]
; DEFAULT-NEXT:    bx lr
;
; BASIC-LABEL: t7:
; BASIC:       @ %bb.0: @ %entry
; BASIC-NEXT:    vld2.32 {d16, d17, d18, d19}, [r0]
; BASIC-NEXT:    vst2.32 {d16, d17, d18, d19}, [r1]
; BASIC-NEXT:    vld1.32 {d18, d19}, [r0]
; BASIC-NEXT:    vorr q8, q9, q9
; BASIC-NEXT:    vuzp.32 q9, q8
; BASIC-NEXT:    vst1.32 {d18, d19}, [r1]
; BASIC-NEXT:    bx lr
entry:
  %0 = bitcast i32* %iptr to i8*                  ; <i8*> [#uses=2]
  %1 = tail call %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2.v4i32.p0i8(i8* %0, i32 1) ; <%struct.__neon_int32x4x2_t> [#uses=2]
  %tmp57 = extractvalue %struct.__neon_int32x4x2_t %1, 0 ; <<4 x i32>> [#uses=1]
  %tmp60 = extractvalue %struct.__neon_int32x4x2_t %1, 1 ; <<4 x i32>> [#uses=1]
  %2 = bitcast i32* %optr to i8*                  ; <i8*> [#uses=2]
  tail call void @llvm.arm.neon.vst2.p0i8.v4i32(i8* %2, <4 x i32> %tmp57, <4 x i32> %tmp60, i32 1)
  %3 = tail call <4 x i32> @llvm.arm.neon.vld1.v4i32.p0i8(i8* %0, i32 1) ; <<4 x i32>> [#uses=1]
  %4 = shufflevector <4 x i32> %3, <4 x i32> undef, <4 x i32> <i32 0, i32 2, i32 0, i32 2> ; <<4 x i32>> [#uses=1]
  tail call void @llvm.arm.neon.vst1.p0i8.v4i32(i8* %2, <4 x i32> %4, i32 1)
  ret void
}

; PR7156
define arm_aapcs_vfpcc i32 @t8() nounwind {
; CHECK-LABEL: t8:
; CHECK:       @ %bb.0: @ %bb.nph55.bb.nph55.split_crit_edge
; CHECK-NEXT:    vrsqrte.f32 q8, q8
; CHECK-NEXT:    vmul.f32 q8, q8, q8
; CHECK-NEXT:    vmul.f32 q8, q8, q8
; CHECK-NEXT:    vst1.32 {d16[1]}, [r0:32]
; CHECK-NEXT:    mov r0, #0
; CHECK-NEXT:    cmp r0, #0
; CHECK-NEXT:    movne r0, #0
; CHECK-NEXT:    bxne lr
; CHECK-NEXT:  LBB7_1: @ %bb7
bb.nph55.bb.nph55.split_crit_edge:
  br label %bb3

bb3:                                              ; preds = %bb3, %bb.nph55.bb.nph55.split_crit_edge
  br i1 undef, label %bb5, label %bb3

bb5:                                              ; preds = %bb3
  br label %bb.i25

bb.i25:                                           ; preds = %bb.i25, %bb5
  %0 = shufflevector <2 x float> undef, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3> ; <<4 x float>> [#uses=1]
  %1 = call <4 x float> @llvm.arm.neon.vrsqrte.v4f32(<4 x float> %0) nounwind ; <<4 x float>> [#uses=1]
  %2 = fmul <4 x float> %1, %1                 ; <<4 x float>> [#uses=1]
  %3 = fmul <4 x float> %2, %2                 ; <<4 x float>> [#uses=1]
  %tmp26.i = bitcast <4 x float> %3 to <2 x double> ; <<2 x double>> [#uses=1]
  %4 = extractelement <2 x double> %tmp26.i, i32 0 ; <double> [#uses=1]
  %5 = bitcast double %4 to <2 x float>           ; <<2 x float>> [#uses=1]
  %6 = extractelement <2 x float> %5, i32 1       ; <float> [#uses=1]
  store float %6, float* undef, align 4
  br i1 undef, label %bb6, label %bb.i25

bb6:                                              ; preds = %bb.i25
  br i1 undef, label %bb7, label %bb14

bb7:                                              ; preds = %bb6
  br label %bb.i49

bb.i49:                                           ; preds = %bb.i49, %bb7
  br i1 undef, label %bb.i19, label %bb.i49

bb.i19:                                           ; preds = %bb.i19, %bb.i49
  br i1 undef, label %exit, label %bb.i19

exit:          ; preds = %bb.i19
  unreachable

bb14:                                             ; preds = %bb6
  ret i32 0
}

%0 = type { %1, %1, %1, %1 }
%1 = type { %2 }
%2 = type { <4 x float> }
%3 = type { %0, %1 }

; PR7157
define arm_aapcs_vfpcc float @t9(%0* nocapture, %3* nocapture) nounwind {
; CHECK-LABEL: t9:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vmov.i32 d16, #0x0
; CHECK-NEXT:    vst1.64 {d16, d17}, [r0:128]
; CHECK-NEXT:    vorr d17, d16, d16
; CHECK-NEXT:    vst1.64 {d16, d17}, [r0:128]
; CHECK-NEXT:  LBB8_1: @ =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    b LBB8_1
  %3 = bitcast double 0.000000e+00 to <2 x float> ; <<2 x float>> [#uses=2]
  %4 = shufflevector <2 x float> %3, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3> ; <<4 x float>> [#uses=1]
  store <4 x float> %4, <4 x float>* undef, align 16
  %5 = shufflevector <2 x float> %3, <2 x float> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3> ; <<4 x float>> [#uses=1]
  store <4 x float> %5, <4 x float>* undef, align 16
  br label %8

; <label>:6                                       ; preds = %8
  br label %7

; <label>:7                                       ; preds = %6
  br label %8

; <label>:8                                       ; preds = %7, %2
  br label %6

; <label>:9                                       ; preds = %8
  ret float undef

; <label>:10                                      ; preds = %6
  ret float 9.990000e+02
}

; PR7162
define arm_aapcs_vfpcc i32 @t10(float %x) nounwind {
; DEFAULT-LABEL: t10:
; DEFAULT:       @ %bb.0: @ %entry
; DEFAULT-NEXT:    vmov.i32 d2, #0x0
; DEFAULT-NEXT:    @ kill: def $s0 killed $s0 def $d0
; DEFAULT-NEXT:    vdup.32 q0, d0[0]
; DEFAULT-NEXT:    vmov.i32 q9, #0x3f000000
; DEFAULT-NEXT:    vmov.f32 s0, s4
; DEFAULT-NEXT:    vmul.f32 q8, q0, q0
; DEFAULT-NEXT:    vadd.f32 q8, q8, q8
; DEFAULT-NEXT:    vadd.f32 q0, q8, q8
; DEFAULT-NEXT:    vmul.f32 q8, q9, d1[0]
; DEFAULT-NEXT:    vmul.f32 q8, q8, q8
; DEFAULT-NEXT:    vadd.f32 q8, q8, q8
; DEFAULT-NEXT:    vmul.f32 q8, q8, q8
; DEFAULT-NEXT:    vst1.32 {d17[1]}, [r0:32]
; DEFAULT-NEXT:    mov r0, #0
; DEFAULT-NEXT:    cmp r0, #0
; DEFAULT-NEXT:    movne r0, #0
; DEFAULT-NEXT:    bxne lr
; DEFAULT-NEXT:  LBB9_1: @ %exit
; DEFAULT-NEXT:    trap
;
; BASIC-LABEL: t10:
; BASIC:       @ %bb.0: @ %entry
; BASIC-NEXT:    @ kill: def $s0 killed $s0 def $d0
; BASIC-NEXT:    vdup.32 q1, d0[0]
; BASIC-NEXT:    vmov.i32 q9, #0x3f000000
; BASIC-NEXT:    vmov.i32 d0, #0x0
; BASIC-NEXT:    vmov.f32 s4, s0
; BASIC-NEXT:    vmul.f32 q8, q1, q1
; BASIC-NEXT:    vadd.f32 q8, q8, q8
; BASIC-NEXT:    vadd.f32 q0, q8, q8
; BASIC-NEXT:    vmul.f32 q8, q9, d1[0]
; BASIC-NEXT:    vmul.f32 q8, q8, q8
; BASIC-NEXT:    vadd.f32 q8, q8, q8
; BASIC-NEXT:    vmul.f32 q8, q8, q8
; BASIC-NEXT:    vst1.32 {d17[1]}, [r0:32]
; BASIC-NEXT:    mov r0, #0
; BASIC-NEXT:    cmp r0, #0
; BASIC-NEXT:    movne r0, #0
; BASIC-NEXT:    bxne lr
; BASIC-NEXT:  LBB9_1: @ %exit
; BASIC-NEXT:    trap
entry:
  %0 = shufflevector <4 x float> zeroinitializer, <4 x float> undef, <4 x i32> zeroinitializer ; <<4 x float>> [#uses=1]
  %1 = insertelement <4 x float> %0, float %x, i32 1 ; <<4 x float>> [#uses=1]
  %2 = insertelement <4 x float> %1, float %x, i32 2 ; <<4 x float>> [#uses=1]
  %3 = insertelement <4 x float> %2, float %x, i32 3 ; <<4 x float>> [#uses=1]
  %tmp54.i = bitcast <4 x float> %3 to <2 x double> ; <<2 x double>> [#uses=1]
  %4 = extractelement <2 x double> %tmp54.i, i32 1 ; <double> [#uses=1]
  %5 = bitcast double %4 to <2 x float>           ; <<2 x float>> [#uses=1]
  %6 = shufflevector <2 x float> %5, <2 x float> undef, <4 x i32> zeroinitializer ; <<4 x float>> [#uses=1]
  %7 = fmul <4 x float> %6, %6
  %8 = fadd <4 x float> %7, %7
  %9 = fadd <4 x float> %8, %8
  %10 = shufflevector <4 x float> undef, <4 x float> %9, <4 x i32> <i32 0, i32 1, i32 2, i32 7> ; <<4 x float>> [#uses=1]
  %11 = fmul <4 x float> %10, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01> ; <<4 x float>> [#uses=1]
  %12 = shufflevector <4 x float> %11, <4 x float> undef, <4 x i32> <i32 3, i32 undef, i32 undef, i32 undef> ; <<4 x float>> [#uses=1]
  %13 = shufflevector <4 x float> %12, <4 x float> undef, <4 x i32> zeroinitializer ; <<4 x float>> [#uses=1]
  %14 = fmul <4 x float> %13, %13
  %15 = fadd <4 x float> %14, %14
  %16 = shufflevector <4 x float> undef, <4 x float> %15, <4 x i32> <i32 0, i32 1, i32 6, i32 3> ; <<4 x float>> [#uses=1]
  %17 = fmul <4 x float> %16, %16
  %18 = extractelement <4 x float> %17, i32 2     ; <float> [#uses=1]
  store float %18, float* undef, align 4
  br i1 undef, label %exit, label %bb14

exit:          ; preds = %bb.i19
  unreachable

bb14:                                             ; preds = %bb6
  ret i32 0
}

; This test crashes the coalescer because live variables were not updated properly.
define <8 x i8> @t11(i8* %A1, i8* %A2, i8* %A3, i8* %A4, i8* %A5, i8* %A6, i8* %A7, i8* %A8, i8* %B) nounwind {
; CHECK-LABEL: t11:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vmov r0, r2, d16
; CHECK-NEXT:    ldr r3, [sp, #16]
; CHECK-NEXT:    vmov.i32 d18, #0x0
; CHECK-NEXT:    vst3.8 {d16, d17, d18}, [r1]
; CHECK-NEXT:    vst3.8 {d16, d17, d18}, [r3]
; CHECK-NEXT:    mov r1, r2
; CHECK-NEXT:    bx lr
  %tmp1d = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8.p0i8(i8* %A4, i32 1) ; <%struct.__neon_int8x8x3_t> [#uses=1]
  %tmp2d = extractvalue %struct.__neon_int8x8x3_t %tmp1d, 0 ; <<8 x i8>> [#uses=1]
  %tmp1f = call %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8.p0i8(i8* %A6, i32 1) ; <%struct.__neon_int8x8x3_t> [#uses=1]
  %tmp2f = extractvalue %struct.__neon_int8x8x3_t %tmp1f, 0 ; <<8 x i8>> [#uses=1]
  %tmp2bd = add <8 x i8> zeroinitializer, %tmp2d  ; <<8 x i8>> [#uses=1]
  %tmp2abcd = mul <8 x i8> zeroinitializer, %tmp2bd ; <<8 x i8>> [#uses=1]
  %tmp2ef = sub <8 x i8> zeroinitializer, %tmp2f  ; <<8 x i8>> [#uses=1]
  %tmp2efgh = mul <8 x i8> %tmp2ef, undef         ; <<8 x i8>> [#uses=2]
  call void @llvm.arm.neon.vst3.p0i8.v8i8(i8* %A2, <8 x i8> undef, <8 x i8> undef, <8 x i8> %tmp2efgh, i32 1)
  %tmp2 = sub <8 x i8> %tmp2efgh, %tmp2abcd       ; <<8 x i8>> [#uses=1]
  %tmp7 = mul <8 x i8> undef, %tmp2               ; <<8 x i8>> [#uses=1]
  tail call void @llvm.arm.neon.vst3.p0i8.v8i8(i8* %B, <8 x i8> undef, <8 x i8> undef, <8 x i8> %tmp7, i32 1)
  ret <8 x i8> undef
}

declare <4 x i32> @llvm.arm.neon.vld1.v4i32.p0i8(i8*, i32) nounwind readonly

declare <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8*, i32) nounwind readonly

declare <4 x i16> @llvm.arm.neon.vshiftn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone

declare void @llvm.arm.neon.vst1.p0i8.v4i32(i8*, <4 x i32>, i32) nounwind

declare void @llvm.arm.neon.vst1.p0i8.v8i16(i8*, <8 x i16>, i32) nounwind

declare void @llvm.arm.neon.vst3.p0i8.v8i8(i8*, <8 x i8>, <8 x i8>, <8 x i8>, i32)
nounwind

declare %struct.__neon_int8x8x3_t @llvm.arm.neon.vld3.v8i8.p0i8(i8*, i32) nounwind readonly

declare %struct.__neon_int32x4x2_t @llvm.arm.neon.vld2.v4i32.p0i8(i8*, i32) nounwind readonly

declare %struct.__neon_int8x8x2_t @llvm.arm.neon.vld2lane.v8i8.p0i8(i8*, <8 x i8>, <8 x i8>, i32, i32) nounwind readonly

declare %struct.__neon_int16x8x2_t @llvm.arm.neon.vld2lane.v8i16.p0i8(i8*, <8 x i16>, <8 x i16>, i32, i32) nounwind readonly

declare void @llvm.arm.neon.vst2.p0i8.v4i32(i8*, <4 x i32>, <4 x i32>, i32) nounwind

declare <4 x float> @llvm.arm.neon.vrsqrte.v4f32(<4 x float>) nounwind readnone

declare void @llvm.trap() nounwind
