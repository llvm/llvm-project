; RUN: opt < %s -mattr=+neon -interleaved-access -S | FileCheck %s -check-prefixes=NEON,ALL
; RUN: opt < %s -interleaved-access -S | FileCheck %s -check-prefixes=NO_NEON,ALL

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"
target triple = "arm---eabi"

define void @load_factor2(<16 x i8>* %ptr) {
; NEON-LABEL:    @load_factor2(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <16 x i8>* %ptr to i8*
; NEON-NEXT:       [[VLDN:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.arm.neon.vld2.v8i8.p0i8(i8* [[TMP1]], i32 4)
; NEON-NEXT:       [[TMP2:%.*]] = extractvalue { <8 x i8>, <8 x i8> } [[VLDN]], 1
; NEON-NEXT:       [[TMP3:%.*]] = extractvalue { <8 x i8>, <8 x i8> } [[VLDN]], 0
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @load_factor2(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = load <16 x i8>, <16 x i8>* %ptr, align 4
  %v0 = shufflevector <16 x i8> %interleaved.vec, <16 x i8> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %v1 = shufflevector <16 x i8> %interleaved.vec, <16 x i8> undef, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  ret void
}

define void @load_factor3(<6 x i32>* %ptr) {
; NEON-LABEL:    @load_factor3(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <6 x i32>* %ptr to i8*
; NEON-NEXT:       [[VLDN:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld3.v2i32.p0i8(i8* [[TMP1]], i32 4)
; NEON-NEXT:       [[TMP2:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } [[VLDN]], 2
; NEON-NEXT:       [[TMP3:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } [[VLDN]], 1
; NEON-NEXT:       [[TMP4:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } [[VLDN]], 0
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @load_factor3(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = load <6 x i32>, <6 x i32>* %ptr, align 4
  %v0 = shufflevector <6 x i32> %interleaved.vec, <6 x i32> undef, <2 x i32> <i32 0, i32 3>
  %v1 = shufflevector <6 x i32> %interleaved.vec, <6 x i32> undef, <2 x i32> <i32 1, i32 4>
  %v2 = shufflevector <6 x i32> %interleaved.vec, <6 x i32> undef, <2 x i32> <i32 2, i32 5>
  ret void
}

define void @load_factor4(<16 x i32>* %ptr) {
; NEON-LABEL:    @load_factor4(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <16 x i32>* %ptr to i8*
; NEON-NEXT:       [[VLDN:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4.v4i32.p0i8(i8* [[TMP1]], i32 4)
; NEON-NEXT:       [[TMP2:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 3
; NEON-NEXT:       [[TMP3:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 2
; NEON-NEXT:       [[TMP4:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 1
; NEON-NEXT:       [[TMP5:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 0
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @load_factor4(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = load <16 x i32>, <16 x i32>* %ptr, align 4
  %v0 = shufflevector <16 x i32> %interleaved.vec, <16 x i32> undef, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %v1 = shufflevector <16 x i32> %interleaved.vec, <16 x i32> undef, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %v2 = shufflevector <16 x i32> %interleaved.vec, <16 x i32> undef, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %v3 = shufflevector <16 x i32> %interleaved.vec, <16 x i32> undef, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  ret void
}

define void @store_factor2(<16 x i8>* %ptr, <8 x i8> %v0, <8 x i8> %v1) {
; NEON-LABEL:    @store_factor2(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <16 x i8>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <8 x i8> %v0, <8 x i8> %v1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <8 x i8> %v0, <8 x i8> %v1, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
; NEON-NEXT:       call void @llvm.arm.neon.vst2.p0i8.v8i8(i8* [[TMP1]], <8 x i8> [[TMP2]], <8 x i8> [[TMP3]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_factor2(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = shufflevector <8 x i8> %v0, <8 x i8> %v1, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  store <16 x i8> %interleaved.vec, <16 x i8>* %ptr, align 4
  ret void
}

define void @store_factor3(<12 x i32>* %ptr, <4 x i32> %v0, <4 x i32> %v1, <4 x i32> %v2) {
; NEON-LABEL:    @store_factor3(
; NEON:            [[TMP1:%.*]] = bitcast <12 x i32>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <8 x i32> %s0, <8 x i32> %s1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <8 x i32> %s0, <8 x i32> %s1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <8 x i32> %s0, <8 x i32> %s1, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
; NEON-NEXT:       call void @llvm.arm.neon.vst3.p0i8.v4i32(i8* [[TMP1]], <4 x i32> [[TMP2]], <4 x i32> [[TMP3]], <4 x i32> [[TMP4]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_factor3(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %s0 = shufflevector <4 x i32> %v0, <4 x i32> %v1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %s1 = shufflevector <4 x i32> %v2, <4 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %interleaved.vec = shufflevector <8 x i32> %s0, <8 x i32> %s1, <12 x i32> <i32 0, i32 4, i32 8, i32 1, i32 5, i32 9, i32 2, i32 6, i32 10, i32 3, i32 7, i32 11>
  store <12 x i32> %interleaved.vec, <12 x i32>* %ptr, align 4
  ret void
}

define void @store_factor4(<16 x i32>* %ptr, <4 x i32> %v0, <4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3) {
; NEON-LABEL:    @store_factor4(
; NEON:            [[TMP1:%.*]] = bitcast <16 x i32>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <8 x i32> %s0, <8 x i32> %s1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <8 x i32> %s0, <8 x i32> %s1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <8 x i32> %s0, <8 x i32> %s1, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
; NEON-NEXT:       [[TMP5:%.*]] = shufflevector <8 x i32> %s0, <8 x i32> %s1, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
; NEON-NEXT:       call void @llvm.arm.neon.vst4.p0i8.v4i32(i8* [[TMP1]], <4 x i32> [[TMP2]], <4 x i32> [[TMP3]], <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_factor4(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %s0 = shufflevector <4 x i32> %v0, <4 x i32> %v1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %s1 = shufflevector <4 x i32> %v2, <4 x i32> %v3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %interleaved.vec = shufflevector <8 x i32> %s0, <8 x i32> %s1, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  store <16 x i32> %interleaved.vec, <16 x i32>* %ptr, align 4
  ret void
}

define void @load_ptrvec_factor2(<4 x i32*>* %ptr) {
; NEON-LABEL:    @load_ptrvec_factor2(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <4 x i32*>* %ptr to i8*
; NEON-NEXT:       [[VLDN:%.*]] = call { <2 x i32>, <2 x i32> } @llvm.arm.neon.vld2.v2i32.p0i8(i8* [[TMP1]], i32 4)
; NEON-NEXT:       [[TMP2:%.*]] = extractvalue { <2 x i32>, <2 x i32> } [[VLDN]], 0
; NEON-NEXT:       [[TMP3:%.*]] = inttoptr <2 x i32> [[TMP2]] to <2 x i32*>
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @load_ptrvec_factor2(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = load <4 x i32*>, <4 x i32*>* %ptr, align 4
  %v0 = shufflevector <4 x i32*> %interleaved.vec, <4 x i32*> undef, <2 x i32> <i32 0, i32 2>
  ret void
}

define void @load_ptrvec_factor3(<6 x i32*>* %ptr) {
; NEON-LABEL:    @load_ptrvec_factor3(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <6 x i32*>* %ptr to i8*
; NEON-NEXT:       [[VLDN:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld3.v2i32.p0i8(i8* [[TMP1]], i32 4)
; NEON-NEXT:       [[TMP2:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } [[VLDN]], 2
; NEON-NEXT:       [[TMP3:%.*]] = inttoptr <2 x i32> [[TMP2]] to <2 x i32*>
; NEON-NEXT:       [[TMP4:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } [[VLDN]], 1
; NEON-NEXT:       [[TMP5:%.*]] = inttoptr <2 x i32> [[TMP4]] to <2 x i32*>
; NEON-NEXT:       [[TMP6:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } [[VLDN]], 0
; NEON-NEXT:       [[TMP7:%.*]] = inttoptr <2 x i32> [[TMP6]] to <2 x i32*>
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @load_ptrvec_factor3(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = load <6 x i32*>, <6 x i32*>* %ptr, align 4
  %v0 = shufflevector <6 x i32*> %interleaved.vec, <6 x i32*> undef, <2 x i32> <i32 0, i32 3>
  %v1 = shufflevector <6 x i32*> %interleaved.vec, <6 x i32*> undef, <2 x i32> <i32 1, i32 4>
  %v2 = shufflevector <6 x i32*> %interleaved.vec, <6 x i32*> undef, <2 x i32> <i32 2, i32 5>
  ret void
}

define void @load_ptrvec_factor4(<8 x i32*>* %ptr) {
; NEON-LABEL:    @load_ptrvec_factor4(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <8 x i32*>* %ptr to i8*
; NEON-NEXT:       [[VLDN:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld4.v2i32.p0i8(i8* [[TMP1]], i32 4)
; NEON-NEXT:       [[TMP2:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[VLDN]], 3
; NEON-NEXT:       [[TMP3:%.*]] = inttoptr <2 x i32> [[TMP2]] to <2 x i32*>
; NEON-NEXT:       [[TMP4:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[VLDN]], 2
; NEON-NEXT:       [[TMP5:%.*]] = inttoptr <2 x i32> [[TMP4]] to <2 x i32*>
; NEON-NEXT:       [[TMP6:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[VLDN]], 1
; NEON-NEXT:       [[TMP7:%.*]] = inttoptr <2 x i32> [[TMP6]] to <2 x i32*>
; NEON-NEXT:       [[TMP8:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[VLDN]], 0
; NEON-NEXT:       [[TMP9:%.*]] = inttoptr <2 x i32> [[TMP8]] to <2 x i32*>
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @load_ptrvec_factor4(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = load <8 x i32*>, <8 x i32*>* %ptr, align 4
  %v0 = shufflevector <8 x i32*> %interleaved.vec, <8 x i32*> undef, <2 x i32> <i32 0, i32 4>
  %v1 = shufflevector <8 x i32*> %interleaved.vec, <8 x i32*> undef, <2 x i32> <i32 1, i32 5>
  %v2 = shufflevector <8 x i32*> %interleaved.vec, <8 x i32*> undef, <2 x i32> <i32 2, i32 6>
  %v3 = shufflevector <8 x i32*> %interleaved.vec, <8 x i32*> undef, <2 x i32> <i32 3, i32 7>
  ret void
}

define void @store_ptrvec_factor2(<4 x i32*>* %ptr, <2 x i32*> %v0, <2 x i32*> %v1) {
; NEON-LABEL:    @store_ptrvec_factor2(
; NEON-NEXT:       [[TMP1:%.*]] = ptrtoint <2 x i32*> %v0 to <2 x i32>
; NEON-NEXT:       [[TMP2:%.*]] = ptrtoint <2 x i32*> %v1 to <2 x i32>
; NEON-NEXT:       [[TMP3:%.*]] = bitcast <4 x i32*>* %ptr to i8*
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <2 x i32> [[TMP1]], <2 x i32> [[TMP2]], <2 x i32> <i32 0, i32 1>
; NEON-NEXT:       [[TMP5:%.*]] = shufflevector <2 x i32> [[TMP1]], <2 x i32> [[TMP2]], <2 x i32> <i32 2, i32 3>
; NEON-NEXT:       call void @llvm.arm.neon.vst2.p0i8.v2i32(i8* [[TMP3]], <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_ptrvec_factor2(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = shufflevector <2 x i32*> %v0, <2 x i32*> %v1, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x i32*> %interleaved.vec, <4 x i32*>* %ptr, align 4
  ret void
}

define void @store_ptrvec_factor3(<6 x i32*>* %ptr, <2 x i32*> %v0, <2 x i32*> %v1, <2 x i32*> %v2) {
; NEON-LABEL:    @store_ptrvec_factor3(
; NEON:            [[TMP1:%.*]] = ptrtoint <4 x i32*> %s0 to <4 x i32>
; NEON-NEXT:       [[TMP2:%.*]] = ptrtoint <4 x i32*> %s1 to <4 x i32>
; NEON-NEXT:       [[TMP3:%.*]] = bitcast <6 x i32*>* %ptr to i8*
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> [[TMP2]], <2 x i32> <i32 0, i32 1>
; NEON-NEXT:       [[TMP5:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> [[TMP2]], <2 x i32> <i32 2, i32 3>
; NEON-NEXT:       [[TMP6:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> [[TMP2]], <2 x i32> <i32 4, i32 5>
; NEON-NEXT:       call void @llvm.arm.neon.vst3.p0i8.v2i32(i8* [[TMP3]], <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], <2 x i32> [[TMP6]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_ptrvec_factor3(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %s0 = shufflevector <2 x i32*> %v0, <2 x i32*> %v1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %s1 = shufflevector <2 x i32*> %v2, <2 x i32*> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %interleaved.vec = shufflevector <4 x i32*> %s0, <4 x i32*> %s1, <6 x i32> <i32 0, i32 2, i32 4, i32 1, i32 3, i32 5>
  store <6 x i32*> %interleaved.vec, <6 x i32*>* %ptr, align 4
  ret void
}

define void @store_ptrvec_factor4(<8 x i32*>* %ptr, <2 x i32*> %v0, <2 x i32*> %v1, <2 x i32*> %v2, <2 x i32*> %v3) {
; NEON-LABEL:    @store_ptrvec_factor4(
; NEON:            [[TMP1:%.*]] = ptrtoint <4 x i32*> %s0 to <4 x i32>
; NEON-NEXT:       [[TMP2:%.*]] = ptrtoint <4 x i32*> %s1 to <4 x i32>
; NEON-NEXT:       [[TMP3:%.*]] = bitcast <8 x i32*>* %ptr to i8*
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> [[TMP2]], <2 x i32> <i32 0, i32 1>
; NEON-NEXT:       [[TMP5:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> [[TMP2]], <2 x i32> <i32 2, i32 3>
; NEON-NEXT:       [[TMP6:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> [[TMP2]], <2 x i32> <i32 4, i32 5>
; NEON-NEXT:       [[TMP7:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> [[TMP2]], <2 x i32> <i32 6, i32 7>
; NEON-NEXT:       call void @llvm.arm.neon.vst4.p0i8.v2i32(i8* [[TMP3]], <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], <2 x i32> [[TMP6]], <2 x i32> [[TMP7]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_ptrvec_factor4(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %s0 = shufflevector <2 x i32*> %v0, <2 x i32*> %v1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %s1 = shufflevector <2 x i32*> %v2, <2 x i32*> %v3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %interleaved.vec = shufflevector <4 x i32*> %s0, <4 x i32*> %s1, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 1, i32 3, i32 5, i32 7>
  store <8 x i32*> %interleaved.vec, <8 x i32*>* %ptr, align 4
  ret void
}

define void @load_undef_mask_factor2(<8 x i32>* %ptr) {
; NEON-LABEL:    @load_undef_mask_factor2(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <8 x i32>* %ptr to i8*
; NEON-NEXT:       [[VLDN:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2.v4i32.p0i8(i8* [[TMP1]], i32 4)
; NEON-NEXT:       [[TMP2:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN]], 1
; NEON-NEXT:       [[TMP3:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN]], 0
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @load_undef_mask_factor2(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = load <8 x i32>, <8 x i32>* %ptr, align 4
  %v0 = shufflevector <8 x i32> %interleaved.vec, <8 x i32> undef, <4 x i32> <i32 undef, i32 2, i32 undef, i32 6>
  %v1 = shufflevector <8 x i32> %interleaved.vec, <8 x i32> undef, <4 x i32> <i32 undef, i32 3, i32 undef, i32 7>
  ret void
}

define void @load_undef_mask_factor3(<12 x i32>* %ptr) {
; NEON-LABEL:    @load_undef_mask_factor3(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <12 x i32>* %ptr to i8*
; NEON-NEXT:       [[VLDN:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3.v4i32.p0i8(i8* [[TMP1]], i32 4)
; NEON-NEXT:       [[TMP2:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 2
; NEON-NEXT:       [[TMP3:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 1
; NEON-NEXT:       [[TMP4:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 0
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @load_undef_mask_factor3(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = load <12 x i32>, <12 x i32>* %ptr, align 4
  %v0 = shufflevector <12 x i32> %interleaved.vec, <12 x i32> undef, <4 x i32> <i32 0, i32 3, i32 6, i32 9>
  %v1 = shufflevector <12 x i32> %interleaved.vec, <12 x i32> undef, <4 x i32> <i32 1, i32 4, i32 7, i32 10>
  %v2 = shufflevector <12 x i32> %interleaved.vec, <12 x i32> undef, <4 x i32> <i32 2, i32 undef, i32 undef, i32 undef>
  ret void
}

define void @load_undef_mask_factor4(<16 x i32>* %ptr) {
; NEON-LABEL:    @load_undef_mask_factor4(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <16 x i32>* %ptr to i8*
; NEON-NEXT:       [[VLDN:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4.v4i32.p0i8(i8* [[TMP1]], i32 4)
; NEON-NEXT:       [[TMP2:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 3
; NEON-NEXT:       [[TMP3:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 2
; NEON-NEXT:       [[TMP4:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 1
; NEON-NEXT:       [[TMP5:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 0
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @load_undef_mask_factor4(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = load <16 x i32>, <16 x i32>* %ptr, align 4
  %v0 = shufflevector <16 x i32> %interleaved.vec, <16 x i32> undef, <4 x i32> <i32 0, i32 4, i32 undef, i32 undef>
  %v1 = shufflevector <16 x i32> %interleaved.vec, <16 x i32> undef, <4 x i32> <i32 1, i32 5, i32 undef, i32 undef>
  %v2 = shufflevector <16 x i32> %interleaved.vec, <16 x i32> undef, <4 x i32> <i32 2, i32 6, i32 undef, i32 undef>
  %v3 = shufflevector <16 x i32> %interleaved.vec, <16 x i32> undef, <4 x i32> <i32 3, i32 7, i32 undef, i32 undef>
  ret void
}

define void @store_undef_mask_factor2(<8 x i32>* %ptr, <4 x i32> %v0, <4 x i32> %v1) {
; NEON-LABEL:    @store_undef_mask_factor2(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <8 x i32>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <4 x i32> %v0, <4 x i32> %v1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <4 x i32> %v0, <4 x i32> %v1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       call void @llvm.arm.neon.vst2.p0i8.v4i32(i8* [[TMP1]], <4 x i32> [[TMP2]], <4 x i32> [[TMP3]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_undef_mask_factor2(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = shufflevector <4 x i32> %v0, <4 x i32> %v1, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 2, i32 6, i32 3, i32 7>
  store <8 x i32> %interleaved.vec, <8 x i32>* %ptr, align 4
  ret void
}

define void @store_undef_mask_factor3(<12 x i32>* %ptr, <4 x i32> %v0, <4 x i32> %v1, <4 x i32> %v2) {
; NEON-LABEL:    @store_undef_mask_factor3(
; NEON:            [[TMP1:%.*]] = bitcast <12 x i32>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <8 x i32> %s0, <8 x i32> %s1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <8 x i32> %s0, <8 x i32> %s1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <8 x i32> %s0, <8 x i32> %s1, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
; NEON-NEXT:       call void @llvm.arm.neon.vst3.p0i8.v4i32(i8* [[TMP1]], <4 x i32> [[TMP2]], <4 x i32> [[TMP3]], <4 x i32> [[TMP4]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_undef_mask_factor3(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %s0 = shufflevector <4 x i32> %v0, <4 x i32> %v1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %s1 = shufflevector <4 x i32> %v2, <4 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %interleaved.vec = shufflevector <8 x i32> %s0, <8 x i32> %s1, <12 x i32> <i32 0, i32 4, i32 undef, i32 1, i32 undef, i32 9, i32 2, i32 6, i32 10, i32 3, i32 7, i32 11>
  store <12 x i32> %interleaved.vec, <12 x i32>* %ptr, align 4
  ret void
}

define void @store_undef_mask_factor4(<16 x i32>* %ptr, <4 x i32> %v0, <4 x i32> %v1, <4 x i32> %v2, <4 x i32> %v3) {
; NEON-LABEL:    @store_undef_mask_factor4(
; NEON:            [[TMP1:%.*]] = bitcast <16 x i32>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <8 x i32> %s0, <8 x i32> %s1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <8 x i32> %s0, <8 x i32> %s1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <8 x i32> %s0, <8 x i32> %s1, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
; NEON-NEXT:       [[TMP5:%.*]] = shufflevector <8 x i32> %s0, <8 x i32> %s1, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
; NEON-NEXT:       call void @llvm.arm.neon.vst4.p0i8.v4i32(i8* [[TMP1]], <4 x i32> [[TMP2]], <4 x i32> [[TMP3]], <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_undef_mask_factor4(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %s0 = shufflevector <4 x i32> %v0, <4 x i32> %v1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %s1 = shufflevector <4 x i32> %v2, <4 x i32> %v3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %interleaved.vec = shufflevector <8 x i32> %s0, <8 x i32> %s1, <16 x i32> <i32 0, i32 4, i32 8, i32 undef, i32 undef, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  store <16 x i32> %interleaved.vec, <16 x i32>* %ptr, align 4
  ret void
}

define void @load_address_space(<4 x i32> addrspace(1)* %ptr) {
; NEON-LABEL:    @load_address_space(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <4 x i32> addrspace(1)* %ptr to i8 addrspace(1)*
; NEON-NEXT:       [[VLDN:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld3.v2i32.p1i8(i8 addrspace(1)* [[TMP1]], i32 0)
; NEON-NEXT:       [[TMP2:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } [[VLDN]], 2
; NEON-NEXT:       [[TMP3:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } [[VLDN]], 1
; NEON-NEXT:       [[TMP4:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } [[VLDN]], 0
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @load_address_space(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = load <4 x i32>, <4 x i32> addrspace(1)* %ptr
  %v0 = shufflevector <4 x i32> %interleaved.vec, <4 x i32> undef, <2 x i32> <i32 0, i32 3>
  %v1 = shufflevector <4 x i32> %interleaved.vec, <4 x i32> undef, <2 x i32> <i32 1, i32 4>
  %v2 = shufflevector <4 x i32> %interleaved.vec, <4 x i32> undef, <2 x i32> <i32 2, i32 5>
  ret void
}

define void @store_address_space(<4 x i32> addrspace(1)* %ptr, <2 x i32> %v0, <2 x i32> %v1) {
; NEON-LABEL:    @store_address_space(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <4 x i32> addrspace(1)* %ptr to i8 addrspace(1)*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <2 x i32> %v0, <2 x i32> %v1, <2 x i32> <i32 0, i32 1>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <2 x i32> %v0, <2 x i32> %v1, <2 x i32> <i32 2, i32 3>
; NEON-NEXT:       call void @llvm.arm.neon.vst2.p1i8.v2i32(i8 addrspace(1)* [[TMP1]], <2 x i32> [[TMP2]], <2 x i32> [[TMP3]], i32 0)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_address_space(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = shufflevector <2 x i32> %v0, <2 x i32> %v1, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x i32> %interleaved.vec, <4 x i32> addrspace(1)* %ptr
  ret void
}

define void @load_f16_factor2(<8 x half>* %ptr) {
; ALL-LABEL: @load_f16_factor2(
; ALL-NOT:     @llvm.arm.neon
; ALL:         ret void
;
  %interleaved.vec = load <8 x half>, <8 x half>* %ptr, align 4
  %v0 = shufflevector <8 x half> %interleaved.vec, <8 x half> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %v1 = shufflevector <8 x half> %interleaved.vec, <8 x half> undef, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  ret void
}

define void @store_f16_factor2(<8 x half>* %ptr, <4 x half> %v0, <4 x half> %v1) {
; ALL-LABEL: @store_f16_factor2(
; ALL-NOT:     @llvm.arm.neon
; ALL:         ret void
;
  %interleaved.vec = shufflevector <4 x half> %v0, <4 x half> %v1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x half> %interleaved.vec, <8 x half>* %ptr, align 4
  ret void
}

define void @load_illegal_factor2(<3 x float>* %ptr) nounwind {
; ALL-LABEL:    @load_illegal_factor2(
; ALL-NOT:        @llvm.arm.neon
; ALL:            ret void
;
  %interleaved.vec = load <3 x float>, <3 x float>* %ptr, align 16
  %v0 = shufflevector <3 x float> %interleaved.vec, <3 x float> undef, <3 x i32> <i32 0, i32 2, i32 undef>
  ret void
}

define void @store_illegal_factor2(<3 x float>* %ptr, <3 x float> %v0) nounwind {
; ALL-LABEL: @store_illegal_factor2(
; ALL-NOT:     @llvm.arm.neon
; ALL:         ret void
;
  %interleaved.vec = shufflevector <3 x float> %v0, <3 x float> undef, <3 x i32> <i32 0, i32 2, i32 undef>
  store <3 x float> %interleaved.vec, <3 x float>* %ptr, align 16
  ret void
}

define void @store_general_mask_factor4(<8 x i32>* %ptr, <32 x i32> %v0, <32 x i32> %v1) {
; NEON-LABEL:    @store_general_mask_factor4(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <8 x i32>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 4, i32 5>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 16, i32 17>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 32, i32 33>
; NEON-NEXT:       [[TMP5:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 8, i32 9>
; NEON-NEXT:       call void @llvm.arm.neon.vst4.p0i8.v2i32(i8* [[TMP1]], <2 x i32> [[TMP2]], <2 x i32> [[TMP3]], <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_general_mask_factor4(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = shufflevector <32 x i32> %v0, <32 x i32> %v1, <8 x i32> <i32 4, i32 16, i32 32, i32 8, i32 5, i32 17, i32 33, i32 9>
  store <8 x i32> %interleaved.vec, <8 x i32>* %ptr, align 4
  ret void
}

define void @store_general_mask_factor4_undefbeg(<8 x i32>* %ptr, <32 x i32> %v0, <32 x i32> %v1) {
; NEON-LABEL:    @store_general_mask_factor4_undefbeg(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <8 x i32>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 4, i32 5>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 16, i32 17>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 32, i32 33>
; NEON-NEXT:       [[TMP5:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 8, i32 9>
; NEON-NEXT:       call void @llvm.arm.neon.vst4.p0i8.v2i32(i8* [[TMP1]], <2 x i32> [[TMP2]], <2 x i32> [[TMP3]], <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_general_mask_factor4_undefbeg(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = shufflevector <32 x i32> %v0, <32 x i32> %v1, <8 x i32> <i32 undef, i32 16, i32 32, i32 8, i32 5, i32 17, i32 33, i32 9>
  store <8 x i32> %interleaved.vec, <8 x i32>* %ptr, align 4
  ret void
}

define void @store_general_mask_factor4_undefend(<8 x i32>* %ptr, <32 x i32> %v0, <32 x i32> %v1) {
; NEON-LABEL:    @store_general_mask_factor4_undefend(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <8 x i32>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 4, i32 5>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 16, i32 17>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 32, i32 33>
; NEON-NEXT:       [[TMP5:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 8, i32 9>
; NEON-NEXT:       call void @llvm.arm.neon.vst4.p0i8.v2i32(i8* [[TMP1]], <2 x i32> [[TMP2]], <2 x i32> [[TMP3]], <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_general_mask_factor4_undefend(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = shufflevector <32 x i32> %v0, <32 x i32> %v1, <8 x i32> <i32 4, i32 16, i32 32, i32 8, i32 5, i32 17, i32 33, i32 undef>
  store <8 x i32> %interleaved.vec, <8 x i32>* %ptr, align 4
  ret void
}

define void @store_general_mask_factor4_undefmid(<8 x i32>* %ptr, <32 x i32> %v0, <32 x i32> %v1) {
; NEON-LABEL:    @store_general_mask_factor4_undefmid(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <8 x i32>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 4, i32 5>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 16, i32 17>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 32, i32 33>
; NEON-NEXT:       [[TMP5:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 8, i32 9>
; NEON-NEXT:       call void @llvm.arm.neon.vst4.p0i8.v2i32(i8* [[TMP1]], <2 x i32> [[TMP2]], <2 x i32> [[TMP3]], <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_general_mask_factor4_undefmid(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = shufflevector <32 x i32> %v0, <32 x i32> %v1, <8 x i32> <i32 4, i32 undef, i32 32, i32 8, i32 5, i32 17, i32 undef, i32 9>
  store <8 x i32> %interleaved.vec, <8 x i32>* %ptr, align 4
  ret void
}

define void @store_general_mask_factor4_undefmulti(<8 x i32>* %ptr, <32 x i32> %v0, <32 x i32> %v1) {
; NEON-LABEL:    @store_general_mask_factor4_undefmulti(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <8 x i32>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 4, i32 5>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 0, i32 1>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 0, i32 1>
; NEON-NEXT:       [[TMP5:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <2 x i32> <i32 8, i32 9>
; NEON-NEXT:       call void @llvm.arm.neon.vst4.p0i8.v2i32(i8* [[TMP1]], <2 x i32> [[TMP2]], <2 x i32> [[TMP3]], <2 x i32> [[TMP4]], <2 x i32> [[TMP5]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_general_mask_factor4_undefmulti(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = shufflevector <32 x i32> %v0, <32 x i32> %v1, <8 x i32> <i32 4, i32 undef, i32 undef, i32 8, i32 undef, i32 undef, i32 undef, i32 9>
  store <8 x i32> %interleaved.vec, <8 x i32>* %ptr, align 4
  ret void
}

define void @store_general_mask_factor3(<12 x i32>* %ptr, <32 x i32> %v0, <32 x i32> %v1) {
; NEON-LABEL:    @store_general_mask_factor3(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <12 x i32>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <4 x i32> <i32 32, i32 33, i32 34, i32 35>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <4 x i32> <i32 16, i32 17, i32 18, i32 19>
; NEON-NEXT:       call void @llvm.arm.neon.vst3.p0i8.v4i32(i8* [[TMP1]], <4 x i32> [[TMP2]], <4 x i32> [[TMP3]], <4 x i32> [[TMP4]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_general_mask_factor3(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = shufflevector <32 x i32> %v0, <32 x i32> %v1, <12 x i32> <i32 4, i32 32, i32 16, i32 5, i32 33, i32 17, i32 6, i32 34, i32 18, i32 7, i32 35, i32 19>
  store <12 x i32> %interleaved.vec, <12 x i32>* %ptr, align 4
  ret void
}

define void @store_general_mask_factor3_undefmultimid(<12 x i32>* %ptr, <32 x i32> %v0, <32 x i32> %v1) {
; NEON-LABEL:    @store_general_mask_factor3_undefmultimid(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <12 x i32>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <4 x i32> <i32 32, i32 33, i32 34, i32 35>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <4 x i32> <i32 16, i32 17, i32 18, i32 19>
; NEON-NEXT:       call void @llvm.arm.neon.vst3.p0i8.v4i32(i8* [[TMP1]], <4 x i32> [[TMP2]], <4 x i32> [[TMP3]], <4 x i32> [[TMP4]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_general_mask_factor3_undefmultimid(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = shufflevector <32 x i32> %v0, <32 x i32> %v1, <12 x i32> <i32 4, i32 32, i32 16, i32 undef, i32 33, i32 17, i32 undef, i32 34, i32 18, i32 7, i32 35, i32 19>
  store <12 x i32> %interleaved.vec, <12 x i32>* %ptr, align 4
  ret void
}

define void @store_general_mask_factor3_undef_fail(<12 x i32>* %ptr, <32 x i32> %v0, <32 x i32> %v1) {
; ALL-LABEL: @store_general_mask_factor3_undef_fail(
; ALL-NOT:     @llvm.arm.neon
; ALL:         ret void
;
  %interleaved.vec = shufflevector <32 x i32> %v0, <32 x i32> %v1, <12 x i32> <i32 4, i32 32, i32 16, i32 undef, i32 33, i32 17, i32 undef, i32 34, i32 18, i32 8, i32 35, i32 19>
  store <12 x i32> %interleaved.vec, <12 x i32>* %ptr, align 4
  ret void
}

define void @store_general_mask_factor3_undeflane(<12 x i32>* %ptr, <32 x i32> %v0, <32 x i32> %v1) {
; NEON-LABEL:    @store_general_mask_factor3_undeflane(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <12 x i32>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <4 x i32> <i32 32, i32 33, i32 34, i32 35>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <4 x i32> <i32 16, i32 17, i32 18, i32 19>
; NEON-NEXT:       call void @llvm.arm.neon.vst3.p0i8.v4i32(i8* [[TMP1]], <4 x i32> [[TMP2]], <4 x i32> [[TMP3]], <4 x i32> [[TMP4]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_general_mask_factor3_undeflane(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = shufflevector <32 x i32> %v0, <32 x i32> %v1, <12 x i32> <i32 undef, i32 32, i32 16, i32 undef, i32 33, i32 17, i32 undef, i32 34, i32 18, i32 undef, i32 35, i32 19>
  store <12 x i32> %interleaved.vec, <12 x i32>* %ptr, align 4
  ret void
}

define void @store_general_mask_factor3_endstart_fail(<12 x i32>* %ptr, <32 x i32> %v0, <32 x i32> %v1) {
; ALL-LABEL:    @store_general_mask_factor3_endstart_fail(
; ALL-NOT:        @llvm.arm.neon
; ALL:            ret void
;
  %interleaved.vec = shufflevector <32 x i32> %v0, <32 x i32> %v1, <12 x i32> <i32 undef, i32 32, i32 16, i32 undef, i32 33, i32 17, i32 undef, i32 34, i32 18, i32 2, i32 35, i32 19>
  store <12 x i32> %interleaved.vec, <12 x i32>* %ptr, align 4
  ret void
}

define void @store_general_mask_factor3_endstart_pass(<12 x i32>* %ptr, <32 x i32> %v0, <32 x i32> %v1) {
; NEON-LABEL:    @store_general_mask_factor3_endstart_pass(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <12 x i32>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <4 x i32> <i32 32, i32 33, i32 34, i32 35>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <4 x i32> <i32 16, i32 17, i32 18, i32 19>
; NEON-NEXT:       call void @llvm.arm.neon.vst3.p0i8.v4i32(i8* [[TMP1]], <4 x i32> [[TMP2]], <4 x i32> [[TMP3]], <4 x i32> [[TMP4]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_general_mask_factor3_endstart_pass(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = shufflevector <32 x i32> %v0, <32 x i32> %v1, <12 x i32> <i32 undef, i32 32, i32 16, i32 undef, i32 33, i32 17, i32 undef, i32 34, i32 18, i32 7, i32 35, i32 19>
  store <12 x i32> %interleaved.vec, <12 x i32>* %ptr, align 4
  ret void
}

define void @store_general_mask_factor3_midstart_fail(<12 x i32>* %ptr, <32 x i32> %v0, <32 x i32> %v1) {
; ALL-LABEL:    @store_general_mask_factor3_midstart_fail(
; ALL-NOT:        @llvm.arm.neon
; ALL:            ret void
;
  %interleaved.vec = shufflevector <32 x i32> %v0, <32 x i32> %v1, <12 x i32> <i32 undef, i32 32, i32 16, i32 0, i32 33, i32 17, i32 undef, i32 34, i32 18, i32 undef, i32 35, i32 19>
  store <12 x i32> %interleaved.vec, <12 x i32>* %ptr, align 4
  ret void
}

define void @store_general_mask_factor3_midstart_pass(<12 x i32>* %ptr, <32 x i32> %v0, <32 x i32> %v1) {
; NEON-LABEL:    @store_general_mask_factor3_midstart_pass(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <12 x i32>* %ptr to i8*
; NEON-NEXT:       [[TMP2:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <4 x i32> <i32 32, i32 33, i32 34, i32 35>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <32 x i32> %v0, <32 x i32> %v1, <4 x i32> <i32 16, i32 17, i32 18, i32 19>
; NEON-NEXT:       call void @llvm.arm.neon.vst3.p0i8.v4i32(i8* [[TMP1]], <4 x i32> [[TMP2]], <4 x i32> [[TMP3]], <4 x i32> [[TMP4]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_general_mask_factor3_midstart_pass(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = shufflevector <32 x i32> %v0, <32 x i32> %v1, <12 x i32> <i32 undef, i32 32, i32 16, i32 1, i32 33, i32 17, i32 undef, i32 34, i32 18, i32 undef, i32 35, i32 19>
  store <12 x i32> %interleaved.vec, <12 x i32>* %ptr, align 4
  ret void
}

@g = external global <4 x float>

; The following does not give a valid interleaved store
; ALL-LABEL: define void @no_interleave
; ALL-NOT: call void @llvm.arm.neon.vst2
; ALL: shufflevector
; ALL: store
; ALL: ret void
define void @no_interleave(<4 x float> %a0) {
  %v0 = shufflevector <4 x float> %a0, <4 x float> %a0, <4 x i32> <i32 0, i32 7, i32 1, i32 undef>
  store <4 x float> %v0, <4 x float>* @g, align 16
  ret void
}

define void @load_factor2_wide2(<16 x i32>* %ptr) {
; NEON-LABEL:    @load_factor2_wide2(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <16 x i32>* %ptr to i32*
; NEON-NEXT:       [[TMP2:%.*]] = bitcast i32* [[TMP1]] to i8*
; NEON-NEXT:       [[VLDN:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2.v4i32.p0i8(i8* [[TMP2]], i32 4)
; NEON-NEXT:       [[TMP3:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN]], 1
; NEON-NEXT:       [[TMP4:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN]], 0
; NEON-NEXT:       [[TMP5:%.*]] = getelementptr i32, i32* [[TMP1]], i32 8
; NEON-NEXT:       [[TMP6:%.*]] = bitcast i32* [[TMP5]] to i8*
; NEON-NEXT:       [[VLDN1:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2.v4i32.p0i8(i8* [[TMP6]], i32 4)
; NEON-NEXT:       [[TMP7:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN1]], 1
; NEON-NEXT:       [[TMP8:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN1]], 0
; NEON-NEXT:       [[TMP9:%.*]] = shufflevector <4 x i32> [[TMP3]], <4 x i32> [[TMP7]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP10:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP8]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @load_factor2_wide2(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = load <16 x i32>, <16 x i32>* %ptr, align 4
  %v0 = shufflevector <16 x i32> %interleaved.vec, <16 x i32> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %v1 = shufflevector <16 x i32> %interleaved.vec, <16 x i32> undef, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  ret void
}

define void @load_factor2_wide3(<24 x i32>* %ptr) {
; NEON-LABEL:    @load_factor2_wide3(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <24 x i32>* [[PTR:%.*]] to i32*
; NEON-NEXT:       [[TMP2:%.*]] = bitcast i32* [[TMP1]] to i8*
; NEON-NEXT:       [[VLDN:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2.v4i32.p0i8(i8* [[TMP2]], i32 4)
; NEON-NEXT:       [[TMP3:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN]], 1
; NEON-NEXT:       [[TMP4:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN]], 0
; NEON-NEXT:       [[TMP5:%.*]] = getelementptr i32, i32* [[TMP1]], i32 8
; NEON-NEXT:       [[TMP6:%.*]] = bitcast i32* [[TMP5]] to i8*
; NEON-NEXT:       [[VLDN1:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2.v4i32.p0i8(i8* [[TMP6]], i32 4)
; NEON-NEXT:       [[TMP7:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN1]], 1
; NEON-NEXT:       [[TMP8:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN1]], 0
; NEON-NEXT:       [[TMP9:%.*]] = getelementptr i32, i32* [[TMP5]], i32 8
; NEON-NEXT:       [[TMP10:%.*]] = bitcast i32* [[TMP9]] to i8*
; NEON-NEXT:       [[VLDN2:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2.v4i32.p0i8(i8* [[TMP10]], i32 4)
; NEON-NEXT:       [[TMP11:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN2]], 1
; NEON-NEXT:       [[TMP12:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN2]], 0
; NEON-NEXT:       [[TMP13:%.*]] = shufflevector <4 x i32> [[TMP3]], <4 x i32> [[TMP7]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP14:%.*]] = shufflevector <4 x i32> [[TMP11]], <4 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
; NEON-NEXT:       [[TMP15:%.*]] = shufflevector <8 x i32> [[TMP13]], <8 x i32> [[TMP14]], <12 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
; NEON-NEXT:       [[TMP16:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP8]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP17:%.*]] = shufflevector <4 x i32> [[TMP12]], <4 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
; NEON-NEXT:       [[TMP18:%.*]] = shufflevector <8 x i32> [[TMP16]], <8 x i32> [[TMP17]], <12 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @load_factor2_wide3(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = load <24 x i32>, <24 x i32>* %ptr, align 4
  %v0 = shufflevector <24 x i32> %interleaved.vec, <24 x i32> undef, <12 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14, i32 16, i32 18, i32 20, i32 22>
  %v1 = shufflevector <24 x i32> %interleaved.vec, <24 x i32> undef, <12 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15, i32 17, i32 19, i32 21, i32 23>
  ret void
}

define void @load_factor3_wide(<24 x i32>* %ptr) {
; NEON-LABEL:    @load_factor3_wide(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <24 x i32>* %ptr to i32*
; NEON-NEXT:       [[TMP2:%.*]] = bitcast i32* [[TMP1]] to i8*
; NEON-NEXT:       [[VLDN:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3.v4i32.p0i8(i8* [[TMP2]], i32 4)
; NEON-NEXT:       [[TMP3:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 2
; NEON-NEXT:       [[TMP4:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 1
; NEON-NEXT:       [[TMP5:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 0
; NEON-NEXT:       [[TMP6:%.*]] = getelementptr i32, i32* [[TMP1]], i32 12
; NEON-NEXT:       [[TMP7:%.*]] = bitcast i32* [[TMP6]] to i8*
; NEON-NEXT:       [[VLDN1:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld3.v4i32.p0i8(i8* [[TMP7]], i32 4)
; NEON-NEXT:       [[TMP8:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN1]], 2
; NEON-NEXT:       [[TMP9:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN1]], 1
; NEON-NEXT:       [[TMP10:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN1]], 0
; NEON-NEXT:       [[TMP11:%.*]] = shufflevector <4 x i32> [[TMP3]], <4 x i32> [[TMP8]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP12:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP9]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP13:%.*]] = shufflevector <4 x i32> [[TMP5]], <4 x i32> [[TMP10]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @load_factor3_wide(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = load <24 x i32>, <24 x i32>* %ptr, align 4
  %v0 = shufflevector <24 x i32> %interleaved.vec, <24 x i32> undef, <8 x i32> <i32 0, i32 3, i32 6, i32 9, i32 12, i32 15, i32 18, i32 21>
  %v1 = shufflevector <24 x i32> %interleaved.vec, <24 x i32> undef, <8 x i32> <i32 1, i32 4, i32 7, i32 10, i32 13, i32 16, i32 19, i32 22>
  %v2 = shufflevector <24 x i32> %interleaved.vec, <24 x i32> undef, <8 x i32> <i32 2, i32 5, i32 8, i32 11, i32 14, i32 17, i32 20, i32 23>
  ret void
}

define void @load_factor4_wide(<32 x i32>* %ptr) {
; NEON-LABEL:    @load_factor4_wide(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <32 x i32>* %ptr to i32*
; NEON-NEXT:       [[TMP2:%.*]] = bitcast i32* [[TMP1]] to i8*
; NEON-NEXT:       [[VLDN:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4.v4i32.p0i8(i8* [[TMP2]], i32 4)
; NEON-NEXT:       [[TMP3:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 3
; NEON-NEXT:       [[TMP4:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 2
; NEON-NEXT:       [[TMP5:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 1
; NEON-NEXT:       [[TMP6:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN]], 0
; NEON-NEXT:       [[TMP7:%.*]] = getelementptr i32, i32* [[TMP1]], i32 16
; NEON-NEXT:       [[TMP8:%.*]] = bitcast i32* [[TMP7]] to i8*
; NEON-NEXT:       [[VLDN1:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.arm.neon.vld4.v4i32.p0i8(i8* [[TMP8]], i32 4)
; NEON-NEXT:       [[TMP9:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN1]], 3
; NEON-NEXT:       [[TMP10:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN1]], 2
; NEON-NEXT:       [[TMP11:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN1]], 1
; NEON-NEXT:       [[TMP12:%.*]] = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLDN1]], 0
; NEON-NEXT:       [[TMP13:%.*]] = shufflevector <4 x i32> [[TMP3]], <4 x i32> [[TMP9]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP14:%.*]] = shufflevector <4 x i32> [[TMP4]], <4 x i32> [[TMP10]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP15:%.*]] = shufflevector <4 x i32> [[TMP5]], <4 x i32> [[TMP11]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP16:%.*]] = shufflevector <4 x i32> [[TMP6]], <4 x i32> [[TMP12]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @load_factor4_wide(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = load <32 x i32>, <32 x i32>* %ptr, align 4
  %v0 = shufflevector <32 x i32> %interleaved.vec, <32 x i32> undef, <8 x i32> <i32 0, i32 4, i32 8, i32 12, i32 16, i32 20, i32 24, i32 28>
  %v1 = shufflevector <32 x i32> %interleaved.vec, <32 x i32> undef, <8 x i32> <i32 1, i32 5, i32 9, i32 13, i32 17, i32 21, i32 25, i32 29>
  %v2 = shufflevector <32 x i32> %interleaved.vec, <32 x i32> undef, <8 x i32> <i32 2, i32 6, i32 10, i32 14, i32 18, i32 22, i32 26, i32 30>
  %v3 = shufflevector <32 x i32> %interleaved.vec, <32 x i32> undef, <8 x i32> <i32 3, i32 7, i32 11, i32 15, i32 19, i32 23, i32 27, i32 31>
  ret void
}

define void @store_factor2_wide(<16 x i32>* %ptr, <8 x i32> %v0, <8 x i32> %v1) {
; NEON-LABEL:    @store_factor2_wide(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <16 x i32>* %ptr to i32*
; NEON-NEXT:       [[TMP2:%.*]] = bitcast i32* [[TMP1]] to i8*
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <8 x i32> %v0, <8 x i32> %v1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <8 x i32> %v0, <8 x i32> %v1, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
; NEON-NEXT:       call void @llvm.arm.neon.vst2.p0i8.v4i32(i8* [[TMP2]], <4 x i32> [[TMP3]], <4 x i32> [[TMP4]], i32 4)
; NEON-NEXT:       [[TMP5:%.*]] = getelementptr i32, i32* [[TMP1]], i32 8
; NEON-NEXT:       [[TMP6:%.*]] = bitcast i32* [[TMP5]] to i8*
; NEON-NEXT:       [[TMP7:%.*]] = shufflevector <8 x i32> %v0, <8 x i32> %v1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP8:%.*]] = shufflevector <8 x i32> %v0, <8 x i32> %v1, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
; NEON-NEXT:       call void @llvm.arm.neon.vst2.p0i8.v4i32(i8* [[TMP6]], <4 x i32> [[TMP7]], <4 x i32> [[TMP8]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_factor2_wide(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = shufflevector <8 x i32> %v0, <8 x i32> %v1, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  store <16 x i32> %interleaved.vec, <16 x i32>* %ptr, align 4
  ret void
}

define void @store_factor3_wide(<24 x i32>* %ptr, <8 x i32> %v0, <8 x i32> %v1, <8 x i32> %v2) {
; NEON-LABEL:    @store_factor3_wide(
; NEON:            [[TMP1:%.*]] = bitcast <24 x i32>* %ptr to i32*
; NEON-NEXT:       [[TMP2:%.*]] = bitcast i32* [[TMP1]] to i8*
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
; NEON-NEXT:       [[TMP5:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <4 x i32> <i32 16, i32 17, i32 18, i32 19>
; NEON-NEXT:       call void @llvm.arm.neon.vst3.p0i8.v4i32(i8* [[TMP2]], <4 x i32> [[TMP3]], <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], i32 4)
; NEON-NEXT:       [[TMP6:%.*]] = getelementptr i32, i32* [[TMP1]], i32 12
; NEON-NEXT:       [[TMP7:%.*]] = bitcast i32* [[TMP6]] to i8*
; NEON-NEXT:       [[TMP8:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP9:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
; NEON-NEXT:       [[TMP10:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <4 x i32> <i32 20, i32 21, i32 22, i32 23>
; NEON-NEXT:       call void @llvm.arm.neon.vst3.p0i8.v4i32(i8* [[TMP7]], <4 x i32> [[TMP8]], <4 x i32> [[TMP9]], <4 x i32> [[TMP10]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_factor3_wide(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %s0 = shufflevector <8 x i32> %v0, <8 x i32> %v1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %s1 = shufflevector <8 x i32> %v2, <8 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %interleaved.vec = shufflevector <16 x i32> %s0, <16 x i32> %s1, <24 x i32> <i32 0, i32 8, i32 16, i32 1, i32 9, i32 17, i32 2, i32 10, i32 18, i32 3, i32 11, i32 19, i32 4, i32 12, i32 20, i32 5, i32 13, i32 21, i32 6, i32 14, i32 22, i32 7, i32 15, i32 23>
  store <24 x i32> %interleaved.vec, <24 x i32>* %ptr, align 4
  ret void
}

define void @store_factor4_wide(<32 x i32>* %ptr, <8 x i32> %v0, <8 x i32> %v1, <8 x i32> %v2, <8 x i32> %v3) {
; NEON-LABEL:    @store_factor4_wide(
; NEON:            [[TMP1:%.*]] = bitcast <32 x i32>* %ptr to i32*
; NEON-NEXT:       [[TMP2:%.*]] = bitcast i32* [[TMP1]] to i8*
; NEON-NEXT:       [[TMP3:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; NEON-NEXT:       [[TMP4:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <4 x i32> <i32 8, i32 9, i32 10, i32 11>
; NEON-NEXT:       [[TMP5:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <4 x i32> <i32 16, i32 17, i32 18, i32 19>
; NEON-NEXT:       [[TMP6:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <4 x i32> <i32 24, i32 25, i32 26, i32 27>
; NEON-NEXT:       call void @llvm.arm.neon.vst4.p0i8.v4i32(i8* [[TMP2]], <4 x i32> [[TMP3]], <4 x i32> [[TMP4]], <4 x i32> [[TMP5]], <4 x i32> [[TMP6]], i32 4)
; NEON-NEXT:       [[TMP7:%.*]] = getelementptr i32, i32* [[TMP1]], i32 16
; NEON-NEXT:       [[TMP8:%.*]] = bitcast i32* [[TMP7]] to i8*
; NEON-NEXT:       [[TMP9:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP10:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <4 x i32> <i32 12, i32 13, i32 14, i32 15>
; NEON-NEXT:       [[TMP11:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <4 x i32> <i32 20, i32 21, i32 22, i32 23>
; NEON-NEXT:       [[TMP12:%.*]] = shufflevector <16 x i32> %s0, <16 x i32> %s1, <4 x i32> <i32 28, i32 29, i32 30, i32 31>
; NEON-NEXT:       call void @llvm.arm.neon.vst4.p0i8.v4i32(i8* [[TMP8]], <4 x i32> [[TMP9]], <4 x i32> [[TMP10]], <4 x i32> [[TMP11]], <4 x i32> [[TMP12]], i32 4)
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @store_factor4_wide(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %s0 = shufflevector <8 x i32> %v0, <8 x i32> %v1, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %s1 = shufflevector <8 x i32> %v2, <8 x i32> %v3, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %interleaved.vec = shufflevector <16 x i32> %s0, <16 x i32> %s1, <32 x i32> <i32 0, i32 8, i32 16, i32 24, i32 1, i32 9, i32 17, i32 25, i32 2, i32 10, i32 18, i32 26, i32 3, i32 11, i32 19, i32 27, i32 4, i32 12, i32 20, i32 28, i32 5, i32 13, i32 21, i32 29, i32 6, i32 14, i32 22, i32 30, i32 7, i32 15, i32 23, i32 31>
  store <32 x i32> %interleaved.vec, <32 x i32>* %ptr, align 4
  ret void
}

define void @load_factor2_fp128(<4 x fp128>* %ptr) {
; ALL-LABEL: @load_factor2_fp128(
; ALL-NOT:     @llvm.arm.neon
; ALL:         ret void
;
  %interleaved.vec = load <4 x fp128>, <4 x fp128>* %ptr, align 16
  %v0 = shufflevector <4 x fp128> %interleaved.vec, <4 x fp128> undef, <2 x i32> <i32 0, i32 2>
  %v1 = shufflevector <4 x fp128> %interleaved.vec, <4 x fp128> undef, <2 x i32> <i32 1, i32 3>
  ret void
}

define void @load_factor2_wide_pointer(<16 x i32*>* %ptr) {
; NEON-LABEL:    @load_factor2_wide_pointer(
; NEON-NEXT:       [[TMP1:%.*]] = bitcast <16 x i32*>* %ptr to i32*
; NEON-NEXT:       [[TMP2:%.*]] = bitcast i32* [[TMP1]] to i8*
; NEON-NEXT:       [[VLDN:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2.v4i32.p0i8(i8* [[TMP2]], i32 4)
; NEON-NEXT:       [[TMP3:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN]], 1
; NEON-NEXT:       [[TMP4:%.*]] = inttoptr <4 x i32> [[TMP3]] to <4 x i32*>
; NEON-NEXT:       [[TMP5:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN]], 0
; NEON-NEXT:       [[TMP6:%.*]] = inttoptr <4 x i32> [[TMP5]] to <4 x i32*>
; NEON-NEXT:       [[TMP7:%.*]] = getelementptr i32, i32* [[TMP1]], i32 8
; NEON-NEXT:       [[TMP8:%.*]] = bitcast i32* [[TMP7]] to i8*
; NEON-NEXT:       [[VLDN1:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.arm.neon.vld2.v4i32.p0i8(i8* [[TMP8]], i32 4)
; NEON-NEXT:       [[TMP9:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN1]], 1
; NEON-NEXT:       [[TMP10:%.*]] = inttoptr <4 x i32> [[TMP9]] to <4 x i32*>
; NEON-NEXT:       [[TMP11:%.*]] = extractvalue { <4 x i32>, <4 x i32> } [[VLDN1]], 0
; NEON-NEXT:       [[TMP12:%.*]] = inttoptr <4 x i32> [[TMP11]] to <4 x i32*>
; NEON-NEXT:       [[TMP13:%.*]] = shufflevector <4 x i32*> [[TMP4]], <4 x i32*> [[TMP10]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       [[TMP14:%.*]] = shufflevector <4 x i32*> [[TMP6]], <4 x i32*> [[TMP12]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; NEON-NEXT:       ret void
; NO_NEON-LABEL: @load_factor2_wide_pointer(
; NO_NEON-NOT:     @llvm.arm.neon
; NO_NEON:         ret void
;
  %interleaved.vec = load <16 x i32*>, <16 x i32*>* %ptr, align 4
  %v0 = shufflevector <16 x i32*> %interleaved.vec, <16 x i32*> undef, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %v1 = shufflevector <16 x i32*> %interleaved.vec, <16 x i32*> undef, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  ret void
}
