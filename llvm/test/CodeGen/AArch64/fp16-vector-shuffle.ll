; RUN: llc < %s -mtriple=aarch64-none-eabi | FileCheck %s

; float16x4_t select_64(float16x4_t a, float16x4_t b, uint16x4_t c) { return vbsl_u16(c, a, b); }
define <4 x half> @select_64(<4 x half> %a, <4 x half> %b, <4 x i16> %c) #0 {
; CHECK-LABEL: select_64:
; CHECK: bsl
entry:
  %0 = bitcast <4 x half> %a to <4 x i16>
  %1 = bitcast <4 x half> %b to <4 x i16>
  %vbsl3.i = and <4 x i16> %0, %c
  %2 = xor <4 x i16> %c, <i16 -1, i16 -1, i16 -1, i16 -1>
  %vbsl4.i = and <4 x i16> %1, %2
  %vbsl5.i = or <4 x i16> %vbsl3.i, %vbsl4.i
  %3 = bitcast <4 x i16> %vbsl5.i to <4 x half>
  ret <4 x half> %3
}

; float16x8_t select_128(float16x8_t a, float16x8_t b, uint16x8_t c) { return vbslq_u16(c, a, b); }
define <8 x half> @select_128(<8 x half> %a, <8 x half> %b, <8 x i16> %c) #0 {
; CHECK-LABEL: select_128:
; CHECK: bsl
entry:
  %0 = bitcast <8 x half> %a to <8 x i16>
  %1 = bitcast <8 x half> %b to <8 x i16>
  %vbsl3.i = and <8 x i16> %0, %c
  %2 = xor <8 x i16> %c, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  %vbsl4.i = and <8 x i16> %1, %2
  %vbsl5.i = or <8 x i16> %vbsl3.i, %vbsl4.i
  %3 = bitcast <8 x i16> %vbsl5.i to <8 x half>
  ret <8 x half> %3
}

; float16x4_t lane_64_64(float16x4_t a, float16x4_t b) {
;  return vcopy_lane_s16(a, 1, b, 2);
; }
define <4 x half> @lane_64_64(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-LABEL: lane_64_64:
; CHECK: ins
entry:
  %0 = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 0, i32 6, i32 2, i32 3>
  ret <4 x half> %0
}

; float16x8_t lane_128_64(float16x8_t a, float16x4_t b) {
;   return vcopyq_lane_s16(a, 1, b, 2);
; }
define <8 x half> @lane_128_64(<8 x half> %a, <4 x half> %b) #0 {
; CHECK-LABEL: lane_128_64:
; CHECK: ins
entry:
  %0 = bitcast <4 x half> %b to <4 x i16>
  %vget_lane = extractelement <4 x i16> %0, i32 2
  %1 = bitcast <8 x half> %a to <8 x i16>
  %vset_lane = insertelement <8 x i16> %1, i16 %vget_lane, i32 1
  %2 = bitcast <8 x i16> %vset_lane to <8 x half>
  ret <8 x half> %2
}

; float16x4_t lane_64_128(float16x4_t a, float16x8_t b) {
;   return vcopy_laneq_s16(a, 3, b, 5);
; }
define <4 x half> @lane_64_128(<4 x half> %a, <8 x half> %b) #0 {
; CHECK-LABEL: lane_64_128:
; CHECK: ins
entry:
  %0 = bitcast <8 x half> %b to <8 x i16>
  %vgetq_lane = extractelement <8 x i16> %0, i32 5
  %1 = bitcast <4 x half> %a to <4 x i16>
  %vset_lane = insertelement <4 x i16> %1, i16 %vgetq_lane, i32 3
  %2 = bitcast <4 x i16> %vset_lane to <4 x half>
  ret <4 x half> %2
}

; float16x8_t lane_128_128(float16x8_t a, float16x8_t b) {
;   return vcopyq_laneq_s16(a, 3, b, 5);
; }
define <8 x half> @lane_128_128(<8 x half> %a, <8 x half> %b) #0 {
; CHECK-LABEL: lane_128_128:
; CHECK: ins
entry:
  %0 = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 13, i32 4, i32 5, i32 6, i32 7>
  ret <8 x half> %0
}

; float16x4_t ext_64(float16x4_t a, float16x4_t b) {
;   return vext_s16(a, b, 3);
; }
define <4 x half> @ext_64(<4 x half> %a, <4 x half> %b) #0 {
; CHECK-LABEL: ext_64:
; CHECK: ext
entry:
  %0 = shufflevector <4 x half> %a, <4 x half> %b, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  ret <4 x half> %0
}

; float16x8_t ext_128(float16x8_t a, float16x8_t b) {
;   return vextq_s16(a, b, 3);
; }
define <8 x half> @ext_128(<8 x half> %a, <8 x half> %b) #0 {
; CHECK-LABEL: ext_128:
; CHECK: ext
entry:
  %0 = shufflevector <8 x half> %a, <8 x half> %b, <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
  ret <8 x half> %0
}

; float16x4_t rev32_64(float16x4_t a) {
;   return vrev32_s16(a);
; }
define <4 x half> @rev32_64(<4 x half> %a) #0 {
entry:
; CHECK-LABEL: rev32_64:
; CHECK: rev32
  %0 = shufflevector <4 x half> %a, <4 x half> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  ret <4 x half> %0
}

; float16x4_t rev64_64(float16x4_t a) {
;   return vrev64_s16(a);
; }
define <4 x half> @rev64_64(<4 x half> %a) #0 {
entry:
; CHECK-LABEL: rev64_64:
; CHECK: rev64
  %0 = shufflevector <4 x half> %a, <4 x half> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x half> %0
}

; float16x8_t rev32_128(float16x8_t a) {
;   return vrev32q_s16(a);
; }
define <8 x half> @rev32_128(<8 x half> %a) #0 {
entry:
; CHECK-LABEL: rev32_128:
; CHECK: rev32
  %0 = shufflevector <8 x half> %a, <8 x half> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
  ret <8 x half> %0
}

; float16x8_t rev64_128(float16x8_t a) {
;   return vrev64q_s16(a);
; }
define <8 x half> @rev64_128(<8 x half> %a) #0 {
entry:
; CHECK-LABEL: rev64_128:
; CHECK: rev64
  %0 = shufflevector <8 x half> %a, <8 x half> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
  ret <8 x half> %0
}

; float16x4_t create_64(long long a) { return vcreate_f16(a); }
define <4 x half> @create_64(i64 %a) #0 {
; CHECK-LABEL: create_64:
; CHECK: fmov
entry:
  %0 = bitcast i64 %a to <4 x half>
  ret <4 x half> %0
}

; float16x4_t dup_64(__fp16 a) { return vdup_n_f16(a); }
define <4 x half> @dup_64(half %a) #0 {
; CHECK-LABEL: dup_64:
; CHECK: dup
entry:
  %vecinit = insertelement <4 x half> undef, half %a, i32 0
  %vecinit1 = insertelement <4 x half> %vecinit, half %a, i32 1
  %vecinit2 = insertelement <4 x half> %vecinit1, half %a, i32 2
  %vecinit3 = insertelement <4 x half> %vecinit2, half %a, i32 3
  ret <4 x half> %vecinit3
}

; float16x8_t dup_128(__fp16 a) { return vdupq_n_f16(a); }
define <8 x half> @dup_128(half %a) #0 {
entry:
; CHECK-LABEL: dup_128:
; CHECK: dup
  %vecinit = insertelement <8 x half> undef, half %a, i32 0
  %vecinit1 = insertelement <8 x half> %vecinit, half %a, i32 1
  %vecinit2 = insertelement <8 x half> %vecinit1, half %a, i32 2
  %vecinit3 = insertelement <8 x half> %vecinit2, half %a, i32 3
  %vecinit4 = insertelement <8 x half> %vecinit3, half %a, i32 4
  %vecinit5 = insertelement <8 x half> %vecinit4, half %a, i32 5
  %vecinit6 = insertelement <8 x half> %vecinit5, half %a, i32 6
  %vecinit7 = insertelement <8 x half> %vecinit6, half %a, i32 7
  ret <8 x half> %vecinit7
}

; float16x4_t dup_lane_64(float16x4_t a) { return vdup_lane_f16(a, 2); }
define <4 x half> @dup_lane_64(<4 x half> %a) #0 {
entry:
; CHECK-LABEL: dup_lane_64:
; CHECK: dup
  %shuffle = shufflevector <4 x half> %a, <4 x half> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  ret <4 x half> %shuffle
}

; float16x8_t dup_lane_128(float16x4_t a) { return vdupq_lane_f16(a, 2); }
define <8 x half> @dup_lane_128(<4 x half> %a) #0 {
entry:
; CHECK-LABEL: dup_lane_128:
; CHECK: dup
  %shuffle = shufflevector <4 x half> %a, <4 x half> undef, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  ret <8 x half> %shuffle
}

; float16x4_t dup_laneq_64(float16x8_t a) { return vdup_laneq_f16(a, 2); }
define <4 x half> @dup_laneq_64(<8 x half> %a) #0 {
entry:
; CHECK-LABEL: dup_laneq_64:
; CHECK: dup
  %shuffle = shufflevector <8 x half> %a, <8 x half> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  ret <4 x half> %shuffle
}

; float16x8_t dup_laneq_128(float16x8_t a) { return vdupq_laneq_f16(a, 2); }
define <8 x half> @dup_laneq_128(<8 x half> %a) #0 {
entry:
; CHECK-LABEL: dup_laneq_128:
; CHECK: dup
  %shuffle = shufflevector <8 x half> %a, <8 x half> undef, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  ret <8 x half> %shuffle
}

; float16x8_t vcombine(float16x4_t a, float16x4_t b) { return vcombine_f16(a, b); }
define <8 x half> @vcombine(<4 x half> %a, <4 x half> %b) #0 {
entry:
; CHECK-LABEL: vcombine:
; CHECK: ins
  %shuffle.i = shufflevector <4 x half> %a, <4 x half> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x half> %shuffle.i
}

; float16x4_t get_high(float16x8_t a) { return vget_high_f16(a); }
define <4 x half> @get_high(<8 x half> %a) #0 {
; CHECK-LABEL: get_high:
; CHECK: ext
entry:
  %shuffle.i = shufflevector <8 x half> %a, <8 x half> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  ret <4 x half> %shuffle.i
}


; float16x4_t get_low(float16x8_t a) { return vget_low_f16(a); }
define <4 x half> @get_low(<8 x half> %a) #0 {
; CHECK-LABEL: get_low:
; CHECK-NOT: ext
entry:
  %shuffle.i = shufflevector <8 x half> %a, <8 x half> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x half> %shuffle.i
}

; float16x4_t set_lane_64(float16x4_t a, __fp16 b) { return vset_lane_f16(b, a, 2); }
define <4 x half> @set_lane_64(<4 x half> %a, half %b) #0 {
; CHECK-LABEL: set_lane_64:
; CHECK: fmov
; CHECK: ins
entry:
  %0 = bitcast half %b to i16
  %1 = bitcast <4 x half> %a to <4 x i16>
  %vset_lane = insertelement <4 x i16> %1, i16 %0, i32 2
  %2 = bitcast <4 x i16> %vset_lane to <4 x half>
  ret <4 x half> %2
}


; float16x8_t set_lane_128(float16x8_t a, __fp16 b) { return vsetq_lane_f16(b, a, 2); }
define <8 x half> @set_lane_128(<8 x half> %a, half %b) #0 {
; CHECK-LABEL: set_lane_128:
; CHECK: fmov
; CHECK: ins
entry:
  %0 = bitcast half %b to i16
  %1 = bitcast <8 x half> %a to <8 x i16>
  %vset_lane = insertelement <8 x i16> %1, i16 %0, i32 2
  %2 = bitcast <8 x i16> %vset_lane to <8 x half>
  ret <8 x half> %2
}

; __fp16 get_lane_64(float16x4_t a) { return vget_lane_f16(a, 2); }
define half @get_lane_64(<4 x half> %a) #0 {
; CHECK-LABEL: get_lane_64:
; CHECK: umov
; CHECK: fmov
entry:
  %0 = bitcast <4 x half> %a to <4 x i16>
  %vget_lane = extractelement <4 x i16> %0, i32 2
  %1 = bitcast i16 %vget_lane to half
  ret half %1
}

; __fp16 get_lane_128(float16x8_t a) { return vgetq_lane_f16(a, 2); }
define half @get_lane_128(<8 x half> %a) #0 {
; CHECK-LABEL: get_lane_128:
; CHECK: umov
; CHECK: fmov
entry:
  %0 = bitcast <8 x half> %a to <8 x i16>
  %vgetq_lane = extractelement <8 x i16> %0, i32 2
  %1 = bitcast i16 %vgetq_lane to half
  ret half %1
}
