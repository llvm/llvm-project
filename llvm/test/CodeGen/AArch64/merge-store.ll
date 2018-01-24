; RUN: llc < %s -mtriple=aarch64-unknown-unknown -mcpu=cyclone -mattr=+slow-misaligned-128store | FileCheck %s --check-prefix=SPLITTING --check-prefix=CHECK
; RUN: llc < %s -mtriple=aarch64-eabi -mattr=-slow-misaligned-128store | FileCheck %s --check-prefix=MISALIGNED --check-prefix=CHECK

@g0 = external global <3 x float>, align 16
@g1 = external global <3 x float>, align 4

; CHECK: ldr q[[R0:[0-9]+]], {{\[}}[[R1:x[0-9]+]], :lo12:g0
; CHECK: str d[[R0]]

define void @blam() {
  %tmp4 = getelementptr inbounds <3 x float>, <3 x float>* @g1, i64 0, i64 0
  %tmp5 = load <3 x float>, <3 x float>* @g0, align 16
  %tmp6 = extractelement <3 x float> %tmp5, i64 0
  store float %tmp6, float* %tmp4
  %tmp7 = getelementptr inbounds float, float* %tmp4, i64 1
  %tmp8 = load <3 x float>, <3 x float>* @g0, align 16
  %tmp9 = extractelement <3 x float> %tmp8, i64 1
  store float %tmp9, float* %tmp7
  ret void;
}


; PR21711 - Merge vector stores into wider vector stores.

; On Cyclone, the stores should not get merged into a 16-byte store because
; unaligned 16-byte stores are slow. This test would infinite loop when
; the fastness of unaligned accesses was not specified correctly.

define void @merge_vec_extract_stores(<4 x float> %v1, <2 x float>* %ptr) {
  %idx0 = getelementptr inbounds <2 x float>, <2 x float>* %ptr, i64 3
  %idx1 = getelementptr inbounds <2 x float>, <2 x float>* %ptr, i64 4

  %shuffle0 = shufflevector <4 x float> %v1, <4 x float> undef, <2 x i32> <i32 0, i32 1>
  %shuffle1 = shufflevector <4 x float> %v1, <4 x float> undef, <2 x i32> <i32 2, i32 3>

  store <2 x float> %shuffle0, <2 x float>* %idx0, align 8
  store <2 x float> %shuffle1, <2 x float>* %idx1, align 8
  ret void

; MISALIGNED-LABEL:    merge_vec_extract_stores
; MISALIGNED:          stur   q0, [x0, #24]
; MISALIGNED-NEXT:     ret

; FIXME: Ideally we would like to use a generic target for this test, but this relies
; on suppressing store pairs.

; SPLITTING-LABEL:    merge_vec_extract_stores
; SPLITTING:          ext   v1.16b, v0.16b, v0.16b, #8
; SPLITTING-NEXT:     str   d0, [x0, #24]
; SPLITTING-NEXT:     str   d1, [x0, #32]
; SPLITTING-NEXT:     ret
}
