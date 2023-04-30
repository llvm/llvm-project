; RUN: llc -mtriple=aarch64-apple-darwin -mattr=+neon -verify-machineinstrs < %s | FileCheck %s

define void @store_16_elements(<16 x i8> %vec, ptr %out) {
; Bits used in mask
; CHECK-LABEL: lCPI0_0
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	32
; CHECK-NEXT: .byte	64
; CHECK-NEXT: .byte	128
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	8
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	32
; CHECK-NEXT: .byte	64
; CHECK-NEXT: .byte	128

; Actual conversion
; CHECK-LABEL: store_16_elements
; CHECK:       ; %bb.0:
; CHECK-NEXT:  Lloh0:
; CHECK-NEXT:  adrp	    x8, lCPI0_0@PAGE
; CHECK-NEXT:  cmeq.16b	v0, v0, #0
; CHECK-NEXT:  Lloh1:
; CHECK-NEXT:  ldr	    q1, [x8, lCPI0_0@PAGEOFF]
; CHECK-NEXT:  bic.16b	v0, v1, v0
; CHECK-NEXT:  ext.16b	v1, v0, v0, #8
; CHECK-NEXT:  addv.8b	b0, v0
; CHECK-NEXT:  addv.8b	b1, v1
; CHECK-NEXT:  fmov	    w9, s0
; CHECK-NEXT:  fmov	    w8, s1
; CHECK-NEXT:  orr	    w8, w9, w8, lsl #8
; CHECK-NEXT:  strh	    w8, [x0]
; CHECK-NEXT:  ret

  %cmp_result = icmp ne <16 x i8> %vec, zeroinitializer
  store <16 x i1> %cmp_result, ptr %out
  ret void
}

define void @store_8_elements(<8 x i16> %vec, ptr %out) {
; CHECK-LABEL: lCPI1_0:
; CHECK-NEXT: .short	1
; CHECK-NEXT: .short	2
; CHECK-NEXT: .short	4
; CHECK-NEXT: .short	8
; CHECK-NEXT: .short	16
; CHECK-NEXT: .short	32
; CHECK-NEXT: .short	64
; CHECK-NEXT: .short	128

; CHECK-LABEL: store_8_elements
; CHECK:       ; %bb.0:
; CHECK-NEXT:  Lloh2:
; CHECK-NEXT:  adrp	    x8, lCPI1_0@PAGE
; CHECK-NEXT:  cmeq.8h	v0, v0, #0
; CHECK-NEXT:  Lloh3:
; CHECK-NEXT:  ldr	    q1, [x8, lCPI1_0@PAGEOFF]
; CHECK-NEXT:  bic.16b	v0, v1, v0
; CHECK-NEXT:  addv.8h	h0, v0
; CHECK-NEXT:  fmov	    w8, s0
; CHECK-NEXT:  strb	    w8, [x0]
; CHECK-NEXT:  ret

  %cmp_result = icmp ne <8 x i16> %vec, zeroinitializer
  store <8 x i1> %cmp_result, ptr %out
  ret void
}

define void @store_4_elements(<4 x i32> %vec, ptr %out) {
; CHECK-LABEL: lCPI2_0:
; CHECK-NEXT:  .long	1
; CHECK-NEXT:  .long	2
; CHECK-NEXT:  .long	4
; CHECK-NEXT:  .long	8

; CHECK-LABEL: store_4_elements
; CHECK:       ; %bb.0:
; CHECK-NEXT:  Lloh4:
; CHECK-NEXT:  adrp	x8, lCPI2_0@PAGE
; CHECK-NEXT:  cmeq.4s	v0, v0, #0
; CHECK-NEXT:  Lloh5:
; CHECK-NEXT:  ldr	    q1, [x8, lCPI2_0@PAGEOFF]
; CHECK-NEXT:  bic.16b	v0, v1, v0
; CHECK-NEXT:  addv.4s	s0, v0
; CHECK-NEXT:  fmov	    w8, s0
; CHECK-NEXT:  strb	    w8, [x0]
; CHECK-NEXT:  ret

  %cmp_result = icmp ne <4 x i32> %vec, zeroinitializer
  store <4 x i1> %cmp_result, ptr %out
  ret void
}

define void @store_2_elements(<2 x i64> %vec, ptr %out) {
; CHECK-LABEL: lCPI3_0:
; CHECK-NEXT: .quad	1
; CHECK-NEXT: .quad	2

; CHECK-LABEL: store_2_elements
; CHECK:       ; %bb.0:
; CHECK-NEXT:  Lloh6:
; CHECK-NEXT:  adrp	x8, lCPI3_0@PAGE
; CHECK-NEXT:  cmeq.2d	v0, v0, #0
; CHECK-NEXT:  Lloh7:
; CHECK-NEXT:  ldr	    q1, [x8, lCPI3_0@PAGEOFF]
; CHECK-NEXT:  bic.16b	v0, v1, v0
; CHECK-NEXT:  addp.2d	d0, v0
; CHECK-NEXT:  fmov	    x8, d0
; CHECK-NEXT:  strb	    w8, [x0]
; CHECK-NEXT:  ret

  %cmp_result = icmp ne <2 x i64> %vec, zeroinitializer
  store <2 x i1> %cmp_result, ptr %out
  ret void
}

define void @add_trunc_compare_before_store(<4 x i32> %vec, ptr %out) {
; CHECK-LABEL: lCPI4_0:
; CHECK-NEXT:  .long	1
; CHECK-NEXT:  .long	2
; CHECK-NEXT:  .long	4
; CHECK-NEXT:  .long	8

; CHECK-LABEL: add_trunc_compare_before_store
; CHECK:       ; %bb.0:
; CHECK-NEXT:  Lloh8:
; CHECK-NEXT:  adrp	    x8, lCPI4_0@PAGE
; CHECK-NEXT:  shl.4s	v0, v0, #31
; CHECK-NEXT:  cmlt.4s	v0, v0, #0
; CHECK-NEXT:  Lloh9:
; CHECK-NEXT:  ldr	    q1, [x8, lCPI4_0@PAGEOFF]
; CHECK-NEXT:  and.16b	v0, v0, v1
; CHECK-NEXT:  addv.4s	s0, v0
; CHECK-NEXT:  fmov 	w8, s0
; CHECK-NEXT:  strb	    w8, [x0]
; CHECK-NEXT:  ret

  %trunc = trunc <4 x i32> %vec to <4 x i1>
  store <4 x i1> %trunc, ptr %out
  ret void
}

define void @add_trunc_mask_unknown_vector_type(<4 x i1> %vec, ptr %out) {
; CHECK-LABEL: lCPI5_0:
; CHECK: .short	1
; CHECK: .short	2
; CHECK: .short	4
; CHECK: .short	8

; CHECK-LABEL: add_trunc_mask_unknown_vector_type
; CHECK:       ; %bb.0:
; CHECK-NEXT: Lloh10:
; CHECK-NEXT:  adrp	    x8, lCPI5_0@PAGE
; CHECK-NEXT:  shl.4h	v0, v0, #15
; CHECK-NEXT:  cmlt.4h	v0, v0, #0
; CHECK-NEXT: Lloh11:
; CHECK-NEXT:  ldr	    d1, [x8, lCPI5_0@PAGEOFF]
; CHECK-NEXT:  and.8b    v0, v0, v1
; CHECK-NEXT:  addv.4h	h0, v0
; CHECK-NEXT:  fmov	    w8, s0
; CHECK-NEXT:  strb	    w8, [x0]
; CHECK-NEXT:  ret

  store <4 x i1> %vec, ptr %out
  ret void
}

define void @store_8_elements_64_bit_vector(<8 x i8> %vec, ptr %out) {
; CHECK-LABEL: lCPI6_0:
; CHECK-NEXT:  .byte	1
; CHECK-NEXT:  .byte	2
; CHECK-NEXT:  .byte	4
; CHECK-NEXT:  .byte	8
; CHECK-NEXT:  .byte	16
; CHECK-NEXT:  .byte	32
; CHECK-NEXT:  .byte	64
; CHECK-NEXT:  .byte	128

; CHECK-LABEL: store_8_elements_64_bit_vector
; CHECK:       ; %bb.0:
; CHECK-NEXT:  Lloh12:
; CHECK-NEXT:  adrp	x8, lCPI6_0@PAGE
; CHECK-NEXT:  cmeq.8b	v0, v0, #0
; CHECK-NEXT:  Lloh13:
; CHECK-NEXT:  ldr	    d1, [x8, lCPI6_0@PAGEOFF]
; CHECK-NEXT:  bic.8b	v0, v1, v0
; CHECK-NEXT:  addv.8b	b0, v0
; CHECK-NEXT:  st1.b	{ v0 }[0], [x0]
; CHECK-NEXT:  ret

  %cmp_result = icmp ne <8 x i8> %vec, zeroinitializer
  store <8 x i1> %cmp_result, ptr %out
  ret void
}

define void @store_4_elements_64_bit_vector(<4 x i16> %vec, ptr %out) {
; CHECK-LABEL: lCPI7_0:
; CHECK-NEXT:  .short	1
; CHECK-NEXT:  .short	2
; CHECK-NEXT:  .short	4
; CHECK-NEXT:  .short	8

; CHECK-LABEL: store_4_elements_64_bit_vector
; CHECK:       ; %bb.0:
; CHECK-NEXT:  Lloh14:
; CHECK-NEXT:  adrp	x8, lCPI7_0@PAGE
; CHECK-NEXT:  cmeq.4h	v0, v0, #0
; CHECK-NEXT:  Lloh15:
; CHECK-NEXT:  ldr	    d1, [x8, lCPI7_0@PAGEOFF]
; CHECK-NEXT:  bic.8b	v0, v1, v0
; CHECK-NEXT:  addv.4h	h0, v0
; CHECK-NEXT:  fmov	    w8, s0
; CHECK-NEXT:  strb	    w8, [x0]
; CHECK-NEXT:  ret

  %cmp_result = icmp ne <4 x i16> %vec, zeroinitializer
  store <4 x i1> %cmp_result, ptr %out
  ret void
}

define void @store_2_elements_64_bit_vector(<2 x i32> %vec, ptr %out) {
; CHECK-LABEL: lCPI8_0:
; CHECK-NEXT:  .long	1
; CHECK-NEXT:  .long	2

; CHECK-LABEL: store_2_elements_64_bit_vector
; CHECK:       ; %bb.0:
; CHECK-NEXT:  Lloh16:
; CHECK-NEXT:  adrp	x8, lCPI8_0@PAGE
; CHECK-NEXT:  cmeq.2s	v0, v0, #0
; CHECK-NEXT:  Lloh17:
; CHECK-NEXT:  ldr	    d1, [x8, lCPI8_0@PAGEOFF]
; CHECK-NEXT:  bic.8b	v0, v1, v0
; CHECK-NEXT:  addp.2s	v0, v0, v0
; CHECK-NEXT:  fmov	    w8, s0
; CHECK-NEXT:  strb	    w8, [x0]
; CHECK-NEXT:  ret

  %cmp_result = icmp ne <2 x i32> %vec, zeroinitializer
  store <2 x i1> %cmp_result, ptr %out
  ret void
}

define void @no_combine_without_truncate(<16 x i8> %vec, ptr %out) {
; CHECK-LABEL: no_combine_without_truncate
; CHECK:     cmtst.16b v0, v0, v0
; CHECK-NOT: addv.8b	b0, v0

  %cmp_result = icmp ne <16 x i8> %vec, zeroinitializer
  %extended_result = sext <16 x i1> %cmp_result to <16 x i8>
  store <16 x i8> %extended_result, ptr %out
  ret void
}

define void @no_combine_for_non_bool_truncate(<4 x i32> %vec, ptr %out) {
; CHECK-LABEL: no_combine_for_non_bool_truncate
; CHECK:     xtn.4h v0, v0
; CHECK-NOT: addv.4s s0, v0

  %trunc = trunc <4 x i32> %vec to <4 x i8>
  store <4 x i8> %trunc, ptr %out
  ret void
}

define void @no_combine_for_build_vector(i1 %a, i1 %b, i1 %c, i1 %d, ptr %out) {
; CHECK-LABEL: no_combine_for_build_vector
; CHECK-NOT: addv

  %1 =   insertelement <4 x i1> undef, i1 %a, i64 0
  %2 =   insertelement <4 x i1>    %1, i1 %b, i64 1
  %3 =   insertelement <4 x i1>    %2, i1 %c, i64 2
  %vec = insertelement <4 x i1>    %3, i1 %d, i64 3
  store <4 x i1> %vec, ptr %out
  ret void
}
