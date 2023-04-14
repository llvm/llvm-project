; RUN: llc -mtriple=aarch64-apple-darwin -mattr=+neon -verify-machineinstrs < %s | FileCheck %s

; Basic tests from input vector to bitmask
; IR generated from clang for:
; __builtin_convertvector + reinterpret_cast<uint16&>

define i16 @convert_to_bitmask16(<16 x i8> %vec) {
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
; CHECK-LABEL: convert_to_bitmask16
; CHECK:       ; %bb.0:
; CHECK-NEXT: Lloh0:
; CHECK-NEXT:  adrp	    x8, lCPI0_0@PAGE
; CHECK-NEXT:  cmeq.16b v0, v0, #0
; CHECK-NEXT: Lloh1:
; CHECK-NEXT:  ldr	    q1, [x8, lCPI0_0@PAGEOFF]
; CHECK-NEXT:  bic.16b	v0, v1, v0
; CHECK-NEXT:  ext.16b	v1, v0, v0, #8
; CHECK-NEXT:  addv.8b	b0, v0
; CHECK-NEXT:  addv.8b	b1, v1
; CHECK-NEXT:  fmov	    w9, s0
; CHECK-NEXT:  fmov	    w8, s1
; CHECK-NEXT:  orr	    w0, w9, w8, lsl #8
; CHECK-NEXT:  ret

  %cmp_result = icmp ne <16 x i8> %vec, zeroinitializer
  %bitmask = bitcast <16 x i1> %cmp_result to i16
  ret i16 %bitmask
}

define i16 @convert_to_bitmask8(<8 x i16> %vec) {
; CHECK-LABEL: lCPI1_0:
; CHECK-NEXT: .short	1
; CHECK-NEXT: .short	2
; CHECK-NEXT: .short	4
; CHECK-NEXT: .short	8
; CHECK-NEXT: .short	16
; CHECK-NEXT: .short	32
; CHECK-NEXT: .short	64
; CHECK-NEXT: .short	128

; CHECK-LABEL: convert_to_bitmask8
; CHECK:       ; %bb.0:
; CHECK-NEXT: Lloh2:
; CHECK-NEXT:  adrp	    x8, lCPI1_0@PAGE
; CHECK-NEXT:  cmeq.8h	v0, v0, #0
; CHECK-NEXT: Lloh3:
; CHECK-NEXT:  ldr	    q1, [x8, lCPI1_0@PAGEOFF]
; CHECK-NEXT:  bic.16b	v0, v1, v0
; CHECK-NEXT:  addv.8h	h0, v0
; CHECK-NEXT:  fmov	    w8, s0
; CHECK-NEXT:  and	    w0, w8, #0xff
; CHECK-NEXT:  ret

  %cmp_result = icmp ne <8 x i16> %vec, zeroinitializer
  %bitmask = bitcast <8 x i1> %cmp_result to i8
  %extended_bitmask = zext i8 %bitmask to i16
  ret i16 %extended_bitmask
}

define i4 @convert_to_bitmask4(<4 x i32> %vec) {
; CHECK-LABEL: lCPI2_0:
; CHECK-NEXT:  .long	1
; CHECK-NEXT:  .long	2
; CHECK-NEXT:  .long	4
; CHECK-NEXT:  .long	8

; CHECK-LABEL: convert_to_bitmask4
; CHECK:       ; %bb.0:
; CHECK-NEXT: Lloh4:
; CHECK-NEXT:  adrp	    x8, lCPI2_0@PAGE
; CHECK-NEXT:  cmeq.4s	v0, v0, #0
; CHECK-NEXT: Lloh5:
; CHECK-NEXT:  ldr	    q1, [x8, lCPI2_0@PAGEOFF]
; CHECK-NEXT:  bic.16b	v0, v1, v0
; CHECK-NEXT:  addv.4s	s0, v0
; CHECK-NEXT:  fmov	    w0, s0
; CHECK-NEXT:  ret

  %cmp_result = icmp ne <4 x i32> %vec, zeroinitializer
  %bitmask = bitcast <4 x i1> %cmp_result to i4
  ret i4 %bitmask
}

define i8 @convert_to_bitmask2(<2 x i64> %vec) {
; CHECK-LABEL: lCPI3_0:
; CHECK-NEXT: .quad	1
; CHECK-NEXT: .quad	2

; CHECK-LABEL: convert_to_bitmask2
; CHECK:       ; %bb.0:
; CHECK-NEXT: Lloh6:
; CHECK-NEXT:  adrp	    x8, lCPI3_0@PAGE
; CHECK-NEXT:  cmeq.2d	v0, v0, #0
; CHECK-NEXT: Lloh7:
; CHECK-NEXT:  ldr	    q1, [x8, lCPI3_0@PAGEOFF]
; CHECK-NEXT:  bic.16b	v0, v1, v0
; CHECK-NEXT:  addp.2d	d0, v0
; CHECK-NEXT:  fmov	    x8, d0
; CHECK-NEXT:  and	    w0, w8, #0x3
; CHECK-NEXT:  ret

  %cmp_result = icmp ne <2 x i64> %vec, zeroinitializer
  %bitmask = bitcast <2 x i1> %cmp_result to i2
  %extended_bitmask = zext i2 %bitmask to i8
  ret i8 %extended_bitmask
}

; Clang's __builtin_convertvector adds an undef vector concat for vectors with <8 elements.
define i8 @clang_builtins_undef_concat_convert_to_bitmask4(<4 x i32> %vec) {
; CHECK-LABEL: lCPI4_0:
; CHECK-NEXT:  .long	1
; CHECK-NEXT:  .long	2
; CHECK-NEXT:  .long	4
; CHECK-NEXT:  .long	8

; CHECK-LABEL: clang_builtins_undef_concat_convert_to_bitmask4
; CHECK:       ; %bb.0:
; CHECK-NEXT: Lloh8:
; CHECK-NEXT:  adrp	    x8, lCPI4_0@PAGE
; CHECK-NEXT:  cmeq.4s	v0, v0, #0
; CHECK-NEXT: Lloh9:
; CHECK-NEXT:  ldr	    q1, [x8, lCPI4_0@PAGEOFF]
; CHECK-NEXT:  bic.16b	v0, v1, v0
; CHECK-NEXT:  addv.4s	s0, v0
; CHECK-NEXT:  fmov	    w0, s0
; CHECK-NEXT:  ret

  %cmp_result = icmp ne <4 x i32> %vec, zeroinitializer
  %vector_pad = shufflevector <4 x i1> %cmp_result, <4 x i1> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef>
  %bitmask = bitcast <8 x i1> %vector_pad to i8
  ret i8 %bitmask
}


define i4 @convert_to_bitmask_no_compare(<4 x i32> %vec1, <4 x i32> %vec2) {
; CHECK-LABEL: lCPI5_0:
; CHECK-NEXT:  .long	1
; CHECK-NEXT:  .long	2
; CHECK-NEXT:  .long	4
; CHECK-NEXT:  .long	8

; CHECK-LABEL: convert_to_bitmask_no_compare
; CHECK:       ; %bb.0:
; CHECK-NEXT: Lloh10:
; CHECK-NEXT:  adrp	    x8, lCPI5_0@PAGE
; CHECK-NEXT:  and.16b  v0, v0, v1
; CHECK-NEXT:  shl.4s	v0, v0, #31
; CHECK-NEXT: Lloh11:
; CHECK-NEXT:  ldr	    q1, [x8, lCPI5_0@PAGEOFF]
; CHECK-NEXT:  cmlt.4s	v0, v0, #0
; CHECK-NEXT:  and.16b	v0, v0, v1
; CHECK-NEXT:  addv.4s	s0, v0
; CHECK-NEXT:  fmov	    w0, s0
; CHECK-NEXT:  ret

  %cmp = and <4 x i32> %vec1, %vec2
  %trunc = trunc <4 x i32> %cmp to <4 x i1>
  %bitmask = bitcast <4 x i1> %trunc to i4
  ret i4 %bitmask
}

define i4 @convert_to_bitmask_with_compare_chain(<4 x i32> %vec1, <4 x i32> %vec2) {
; CHECK-LABEL: lCPI6_0:
; CHECK-NEXT:  .long	1
; CHECK-NEXT:  .long	2
; CHECK-NEXT:  .long	4
; CHECK-NEXT:  .long	8

; CHECK-LABEL: convert_to_bitmask_with_compare_chain
; CHECK:       ; %bb.0:
; CHECK-NEXT: Lloh12:
; CHECK-NEXT:  adrp	    x8, lCPI6_0@PAGE
; CHECK-NEXT:  cmeq.4s	v2, v0, #0
; CHECK-NEXT:  cmeq.4s	v0, v0, v1
; CHECK-NEXT: Lloh13:
; CHECK-NEXT:  ldr	    q1, [x8, lCPI6_0@PAGEOFF]
; CHECK-NEXT:  bic.16b	v0, v0, v2
; CHECK-NEXT:  and.16b	v0, v0, v1
; CHECK-NEXT:  addv.4s	s0, v0
; CHECK-NEXT:  fmov	    w0, s0
; CHECK-NEXT:  ret

  %cmp1 = icmp ne <4 x i32> %vec1, zeroinitializer
  %cmp2 = icmp eq <4 x i32> %vec1, %vec2
  %cmp3 = and <4 x i1> %cmp1, %cmp2
  %bitmask = bitcast <4 x i1> %cmp3 to i4
  ret i4 %bitmask
}

define i4 @convert_to_bitmask_with_trunc_in_chain(<4 x i32> %vec1, <4 x i32> %vec2) {
; CHECK-LABEL: lCPI7_0:
; CHECK-NEXT:  .long	1
; CHECK-NEXT:  .long	2
; CHECK-NEXT:  .long	4
; CHECK-NEXT:  .long	8

; CHECK-LABEL: convert_to_bitmask_with_trunc_in_chain
; CHECK:       ; %bb.0:
; CHECK-NEXT:  cmeq.4s	v0, v0, #0
; CHECK-NEXT: Lloh14:
; CHECK-NEXT:  adrp	    x8, lCPI7_0@PAGE
; CHECK-NEXT:  bic.16b	v0, v1, v0
; CHECK-NEXT:  shl.4s	v0, v0, #31
; CHECK-NEXT: Lloh15:
; CHECK-NEXT:  ldr	    q1, [x8, lCPI7_0@PAGEOFF]
; CHECK-NEXT:  cmlt.4s	v0, v0, #0
; CHECK-NEXT:  and.16b	v0, v0, v1
; CHECK-NEXT:  addv.4s	s0, v0
; CHECK-NEXT:  fmov	    w0, s0
; CHECK-NEXT:  ret

  %cmp1 = icmp ne <4 x i32> %vec1, zeroinitializer
  %trunc_vec = trunc <4 x i32> %vec2 to <4 x i1>
  %and_res = and <4 x i1> %cmp1, %trunc_vec
  %bitmask = bitcast <4 x i1> %and_res to i4
  ret i4 %bitmask
}

define i4 @convert_to_bitmask_with_unknown_type_in_long_chain(<4 x i32> %vec1, <4 x i32> %vec2) {
; CHECK-LABEL: lCPI8_0:
; CHECK-NEXT:  .short	1
; CHECK-NEXT:  .short	2
; CHECK-NEXT:  .short	4
; CHECK-NEXT:  .short	8

; CHECK-LABEL: convert_to_bitmask_with_unknown_type_in_long_chain
; CHECK:      ; %bb.0:
; CHECK-NEXT: 	cmeq.4s	v0, v0, #0
; CHECK-NEXT: Lloh16:
; CHECK-NEXT: 	adrp	x8, lCPI8_0@PAGE
; CHECK-NEXT: 	cmeq.4s	v1, v1, #0
; CHECK-NEXT: 	movi	d2, #0x000000ffffffff
; CHECK-NEXT: 	bic.16b	v0, v1, v0
; CHECK-NEXT: 	movi	d1, #0xffff0000ffff0000
; CHECK-NEXT: 	xtn.4h	v0, v0
; CHECK-NEXT: 	movi	d3, #0x00ffffffffffff
; CHECK-NEXT: 	orr.8b	v0, v0, v2
; CHECK-NEXT: 	movi	d2, #0x00ffffffff0000
; CHECK-NEXT: 	eor.8b	v1, v0, v1
; CHECK-NEXT: 	mov.h	v1[2], wzr
; CHECK-NEXT: 	eor.8b	v0, v0, v2
; CHECK-NEXT: 	orr.8b	v0, v0, v3
; CHECK-NEXT: 	orr.8b	v0, v1, v0
; CHECK-NEXT: Lloh17:
; CHECK-NEXT: 	ldr	d1, [x8, lCPI8_0@PAGEOFF]
; CHECK-NEXT: 	shl.4h	v0, v0, #15
; CHECK-NEXT: 	cmlt.4h	v0, v0, #0
; CHECK-NEXT: 	and.8b	v0, v0, v1
; CHECK-NEXT: 	addv.4h	h0, v0
; CHECK-NEXT: 	fmov	w0, s0
; CHECK-NEXT: 	ret

  %cmp1 = icmp ne <4 x i32> %vec1, zeroinitializer
  %cmp2 = icmp eq <4 x i32> %vec2, zeroinitializer

  ; Artificially make this a long chain to hide the original type
  %chain1 = and <4 x i1> %cmp1, %cmp2;
  %chain2 = or <4 x i1> %chain1, <i1 1, i1 1, i1 0, i1 0>;
  %chain3 = xor <4 x i1> %chain2, <i1 0, i1 1, i1 0, i1 1>;
  %chain4 = and <4 x i1> %chain3, <i1 1, i1 1, i1 0, i1 1>;
  %chain5 = or <4 x i1> %chain4, <i1 1, i1 1, i1 1, i1 0>;
  %chain6 = xor <4 x i1> <i1 0, i1 1, i1 1, i1 0>, %chain2;
  %chain7 = or <4 x i1> %chain5, %chain6;
  %bitmask = bitcast <4 x i1> %chain7 to i4
  ret i4 %bitmask
}

define i4 @convert_to_bitmask_with_different_types_in_chain(<4 x i16> %vec1, <4 x i32> %vec2) {
; CHECK-LABEL: lCPI9_0:
; CHECK-NEXT:  .short	1
; CHECK-NEXT:  .short	2
; CHECK-NEXT:  .short	4
; CHECK-NEXT:  .short	8

; CHECK-LABEL: convert_to_bitmask_with_different_types_in_chain
; CHECK:      ; %bb.0:
; CHECK-NEXT: Lloh18:
; CHECK-NEXT: 	adrp	x8, lCPI9_0@PAGE
; CHECK-NEXT: 	cmeq.4h	v0, v0, #0
; CHECK-NEXT: 	cmeq.4s	v1, v1, #0
; CHECK-NEXT: 	xtn.4h	v1, v1
; CHECK-NEXT: Lloh19:
; CHECK-NEXT: 	ldr	d2, [x8, lCPI9_0@PAGEOFF]
; CHECK-NEXT: 	orn.8b	v0, v1, v0
; CHECK-NEXT: 	and.8b	v0, v0, v2
; CHECK-NEXT: 	addv.4h	h0, v0
; CHECK-NEXT: 	fmov	w0, s0
; CHECK-NEXT: 	ret

  %cmp1 = icmp ne <4 x i16> %vec1, zeroinitializer
  %cmp2 = icmp eq <4 x i32> %vec2, zeroinitializer
  %chain1 = or <4 x i1> %cmp1, %cmp2
  %bitmask = bitcast <4 x i1> %chain1 to i4
  ret i4 %bitmask
}

define i16 @convert_to_bitmask_without_knowing_type(<16 x i1> %vec) {
; CHECK-LABEL: convert_to_bitmask_without_knowing_type:
; CHECK:       ; %bb.0:
; CHECK-NEXT:  Lloh20:
; CHECK-NEXT:    adrp x8, lCPI10_0@PAGE
; CHECK-NEXT:    shl.16b v0, v0, #7
; CHECK-NEXT:    cmlt.16b v0, v0, #0
; CHECK-NEXT:  Lloh21:
; CHECK-NEXT:    ldr q1, [x8, lCPI10_0@PAGEOFF]
; CHECK-NEXT:    and.16b v0, v0, v1
; CHECK-NEXT:    ext.16b v1, v0, v0, #8
; CHECK-NEXT:    addv.8b b0, v0
; CHECK-NEXT:    addv.8b b1, v1
; CHECK-NEXT:    fmov w9, s0
; CHECK-NEXT:    fmov w8, s1
; CHECK-NEXT:    orr w0, w9, w8, lsl #8
; CHECK-NEXT:    ret

  %bitmask = bitcast <16 x i1> %vec to i16
  ret i16 %bitmask
}

define i2 @convert_to_bitmask_2xi32(<2 x i32> %vec) {
; CHECK-LABEL: convert_to_bitmask_2xi32
; CHECK:       ; %bb.0:
; CHECK-NEXT:  Lloh22:
; CHECK-NEXT:  	adrp	x8, lCPI11_0@PAGE
; CHECK-NEXT:  	cmeq.2s	v0, v0, #0
; CHECK-NEXT:  Lloh23:
; CHECK-NEXT:  	ldr	d1, [x8, lCPI11_0@PAGEOFF]
; CHECK-NEXT:  	bic.8b	v0, v1, v0
; CHECK-NEXT:  	addp.2s	v0, v0, v0
; CHECK-NEXT:  	fmov	w0, s0
; CHECK-NEXT:  	ret

  %cmp_result = icmp ne <2 x i32> %vec, zeroinitializer
  %bitmask = bitcast <2 x i1> %cmp_result to i2
  ret i2 %bitmask
}

define i4 @convert_to_bitmask_4xi8(<4 x i8> %vec) {
; CHECK-LABEL: convert_to_bitmask_4xi8
; CHECK:       ; %bb.0:
; CHECK-NEXT:  Lloh24:
; CHECK-NEXT:  	adrp	x8, lCPI12_0@PAGE
; CHECK-NEXT:  	bic.4h	v0, #255, lsl #8
; CHECK-NEXT:  	cmeq.4h	v0, v0, #0
; CHECK-NEXT:  Lloh25:
; CHECK-NEXT:  	ldr	d1, [x8, lCPI12_0@PAGEOFF]
; CHECK-NEXT:  	bic.8b	v0, v1, v0
; CHECK-NEXT:  	addv.4h	h0, v0
; CHECK-NEXT:  	fmov	w0, s0
; CHECK-NEXT:  	ret

  %cmp_result = icmp ne <4 x i8> %vec, zeroinitializer
  %bitmask = bitcast <4 x i1> %cmp_result to i4
  ret i4 %bitmask
}

define i8 @convert_to_bitmask_8xi2(<8 x i2> %vec) {
; CHECK-LABEL: convert_to_bitmask_8xi2
; CHECK:       ; %bb.0:
; CHECK-NEXT:  	movi.8b v1, #3
; CHECK-NEXT:  Lloh26:
; CHECK-NEXT:  	adrp	x8, lCPI13_0@PAGE
; CHECK-NEXT:  	and.8b	v0, v0, v1
; CHECK-NEXT:  Lloh27:
; CHECK-NEXT:  	ldr	d1, [x8, lCPI13_0@PAGEOFF]
; CHECK-NEXT:  	cmeq.8b	v0, v0, #0
; CHECK-NEXT:  	bic.8b	v0, v1, v0
; CHECK-NEXT:  	addv.8b	b0, v0
; CHECK-NEXT:  	fmov	w0, s0
; CHECK-NEXT:  	ret

  %cmp_result = icmp ne <8 x i2> %vec, zeroinitializer
  %bitmask = bitcast <8 x i1> %cmp_result to i8
  ret i8 %bitmask
}

define i4 @convert_to_bitmask_float(<4 x float> %vec) {
; CHECK-LABEL: lCPI14_0:
; CHECK-NEXT:  .long	1
; CHECK-NEXT:  .long	2
; CHECK-NEXT:  .long	4
; CHECK-NEXT:  .long	8

; CHECK-LABEL: convert_to_bitmask_float
; CHECK:       ; %bb.0:
; CHECK-NEXT: Lloh28:
; CHECK-NEXT:  adrp	    x8, lCPI14_0@PAGE
; CHECK-NEXT:  fcmgt.4s	v1, v0, #0.0
; CHECK-NEXT:  fcmlt.4s	v0, v0, #0.0
; CHECK-NEXT: Lloh29:
; CHECK-NEXT:  ldr	    q2, [x8, lCPI14_0@PAGEOFF]
; CHECK-NEXT:  orr.16b	v0, v0, v1
; CHECK-NEXT:  and.16b	v0, v0, v2
; CHECK-NEXT:  addv.4s	s0, v0
; CHECK-NEXT:  fmov	    w0, s0
; CHECK-NEXT:  ret

  %cmp_result = fcmp one <4 x float> %vec, zeroinitializer
  %bitmask = bitcast <4 x i1> %cmp_result to i4
  ret i4 %bitmask
}

; TODO(lawben): Change this in follow-up patch to #D145301, as truncating stores fix this.
; Larger vector types don't map directly.
define i8 @no_convert_large_vector(<8 x i32> %vec) {
; CHECK-LABEL: convert_large_vector:
; CHECK:       cmeq.4s	v1, v1, #0
; CHECK-NOT:   addv

   %cmp_result = icmp ne <8 x i32> %vec, zeroinitializer
   %bitmask = bitcast <8 x i1> %cmp_result to i8
   ret i8 %bitmask
}

; This may still be converted as a v8i8 after the vector concat (but not as v4iX).
define i8 @no_direct_convert_for_bad_concat(<4 x i32> %vec) {
; CHECK-LABEL: no_direct_convert_for_bad_concat:
; CHECK:     cmtst.4s v0, v0, v0
; CHECK-NOT: addv.4

  %cmp_result = icmp ne <4 x i32> %vec, zeroinitializer
  %vector_pad = shufflevector <4 x i1> poison, <4 x i1> %cmp_result, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 4, i32 5, i32 6, i32 7>
  %bitmask = bitcast <8 x i1> %vector_pad to i8
  ret i8 %bitmask
}

define <8 x i1> @no_convert_without_direct_bitcast(<8 x i16> %vec) {
; CHECK-LABEL: no_convert_without_direct_bitcast:
; CHECK:     cmtst.8h v0, v0, v0
; CHECK-NOT: addv.4s	s0, v0

   %cmp_result = icmp ne <8 x i16> %vec, zeroinitializer
   ret <8 x i1> %cmp_result
}
