; RUN: llc -mtriple=aarch64_be--linux-gnu < %s | FileCheck %s

@vec_v8i16 = global <8 x i16> <i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 8>

; CHECK-LABEL: movi_modimm_t1:
define i16 @movi_modimm_t1() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    movi	   v[[REG2:[0-9]+]].4s, #1
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: movi_modimm_t2:
define i16 @movi_modimm_t2() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    movi	   v[[REG2:[0-9]+]].4s, #1, lsl #8
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 256, i16 0, i16 256, i16 0, i16 256, i16 0, i16 256, i16 0>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: movi_modimm_t3:
define i16 @movi_modimm_t3() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    movi	   v[[REG2:[0-9]+]].4s, #1, lsl #16
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: movi_modimm_t4:
define i16 @movi_modimm_t4() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    movi	   v[[REG2:[0-9]+]].4s, #1, lsl #24
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 0, i16 256, i16 0, i16 256, i16 0, i16 256, i16 0, i16 256>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: movi_modimm_t5:
define i16 @movi_modimm_t5() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    movi	   v[[REG2:[0-9]+]].8h, #1
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: movi_modimm_t6:
define i16 @movi_modimm_t6() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    movi	   v[[REG2:[0-9]+]].8h, #1, lsl #8
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: movi_modimm_t7:
define i16 @movi_modimm_t7() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    movi	   v[[REG2:[0-9]+]].4s, #1, msl #8
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 511, i16 0, i16 511, i16 0, i16 511, i16 0, i16 511, i16 0>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: movi_modimm_t8:
define i16 @movi_modimm_t8() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    movi	   v[[REG2:[0-9]+]].4s, #1, msl #16
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 65535, i16 1, i16 65535, i16 1, i16 65535, i16 1, i16 65535, i16 1>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: movi_modimm_t9:
define i16 @movi_modimm_t9() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    movi	   v[[REG2:[0-9]+]].16b, #1
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 257, i16 257, i16 257, i16 257, i16 257, i16 257, i16 257, i16 257>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: movi_modimm_t10:
define i16 @movi_modimm_t10() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    movi	   v[[REG2:[0-9]+]].2d, #0x00ffff0000ffff
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 -1, i16 0, i16 -1, i16 0, i16 -1, i16 0, i16 -1, i16 0>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: fmov_modimm_t11:
define i16 @fmov_modimm_t11() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    fmov    v[[REG2:[0-9]+]].4s, #3.00000000
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 0, i16 16448, i16 0, i16 16448, i16 0, i16 16448, i16 0, i16 16448>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: fmov_modimm_t12:
define i16 @fmov_modimm_t12() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    fmov    v[[REG2:[0-9]+]].2d, #0.17968750
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 0, i16 0, i16 0, i16 16327, i16 0, i16 0, i16 0, i16 16327>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: mvni_modimm_t1:
define i16 @mvni_modimm_t1() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    mvni	   v[[REG2:[0-9]+]].4s, #1
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 65534, i16 65535, i16 65534, i16 65535, i16 65534, i16 65535, i16 65534, i16 65535>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: mvni_modimm_t2:
define i16 @mvni_modimm_t2() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    mvni	   v[[REG2:[0-9]+]].4s, #1, lsl #8
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 65279, i16 65535, i16 65279, i16 65535, i16 65279, i16 65535, i16 65279, i16 65535>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: mvni_modimm_t3:
define i16 @mvni_modimm_t3() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    mvni	   v[[REG2:[0-9]+]].4s, #1, lsl #16
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 65535, i16 65534, i16 65535, i16 65534, i16 65535, i16 65534, i16 65535, i16 65534>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: mvni_modimm_t4:
define i16 @mvni_modimm_t4() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    mvni	   v[[REG2:[0-9]+]].4s, #1, lsl #24
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 65535, i16 65279, i16 65535, i16 65279, i16 65535, i16 65279, i16 65535, i16 65279>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: mvni_modimm_t5:
define i16 @mvni_modimm_t5() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    mvni	   v[[REG2:[0-9]+]].8h, #1
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 65534, i16 65534, i16 65534, i16 65534, i16 65534, i16 65534, i16 65534, i16 65534>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: mvni_modimm_t6:
define i16 @mvni_modimm_t6() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    mvni	   v[[REG2:[0-9]+]].8h, #1, lsl #8
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 65279, i16 65279, i16 65279, i16 65279, i16 65279, i16 65279, i16 65279, i16 65279>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: mvni_modimm_t7:
define i16 @mvni_modimm_t7() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    mvni	   v[[REG2:[0-9]+]].4s, #1, msl #8
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 65024, i16 65535, i16 65024, i16 65535, i16 65024, i16 65535, i16 65024, i16 65535>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: mvni_modimm_t8:
define i16 @mvni_modimm_t8() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    mvni	   v[[REG2:[0-9]+]].4s, #1, msl #16
  ; CHECK-NEXT:    add	   v[[REG1]].8h, v[[REG1]].8h, v[[REG2]].8h
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = add <8 x i16> %in, <i16 0, i16 65534, i16 0, i16 65534, i16 0, i16 65534, i16 0, i16 65534>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: bic_modimm_t1:
define i16 @bic_modimm_t1() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    bic	   v[[REG2:[0-9]+]].4s, #1
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = and <8 x i16> %in, <i16 65534, i16 65535, i16 65534, i16 65535, i16 65534, i16 65535, i16 65534, i16 65535>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: bic_modimm_t2:
define i16 @bic_modimm_t2() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    bic	   v[[REG2:[0-9]+]].4s, #1, lsl #8
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = and <8 x i16> %in, <i16 65279, i16 65535, i16 65279, i16 65535, i16 65279, i16 65535, i16 65279, i16 65535>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: bic_modimm_t3:
define i16 @bic_modimm_t3() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    bic	   v[[REG2:[0-9]+]].4s, #1, lsl #16
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = and <8 x i16> %in, <i16 65535, i16 65534, i16 65535, i16 65534, i16 65535, i16 65534, i16 65535, i16 65534>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: bic_modimm_t4:
define i16 @bic_modimm_t4() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    bic	   v[[REG2:[0-9]+]].4s, #1, lsl #24
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = and <8 x i16> %in, <i16 65535, i16 65279, i16 65535, i16 65279, i16 65535, i16 65279, i16 65535, i16 65279>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: bic_modimm_t5:
define i16 @bic_modimm_t5() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    bic	   v[[REG2:[0-9]+]].8h, #1
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = and <8 x i16> %in, <i16 65534, i16 65534, i16 65534, i16 65534, i16 65534, i16 65534, i16 65534, i16 65534>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: bic_modimm_t6:
define i16 @bic_modimm_t6() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    bic	   v[[REG2:[0-9]+]].8h, #1, lsl #8
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = and <8 x i16> %in, <i16 65279, i16 65279, i16 65279, i16 65279, i16 65279, i16 65279, i16 65279, i16 65279>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: orr_modimm_t1:
define i16 @orr_modimm_t1() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    orr	   v[[REG2:[0-9]+]].4s, #1
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = or <8 x i16> %in, <i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: orr_modimm_t2:
define i16 @orr_modimm_t2() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    orr     v[[REG2:[0-9]+]].4s, #1, lsl #8
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = or <8 x i16> %in, <i16 256, i16 0, i16 256, i16 0, i16 256, i16 0, i16 256, i16 0>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: orr_modimm_t3:
define i16 @orr_modimm_t3() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    orr	   v[[REG2:[0-9]+]].4s, #1, lsl #16
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = or <8 x i16> %in, <i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: orr_modimm_t4:
define i16 @orr_modimm_t4() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    orr	   v[[REG2:[0-9]+]].4s, #1, lsl #24
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = or <8 x i16> %in, <i16 0, i16 256, i16 0, i16 256, i16 0, i16 256, i16 0, i16 256>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: orr_modimm_t5:
define i16 @orr_modimm_t5() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    orr	   v[[REG2:[0-9]+]].8h, #1
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = or <8 x i16> %in, <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

; CHECK-LABEL: orr_modimm_t6:
define i16 @orr_modimm_t6() nounwind {
  ; CHECK:         ld1     { v[[REG1:[0-9]+]].8h }, [x{{[0-9]+}}]
  ; CHECK-NEXT:    orr	   v[[REG2:[0-9]+]].8h, #1, lsl #8
  ; CHECK-NEXT:    umov	   w{{[0-9]+}}, v[[REG1]].h[0]
  %in = load <8 x i16>, <8 x i16>* @vec_v8i16
  %rv = or <8 x i16> %in, <i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256>
  %el = extractelement <8 x i16> %rv, i32 0
  ret i16 %el
}

declare i8 @f_v8i8(<8 x i8> %arg)
declare i16 @f_v4i16(<4 x i16> %arg)
declare i32 @f_v2i32(<2 x i32> %arg)
declare i64 @f_v1i64(<1 x i64> %arg)
declare i8 @f_v16i8(<16 x i8> %arg)
declare i16 @f_v8i16(<8 x i16> %arg)
declare i32 @f_v4i32(<4 x i32> %arg)
declare i64 @f_v2i64(<2 x i64> %arg)

; CHECK-LABEL: modimm_t1_call:
define void @modimm_t1_call() {
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #8
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.8b, v[[REG1]].8b
  ; CHECK-NEXT:    bl      f_v8i8
  call i8 @f_v8i8(<8 x i8> <i8 8, i8 0, i8 0, i8 0, i8 8, i8 0, i8 0, i8 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #7
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.4h, v[[REG1]].4h
  ; CHECK-NEXT:    bl      f_v4i16
  call i16 @f_v4i16(<4 x i16> <i16 7, i16 0, i16 7, i16 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #6
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.2s, v[[REG1]].2s
  ; CHECK-NEXT:    bl      f_v2i32
  call i32 @f_v2i32(<2 x i32> <i32 6, i32 6>)
  ; CHECK:         movi    v{{[0-9]+}}.2s, #5
  ; CHECK-NEXT:    bl      f_v1i64
  call i64 @f_v1i64(<1 x i64> <i64 21474836485>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #5
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].16b, v[[REG1]].16b
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v16i8
  call i8 @f_v16i8(<16 x i8> <i8 5, i8 0, i8 0, i8 0, i8 5, i8 0, i8 0, i8 0, i8 5, i8 0, i8 0, i8 0, i8 5, i8 0, i8 0, i8 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #4
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].8h, v[[REG1]].8h
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v8i16
  call i16 @f_v8i16(<8 x i16> <i16 4, i16 0, i16 4, i16 0, i16 4, i16 0, i16 4, i16 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #3
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].4s, v[[REG1]].4s
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v4i32
  call i32 @f_v4i32(<4 x i32> <i32 3, i32 3, i32 3, i32 3>)
  ; CHECK:         movi    v[[REG:[0-9]+]].4s, #2
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v2i64
  call i64 @f_v2i64(<2 x i64> <i64 8589934594, i64 8589934594>)

  ret void
}

; CHECK-LABEL: modimm_t2_call:
define void @modimm_t2_call() {
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #8, lsl #8
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.8b, v[[REG1]].8b
  ; CHECK-NEXT:    bl      f_v8i8
  call i8 @f_v8i8(<8 x i8> <i8 0, i8 8, i8 0, i8 0, i8 0, i8 8, i8 0, i8 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #7, lsl #8
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.4h, v[[REG1]].4h
  ; CHECK-NEXT:    bl      f_v4i16
  call i16 @f_v4i16(<4 x i16> <i16 1792, i16 0, i16 1792, i16 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #6, lsl #8
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.2s, v[[REG1]].2s
  ; CHECK-NEXT:    bl      f_v2i32
  call i32 @f_v2i32(<2 x i32> <i32 1536, i32 1536>)
  ; CHECK:         movi    v{{[0-9]+}}.2s, #5, lsl #8
  ; CHECK-NEXT:    bl      f_v1i64
  call i64 @f_v1i64(<1 x i64> <i64 5497558140160>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #5, lsl #8
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].16b, v[[REG1]].16b
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v16i8
  call i8 @f_v16i8(<16 x i8> <i8 0, i8 5, i8 0, i8 0, i8 0, i8 5, i8 0, i8 0, i8 0, i8 5, i8 0, i8 0, i8 0, i8 5, i8 0, i8 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #4, lsl #8
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].8h, v[[REG1]].8h
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v8i16
  call i16 @f_v8i16(<8 x i16> <i16 1024, i16 0, i16 1024, i16 0, i16 1024, i16 0, i16 1024, i16 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #3, lsl #8
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].4s, v[[REG1]].4s
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v4i32
  call i32 @f_v4i32(<4 x i32> <i32 768, i32 768, i32 768, i32 768>)
  ; CHECK:         movi    v[[REG:[0-9]+]].4s, #2, lsl #8
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v2i64
  call i64 @f_v2i64(<2 x i64> <i64 2199023256064, i64 2199023256064>)

  ret void
}

; CHECK-LABEL: modimm_t3_call:
define void @modimm_t3_call() {
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #8, lsl #16
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.8b, v[[REG1]].8b
  ; CHECK-NEXT:    bl      f_v8i8
  call i8 @f_v8i8(<8 x i8> <i8 0, i8 0, i8 8, i8 0, i8 0, i8 0, i8 8, i8 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #7, lsl #16
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.4h, v[[REG1]].4h
  ; CHECK-NEXT:    bl      f_v4i16
  call i16 @f_v4i16(<4 x i16> <i16 0, i16 7, i16 0, i16 7>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #6, lsl #16
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.2s, v[[REG1]].2s
  ; CHECK-NEXT:    bl      f_v2i32
  call i32 @f_v2i32(<2 x i32> <i32 393216, i32 393216>)
  ; CHECK:         movi    v{{[0-9]+}}.2s, #5, lsl #16
  ; CHECK-NEXT:    bl      f_v1i64
  call i64 @f_v1i64(<1 x i64> <i64 1407374883880960>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #5, lsl #16
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].16b, v[[REG1]].16b
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v16i8
  call i8 @f_v16i8(<16 x i8> <i8 0, i8 0, i8 5, i8 0, i8 0, i8 0, i8 5, i8 0, i8 0, i8 0, i8 5, i8 0, i8 0, i8 0, i8 5, i8 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #4, lsl #16
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].8h, v[[REG1]].8h
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v8i16
  call i16 @f_v8i16(<8 x i16> <i16 0, i16 4, i16 0, i16 4, i16 0, i16 4, i16 0, i16 4>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #3, lsl #16
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].4s, v[[REG1]].4s
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v4i32
  call i32 @f_v4i32(<4 x i32> <i32 196608, i32 196608, i32 196608, i32 196608>)
  ; CHECK:         movi    v[[REG:[0-9]+]].4s, #2, lsl #16
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v2i64
  call i64 @f_v2i64(<2 x i64> <i64 562949953552384, i64 562949953552384>)

  ret void
}

; CHECK-LABEL: modimm_t4_call:
define void @modimm_t4_call() {
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #8, lsl #24
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.8b, v[[REG1]].8b
  ; CHECK-NEXT:    bl      f_v8i8
  call i8 @f_v8i8(<8 x i8> <i8 0, i8 0, i8 0, i8 8, i8 0, i8 0, i8 0, i8 8>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #7, lsl #24
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.4h, v[[REG1]].4h
  ; CHECK-NEXT:    bl      f_v4i16
  call i16 @f_v4i16(<4 x i16> <i16 0, i16 1792, i16 0, i16 1792>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #6, lsl #24
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.2s, v[[REG1]].2s
  ; CHECK-NEXT:    bl      f_v2i32
  call i32 @f_v2i32(<2 x i32> <i32 100663296, i32 100663296>)
  ; CHECK:         movi    v{{[0-9]+}}.2s, #5, lsl #24
  ; CHECK-NEXT:    bl      f_v1i64
  call i64 @f_v1i64(<1 x i64> <i64 360287970273525760>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #5, lsl #24
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].16b, v[[REG1]].16b
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v16i8
  call i8 @f_v16i8(<16 x i8> <i8 0, i8 0, i8 0, i8 5, i8 0, i8 0, i8 0, i8 5, i8 0, i8 0, i8 0, i8 5, i8 0, i8 0, i8 0, i8 5>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #4, lsl #24
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].8h, v[[REG1]].8h
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v8i16
  call i16 @f_v8i16(<8 x i16> <i16 0, i16 1024, i16 0, i16 1024, i16 0, i16 1024, i16 0, i16 1024>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #3, lsl #24
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].4s, v[[REG1]].4s
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v4i32
  call i32 @f_v4i32(<4 x i32> <i32 50331648, i32 50331648, i32 50331648, i32 50331648>)
  ; CHECK:         movi    v[[REG:[0-9]+]].4s, #2, lsl #24
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v2i64
  call i64 @f_v2i64(<2 x i64> <i64 144115188109410304, i64 144115188109410304>)

  ret void
}

; CHECK-LABEL: modimm_t5_call:
define void @modimm_t5_call() {
  ; CHECK:         movi    v[[REG1:[0-9]+]].4h, #8
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.8b, v[[REG1]].8b
  ; CHECK-NEXT:    bl      f_v8i8
  call i8 @f_v8i8(<8 x i8> <i8 8, i8 0, i8 8, i8 0, i8 8, i8 0, i8 8, i8 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4h, #7
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.4h, v[[REG1]].4h
  ; CHECK-NEXT:    bl      f_v4i16
  call i16 @f_v4i16(<4 x i16> <i16 7, i16 7, i16 7, i16 7>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4h, #6
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.2s, v[[REG1]].2s
  ; CHECK-NEXT:    bl      f_v2i32
  call i32 @f_v2i32(<2 x i32> <i32 393222, i32 393222>)
  ; CHECK:         movi    v{{[0-9]+}}.4h, #5
  ; CHECK-NEXT:    bl      f_v1i64
  call i64 @f_v1i64(<1 x i64> <i64 1407396358717445>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].8h, #5
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].16b, v[[REG1]].16b
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v16i8
  call i8 @f_v16i8(<16 x i8> <i8 5, i8 0, i8 5, i8 0, i8 5, i8 0, i8 5, i8 0, i8 5, i8 0, i8 5, i8 0, i8 5, i8 0, i8 5, i8 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].8h, #4
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].8h, v[[REG1]].8h
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v8i16
  call i16 @f_v8i16(<8 x i16> <i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4, i16 4>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].8h, #3
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].4s, v[[REG1]].4s
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v4i32
  call i32 @f_v4i32(<4 x i32> <i32 196611, i32 196611, i32 196611, i32 196611>)
  ; CHECK:         movi    v[[REG:[0-9]+]].8h, #2
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v2i64
  call i64 @f_v2i64(<2 x i64> <i64 562958543486978, i64 562958543486978>)

  ret void
}

; CHECK-LABEL: modimm_t6_call:
define void @modimm_t6_call() {
  ; CHECK:         movi    v[[REG1:[0-9]+]].4h, #8, lsl #8
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.8b, v[[REG1]].8b
  ; CHECK-NEXT:    bl      f_v8i8
  call i8 @f_v8i8(<8 x i8> <i8 0, i8 8, i8 0, i8 8, i8 0, i8 8, i8 0, i8 8>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4h, #7, lsl #8
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.4h, v[[REG1]].4h
  ; CHECK-NEXT:    bl      f_v4i16
  call i16 @f_v4i16(<4 x i16> <i16 1792, i16 1792, i16 1792, i16 1792>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4h, #6, lsl #8
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.2s, v[[REG1]].2s
  ; CHECK-NEXT:    bl      f_v2i32
  call i32 @f_v2i32(<2 x i32> <i32 100664832, i32 100664832>)
  ; CHECK:         movi    v{{[0-9]+}}.4h, #5, lsl #8
  ; CHECK-NEXT:    bl      f_v1i64
  call i64 @f_v1i64(<1 x i64> <i64 360293467831665920>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].8h, #5, lsl #8
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].16b, v[[REG1]].16b
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v16i8
  call i8 @f_v16i8(<16 x i8> <i8 0, i8 5, i8 0, i8 5, i8 0, i8 5, i8 0, i8 5, i8 0, i8 5, i8 0, i8 5, i8 0, i8 5, i8 0, i8 5>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].8h, #4, lsl #8
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].8h, v[[REG1]].8h
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v8i16
  call i16 @f_v8i16(<8 x i16> <i16 1024, i16 1024, i16 1024, i16 1024, i16 1024, i16 1024, i16 1024, i16 1024>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].8h, #3, lsl #8
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].4s, v[[REG1]].4s
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v4i32
  call i32 @f_v4i32(<4 x i32> <i32 50332416, i32 50332416, i32 50332416, i32 50332416>)
  ; CHECK:         movi    v[[REG:[0-9]+]].8h, #2, lsl #8
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v2i64
  call i64 @f_v2i64(<2 x i64> <i64 144117387132666368, i64 144117387132666368>)

  ret void
}

; CHECK-LABEL: modimm_t7_call:
define void @modimm_t7_call() {
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #8, msl #8
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.8b, v[[REG1]].8b
  ; CHECK-NEXT:    bl      f_v8i8
  call i8 @f_v8i8(<8 x i8> <i8 255, i8 8, i8 0, i8 0, i8 255, i8 8, i8 0, i8 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #7, msl #8
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.4h, v[[REG1]].4h
  ; CHECK-NEXT:    bl      f_v4i16
  call i16 @f_v4i16(<4 x i16> <i16 2047, i16 0, i16 2047, i16 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #6, msl #8
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.2s, v[[REG1]].2s
  ; CHECK-NEXT:    bl      f_v2i32
  call i32 @f_v2i32(<2 x i32> <i32 1791, i32 1791>)
  ; CHECK:         movi    v{{[0-9]+}}.2s, #5, msl #8
  ; CHECK-NEXT:    bl      f_v1i64
  call i64 @f_v1i64(<1 x i64> <i64 6592774800895>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #5, msl #8
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].16b, v[[REG1]].16b
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v16i8
  call i8 @f_v16i8(<16 x i8> <i8 255, i8 5, i8 0, i8 0, i8 255, i8 5, i8 0, i8 0, i8 255, i8 5, i8 0, i8 0, i8 255, i8 5, i8 0, i8 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #4, msl #8
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].8h, v[[REG1]].8h
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v8i16
  call i16 @f_v8i16(<8 x i16> <i16 1279, i16 0, i16 1279, i16 0, i16 1279, i16 0, i16 1279, i16 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #3, msl #8
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].4s, v[[REG1]].4s
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v4i32
  call i32 @f_v4i32(<4 x i32> <i32 1023, i32 1023, i32 1023, i32 1023>)
  ; CHECK:         movi    v[[REG:[0-9]+]].4s, #2, msl #8
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v2i64
  call i64 @f_v2i64(<2 x i64> <i64 3294239916799, i64 3294239916799>)

  ret void
}

; CHECK-LABEL: modimm_t8_call:
define void @modimm_t8_call() {
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #8, msl #16
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.8b, v[[REG1]].8b
  ; CHECK-NEXT:    bl      f_v8i8
  call i8 @f_v8i8(<8 x i8> <i8 255, i8 255, i8 8, i8 0, i8 255, i8 255, i8 8, i8 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #7, msl #16
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.4h, v[[REG1]].4h
  ; CHECK-NEXT:    bl      f_v4i16
  call i16 @f_v4i16(<4 x i16> <i16 65535, i16 7, i16 65535, i16 7>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].2s, #6, msl #16
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.2s, v[[REG1]].2s
  ; CHECK-NEXT:    bl      f_v2i32
  call i32 @f_v2i32(<2 x i32> <i32 458751, i32 458751>)
  ; CHECK:         movi    v{{[0-9]+}}.2s, #5, msl #16
  ; CHECK-NEXT:    bl      f_v1i64
  call i64 @f_v1i64(<1 x i64> <i64 1688845565689855>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #5, msl #16
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].16b, v[[REG1]].16b
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v16i8
  call i8 @f_v16i8(<16 x i8> <i8 255, i8 255, i8 5, i8 0, i8 255, i8 255, i8 5, i8 0, i8 255, i8 255, i8 5, i8 0, i8 255, i8 255, i8 5, i8 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #4, msl #16
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].8h, v[[REG1]].8h
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v8i16
  call i16 @f_v8i16(<8 x i16> <i16 65535, i16 4, i16 65535, i16 4, i16 65535, i16 4, i16 65535, i16 4>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].4s, #3, msl #16
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].4s, v[[REG1]].4s
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v4i32
  call i32 @f_v4i32(<4 x i32> <i32 262143, i32 262143, i32 262143, i32 262143>)
  ; CHECK:         movi    v[[REG:[0-9]+]].4s, #2, msl #16
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v2i64
  call i64 @f_v2i64(<2 x i64> <i64 844420635361279, i64 844420635361279>)

  ret void
}

; CHECK-LABEL: modimm_t9_call:
define void @modimm_t9_call() {
  ; CHECK:         movi    v[[REG1:[0-9]+]].8b, #8
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.8b, v[[REG1]].8b
  ; CHECK-NEXT:    bl      f_v8i8
  call i8 @f_v8i8(<8 x i8> <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].8b, #7
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.4h, v[[REG1]].4h
  ; CHECK-NEXT:    bl      f_v4i16
  call i16 @f_v4i16(<4 x i16> <i16 1799, i16 1799, i16 1799, i16 1799>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].8b, #6
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.2s, v[[REG1]].2s
  ; CHECK-NEXT:    bl      f_v2i32
  call i32 @f_v2i32(<2 x i32> <i32 101058054, i32 101058054>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].16b, #5
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].16b, v[[REG1]].16b
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v16i8
  call i8 @f_v16i8(<16 x i8> <i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5, i8 5>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].16b, #4
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].8h, v[[REG1]].8h
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v8i16
  call i16 @f_v8i16(<8 x i16> <i16 1028, i16 1028, i16 1028, i16 1028, i16 1028, i16 1028, i16 1028, i16 1028>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].16b, #3
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].4s, v[[REG1]].4s
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v4i32
  call i32 @f_v4i32(<4 x i32> <i32 50529027, i32 50529027, i32 50529027, i32 50529027>)

  ret void
}

; CHECK-LABEL: modimm_t10_call:
define void @modimm_t10_call() {
  ; CHECK:         movi    d[[REG1:[0-9]+]], #0x0000ff000000ff
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.8b, v[[REG1]].8b
  ; CHECK-NEXT:    bl      f_v8i8
  call i8 @f_v8i8(<8 x i8> <i8 -1, i8 0, i8 0, i8 0, i8 -1, i8 0, i8 0, i8 0>)
  ; CHECK:         movi    d[[REG1:[0-9]+]], #0x00ffff0000ffff
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.4h, v[[REG1]].4h
  ; CHECK-NEXT:    bl      f_v4i16
  call i16 @f_v4i16(<4 x i16> <i16 -1, i16 0, i16 -1, i16 0>)
  ; CHECK:         movi    d[[REG1:[0-9]+]], #0xffffffffffffffff
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.2s, v[[REG1]].2s
  ; CHECK-NEXT:    bl      f_v2i32
  call i32 @f_v2i32(<2 x i32> <i32 -1, i32 -1>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].2d, #0xffffff00ffffff
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].16b, v[[REG1]].16b
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v16i8
  call i8 @f_v16i8(<16 x i8> <i8 -1, i8 -1, i8 -1, i8 0, i8 -1, i8 -1, i8 -1, i8 0, i8 -1, i8 -1, i8 -1, i8 0, i8 -1, i8 -1, i8 -1, i8 0>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].2d, #0xffffffffffff0000
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].8h, v[[REG1]].8h
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v8i16
  call i16 @f_v8i16(<8 x i16> <i16 0, i16 -1, i16 -1, i16 -1, i16 0, i16 -1, i16 -1, i16 -1>)
  ; CHECK:         movi    v[[REG1:[0-9]+]].2d, #0xffffffff00000000
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].4s, v[[REG1]].4s
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v4i32
  call i32 @f_v4i32(<4 x i32> <i32 0, i32 -1, i32 0, i32 -1>)

  ret void
}

; CHECK-LABEL: modimm_t11_call:
define void @modimm_t11_call() {
  ; CHECK:         fmov    v[[REG1:[0-9]+]].2s, #4.00000000
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.8b, v[[REG1]].8b
  ; CHECK-NEXT:    bl      f_v8i8
  call i8 @f_v8i8(<8 x i8> <i8 0, i8 0, i8 128, i8 64, i8 0, i8 0, i8 128, i8 64>)
  ; CHECK:         fmov    v[[REG1:[0-9]+]].2s, #3.75000000
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.4h, v[[REG1]].4h
  ; CHECK-NEXT:    bl      f_v4i16
  call i16 @f_v4i16(<4 x i16> <i16 0, i16 16496, i16 0, i16 16496>)
  ; CHECK:         fmov    v[[REG1:[0-9]+]].2s, #3.50000000
  ; CHECK-NEXT:    rev64   v{{[0-9]+}}.2s, v[[REG1]].2s
  ; CHECK-NEXT:    bl      f_v2i32
  call i32 @f_v2i32(<2 x i32> <i32 1080033280, i32 1080033280>)
  ; CHECK:         fmov    v{{[0-9]+}}.2s, #0.39062500
  ; CHECK-NEXT:    bl      f_v1i64
  call i64 @f_v1i64(<1 x i64> <i64 4523865826746957824>)
  ; CHECK:         fmov    v[[REG1:[0-9]+]].4s, #3.25000000
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].16b, v[[REG1]].16b
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v16i8
  call i8 @f_v16i8(<16 x i8> <i8 0, i8 0, i8 80, i8 64, i8 0, i8 0, i8 80, i8 64, i8 0, i8 0, i8 80, i8 64, i8 0, i8 0, i8 80, i8 64>)
  ; CHECK:         fmov    v[[REG1:[0-9]+]].4s, #3.00000000
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].8h, v[[REG1]].8h
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v8i16
  call i16 @f_v8i16(<8 x i16> <i16 0, i16 16448, i16 0, i16 16448, i16 0, i16 16448, i16 0, i16 16448>)
  ; CHECK:         fmov    v[[REG1:[0-9]+]].4s, #2.75000000
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].4s, v[[REG1]].4s
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v4i32
  call i32 @f_v4i32(<4 x i32> <i32 1076887552, i32 1076887552, i32 1076887552, i32 1076887552>)
  ; CHECK:         fmov    v[[REG:[0-9]+]].4s, #2.5000000
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v2i64
  call i64 @f_v2i64(<2 x i64> <i64 4620693218757967872, i64 4620693218757967872>)

  ret void
}

; CHECK-LABEL: modimm_t12_call:
define void @modimm_t12_call() {
  ; CHECK:         fmov    v[[REG1:[0-9]+]].2d, #0.18750000
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].16b, v[[REG1]].16b
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v16i8
  call i8 @f_v16i8(<16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 200, i8 63, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 200, i8 63>)
  ; CHECK:         fmov    v[[REG1:[0-9]+]].2d, #0.17968750
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].8h, v[[REG1]].8h
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v8i16
  call i16 @f_v8i16(<8 x i16> <i16 0, i16 0, i16 0, i16 16327, i16 0, i16 0, i16 0, i16 16327>)
  ; CHECK:         fmov    v[[REG1:[0-9]+]].2d, #0.17187500
  ; CHECK-NEXT:    rev64   v[[REG2:[0-9]+]].4s, v[[REG1]].4s
  ; CHECK-NEXT:    ext     v[[REG2]].16b, v[[REG2]].16b, v[[REG2]].16b, #8
  ; CHECK-NEXT:    bl      f_v4i32
  call i32 @f_v4i32(<4 x i32> <i32 0, i32 1069940736, i32 0, i32 1069940736>)

  ret void
}
