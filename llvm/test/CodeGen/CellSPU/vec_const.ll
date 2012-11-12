; RUN: llc < %s -march=cellspu > %t1.s
; RUN: llc < %s -march=cellspu -mattr=large_mem > %t2.s
; RUN: grep -w il  %t1.s | count 3
; RUN: grep ilhu   %t1.s | count 8
; RUN: grep -w ilh %t1.s | count 5
; RUN: grep iohl   %t1.s | count 7
; RUN: grep lqa    %t1.s | count 6
; RUN: grep 24672  %t1.s | count 2
; RUN: grep 16429  %t1.s | count 1
; RUN: grep 63572  %t1.s | count 1
; RUN: grep  4660  %t1.s | count 1
; RUN: grep 22136  %t1.s | count 1
; RUN: grep 43981  %t1.s | count 1
; RUN: grep 61202  %t1.s | count 1
; RUN: grep 16393  %t1.s | count 1
; RUN: grep  8699  %t1.s | count 1
; RUN: grep 21572  %t1.s | count 1
; RUN: grep 11544  %t1.s | count 1
; RUN: grep 1311768467750121234 %t1.s | count 1
; RUN: grep lqd    %t2.s | count 6

target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128"
target triple = "spu-unknown-elf"

; Vector constant load tests:

; IL <reg>, 2
define <4 x i32> @v4i32_constvec() {
        ret <4 x i32> < i32 2, i32 2, i32 2, i32 2 >
}

; Spill to constant pool
define <4 x i32> @v4i32_constpool() {
        ret <4 x i32> < i32 2, i32 1, i32 1, i32 2 >
}

; Max negative range for IL
define <4 x i32> @v4i32_constvec_2() {
        ret <4 x i32> < i32 -32768, i32 -32768, i32 -32768, i32 -32768 >
}

; ILHU <reg>, 73 (0x49)
; 4784128 = 0x490000
define <4 x i32> @v4i32_constvec_3() {
        ret <4 x i32> < i32 4784128, i32 4784128,
                        i32 4784128, i32 4784128 >
}

; ILHU <reg>, 61 (0x3d)
; IOHL <reg>, 15395 (0x3c23)
define <4 x i32> @v4i32_constvec_4() {
        ret <4 x i32> < i32 4013091, i32 4013091,
                        i32 4013091, i32 4013091 >
}

; ILHU <reg>, 0x5050 (20560)
; IOHL <reg>, 0x5050 (20560)
; Tests for whether we expand the size of the bit pattern properly, because
; this could be interpreted as an i8 pattern (0x50)
define <4 x i32> @v4i32_constvec_5() {
        ret <4 x i32> < i32 1347440720, i32 1347440720,
                        i32 1347440720, i32 1347440720 >
}

; ILH
define <8 x i16> @v8i16_constvec_1() {
        ret <8 x i16> < i16 32767, i16 32767, i16 32767, i16 32767,
                        i16 32767, i16 32767, i16 32767, i16 32767 >
}

; ILH
define <8 x i16> @v8i16_constvec_2() {
        ret <8 x i16> < i16 511, i16 511, i16 511, i16 511, i16 511,
                        i16 511, i16 511, i16 511 >
}

; ILH
define <8 x i16> @v8i16_constvec_3() {
        ret <8 x i16> < i16 -512, i16 -512, i16 -512, i16 -512, i16 -512,
                        i16 -512, i16 -512, i16 -512 >
}

; ILH <reg>, 24672 (0x6060)
; Tests whether we expand the size of the bit pattern properly, because
; this could be interpreted as an i8 pattern (0x60)
define <8 x i16> @v8i16_constvec_4() {
        ret <8 x i16> < i16 24672, i16 24672, i16 24672, i16 24672, i16 24672,
                        i16 24672, i16 24672, i16 24672 >
}

; ILH <reg>, 24672 (0x6060)
; Tests whether we expand the size of the bit pattern properly, because
; this is an i8 pattern but has to be expanded out to i16 to load it
; properly into the vector register.
define <16 x i8> @v16i8_constvec_1() {
        ret <16 x i8> < i8 96, i8 96, i8 96, i8 96, i8 96, i8 96, i8 96, i8 96,
                        i8 96, i8 96, i8 96, i8 96, i8 96, i8 96, i8 96, i8 96 >
}

define <4 x float> @v4f32_constvec_1() {
entry:
        ret <4 x float> < float 0x4005BF0A80000000,
                          float 0x4005BF0A80000000,
                          float 0x4005BF0A80000000,
                          float 0x4005BF0A80000000 >
}

define <4 x float> @v4f32_constvec_2() {
entry:
        ret <4 x float> < float 0.000000e+00,
                          float 0.000000e+00,
                          float 0.000000e+00,
                          float 0.000000e+00 >
}


define <4 x float> @v4f32_constvec_3() {
entry:
        ret <4 x float> < float 0x4005BF0A80000000,
                          float 0x3810000000000000,
                          float 0x47EFFFFFE0000000,
                          float 0x400921FB60000000 >
}

;  1311768467750121234 => 0x 12345678 abcdef12
;  HI32_hi:  4660
;  HI32_lo: 22136
;  LO32_hi: 43981
;  LO32_lo: 61202
define <2 x i64> @i64_constvec_1() {
entry:
        ret <2 x i64> < i64 1311768467750121234,
                        i64 1311768467750121234 >
}

define <2 x i64> @i64_constvec_2() {
entry:
        ret <2 x i64> < i64 1, i64 1311768467750121234 >
}

define <2 x double> @f64_constvec_1() {
entry:
 ret <2 x double> < double 0x400921fb54442d18,
                    double 0xbff6a09e667f3bcd >
}

; 0x400921fb 54442d18 ->
;   (ILHU 0x4009 [16393]/IOHL 0x21fb [ 8699])
;   (ILHU 0x5444 [21572]/IOHL 0x2d18 [11544])
define <2 x double> @f64_constvec_2() {
entry:
 ret <2 x double> < double 0x400921fb54442d18,
                    double 0x400921fb54442d18 >
}
