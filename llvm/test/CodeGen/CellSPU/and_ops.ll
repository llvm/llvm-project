; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep and    %t1.s | count 234
; RUN: grep andc   %t1.s | count 85
; RUN: grep andi   %t1.s | count 37
; RUN: grep andhi  %t1.s | count 30
; RUN: grep andbi  %t1.s | count 4

target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

; AND instruction generation:
define <4 x i32> @and_v4i32_1(<4 x i32> %arg1, <4 x i32> %arg2) {
        %A = and <4 x i32> %arg1, %arg2
        ret <4 x i32> %A
}

define <4 x i32> @and_v4i32_2(<4 x i32> %arg1, <4 x i32> %arg2) {
        %A = and <4 x i32> %arg2, %arg1
        ret <4 x i32> %A
}

define <8 x i16> @and_v8i16_1(<8 x i16> %arg1, <8 x i16> %arg2) {
        %A = and <8 x i16> %arg1, %arg2
        ret <8 x i16> %A
}

define <8 x i16> @and_v8i16_2(<8 x i16> %arg1, <8 x i16> %arg2) {
        %A = and <8 x i16> %arg2, %arg1
        ret <8 x i16> %A
}

define <16 x i8> @and_v16i8_1(<16 x i8> %arg1, <16 x i8> %arg2) {
        %A = and <16 x i8> %arg2, %arg1
        ret <16 x i8> %A
}

define <16 x i8> @and_v16i8_2(<16 x i8> %arg1, <16 x i8> %arg2) {
        %A = and <16 x i8> %arg1, %arg2
        ret <16 x i8> %A
}

define i32 @and_i32_1(i32 %arg1, i32 %arg2) {
        %A = and i32 %arg2, %arg1
        ret i32 %A
}

define i32 @and_i32_2(i32 %arg1, i32 %arg2) {
        %A = and i32 %arg1, %arg2
        ret i32 %A
}

define i16 @and_i16_1(i16 %arg1, i16 %arg2) {
        %A = and i16 %arg2, %arg1
        ret i16 %A
}

define i16 @and_i16_2(i16 %arg1, i16 %arg2) {
        %A = and i16 %arg1, %arg2
        ret i16 %A
}

define i8 @and_i8_1(i8 %arg1, i8 %arg2) {
        %A = and i8 %arg2, %arg1
        ret i8 %A
}

define i8 @and_i8_2(i8 %arg1, i8 %arg2) {
        %A = and i8 %arg1, %arg2
        ret i8 %A
}

; ANDC instruction generation:
define <4 x i32> @andc_v4i32_1(<4 x i32> %arg1, <4 x i32> %arg2) {
        %A = xor <4 x i32> %arg2, < i32 -1, i32 -1, i32 -1, i32 -1 >
        %B = and <4 x i32> %arg1, %A
        ret <4 x i32> %B
}

define <4 x i32> @andc_v4i32_2(<4 x i32> %arg1, <4 x i32> %arg2) {
        %A = xor <4 x i32> %arg1, < i32 -1, i32 -1, i32 -1, i32 -1 >
        %B = and <4 x i32> %arg2, %A
        ret <4 x i32> %B
}

define <4 x i32> @andc_v4i32_3(<4 x i32> %arg1, <4 x i32> %arg2) {
        %A = xor <4 x i32> %arg1, < i32 -1, i32 -1, i32 -1, i32 -1 >
        %B = and <4 x i32> %A, %arg2
        ret <4 x i32> %B
}

define <8 x i16> @andc_v8i16_1(<8 x i16> %arg1, <8 x i16> %arg2) {
        %A = xor <8 x i16> %arg2, < i16 -1, i16 -1, i16 -1, i16 -1,
                                    i16 -1, i16 -1, i16 -1, i16 -1 >
        %B = and <8 x i16> %arg1, %A
        ret <8 x i16> %B
}

define <8 x i16> @andc_v8i16_2(<8 x i16> %arg1, <8 x i16> %arg2) {
        %A = xor <8 x i16> %arg1, < i16 -1, i16 -1, i16 -1, i16 -1,
                                    i16 -1, i16 -1, i16 -1, i16 -1 >
        %B = and <8 x i16> %arg2, %A
        ret <8 x i16> %B
}

define <16 x i8> @andc_v16i8_1(<16 x i8> %arg1, <16 x i8> %arg2) {
        %A = xor <16 x i8> %arg1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1 >
        %B = and <16 x i8> %arg2, %A
        ret <16 x i8> %B
}

define <16 x i8> @andc_v16i8_2(<16 x i8> %arg1, <16 x i8> %arg2) {
        %A = xor <16 x i8> %arg2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1 >
        %B = and <16 x i8> %arg1, %A
        ret <16 x i8> %B
}

define <16 x i8> @andc_v16i8_3(<16 x i8> %arg1, <16 x i8> %arg2) {
        %A = xor <16 x i8> %arg2, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1,
                                    i8 -1, i8 -1, i8 -1, i8 -1 >
        %B = and <16 x i8> %A, %arg1
        ret <16 x i8> %B
}

define i32 @andc_i32_1(i32 %arg1, i32 %arg2) {
        %A = xor i32 %arg2, -1
        %B = and i32 %A, %arg1
        ret i32 %B
}

define i32 @andc_i32_2(i32 %arg1, i32 %arg2) {
        %A = xor i32 %arg1, -1
        %B = and i32 %A, %arg2
        ret i32 %B
}

define i32 @andc_i32_3(i32 %arg1, i32 %arg2) {
        %A = xor i32 %arg2, -1
        %B = and i32 %arg1, %A
        ret i32 %B
}

define i16 @andc_i16_1(i16 %arg1, i16 %arg2) {
        %A = xor i16 %arg2, -1
        %B = and i16 %A, %arg1
        ret i16 %B
}

define i16 @andc_i16_2(i16 %arg1, i16 %arg2) {
        %A = xor i16 %arg1, -1
        %B = and i16 %A, %arg2
        ret i16 %B
}

define i16 @andc_i16_3(i16 %arg1, i16 %arg2) {
        %A = xor i16 %arg2, -1
        %B = and i16 %arg1, %A
        ret i16 %B
}

define i8 @andc_i8_1(i8 %arg1, i8 %arg2) {
        %A = xor i8 %arg2, -1
        %B = and i8 %A, %arg1
        ret i8 %B
}

define i8 @andc_i8_2(i8 %arg1, i8 %arg2) {
        %A = xor i8 %arg1, -1
        %B = and i8 %A, %arg2
        ret i8 %B
}

define i8 @andc_i8_3(i8 %arg1, i8 %arg2) {
        %A = xor i8 %arg2, -1
        %B = and i8 %arg1, %A
        ret i8 %B
}

; ANDI instruction generation (i32 data type):
define <4 x i32> @andi_v4i32_1(<4 x i32> %in) {
        %tmp2 = and <4 x i32> %in, < i32 511, i32 511, i32 511, i32 511 >
        ret <4 x i32> %tmp2
}

define <4 x i32> @andi_v4i32_2(<4 x i32> %in) {
        %tmp2 = and <4 x i32> %in, < i32 510, i32 510, i32 510, i32 510 >
        ret <4 x i32> %tmp2
}

define <4 x i32> @andi_v4i32_3(<4 x i32> %in) {
        %tmp2 = and <4 x i32> %in, < i32 -1, i32 -1, i32 -1, i32 -1 >
        ret <4 x i32> %tmp2
}

define <4 x i32> @andi_v4i32_4(<4 x i32> %in) {
        %tmp2 = and <4 x i32> %in, < i32 -512, i32 -512, i32 -512, i32 -512 >
        ret <4 x i32> %tmp2
}

define i32 @andi_u32(i32 zeroext  %in) zeroext  {
        %tmp37 = and i32 %in, 37
        ret i32 %tmp37
}

define i32 @andi_i32(i32 signext  %in) signext  {
        %tmp38 = and i32 %in, 37
        ret i32 %tmp38
}

define i32 @andi_i32_1(i32 %in) {
        %tmp37 = and i32 %in, 37
        ret i32 %tmp37
}

; ANDHI instruction generation (i16 data type):
define <8 x i16> @andhi_v8i16_1(<8 x i16> %in) {
        %tmp2 = and <8 x i16> %in, < i16 511, i16 511, i16 511, i16 511,
                                     i16 511, i16 511, i16 511, i16 511 >
        ret <8 x i16> %tmp2
}

define <8 x i16> @andhi_v8i16_2(<8 x i16> %in) {
        %tmp2 = and <8 x i16> %in, < i16 510, i16 510, i16 510, i16 510,
                                     i16 510, i16 510, i16 510, i16 510 >
        ret <8 x i16> %tmp2
}

define <8 x i16> @andhi_v8i16_3(<8 x i16> %in) {
        %tmp2 = and <8 x i16> %in, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1,
                                     i16 -1, i16 -1, i16 -1 >
        ret <8 x i16> %tmp2
}

define <8 x i16> @andhi_v8i16_4(<8 x i16> %in) {
        %tmp2 = and <8 x i16> %in, < i16 -512, i16 -512, i16 -512, i16 -512,
                                     i16 -512, i16 -512, i16 -512, i16 -512 >
        ret <8 x i16> %tmp2
}

define i16 @andhi_u16(i16 zeroext  %in) zeroext  {
        %tmp37 = and i16 %in, 37         ; <i16> [#uses=1]
        ret i16 %tmp37
}

define i16 @andhi_i16(i16 signext  %in) signext  {
        %tmp38 = and i16 %in, 37         ; <i16> [#uses=1]
        ret i16 %tmp38
}

; i8 data type (s/b ANDBI if 8-bit registers were supported):
define <16 x i8> @and_v16i8(<16 x i8> %in) {
        ; ANDBI generated for vector types
        %tmp2 = and <16 x i8> %in, < i8 42, i8 42, i8 42, i8 42, i8 42, i8 42,
                                     i8 42, i8 42, i8 42, i8 42, i8 42, i8 42,
                                     i8 42, i8 42, i8 42, i8 42 >
        ret <16 x i8> %tmp2
}

define i8 @and_u8(i8 zeroext  %in) zeroext  {
        ; ANDBI generated:
        %tmp37 = and i8 %in, 37
        ret i8 %tmp37
}

define i8 @and_sext8(i8 signext  %in) signext  {
        ; ANDBI generated
        %tmp38 = and i8 %in, 37
        ret i8 %tmp38
}

define i8 @and_i8(i8 %in) {
        ; ANDBI generated
        %tmp38 = and i8 %in, 205
        ret i8 %tmp38
}
