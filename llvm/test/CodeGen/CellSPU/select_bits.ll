; RUN: llc < %s -march=cellspu > %t1.s
; RUN: grep selb   %t1.s | count 56

; CellSPU legalization is over-sensitive to Legalize's traversal order.
; XFAIL: *

target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

;-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
; v2i64
;-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

; (or (and rC, rB), (and (not rC), rA))
define <2 x i64> @selectbits_v2i64_01(<2 x i64> %rA, <2 x i64> %rB, <2 x i64> %rC) {
        %C = and <2 x i64> %rC, %rB
        %A = xor <2 x i64> %rC, < i64 -1, i64 -1 >
        %B = and <2 x i64> %A, %rA
        %D = or <2 x i64> %C, %B
        ret <2 x i64> %D
}

; (or (and rB, rC), (and (not rC), rA))
define <2 x i64> @selectbits_v2i64_02(<2 x i64> %rA, <2 x i64> %rB, <2 x i64> %rC) {
        %C = and <2 x i64> %rB, %rC
        %A = xor <2 x i64> %rC, < i64 -1, i64 -1 >
        %B = and <2 x i64> %A, %rA
        %D = or <2 x i64> %C, %B
        ret <2 x i64> %D
}

; (or (and (not rC), rA), (and rB, rC))
define <2 x i64> @selectbits_v2i64_03(<2 x i64> %rA, <2 x i64> %rB, <2 x i64> %rC) {
        %A = xor <2 x i64> %rC, < i64 -1, i64 -1 >
        %B = and <2 x i64> %A, %rA
        %C = and <2 x i64> %rB, %rC
        %D = or <2 x i64> %C, %B
        ret <2 x i64> %D
}

; (or (and (not rC), rA), (and rC, rB))
define <2 x i64> @selectbits_v2i64_04(<2 x i64> %rA, <2 x i64> %rB, <2 x i64> %rC) {
        %A = xor <2 x i64> %rC, < i64 -1, i64 -1 >
        %B = and <2 x i64> %A, %rA
        %C = and <2 x i64> %rC, %rB
        %D = or <2 x i64> %C, %B
        ret <2 x i64> %D
}

; (or (and rC, rB), (and rA, (not rC)))
define <2 x i64> @selectbits_v2i64_05(<2 x i64> %rA, <2 x i64> %rB, <2 x i64> %rC) {
        %C = and <2 x i64> %rC, %rB
        %A = xor <2 x i64> %rC, < i64 -1, i64 -1 >
        %B = and <2 x i64> %rA, %A
        %D = or <2 x i64> %C, %B
        ret <2 x i64> %D
}

; (or (and rB, rC), (and rA, (not rC)))
define <2 x i64> @selectbits_v2i64_06(<2 x i64> %rA, <2 x i64> %rB, <2 x i64> %rC) {
        %C = and <2 x i64> %rB, %rC
        %A = xor <2 x i64> %rC, < i64 -1, i64 -1 >
        %B = and <2 x i64> %rA, %A
        %D = or <2 x i64> %C, %B
        ret <2 x i64> %D
}

; (or (and rA, (not rC)), (and rB, rC))
define <2 x i64> @selectbits_v2i64_07(<2 x i64> %rA, <2 x i64> %rB, <2 x i64> %rC) {
        %A = xor <2 x i64> %rC, < i64 -1, i64 -1 >
        %B = and <2 x i64> %rA, %A
        %C = and <2 x i64> %rB, %rC
        %D = or <2 x i64> %C, %B
        ret <2 x i64> %D
}

; (or (and rA, (not rC)), (and rC, rB))
define <2 x i64> @selectbits_v2i64_08(<2 x i64> %rA, <2 x i64> %rB, <2 x i64> %rC) {
        %A = xor <2 x i64> %rC, < i64 -1, i64 -1 >
        %B = and <2 x i64> %rA, %A
        %C = and <2 x i64> %rC, %rB
        %D = or <2 x i64> %C, %B
        ret <2 x i64> %D
}

;-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
; v4i32
;-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

; (or (and rC, rB), (and (not rC), rA))
define <4 x i32> @selectbits_v4i32_01(<4 x i32> %rA, <4 x i32> %rB, <4 x i32> %rC) {
        %C = and <4 x i32> %rC, %rB
        %A = xor <4 x i32> %rC, < i32 -1, i32 -1, i32 -1, i32 -1 >
        %B = and <4 x i32> %A, %rA
        %D = or <4 x i32> %C, %B
        ret <4 x i32> %D
}

; (or (and rB, rC), (and (not rC), rA))
define <4 x i32> @selectbits_v4i32_02(<4 x i32> %rA, <4 x i32> %rB, <4 x i32> %rC) {
        %C = and <4 x i32> %rB, %rC
        %A = xor <4 x i32> %rC, < i32 -1, i32 -1, i32 -1, i32 -1 >
        %B = and <4 x i32> %A, %rA
        %D = or <4 x i32> %C, %B
        ret <4 x i32> %D
}

; (or (and (not rC), rA), (and rB, rC))
define <4 x i32> @selectbits_v4i32_03(<4 x i32> %rA, <4 x i32> %rB, <4 x i32> %rC) {
        %A = xor <4 x i32> %rC, < i32 -1, i32 -1, i32 -1, i32 -1 >
        %B = and <4 x i32> %A, %rA
        %C = and <4 x i32> %rB, %rC
        %D = or <4 x i32> %C, %B
        ret <4 x i32> %D
}

; (or (and (not rC), rA), (and rC, rB))
define <4 x i32> @selectbits_v4i32_04(<4 x i32> %rA, <4 x i32> %rB, <4 x i32> %rC) {
        %A = xor <4 x i32> %rC, < i32 -1, i32 -1, i32 -1, i32 -1>
        %B = and <4 x i32> %A, %rA
        %C = and <4 x i32> %rC, %rB
        %D = or <4 x i32> %C, %B
        ret <4 x i32> %D
}

; (or (and rC, rB), (and rA, (not rC)))
define <4 x i32> @selectbits_v4i32_05(<4 x i32> %rA, <4 x i32> %rB, <4 x i32> %rC) {
        %C = and <4 x i32> %rC, %rB
        %A = xor <4 x i32> %rC, < i32 -1, i32 -1, i32 -1, i32 -1>
        %B = and <4 x i32> %rA, %A
        %D = or <4 x i32> %C, %B
        ret <4 x i32> %D
}

; (or (and rB, rC), (and rA, (not rC)))
define <4 x i32> @selectbits_v4i32_06(<4 x i32> %rA, <4 x i32> %rB, <4 x i32> %rC) {
        %C = and <4 x i32> %rB, %rC
        %A = xor <4 x i32> %rC, < i32 -1, i32 -1, i32 -1, i32 -1>
        %B = and <4 x i32> %rA, %A
        %D = or <4 x i32> %C, %B
        ret <4 x i32> %D
}

; (or (and rA, (not rC)), (and rB, rC))
define <4 x i32> @selectbits_v4i32_07(<4 x i32> %rA, <4 x i32> %rB, <4 x i32> %rC) {
        %A = xor <4 x i32> %rC, < i32 -1, i32 -1, i32 -1, i32 -1>
        %B = and <4 x i32> %rA, %A
        %C = and <4 x i32> %rB, %rC
        %D = or <4 x i32> %C, %B
        ret <4 x i32> %D
}

; (or (and rA, (not rC)), (and rC, rB))
define <4 x i32> @selectbits_v4i32_08(<4 x i32> %rA, <4 x i32> %rB, <4 x i32> %rC) {
        %A = xor <4 x i32> %rC, < i32 -1, i32 -1, i32 -1, i32 -1>
        %B = and <4 x i32> %rA, %A
        %C = and <4 x i32> %rC, %rB
        %D = or <4 x i32> %C, %B
        ret <4 x i32> %D
}

;-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
; v8i16
;-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

; (or (and rC, rB), (and (not rC), rA))
define <8 x i16> @selectbits_v8i16_01(<8 x i16> %rA, <8 x i16> %rB, <8 x i16> %rC) {
        %C = and <8 x i16> %rC, %rB
        %A = xor <8 x i16> %rC, < i16 -1, i16 -1, i16 -1, i16 -1,
                                  i16 -1, i16 -1, i16 -1, i16 -1 >
        %B = and <8 x i16> %A, %rA
        %D = or <8 x i16> %C, %B
        ret <8 x i16> %D
}

; (or (and rB, rC), (and (not rC), rA))
define <8 x i16> @selectbits_v8i16_02(<8 x i16> %rA, <8 x i16> %rB, <8 x i16> %rC) {
        %C = and <8 x i16> %rB, %rC
        %A = xor <8 x i16> %rC, < i16 -1, i16 -1, i16 -1, i16 -1,
                                  i16 -1, i16 -1, i16 -1, i16 -1 >
        %B = and <8 x i16> %A, %rA
        %D = or <8 x i16> %C, %B
        ret <8 x i16> %D
}

; (or (and (not rC), rA), (and rB, rC))
define <8 x i16> @selectbits_v8i16_03(<8 x i16> %rA, <8 x i16> %rB, <8 x i16> %rC) {
        %A = xor <8 x i16> %rC, < i16 -1, i16 -1, i16 -1, i16 -1,
                                  i16 -1, i16 -1, i16 -1, i16 -1 >
        %B = and <8 x i16> %A, %rA
        %C = and <8 x i16> %rB, %rC
        %D = or <8 x i16> %C, %B
        ret <8 x i16> %D
}

; (or (and (not rC), rA), (and rC, rB))
define <8 x i16> @selectbits_v8i16_04(<8 x i16> %rA, <8 x i16> %rB, <8 x i16> %rC) {
        %A = xor <8 x i16> %rC, < i16 -1, i16 -1, i16 -1, i16 -1,
                                  i16 -1, i16 -1, i16 -1, i16 -1 >
        %B = and <8 x i16> %A, %rA
        %C = and <8 x i16> %rC, %rB
        %D = or <8 x i16> %C, %B
        ret <8 x i16> %D
}

; (or (and rC, rB), (and rA, (not rC)))
define <8 x i16> @selectbits_v8i16_05(<8 x i16> %rA, <8 x i16> %rB, <8 x i16> %rC) {
        %C = and <8 x i16> %rC, %rB
        %A = xor <8 x i16> %rC, < i16 -1, i16 -1, i16 -1, i16 -1,
                                  i16 -1, i16 -1, i16 -1, i16 -1 >
        %B = and <8 x i16> %rA, %A
        %D = or <8 x i16> %C, %B
        ret <8 x i16> %D
}

; (or (and rB, rC), (and rA, (not rC)))
define <8 x i16> @selectbits_v8i16_06(<8 x i16> %rA, <8 x i16> %rB, <8 x i16> %rC) {
        %C = and <8 x i16> %rB, %rC
        %A = xor <8 x i16> %rC, < i16 -1, i16 -1, i16 -1, i16 -1,
                                  i16 -1, i16 -1, i16 -1, i16 -1 >
        %B = and <8 x i16> %rA, %A
        %D = or <8 x i16> %C, %B
        ret <8 x i16> %D
}

; (or (and rA, (not rC)), (and rB, rC))
define <8 x i16> @selectbits_v8i16_07(<8 x i16> %rA, <8 x i16> %rB, <8 x i16> %rC) {
        %A = xor <8 x i16> %rC, < i16 -1, i16 -1, i16 -1, i16 -1,
                                  i16 -1, i16 -1, i16 -1, i16 -1 >
        %B = and <8 x i16> %rA, %A
        %C = and <8 x i16> %rB, %rC
        %D = or <8 x i16> %C, %B
        ret <8 x i16> %D
}

; (or (and rA, (not rC)), (and rC, rB))
define <8 x i16> @selectbits_v8i16_08(<8 x i16> %rA, <8 x i16> %rB, <8 x i16> %rC) {
        %A = xor <8 x i16> %rC, < i16 -1, i16 -1, i16 -1, i16 -1,
                                  i16 -1, i16 -1, i16 -1, i16 -1 >
        %B = and <8 x i16> %rA, %A
        %C = and <8 x i16> %rC, %rB
        %D = or <8 x i16> %C, %B
        ret <8 x i16> %D
}

;-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
; v16i8
;-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

; (or (and rC, rB), (and (not rC), rA))
define <16 x i8> @selectbits_v16i8_01(<16 x i8> %rA, <16 x i8> %rB, <16 x i8> %rC) {
        %C = and <16 x i8> %rC, %rB
        %A = xor <16 x i8> %rC, < i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1 >
        %B = and <16 x i8> %A, %rA
        %D = or <16 x i8> %C, %B
        ret <16 x i8> %D
}

; (or (and rB, rC), (and (not rC), rA))
define <16 x i8> @selectbits_v16i8_02(<16 x i8> %rA, <16 x i8> %rB, <16 x i8> %rC) {
        %C = and <16 x i8> %rB, %rC
        %A = xor <16 x i8> %rC, < i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1 >
        %B = and <16 x i8> %A, %rA
        %D = or <16 x i8> %C, %B
        ret <16 x i8> %D
}

; (or (and (not rC), rA), (and rB, rC))
define <16 x i8> @selectbits_v16i8_03(<16 x i8> %rA, <16 x i8> %rB, <16 x i8> %rC) {
        %A = xor <16 x i8> %rC, < i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1 >
        %B = and <16 x i8> %A, %rA
        %C = and <16 x i8> %rB, %rC
        %D = or <16 x i8> %C, %B
        ret <16 x i8> %D
}

; (or (and (not rC), rA), (and rC, rB))
define <16 x i8> @selectbits_v16i8_04(<16 x i8> %rA, <16 x i8> %rB, <16 x i8> %rC) {
        %A = xor <16 x i8> %rC, < i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1 >
        %B = and <16 x i8> %A, %rA
        %C = and <16 x i8> %rC, %rB
        %D = or <16 x i8> %C, %B
        ret <16 x i8> %D
}

; (or (and rC, rB), (and rA, (not rC)))
define <16 x i8> @selectbits_v16i8_05(<16 x i8> %rA, <16 x i8> %rB, <16 x i8> %rC) {
        %C = and <16 x i8> %rC, %rB
        %A = xor <16 x i8> %rC, < i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1 >
        %B = and <16 x i8> %rA, %A
        %D = or <16 x i8> %C, %B
        ret <16 x i8> %D
}

; (or (and rB, rC), (and rA, (not rC)))
define <16 x i8> @selectbits_v16i8_06(<16 x i8> %rA, <16 x i8> %rB, <16 x i8> %rC) {
        %C = and <16 x i8> %rB, %rC
        %A = xor <16 x i8> %rC, < i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1 >
        %B = and <16 x i8> %rA, %A
        %D = or <16 x i8> %C, %B
        ret <16 x i8> %D
}

; (or (and rA, (not rC)), (and rB, rC))
define <16 x i8> @selectbits_v16i8_07(<16 x i8> %rA, <16 x i8> %rB, <16 x i8> %rC) {
        %A = xor <16 x i8> %rC, < i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1 >
        %B = and <16 x i8> %rA, %A
        %C = and <16 x i8> %rB, %rC
        %D = or <16 x i8> %C, %B
        ret <16 x i8> %D
}

; (or (and rA, (not rC)), (and rC, rB))
define <16 x i8> @selectbits_v16i8_08(<16 x i8> %rA, <16 x i8> %rB, <16 x i8> %rC) {
        %A = xor <16 x i8> %rC, < i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1,
                                  i8 -1, i8 -1, i8 -1, i8 -1 >
        %B = and <16 x i8> %rA, %A
        %C = and <16 x i8> %rC, %rB
        %D = or <16 x i8> %C, %B
        ret <16 x i8> %D
}

;-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
; i32
;-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

; (or (and rC, rB), (and (not rC), rA))
define i32 @selectbits_i32_01(i32 %rA, i32 %rB, i32 %rC) {
        %C = and i32 %rC, %rB
        %A = xor i32 %rC, -1
        %B = and i32 %A, %rA
        %D = or i32 %C, %B
        ret i32 %D
}

; (or (and rB, rC), (and (not rC), rA))
define i32 @selectbits_i32_02(i32 %rA, i32 %rB, i32 %rC) {
        %C = and i32 %rB, %rC
        %A = xor i32 %rC, -1
        %B = and i32 %A, %rA
        %D = or i32 %C, %B
        ret i32 %D
}

; (or (and (not rC), rA), (and rB, rC))
define i32 @selectbits_i32_03(i32 %rA, i32 %rB, i32 %rC) {
        %A = xor i32 %rC, -1
        %B = and i32 %A, %rA
        %C = and i32 %rB, %rC
        %D = or i32 %C, %B
        ret i32 %D
}

; (or (and (not rC), rA), (and rC, rB))
define i32 @selectbits_i32_04(i32 %rA, i32 %rB, i32 %rC) {
        %A = xor i32 %rC, -1
        %B = and i32 %A, %rA
        %C = and i32 %rC, %rB
        %D = or i32 %C, %B
        ret i32 %D
}

; (or (and rC, rB), (and rA, (not rC)))
define i32 @selectbits_i32_05(i32 %rA, i32 %rB, i32 %rC) {
        %C = and i32 %rC, %rB
        %A = xor i32 %rC, -1
        %B = and i32 %rA, %A
        %D = or i32 %C, %B
        ret i32 %D
}

; (or (and rB, rC), (and rA, (not rC)))
define i32 @selectbits_i32_06(i32 %rA, i32 %rB, i32 %rC) {
        %C = and i32 %rB, %rC
        %A = xor i32 %rC, -1
        %B = and i32 %rA, %A
        %D = or i32 %C, %B
        ret i32 %D
}

; (or (and rA, (not rC)), (and rB, rC))
define i32 @selectbits_i32_07(i32 %rA, i32 %rB, i32 %rC) {
        %A = xor i32 %rC, -1
        %B = and i32 %rA, %A
        %C = and i32 %rB, %rC
        %D = or i32 %C, %B
        ret i32 %D
}

; (or (and rA, (not rC)), (and rC, rB))
define i32 @selectbits_i32_08(i32 %rA, i32 %rB, i32 %rC) {
        %A = xor i32 %rC, -1
        %B = and i32 %rA, %A
        %C = and i32 %rC, %rB
        %D = or i32 %C, %B
        ret i32 %D
}

;-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
; i16
;-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

; (or (and rC, rB), (and (not rC), rA))
define i16 @selectbits_i16_01(i16 %rA, i16 %rB, i16 %rC) {
        %C = and i16 %rC, %rB
        %A = xor i16 %rC, -1
        %B = and i16 %A, %rA
        %D = or i16 %C, %B
        ret i16 %D
}

; (or (and rB, rC), (and (not rC), rA))
define i16 @selectbits_i16_02(i16 %rA, i16 %rB, i16 %rC) {
        %C = and i16 %rB, %rC
        %A = xor i16 %rC, -1
        %B = and i16 %A, %rA
        %D = or i16 %C, %B
        ret i16 %D
}

; (or (and (not rC), rA), (and rB, rC))
define i16 @selectbits_i16_03(i16 %rA, i16 %rB, i16 %rC) {
        %A = xor i16 %rC, -1
        %B = and i16 %A, %rA
        %C = and i16 %rB, %rC
        %D = or i16 %C, %B
        ret i16 %D
}

; (or (and (not rC), rA), (and rC, rB))
define i16 @selectbits_i16_04(i16 %rA, i16 %rB, i16 %rC) {
        %A = xor i16 %rC, -1
        %B = and i16 %A, %rA
        %C = and i16 %rC, %rB
        %D = or i16 %C, %B
        ret i16 %D
}

; (or (and rC, rB), (and rA, (not rC)))
define i16 @selectbits_i16_05(i16 %rA, i16 %rB, i16 %rC) {
        %C = and i16 %rC, %rB
        %A = xor i16 %rC, -1
        %B = and i16 %rA, %A
        %D = or i16 %C, %B
        ret i16 %D
}

; (or (and rB, rC), (and rA, (not rC)))
define i16 @selectbits_i16_06(i16 %rA, i16 %rB, i16 %rC) {
        %C = and i16 %rB, %rC
        %A = xor i16 %rC, -1
        %B = and i16 %rA, %A
        %D = or i16 %C, %B
        ret i16 %D
}

; (or (and rA, (not rC)), (and rB, rC))
define i16 @selectbits_i16_07(i16 %rA, i16 %rB, i16 %rC) {
        %A = xor i16 %rC, -1
        %B = and i16 %rA, %A
        %C = and i16 %rB, %rC
        %D = or i16 %C, %B
        ret i16 %D
}

; (or (and rA, (not rC)), (and rC, rB))
define i16 @selectbits_i16_08(i16 %rA, i16 %rB, i16 %rC) {
        %A = xor i16 %rC, -1
        %B = and i16 %rA, %A
        %C = and i16 %rC, %rB
        %D = or i16 %C, %B
        ret i16 %D
}

;-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
; i8
;-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

; (or (and rC, rB), (and (not rC), rA))
define i8 @selectbits_i8_01(i8 %rA, i8 %rB, i8 %rC) {
        %C = and i8 %rC, %rB
        %A = xor i8 %rC, -1
        %B = and i8 %A, %rA
        %D = or i8 %C, %B
        ret i8 %D
}

; (or (and rB, rC), (and (not rC), rA))
define i8 @selectbits_i8_02(i8 %rA, i8 %rB, i8 %rC) {
        %C = and i8 %rB, %rC
        %A = xor i8 %rC, -1
        %B = and i8 %A, %rA
        %D = or i8 %C, %B
        ret i8 %D
}

; (or (and (not rC), rA), (and rB, rC))
define i8 @selectbits_i8_03(i8 %rA, i8 %rB, i8 %rC) {
        %A = xor i8 %rC, -1
        %B = and i8 %A, %rA
        %C = and i8 %rB, %rC
        %D = or i8 %C, %B
        ret i8 %D
}

; (or (and (not rC), rA), (and rC, rB))
define i8 @selectbits_i8_04(i8 %rA, i8 %rB, i8 %rC) {
        %A = xor i8 %rC, -1
        %B = and i8 %A, %rA
        %C = and i8 %rC, %rB
        %D = or i8 %C, %B
        ret i8 %D
}

; (or (and rC, rB), (and rA, (not rC)))
define i8 @selectbits_i8_05(i8 %rA, i8 %rB, i8 %rC) {
        %C = and i8 %rC, %rB
        %A = xor i8 %rC, -1
        %B = and i8 %rA, %A
        %D = or i8 %C, %B
        ret i8 %D
}

; (or (and rB, rC), (and rA, (not rC)))
define i8 @selectbits_i8_06(i8 %rA, i8 %rB, i8 %rC) {
        %C = and i8 %rB, %rC
        %A = xor i8 %rC, -1
        %B = and i8 %rA, %A
        %D = or i8 %C, %B
        ret i8 %D
}

; (or (and rA, (not rC)), (and rB, rC))
define i8 @selectbits_i8_07(i8 %rA, i8 %rB, i8 %rC) {
        %A = xor i8 %rC, -1
        %B = and i8 %rA, %A
        %C = and i8 %rB, %rC
        %D = or i8 %C, %B
        ret i8 %D
}

; (or (and rA, (not rC)), (and rC, rB))
define i8 @selectbits_i8_08(i8 %rA, i8 %rB, i8 %rC) {
        %A = xor i8 %rC, -1
        %B = and i8 %rA, %A
        %C = and i8 %rC, %rB
        %D = or i8 %C, %B
        ret i8 %D
}
