; RUN: llc < %s -march=cellspu > %t1.s
; RUN: grep shufb   %t1.s | count 39
; RUN: grep ilhu    %t1.s | count 27
; RUN: grep iohl    %t1.s | count 27
; RUN: grep lqa     %t1.s | count 10
; RUN: grep shlqby  %t1.s | count 12
; RUN: grep   515   %t1.s | count 1
; RUN: grep  1029   %t1.s | count 2
; RUN: grep  1543   %t1.s | count 2
; RUN: grep  2057   %t1.s | count 2
; RUN: grep  2571   %t1.s | count 2
; RUN: grep  3085   %t1.s | count 2
; RUN: grep  3599   %t1.s | count 2
; RUN: grep 32768   %t1.s | count 1
; RUN: grep 32769   %t1.s | count 1
; RUN: grep 32770   %t1.s | count 1
; RUN: grep 32771   %t1.s | count 1
; RUN: grep 32772   %t1.s | count 1
; RUN: grep 32773   %t1.s | count 1
; RUN: grep 32774   %t1.s | count 1
; RUN: grep 32775   %t1.s | count 1
; RUN: grep 32776   %t1.s | count 1
; RUN: grep 32777   %t1.s | count 1
; RUN: grep 32778   %t1.s | count 1
; RUN: grep 32779   %t1.s | count 1
; RUN: grep 32780   %t1.s | count 1
; RUN: grep 32781   %t1.s | count 1
; RUN: grep 32782   %t1.s | count 1
; RUN: grep 32783   %t1.s | count 1
; RUN: grep 32896   %t1.s | count 24

target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define i32 @i32_extract_0(<4 x i32> %v) {
entry:
  %a = extractelement <4 x i32> %v, i32 0
  ret i32 %a
}

define i32 @i32_extract_1(<4 x i32> %v) {
entry:
  %a = extractelement <4 x i32> %v, i32 1
  ret i32 %a
}

define i32 @i32_extract_2(<4 x i32> %v) {
entry:
  %a = extractelement <4 x i32> %v, i32 2
  ret i32 %a
}

define i32 @i32_extract_3(<4 x i32> %v) {
entry:
  %a = extractelement <4 x i32> %v, i32 3
  ret i32 %a
}

define i16 @i16_extract_0(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 0
  ret i16 %a
}

define i16 @i16_extract_1(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 1
  ret i16 %a
}

define i16 @i16_extract_2(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 2
  ret i16 %a
}

define i16 @i16_extract_3(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 3
  ret i16 %a
}

define i16 @i16_extract_4(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 4
  ret i16 %a
}

define i16 @i16_extract_5(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 5
  ret i16 %a
}

define i16 @i16_extract_6(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 6
  ret i16 %a
}

define i16 @i16_extract_7(<8 x i16> %v) {
entry:
  %a = extractelement <8 x i16> %v, i32 7
  ret i16 %a
}

define i8 @i8_extract_0(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 0
  ret i8 %a
}

define i8 @i8_extract_1(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 1
  ret i8 %a
}

define i8 @i8_extract_2(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 2
  ret i8 %a
}

define i8 @i8_extract_3(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 3
  ret i8 %a
}

define i8 @i8_extract_4(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 4
  ret i8 %a
}

define i8 @i8_extract_5(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 5
  ret i8 %a
}

define i8 @i8_extract_6(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 6
  ret i8 %a
}

define i8 @i8_extract_7(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 7
  ret i8 %a
}

define i8 @i8_extract_8(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 8
  ret i8 %a
}

define i8 @i8_extract_9(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 9
  ret i8 %a
}

define i8 @i8_extract_10(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 10
  ret i8 %a
}

define i8 @i8_extract_11(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 11
  ret i8 %a
}

define i8 @i8_extract_12(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 12
  ret i8 %a
}

define i8 @i8_extract_13(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 13
  ret i8 %a
}

define i8 @i8_extract_14(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 14
  ret i8 %a
}

define i8 @i8_extract_15(<16 x i8> %v) {
entry:
  %a = extractelement <16 x i8> %v, i32 15
  ret i8 %a
}

;;--------------------------------------------------------------------------
;; extract element, variable index:
;;--------------------------------------------------------------------------

define i8 @extract_varadic_i8(i32 %i) nounwind readnone {
entry:
        %0 = extractelement <16 x i8> < i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, i32 %i
        ret i8 %0
}

define i8 @extract_varadic_i8_1(<16 x i8> %v, i32 %i) nounwind readnone {
entry:
        %0 = extractelement <16 x i8> %v, i32 %i
        ret i8 %0
}

define i16 @extract_varadic_i16(i32 %i) nounwind readnone {
entry:
        %0 = extractelement <8 x i16> < i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, i32 %i
        ret i16 %0
}

define i16 @extract_varadic_i16_1(<8 x i16> %v, i32 %i) nounwind readnone {
entry:
        %0 = extractelement <8 x i16> %v, i32 %i
        ret i16 %0
}

define i32 @extract_varadic_i32(i32 %i) nounwind readnone {
entry:
        %0 = extractelement <4 x i32> < i32 0, i32 1, i32 2, i32 3>, i32 %i
        ret i32 %0
}

define i32 @extract_varadic_i32_1(<4 x i32> %v, i32 %i) nounwind readnone {
entry:
        %0 = extractelement <4 x i32> %v, i32 %i
        ret i32 %0
}

define float @extract_varadic_f32(i32 %i) nounwind readnone {
entry:
        %0 = extractelement <4 x float> < float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00 >, i32 %i
        ret float %0
}

define float @extract_varadic_f32_1(<4 x float> %v, i32 %i) nounwind readnone {
entry:
        %0 = extractelement <4 x float> %v, i32 %i
        ret float %0
}

define i64 @extract_varadic_i64(i32 %i) nounwind readnone {
entry:
        %0 = extractelement <2 x i64> < i64 0, i64 1>, i32 %i
        ret i64 %0
}

define i64 @extract_varadic_i64_1(<2 x i64> %v, i32 %i) nounwind readnone {
entry:
        %0 = extractelement <2 x i64> %v, i32 %i
        ret i64 %0
}

define double @extract_varadic_f64(i32 %i) nounwind readnone {
entry:
        %0 = extractelement <2 x double> < double 1.000000e+00, double 2.000000e+00>, i32 %i
        ret double %0
}

define double @extract_varadic_f64_1(<2 x double> %v, i32 %i) nounwind readnone {
entry:
        %0 = extractelement <2 x double> %v, i32 %i
        ret double %0
}
