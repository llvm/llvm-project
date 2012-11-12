; RUN: llc < %s -march=cellspu > %t1.s
; RUN: grep ilhu  %t1.s | count 9
; RUN: grep iohl  %t1.s | count 7
; RUN: grep -w il    %t1.s | count 3
; RUN: grep 16429 %t1.s | count 1
; RUN: grep 63572 %t1.s | count 1
; RUN: grep   128 %t1.s | count 1
; RUN: grep 32639 %t1.s | count 1
; RUN: grep 65535 %t1.s | count 1
; RUN: grep 16457 %t1.s | count 1
; RUN: grep  4059 %t1.s | count 1
; RUN: grep 49077 %t1.s | count 1
; RUN: grep  1267 %t1.s | count 2
; RUN: grep 16309 %t1.s | count 1
; RUN: cat %t1.s | FileCheck %s
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define i32 @test_1() {
  ret i32 4784128               ;; ILHU via pattern (0x49000)
}

define i32 @test_2() {
  ret i32 5308431               ;; ILHU/IOHL via pattern (0x5100f)
}

define i32 @test_3() {
  ret i32 511                   ;; IL via pattern
}

define i32 @test_4() {
  ret i32 -512                  ;; IL via pattern
}

define i32 @test_5()
{
;CHECK: test_5:
;CHECK-NOT: ila $3, 40000
;CHECK: ilhu
;CHECK: iohl
;CHECK: bi $lr
  ret i32 400000
}

;; double             float       floatval
;; 0x4005bf0a80000000 0x402d|f854 2.718282
define float @float_const_1() {
  ret float 0x4005BF0A80000000  ;; ILHU/IOHL
}

;; double             float       floatval
;; 0x3810000000000000 0x0080|0000 0.000000
define float @float_const_2() {
  ret float 0x3810000000000000  ;; IL 128
}

;; double             float       floatval
;; 0x47efffffe0000000 0x7f7f|ffff NaN
define float @float_const_3() {
  ret float 0x47EFFFFFE0000000  ;; ILHU/IOHL via pattern
}

;; double             float       floatval
;; 0x400921fb60000000 0x4049|0fdb 3.141593
define float @float_const_4() {
  ret float 0x400921FB60000000  ;; ILHU/IOHL via pattern
}

;; double             float       floatval
;; 0xbff6a09e60000000 0xbfb5|04f3 -1.414214
define float @float_const_5() {
  ret float 0xBFF6A09E60000000  ;; ILHU/IOHL via pattern
}

;; double             float       floatval
;; 0x3ff6a09e60000000 0x3fb5|04f3 1.414214
define float @float_const_6() {
  ret float 0x3FF6A09E60000000  ;; ILHU/IOHL via pattern
}

define float @float_const_7() {
  ret float 0.000000e+00        ;; IL 0 via pattern
}
