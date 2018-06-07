; RUN: opt < %s -constprop -S | FileCheck %s

; Overflow on a float to int or int to float conversion is undefined (PR21130).

define i8 @overflow_fptosi() {
; CHECK-LABEL: @overflow_fptosi(
; CHECK-NEXT:    ret i8 undef
;
  %i = fptosi double 1.56e+02 to i8
  ret i8 %i
}

define i8 @overflow_fptoui() {
; CHECK-LABEL: @overflow_fptoui(
; CHECK-NEXT:    ret i8 undef
;
  %i = fptoui double 2.56e+02 to i8
  ret i8 %i
}

; The maximum float is approximately 2 ** 128 which is 3.4E38.
; The constant below is 4E38. Use a 130 bit integer to hold that
; number; 129-bits for the value + 1 bit for the sign.

define float @overflow_uitofp() {
; CHECK-LABEL: @overflow_uitofp(
; CHECK-NEXT:    ret float undef
;
  %i = uitofp i130 400000000000000000000000000000000000000 to float
  ret float %i
}

define float @overflow_sitofp() {
; CHECK-LABEL: @overflow_sitofp(
; CHECK-NEXT:    ret float undef
;
  %i = sitofp i130 400000000000000000000000000000000000000 to float
  ret float %i
}

