; RUN: llc -mtriple=arm %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ARM --check-prefix=CHECK-CMP
; RUN: llc -mtriple=thumb-eabi %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-T --check-prefix=CHECK-CMP
; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-T2 --check-prefix=CHECK-CMP


; Check for clipping against 0 that should result in bic
;
; Base tests with different bit widths
;

; x < 0 ? 0 : x
; 32-bit base test
define i32 @sat0_base_32bit(i32 %x) #0 {
; CHECK-LABEL: sat0_base_32bit:
; CHECK-CMP-NOT: cmp
; CHECK-ARM: bic {{r[0-9]}}, [[INPUT:r[0-9]]], [[INPUT]], asr #31
; CHECK-T2: bic.w {{r[0-9]}}, [[INPUT:r[0-9]]], [[INPUT]], asr #31
; CHECK-T: asrs [[IM:r[0-9]]], {{r[0-9]}}, #31
; CHECK-T-NEXT: bics {{r[0-9]}}, [[IM]]
entry:
  %cmpLow = icmp slt i32 %x, 0
  %saturateLow = select i1 %cmpLow, i32 0, i32 %x
  ret i32 %saturateLow
}

; x < 0 ? 0 : x
; 16-bit base test
define i16 @sat0_base_16bit(i16 %x) #0 {
; CHECK-LABEL: sat0_base_16bit:
; CHECK-CMP: cmp
; CHECK-ARM-NOT: bic
; CHECK-T2-NOT: bic.w
; CHECK-T-NOT: bics
entry:
  %cmpLow = icmp slt i16 %x, 0
  %saturateLow = select i1 %cmpLow, i16 0, i16 %x
  ret i16 %saturateLow
}

; x < 0 ? 0 : x
; 8-bit base test
define i8 @sat0_base_8bit(i8 %x) #0 {
; CHECK-LABEL: sat0_base_8bit:
; CHECK-CMP: cmp
; CHECK-ARM-NOT: bic
; CHECK-T2-NOT: bic.w
entry:
  %cmpLow = icmp slt i8 %x, 0
  %saturateLow = select i1 %cmpLow, i8 0, i8 %x
  ret i8 %saturateLow
}

; Test where the conditional is formed in a different way

; x > 0 ? x : 0
define i32 @sat0_lower_1(i32 %x) #0 {
; CHECK-LABEL: sat0_lower_1:
; CHECK-CMP-NOT: cmp
; CHECK-ARM: bic {{r[0-9]}}, [[INPUT:r[0-9]]], [[INPUT]], asr #31
; CHECK-T2: bic.w {{r[0-9]}}, [[INPUT:r[0-9]]], [[INPUT]], asr #31
; CHECK-T: asrs [[IM:r[0-9]]], {{r[0-9]}}, #31
; CHECK-T-NEXT: bics {{r[0-9]}}, [[IM]]
entry:
  %cmpGt = icmp sgt i32 %x, 0
  %saturateLow = select i1 %cmpGt, i32 %x, i32 0
  ret i32 %saturateLow
}


; Check for clipping against -1 that should result in orr
;
; Base tests with different bit widths
;

; x < -1 ? -1 : x
; 32-bit base test
define i32 @sat1_base_32bit(i32 %x) #0 {
; CHECK-LABEL: sat1_base_32bit:
; CHECK-CMP-NOT: cmp
; CHECK-ARM: orr {{r[0-9]}}, [[INPUT:r[0-9]]], [[INPUT]], asr #31
; CHECK-T2: orr.w {{r[0-9]}}, [[INPUT:r[0-9]]], [[INPUT]], asr #31
; CHECK-T: asrs [[IM:r[0-9]]], {{r[0-9]}}, #31
; CHECK-T-NEXT: orrs {{r[0-9]}}, [[IM]]
entry:
  %cmpLow = icmp slt i32 %x, -1
  %saturateLow = select i1 %cmpLow, i32 -1, i32 %x
  ret i32 %saturateLow
}

; x < -1 ? -1 : x
; 16-bit base test
define i16 @sat1_base_16bit(i16 %x) #0 {
; CHECK-LABEL: sat1_base_16bit:
; CHECK-ARM: cmn
; CHECK-T2: cmp
; CHECK-T: cmp
entry:
  %cmpLow = icmp slt i16 %x, -1
  %saturateLow = select i1 %cmpLow, i16 -1, i16 %x
  ret i16 %saturateLow
}

; x < -1 ? -1 : x
; 8-bit base test
define i8 @sat1_base_8bit(i8 %x) #0 {
; CHECK-LABEL: sat1_base_8bit:
; CHECK-ARM: cmn
; CHECK-T2: cmp
; CHECK-T: cmp
entry:
  %cmpLow = icmp slt i8 %x, -1
  %saturateLow = select i1 %cmpLow, i8 -1, i8 %x
  ret i8 %saturateLow
}

; Test where the conditional is formed in a different way

; x > -1 ? x : -1
define i32 @sat1_lower_1(i32 %x) #0 {
; CHECK-LABEL: sat1_lower_1:
; CHECK-ARM: orr {{r[0-9]}}, [[INPUT:r[0-9]]], [[INPUT]], asr #31
; CHECK-T2: orr.w {{r[0-9]}}, [[INPUT:r[0-9]]], [[INPUT]], asr #31
; CHECK-T: asrs [[IM:r[0-9]]], {{r[0-9]}}, #31
; CHECK-T-NEXT: orrs {{r[0-9]}}, [[IM]]
; CHECK-CMP-NOT: cmp
entry:
  %cmpGt = icmp sgt i32 %x, -1
  %saturateLow = select i1 %cmpGt, i32 %x, i32 -1
  ret i32 %saturateLow
}

; The following tests for patterns that should not transform into bitops
; but that are similar enough that could confuse the selector.

; x < 0 ? 0 : y where x and y does not properly match
define i32 @no_sat0_incorrect_variable(i32 %x, i32 %y) #0 {
; CHECK-LABEL: no_sat0_incorrect_variable:
; CHECK-NOT: bic
; CHECK-NOT: asrs
; CHECK-CMP: cmp
entry:
  %cmpLow = icmp slt i32 %x, 0
  %saturateLow = select i1 %cmpLow, i32 0, i32 %y
  ret i32 %saturateLow
}

; x < 0 ? -1 : x
define i32 @no_sat0_incorrect_constant(i32 %x) #0 {
; CHECK-LABEL: no_sat0_incorrect_constant:
; CHECK-NOT: bic
; CHECK-NOT: asrs
; CHECK-CMP: cmp
entry:
  %cmpLow = icmp slt i32 %x, 0
  %saturateLow = select i1 %cmpLow, i32 -1, i32 %x
  ret i32 %saturateLow
}
