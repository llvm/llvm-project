; RUN: llc < %s -march=nvptx -mcpu=sm_20 -O3 | FileCheck %s --check-prefix=OPT
; RUN: llc < %s -march=nvptx -mcpu=sm_20 -O0 | FileCheck %s --check-prefix=NOOPT

; OPT-LABEL: @mulwide16
; NOOPT-LABEL: @mulwide16
define i32 @mulwide16(i16 %a, i16 %b) {
; OPT: mul.wide.s16
; NOOPT: mul.lo.s32
  %val0 = sext i16 %a to i32
  %val1 = sext i16 %b to i32
  %val2 = mul i32 %val0, %val1
  ret i32 %val2
}

; OPT-LABEL: @mulwideu16
; NOOPT-LABEL: @mulwideu16
define i32 @mulwideu16(i16 %a, i16 %b) {
; OPT: mul.wide.u16
; NOOPT: mul.lo.s32
  %val0 = zext i16 %a to i32
  %val1 = zext i16 %b to i32
  %val2 = mul i32 %val0, %val1
  ret i32 %val2
}

; OPT-LABEL: @mulwide32
; NOOPT-LABEL: @mulwide32
define i64 @mulwide32(i32 %a, i32 %b) {
; OPT: mul.wide.s32
; NOOPT: mul.lo.s64
  %val0 = sext i32 %a to i64
  %val1 = sext i32 %b to i64
  %val2 = mul i64 %val0, %val1
  ret i64 %val2
}

; OPT-LABEL: @mulwideu32
; NOOPT-LABEL: @mulwideu32
define i64 @mulwideu32(i32 %a, i32 %b) {
; OPT: mul.wide.u32
; NOOPT: mul.lo.s64
  %val0 = zext i32 %a to i64
  %val1 = zext i32 %b to i64
  %val2 = mul i64 %val0, %val1
  ret i64 %val2
}
