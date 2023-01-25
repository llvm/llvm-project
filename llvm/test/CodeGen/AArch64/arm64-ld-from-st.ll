; RUN: llc < %s -mtriple aarch64--none-eabi -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: Str64Ldr64
; CHECK: mov x0, x1
define i64 @Str64Ldr64(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i64, ptr %P, i64 1
  %0 = load i64, ptr %arrayidx1
  ret i64 %0
}

; CHECK-LABEL: Str64Ldr32_0
; CHECK: mov w0, w1
define i32 @Str64Ldr32_0(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i32, ptr %P, i64 2
  %0 = load i32, ptr %arrayidx1
  ret i32 %0
}

; CHECK-LABEL: Str64Ldr32_1
; CHECK: lsr x0, x1, #32
define i32 @Str64Ldr32_1(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i32, ptr %P, i64 3
  %0 = load i32, ptr %arrayidx1
  ret i32 %0
}

; CHECK-LABEL: Str64Ldr16_0
; CHECK: mov w0, w1
define i16 @Str64Ldr16_0(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 4
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: Str64Ldr16_1
; CHECK: ubfx x0, x1, #16, #16
define i16 @Str64Ldr16_1(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 5
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: Str64Ldr16_2
; CHECK: ubfx x0, x1, #32, #16
define i16 @Str64Ldr16_2(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 6
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: Str64Ldr16_3
; CHECK: lsr x0, x1, #48
define i16 @Str64Ldr16_3(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 7
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: Str64Ldr8_0
; CHECK: mov w0, w1
define i8 @Str64Ldr8_0(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 8
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Str64Ldr8_1
; CHECK: ubfx x0, x1, #8, #8
define i8 @Str64Ldr8_1(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 9
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Str64Ldr8_2
; CHECK: ubfx x0, x1, #16, #8
define i8 @Str64Ldr8_2(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 10
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Str64Ldr8_3
; CHECK: ubfx x0, x1, #24, #8
define i8 @Str64Ldr8_3(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 11
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Str64Ldr8_4
; CHECK: ubfx x0, x1, #32, #8
define i8 @Str64Ldr8_4(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 12
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Str64Ldr8_5
; CHECK: ubfx x0, x1, #40, #8
define i8 @Str64Ldr8_5(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 13
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Str64Ldr8_6
; CHECK: ubfx x0, x1, #48, #8
define i8 @Str64Ldr8_6(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 14
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Str64Ldr8_7
; CHECK: lsr x0, x1, #56
define i8 @Str64Ldr8_7(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 15
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Str32Ldr32
; CHECK: mov w0, w1
define i32 @Str32Ldr32(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i32, ptr %P, i64 1
  %0 = load i32, ptr %arrayidx1
  ret i32 %0
}

; CHECK-LABEL: Str32Ldr16_0
; CHECK: mov w0, w1
define i16 @Str32Ldr16_0(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 2
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: Str32Ldr16_1
; CHECK: lsr	w0, w1, #16
define i16 @Str32Ldr16_1(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 3
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: Str32Ldr8_0
; CHECK: mov w0, w1
define i8 @Str32Ldr8_0(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 4
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Str32Ldr8_1
; CHECK: ubfx w0, w1, #8, #8
define i8 @Str32Ldr8_1(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 5
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Str32Ldr8_2
; CHECK: ubfx w0, w1, #16, #8
define i8 @Str32Ldr8_2(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 6
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Str32Ldr8_3
; CHECK: lsr w0, w1, #24
define i8 @Str32Ldr8_3(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 7
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Str16Ldr16
; CHECK: mov w0, w1
define i16 @Str16Ldr16(ptr nocapture %P, i16 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i16, ptr %P, i64 1
  store i16 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 1
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: Str16Ldr8_0
; CHECK: mov w0, w1
define i8 @Str16Ldr8_0(ptr nocapture %P, i16 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i16, ptr %P, i64 1
  store i16 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 2
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Str16Ldr8_1
; CHECK: ubfx w0, w1, #8, #8
define i8 @Str16Ldr8_1(ptr nocapture %P, i16 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i16, ptr %P, i64 1
  store i16 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 3
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}


; CHECK-LABEL: Unscaled_Str64Ldr64
; CHECK: mov x0, x1
define i64 @Unscaled_Str64Ldr64(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i64, ptr %P, i64 -1
  %0 = load i64, ptr %arrayidx1
  ret i64 %0
}

; CHECK-LABEL: Unscaled_Str64Ldr32_0
; CHECK: mov w0, w1
define i32 @Unscaled_Str64Ldr32_0(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i32, ptr %P, i64 -2
  %0 = load i32, ptr %arrayidx1
  ret i32 %0
}

; CHECK-LABEL: Unscaled_Str64Ldr32_1
; CHECK: lsr x0, x1, #32
define i32 @Unscaled_Str64Ldr32_1(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i32, ptr %P, i64 -1
  %0 = load i32, ptr %arrayidx1
  ret i32 %0
}

; CHECK-LABEL: Unscaled_Str64Ldr16_0
; CHECK: mov w0, w1
define i16 @Unscaled_Str64Ldr16_0(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 -4
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: Unscaled_Str64Ldr16_1
; CHECK: ubfx x0, x1, #16, #16
define i16 @Unscaled_Str64Ldr16_1(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 -3
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: Unscaled_Str64Ldr16_2
; CHECK: ubfx x0, x1, #32, #16
define i16 @Unscaled_Str64Ldr16_2(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 -2
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: Unscaled_Str64Ldr16_3
; CHECK: lsr x0, x1, #48
define i16 @Unscaled_Str64Ldr16_3(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 -1
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: Unscaled_Str64Ldr8_0
; CHECK: mov w0, w1
define i8 @Unscaled_Str64Ldr8_0(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 -8
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Unscaled_Str64Ldr8_1
; CHECK: ubfx x0, x1, #8, #8
define i8 @Unscaled_Str64Ldr8_1(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 -7
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Unscaled_Str64Ldr8_2
; CHECK: ubfx x0, x1, #16, #8
define i8 @Unscaled_Str64Ldr8_2(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 -6
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Unscaled_Str64Ldr8_3
; CHECK: ubfx x0, x1, #24, #8
define i8 @Unscaled_Str64Ldr8_3(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 -5
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Unscaled_Str64Ldr8_4
; CHECK: ubfx x0, x1, #32, #8
define i8 @Unscaled_Str64Ldr8_4(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 -4
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Unscaled_Str64Ldr8_5
; CHECK: ubfx x0, x1, #40, #8
define i8 @Unscaled_Str64Ldr8_5(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 -3
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Unscaled_Str64Ldr8_6
; CHECK: ubfx x0, x1, #48, #8
define i8 @Unscaled_Str64Ldr8_6(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 -2
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Unscaled_Str64Ldr8_7
; CHECK: lsr x0, x1, #56
define i8 @Unscaled_Str64Ldr8_7(ptr nocapture %P, i64 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 -1
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Unscaled_Str32Ldr32
; CHECK: mov w0, w1
define i32 @Unscaled_Str32Ldr32(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 -1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i32, ptr %P, i64 -1
  %0 = load i32, ptr %arrayidx1
  ret i32 %0
}

; CHECK-LABEL: Unscaled_Str32Ldr16_0
; CHECK: mov w0, w1
define i16 @Unscaled_Str32Ldr16_0(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 -1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 -2
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: Unscaled_Str32Ldr16_1
; CHECK: lsr	w0, w1, #16
define i16 @Unscaled_Str32Ldr16_1(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 -1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 -1
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: Unscaled_Str32Ldr8_0
; CHECK: mov w0, w1
define i8 @Unscaled_Str32Ldr8_0(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 -1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 -4
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Unscaled_Str32Ldr8_1
; CHECK: ubfx w0, w1, #8, #8
define i8 @Unscaled_Str32Ldr8_1(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 -1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 -3
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Unscaled_Str32Ldr8_2
; CHECK: ubfx w0, w1, #16, #8
define i8 @Unscaled_Str32Ldr8_2(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 -1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 -2
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Unscaled_Str32Ldr8_3
; CHECK: lsr w0, w1, #24
define i8 @Unscaled_Str32Ldr8_3(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 -1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 -1
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Unscaled_Str16Ldr16
; CHECK: mov w0, w1
define i16 @Unscaled_Str16Ldr16(ptr nocapture %P, i16 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i16, ptr %P, i64 -1
  store i16 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 -1
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: Unscaled_Str16Ldr8_0
; CHECK: mov w0, w1
define i8 @Unscaled_Str16Ldr8_0(ptr nocapture %P, i16 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i16, ptr %P, i64 -1
  store i16 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 -2
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: Unscaled_Str16Ldr8_1
; CHECK: ubfx w0, w1, #8, #8
define i8 @Unscaled_Str16Ldr8_1(ptr nocapture %P, i16 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i16, ptr %P, i64 -1
  store i16 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, ptr %P, i64 -1
  %0 = load i8, ptr %arrayidx1
  ret i8 %0
}

; CHECK-LABEL: StrVolatileLdr
; CHECK: ldrh
define i16 @StrVolatileLdr(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 2
  %0 = load volatile i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: StrNotInRangeLdr
; CHECK: ldrh
define i16 @StrNotInRangeLdr(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 1
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: Unscaled_StrNotInRangeLdr
; CHECK: ldurh
define i16 @Unscaled_StrNotInRangeLdr(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 -1
  store i32 %v, ptr %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 -3
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

; CHECK-LABEL: StrCallLdr
; CHECK: ldrh
define i16 @StrCallLdr(ptr nocapture %P, i32 %v, i64 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 1
  store i32 %v, ptr %arrayidx0
  %c = call i1 @test_dummy()
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 1
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}

declare i1 @test_dummy()

; CHECK-LABEL: StrStrLdr
; CHECK: ldrh
define i16 @StrStrLdr(i32 %v, ptr %P, ptr %P2, i32 %n) {
entry:
  %arrayidx0 = getelementptr inbounds i32, ptr %P, i64 1
  store i32 %v, ptr %arrayidx0
  store i32 %n, ptr %P2
  %arrayidx1 = getelementptr inbounds i16, ptr %P, i64 2
  %0 = load i16, ptr %arrayidx1
  ret i16 %0
}
