; RUN: llc < %s -mtriple aarch64--none-eabi -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: Str64Ldr64
; CHECK: mov x0, x1
define i64 @Str64Ldr64(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i64*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i64, i64* %0, i64 1
  %1 = load i64, i64* %arrayidx1
  ret i64 %1
}

; CHECK-LABEL: Str64Ldr32_0
; CHECK: and x0, x1, #0xffffffff
define i32 @Str64Ldr32_0(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i32*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i32, i32* %0, i64 2
  %1 = load i32, i32* %arrayidx1
  ret i32 %1
}

; CHECK-LABEL: Str64Ldr32_1
; CHECK: lsr x0, x1, #32
define i32 @Str64Ldr32_1(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i32*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i32, i32* %0, i64 3
  %1 = load i32, i32* %arrayidx1
  ret i32 %1
}

; CHECK-LABEL: Str64Ldr16_0
; CHECK: and x0, x1, #0xffff
define i16 @Str64Ldr16_0(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i16*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 4
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: Str64Ldr16_1
; CHECK: ubfx x0, x1, #16, #16
define i16 @Str64Ldr16_1(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i16*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 5
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: Str64Ldr16_2
; CHECK: ubfx x0, x1, #32, #16
define i16 @Str64Ldr16_2(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i16*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 6
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: Str64Ldr16_3
; CHECK: lsr x0, x1, #48
define i16 @Str64Ldr16_3(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i16*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 7
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: Str64Ldr8_0
; CHECK: and x0, x1, #0xff
define i8 @Str64Ldr8_0(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 8
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Str64Ldr8_1
; CHECK: ubfx x0, x1, #8, #8
define i8 @Str64Ldr8_1(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 9
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Str64Ldr8_2
; CHECK: ubfx x0, x1, #16, #8
define i8 @Str64Ldr8_2(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 10
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Str64Ldr8_3
; CHECK: ubfx x0, x1, #24, #8
define i8 @Str64Ldr8_3(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 11
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Str64Ldr8_4
; CHECK: ubfx x0, x1, #32, #8
define i8 @Str64Ldr8_4(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 12
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Str64Ldr8_5
; CHECK: ubfx x0, x1, #40, #8
define i8 @Str64Ldr8_5(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 13
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Str64Ldr8_6
; CHECK: ubfx x0, x1, #48, #8
define i8 @Str64Ldr8_6(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 14
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Str64Ldr8_7
; CHECK: lsr x0, x1, #56
define i8 @Str64Ldr8_7(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 15
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Str32Ldr32
; CHECK: mov w0, w1
define i32 @Str32Ldr32(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i32*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i32, i32* %0, i64 1
  %1 = load i32, i32* %arrayidx1
  ret i32 %1
}

; CHECK-LABEL: Str32Ldr16_0
; CHECK: and w0, w1, #0xffff
define i16 @Str32Ldr16_0(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i16*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 2
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: Str32Ldr16_1
; CHECK: lsr	w0, w1, #16
define i16 @Str32Ldr16_1(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i16*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 3
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: Str32Ldr8_0
; CHECK: and w0, w1, #0xff
define i8 @Str32Ldr8_0(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i8*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 4
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Str32Ldr8_1
; CHECK: ubfx w0, w1, #8, #8
define i8 @Str32Ldr8_1(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i8*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 5
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Str32Ldr8_2
; CHECK: ubfx w0, w1, #16, #8
define i8 @Str32Ldr8_2(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i8*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 6
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Str32Ldr8_3
; CHECK: lsr w0, w1, #24
define i8 @Str32Ldr8_3(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i8*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 7
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Str16Ldr16
; CHECK: and w0, w1, #0xffff
define i16 @Str16Ldr16(i16* nocapture %P, i16 %v, i64 %n) {
entry:
  %0 = bitcast i16* %P to i16*
  %arrayidx0 = getelementptr inbounds i16, i16* %P, i64 1
  store i16 %v, i16* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 1
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: Str16Ldr8_0
; CHECK: and w0, w1, #0xff
define i8 @Str16Ldr8_0(i16* nocapture %P, i16 %v, i64 %n) {
entry:
  %0 = bitcast i16* %P to i8*
  %arrayidx0 = getelementptr inbounds i16, i16* %P, i64 1
  store i16 %v, i16* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 2
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Str16Ldr8_1
; CHECK: ubfx w0, w1, #8, #8
define i8 @Str16Ldr8_1(i16* nocapture %P, i16 %v, i64 %n) {
entry:
  %0 = bitcast i16* %P to i8*
  %arrayidx0 = getelementptr inbounds i16, i16* %P, i64 1
  store i16 %v, i16* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 3
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}


; CHECK-LABEL: Unscaled_Str64Ldr64
; CHECK: mov x0, x1
define i64 @Unscaled_Str64Ldr64(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i64*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i64, i64* %0, i64 -1
  %1 = load i64, i64* %arrayidx1
  ret i64 %1
}

; CHECK-LABEL: Unscaled_Str64Ldr32_0
; CHECK: and x0, x1, #0xffffffff
define i32 @Unscaled_Str64Ldr32_0(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i32*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i32, i32* %0, i64 -2
  %1 = load i32, i32* %arrayidx1
  ret i32 %1
}

; CHECK-LABEL: Unscaled_Str64Ldr32_1
; CHECK: lsr x0, x1, #32
define i32 @Unscaled_Str64Ldr32_1(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i32*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i32, i32* %0, i64 -1
  %1 = load i32, i32* %arrayidx1
  ret i32 %1
}

; CHECK-LABEL: Unscaled_Str64Ldr16_0
; CHECK: and x0, x1, #0xffff
define i16 @Unscaled_Str64Ldr16_0(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i16*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 -4
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: Unscaled_Str64Ldr16_1
; CHECK: ubfx x0, x1, #16, #16
define i16 @Unscaled_Str64Ldr16_1(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i16*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 -3
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: Unscaled_Str64Ldr16_2
; CHECK: ubfx x0, x1, #32, #16
define i16 @Unscaled_Str64Ldr16_2(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i16*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 -2
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: Unscaled_Str64Ldr16_3
; CHECK: lsr x0, x1, #48
define i16 @Unscaled_Str64Ldr16_3(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i16*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 -1
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: Unscaled_Str64Ldr8_0
; CHECK: and x0, x1, #0xff
define i8 @Unscaled_Str64Ldr8_0(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 -8
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Unscaled_Str64Ldr8_1
; CHECK: ubfx x0, x1, #8, #8
define i8 @Unscaled_Str64Ldr8_1(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 -7
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Unscaled_Str64Ldr8_2
; CHECK: ubfx x0, x1, #16, #8
define i8 @Unscaled_Str64Ldr8_2(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 -6
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Unscaled_Str64Ldr8_3
; CHECK: ubfx x0, x1, #24, #8
define i8 @Unscaled_Str64Ldr8_3(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 -5
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Unscaled_Str64Ldr8_4
; CHECK: ubfx x0, x1, #32, #8
define i8 @Unscaled_Str64Ldr8_4(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 -4
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Unscaled_Str64Ldr8_5
; CHECK: ubfx x0, x1, #40, #8
define i8 @Unscaled_Str64Ldr8_5(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 -3
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Unscaled_Str64Ldr8_6
; CHECK: ubfx x0, x1, #48, #8
define i8 @Unscaled_Str64Ldr8_6(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 -2
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Unscaled_Str64Ldr8_7
; CHECK: lsr x0, x1, #56
define i8 @Unscaled_Str64Ldr8_7(i64* nocapture %P, i64 %v, i64 %n) {
entry:
  %0 = bitcast i64* %P to i8*
  %arrayidx0 = getelementptr inbounds i64, i64* %P, i64 -1
  store i64 %v, i64* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 -1
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Unscaled_Str32Ldr32
; CHECK: mov w0, w1
define i32 @Unscaled_Str32Ldr32(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i32*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 -1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i32, i32* %0, i64 -1
  %1 = load i32, i32* %arrayidx1
  ret i32 %1
}

; CHECK-LABEL: Unscaled_Str32Ldr16_0
; CHECK: and w0, w1, #0xffff
define i16 @Unscaled_Str32Ldr16_0(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i16*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 -1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 -2
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: Unscaled_Str32Ldr16_1
; CHECK: lsr	w0, w1, #16
define i16 @Unscaled_Str32Ldr16_1(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i16*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 -1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 -1
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: Unscaled_Str32Ldr8_0
; CHECK: and w0, w1, #0xff
define i8 @Unscaled_Str32Ldr8_0(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i8*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 -1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 -4
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Unscaled_Str32Ldr8_1
; CHECK: ubfx w0, w1, #8, #8
define i8 @Unscaled_Str32Ldr8_1(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i8*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 -1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 -3
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Unscaled_Str32Ldr8_2
; CHECK: ubfx w0, w1, #16, #8
define i8 @Unscaled_Str32Ldr8_2(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i8*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 -1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 -2
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Unscaled_Str32Ldr8_3
; CHECK: lsr w0, w1, #24
define i8 @Unscaled_Str32Ldr8_3(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i8*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 -1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 -1
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Unscaled_Str16Ldr16
; CHECK: and w0, w1, #0xffff
define i16 @Unscaled_Str16Ldr16(i16* nocapture %P, i16 %v, i64 %n) {
entry:
  %0 = bitcast i16* %P to i16*
  %arrayidx0 = getelementptr inbounds i16, i16* %P, i64 -1
  store i16 %v, i16* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 -1
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: Unscaled_Str16Ldr8_0
; CHECK: and w0, w1, #0xff
define i8 @Unscaled_Str16Ldr8_0(i16* nocapture %P, i16 %v, i64 %n) {
entry:
  %0 = bitcast i16* %P to i8*
  %arrayidx0 = getelementptr inbounds i16, i16* %P, i64 -1
  store i16 %v, i16* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 -2
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: Unscaled_Str16Ldr8_1
; CHECK: ubfx w0, w1, #8, #8
define i8 @Unscaled_Str16Ldr8_1(i16* nocapture %P, i16 %v, i64 %n) {
entry:
  %0 = bitcast i16* %P to i8*
  %arrayidx0 = getelementptr inbounds i16, i16* %P, i64 -1
  store i16 %v, i16* %arrayidx0
  %arrayidx1 = getelementptr inbounds i8, i8* %0, i64 -1
  %1 = load i8, i8* %arrayidx1
  ret i8 %1
}

; CHECK-LABEL: StrVolatileLdr
; CHECK: ldrh
define i16 @StrVolatileLdr(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i16*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 2
  %1 = load volatile i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: StrNotInRangeLdr
; CHECK: ldrh
define i16 @StrNotInRangeLdr(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i16*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 1
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: Unscaled_StrNotInRangeLdr
; CHECK: ldurh
define i16 @Unscaled_StrNotInRangeLdr(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i16*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 -1
  store i32 %v, i32* %arrayidx0
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 -3
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

; CHECK-LABEL: StrCallLdr
; CHECK: ldrh
define i16 @StrCallLdr(i32* nocapture %P, i32 %v, i64 %n) {
entry:
  %0 = bitcast i32* %P to i16*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 1
  store i32 %v, i32* %arrayidx0
  %c = call i1 @test_dummy()
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 1
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}

declare i1 @test_dummy()

; CHECK-LABEL: StrStrLdr
; CHECK: ldrh
define i16 @StrStrLdr(i32 %v, i32* %P, i32* %P2, i32 %n) {
entry:
  %0 = bitcast i32* %P to i16*
  %arrayidx0 = getelementptr inbounds i32, i32* %P, i64 1
  store i32 %v, i32* %arrayidx0
  store i32 %n, i32* %P2
  %arrayidx1 = getelementptr inbounds i16, i16* %0, i64 2
  %1 = load i16, i16* %arrayidx1
  ret i16 %1
}
