; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -fast-isel-abort=1 -aarch64-enable-atomic-cfg-tidy=false -disable-cgp-branch-opts -verify-machineinstrs < %s | FileCheck %s

;
; Test folding of the sign-/zero-extend into the load instruction.
;

; Unscaled
define i32 @load_unscaled_zext_i8_to_i32(i64 %a) {
; CHECK-LABEL: load_unscaled_zext_i8_to_i32
; CHECK:       ldurb w0, [x0, #-8]
; CHECK-NOT:   uxtb
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to i8*
  %3 = load i8, i8* %2
  br label %bb2

bb2:
  %4 = zext i8 %3 to i32
  ret i32 %4
}

define i32 @load_unscaled_zext_i16_to_i32(i64 %a) {
; CHECK-LABEL: load_unscaled_zext_i16_to_i32
; CHECK:       ldurh w0, [x0, #-8]
; CHECK-NOT:   uxth
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to i16*
  %3 = load i16, i16* %2
  br label %bb2

bb2:
  %4 = zext i16 %3 to i32
  ret i32 %4
}

define i64 @load_unscaled_zext_i8_to_i64(i64 %a) {
; CHECK-LABEL: load_unscaled_zext_i8_to_i64
; CHECK:       ldurb w0, [x0, #-8]
; CHECK-NOT:   uxtb
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to i8*
  %3 = load i8, i8* %2
  br label %bb2

bb2:
  %4 = zext i8 %3 to i64
  ret i64 %4
}

define i64 @load_unscaled_zext_i16_to_i64(i64 %a) {
; CHECK-LABEL: load_unscaled_zext_i16_to_i64
; CHECK:       ldurh w0, [x0, #-8]
; CHECK-NOT:   uxth
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to i16*
  %3 = load i16, i16* %2
  br label %bb2

bb2:
  %4 = zext i16 %3 to i64
  ret i64 %4
}

define i64 @load_unscaled_zext_i32_to_i64(i64 %a) {
; CHECK-LABEL: load_unscaled_zext_i32_to_i64
; CHECK:       ldur w0, [x0, #-8]
; CHECK-NOT:   uxtw
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to i32*
  %3 = load i32, i32* %2
  br label %bb2

bb2:
  %4 = zext i32 %3 to i64
  ret i64 %4
}

define i32 @load_unscaled_sext_i8_to_i32(i64 %a) {
; CHECK-LABEL: load_unscaled_sext_i8_to_i32
; CHECK:       ldursb w0, [x0, #-8]
; CHECK-NOT:   sxtb
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to i8*
  %3 = load i8, i8* %2
  br label %bb2

bb2:
  %4 = sext i8 %3 to i32
  ret i32 %4
}

define i32 @load_unscaled_sext_i16_to_i32(i64 %a) {
; CHECK-LABEL: load_unscaled_sext_i16_to_i32
; CHECK:       ldursh w0, [x0, #-8]
; CHECK-NOT:   sxth
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to i16*
  %3 = load i16, i16* %2
  br label %bb2

bb2:
  %4 = sext i16 %3 to i32
  ret i32 %4
}

define i64 @load_unscaled_sext_i8_to_i64(i64 %a) {
; CHECK-LABEL: load_unscaled_sext_i8_to_i64
; CHECK:       ldursb x0, [x0, #-8]
; CHECK-NOT:   sxtb
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to i8*
  %3 = load i8, i8* %2
  br label %bb2

bb2:
  %4 = sext i8 %3 to i64
  ret i64 %4
}

define i64 @load_unscaled_sext_i16_to_i64(i64 %a) {
; CHECK-LABEL: load_unscaled_sext_i16_to_i64
; CHECK:       ldursh x0, [x0, #-8]
; CHECK-NOT:   sxth
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to i16*
  %3 = load i16, i16* %2
  br label %bb2

bb2:
  %4 = sext i16 %3 to i64
  ret i64 %4
}

define i64 @load_unscaled_sext_i32_to_i64(i64 %a) {
; CHECK-LABEL: load_unscaled_sext_i32_to_i64
; CHECK:       ldursw x0, [x0, #-8]
; CHECK-NOT:   sxtw
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to i32*
  %3 = load i32, i32* %2
  br label %bb2

bb2:
  %4 = sext i32 %3 to i64
  ret i64 %4
}

; Register
define i32 @load_register_zext_i8_to_i32(i64 %a, i64 %b) {
; CHECK-LABEL: load_register_zext_i8_to_i32
; CHECK:       ldrb w0, [x0, x1]
; CHECK-NOT:   uxtb
  %1 = add i64 %a, %b
  %2 = inttoptr i64 %1 to i8*
  %3 = load i8, i8* %2
  br label %bb2

bb2:
  %4 = zext i8 %3 to i32
  ret i32 %4
}

define i32 @load_register_zext_i16_to_i32(i64 %a, i64 %b) {
; CHECK-LABEL: load_register_zext_i16_to_i32
; CHECK:       ldrh w0, [x0, x1]
; CHECK-NOT:   uxth
  %1 = add i64 %a, %b
  %2 = inttoptr i64 %1 to i16*
  %3 = load i16, i16* %2
  br label %bb2

bb2:
  %4 = zext i16 %3 to i32
  ret i32 %4
}

define i64 @load_register_zext_i8_to_i64(i64 %a, i64 %b) {
; CHECK-LABEL: load_register_zext_i8_to_i64
; CHECK:       ldrb w0, [x0, x1]
; CHECK-NOT:   uxtb
  %1 = add i64 %a, %b
  %2 = inttoptr i64 %1 to i8*
  %3 = load i8, i8* %2
  br label %bb2

bb2:
  %4 = zext i8 %3 to i64
  ret i64 %4
}

define i64 @load_register_zext_i16_to_i64(i64 %a, i64 %b) {
; CHECK-LABEL: load_register_zext_i16_to_i64
; CHECK:       ldrh w0, [x0, x1]
; CHECK-NOT:   uxth
  %1 = add i64 %a, %b
  %2 = inttoptr i64 %1 to i16*
  %3 = load i16, i16* %2
  br label %bb2

bb2:
  %4 = zext i16 %3 to i64
  ret i64 %4
}

define i64 @load_register_zext_i32_to_i64(i64 %a, i64 %b) {
; CHECK-LABEL: load_register_zext_i32_to_i64
; CHECK:       ldr w0, [x0, x1]
; CHECK-NOT:   uxtw
  %1 = add i64 %a, %b
  %2 = inttoptr i64 %1 to i32*
  %3 = load i32, i32* %2
  br label %bb2

bb2:
  %4 = zext i32 %3 to i64
  ret i64 %4
}

define i32 @load_register_sext_i8_to_i32(i64 %a, i64 %b) {
; CHECK-LABEL: load_register_sext_i8_to_i32
; CHECK:       ldrsb w0, [x0, x1]
; CHECK-NOT:   sxtb
  %1 = add i64 %a, %b
  %2 = inttoptr i64 %1 to i8*
  %3 = load i8, i8* %2
  br label %bb2

bb2:
  %4 = sext i8 %3 to i32
  ret i32 %4
}

define i32 @load_register_sext_i16_to_i32(i64 %a, i64 %b) {
; CHECK-LABEL: load_register_sext_i16_to_i32
; CHECK:       ldrsh w0, [x0, x1]
; CHECK-NOT:   sxth
  %1 = add i64 %a, %b
  %2 = inttoptr i64 %1 to i16*
  %3 = load i16, i16* %2
  br label %bb2

bb2:
  %4 = sext i16 %3 to i32
  ret i32 %4
}

define i64 @load_register_sext_i8_to_i64(i64 %a, i64 %b) {
; CHECK-LABEL: load_register_sext_i8_to_i64
; CHECK:       ldrsb x0, [x0, x1]
; CHECK-NOT:   sxtb
  %1 = add i64 %a, %b
  %2 = inttoptr i64 %1 to i8*
  %3 = load i8, i8* %2
  br label %bb2

bb2:
  %4 = sext i8 %3 to i64
  ret i64 %4
}

define i64 @load_register_sext_i16_to_i64(i64 %a, i64 %b) {
; CHECK-LABEL: load_register_sext_i16_to_i64
; CHECK:       ldrsh x0, [x0, x1]
; CHECK-NOT:   sxth
  %1 = add i64 %a, %b
  %2 = inttoptr i64 %1 to i16*
  %3 = load i16, i16* %2
  br label %bb2

bb2:
  %4 = sext i16 %3 to i64
  ret i64 %4
}

define i64 @load_register_sext_i32_to_i64(i64 %a, i64 %b) {
; CHECK-LABEL: load_register_sext_i32_to_i64
; CHECK:       ldrsw x0, [x0, x1]
; CHECK-NOT:   sxtw
  %1 = add i64 %a, %b
  %2 = inttoptr i64 %1 to i32*
  %3 = load i32, i32* %2
  br label %bb2

bb2:
  %4 = sext i32 %3 to i64
  ret i64 %4
}

; Extend
define i32 @load_extend_zext_i8_to_i32(i64 %a, i32 %b) {
; CHECK-LABEL: load_extend_zext_i8_to_i32
; CHECK:       ldrb w0, [x0, w1, sxtw]
; CHECK-NOT:   uxtb
  %1 = sext i32 %b to i64
  %2 = add i64 %a, %1
  %3 = inttoptr i64 %2 to i8*
  %4 = load i8, i8* %3
  br label %bb2

bb2:
  %5 = zext i8 %4 to i32
  ret i32 %5
}

define i32 @load_extend_zext_i16_to_i32(i64 %a, i32 %b) {
; CHECK-LABEL: load_extend_zext_i16_to_i32
; CHECK:       ldrh w0, [x0, w1, sxtw]
; CHECK-NOT:   uxth
  %1 = sext i32 %b to i64
  %2 = add i64 %a, %1
  %3 = inttoptr i64 %2 to i16*
  %4 = load i16, i16* %3
  br label %bb2

bb2:
  %5 = zext i16 %4 to i32
  ret i32 %5
}

define i64 @load_extend_zext_i8_to_i64(i64 %a, i32 %b) {
; CHECK-LABEL: load_extend_zext_i8_to_i64
; CHECK:       ldrb w0, [x0, w1, sxtw]
; CHECK-NOT:   uxtb
  %1 = sext i32 %b to i64
  %2 = add i64 %a, %1
  %3 = inttoptr i64 %2 to i8*
  %4 = load i8, i8* %3
  br label %bb2

bb2:
  %5 = zext i8 %4 to i64
  ret i64 %5
}

define i64 @load_extend_zext_i16_to_i64(i64 %a, i32 %b) {
; CHECK-LABEL: load_extend_zext_i16_to_i64
; CHECK:       ldrh w0, [x0, w1, sxtw]
; CHECK-NOT:   uxth
  %1 = sext i32 %b to i64
  %2 = add i64 %a, %1
  %3 = inttoptr i64 %2 to i16*
  %4 = load i16, i16* %3
  br label %bb2

bb2:
  %5 = zext i16 %4 to i64
  ret i64 %5
}

define i64 @load_extend_zext_i32_to_i64(i64 %a, i32 %b) {
; CHECK-LABEL: load_extend_zext_i32_to_i64
; CHECK:       ldr w0, [x0, w1, sxtw]
; CHECK-NOT:   uxtw
  %1 = sext i32 %b to i64
  %2 = add i64 %a, %1
  %3 = inttoptr i64 %2 to i32*
  %4 = load i32, i32* %3
  br label %bb2

bb2:
  %5 = zext i32 %4 to i64
  ret i64 %5
}

define i32 @load_extend_sext_i8_to_i32(i64 %a, i32 %b) {
; CHECK-LABEL: load_extend_sext_i8_to_i32
; CHECK:       ldrsb w0, [x0, w1, sxtw]
; CHECK-NOT:   sxtb
  %1 = sext i32 %b to i64
  %2 = add i64 %a, %1
  %3 = inttoptr i64 %2 to i8*
  %4 = load i8, i8* %3
  br label %bb2

bb2:
  %5 = sext i8 %4 to i32
  ret i32 %5
}

define i32 @load_extend_sext_i16_to_i32(i64 %a, i32 %b) {
; CHECK-LABEL: load_extend_sext_i16_to_i32
; CHECK:       ldrsh w0, [x0, w1, sxtw]
; CHECK-NOT:   sxth
  %1 = sext i32 %b to i64
  %2 = add i64 %a, %1
  %3 = inttoptr i64 %2 to i16*
  %4 = load i16, i16* %3
  br label %bb2

bb2:
  %5 = sext i16 %4 to i32
  ret i32 %5
}

define i64 @load_extend_sext_i8_to_i64(i64 %a, i32 %b) {
; CHECK-LABEL: load_extend_sext_i8_to_i64
; CHECK:       ldrsb x0, [x0, w1, sxtw]
; CHECK-NOT:   sxtb
  %1 = sext i32 %b to i64
  %2 = add i64 %a, %1
  %3 = inttoptr i64 %2 to i8*
  %4 = load i8, i8* %3
  br label %bb2

bb2:
  %5 = sext i8 %4 to i64
  ret i64 %5
}

define i64 @load_extend_sext_i16_to_i64(i64 %a, i32 %b) {
; CHECK-LABEL: load_extend_sext_i16_to_i64
; CHECK:       ldrsh x0, [x0, w1, sxtw]
; CHECK-NOT:   sxth
  %1 = sext i32 %b to i64
  %2 = add i64 %a, %1
  %3 = inttoptr i64 %2 to i16*
  %4 = load i16, i16* %3
  br label %bb2

bb2:
  %5 = sext i16 %4 to i64
  ret i64 %5
}

define i64 @load_extend_sext_i32_to_i64(i64 %a, i32 %b) {
; CHECK-LABEL: load_extend_sext_i32_to_i64
; CHECK:       ldrsw x0, [x0, w1, sxtw]
; CHECK-NOT:   sxtw
  %1 = sext i32 %b to i64
  %2 = add i64 %a, %1
  %3 = inttoptr i64 %2 to i32*
  %4 = load i32, i32* %3
  br label %bb2

bb2:
  %5 = sext i32 %4 to i64
  ret i64 %5
}

