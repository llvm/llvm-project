; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -verify-machineinstrs < %s | FileCheck %s

;
; Test folding of the sign-/zero-extend into the load instruction.
;

; Unscaled
define i32 @load_unscaled_zext_i8_to_i32(i64 %a) {
; CHECK-LABEL: load_unscaled_zext_i8_to_i32
; CHECK:       ldurb [[REG:w[0-9]+]], [x0, #-8]
; CHECK:       uxtb w0, [[REG]]
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to ptr addrspace(256)
  %3 = load i8, ptr addrspace(256) %2
  %4 = zext i8 %3 to i32
  ret i32 %4
}

define i32 @load_unscaled_zext_i16_to_i32(i64 %a) {
; CHECK-LABEL: load_unscaled_zext_i16_to_i32
; CHECK:       ldurh [[REG:w[0-9]+]], [x0, #-8]
; CHECK:       uxth w0, [[REG]]
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to ptr addrspace(256)
  %3 = load i16, ptr addrspace(256) %2
  %4 = zext i16 %3 to i32
  ret i32 %4
}

define i64 @load_unscaled_zext_i8_to_i64(i64 %a) {
; CHECK-LABEL: load_unscaled_zext_i8_to_i64
; CHECK:       ldurb w[[REG:[0-9]+]], [x0, #-8]
; CHECK:       ubfx x0, x[[REG]], #0, #8
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to ptr addrspace(256)
  %3 = load i8, ptr addrspace(256) %2
  %4 = zext i8 %3 to i64
  ret i64 %4
}

define i64 @load_unscaled_zext_i16_to_i64(i64 %a) {
; CHECK-LABEL: load_unscaled_zext_i16_to_i64
; CHECK:       ldurh w[[REG:[0-9]+]], [x0, #-8]
; CHECK:       ubfx x0, x[[REG]], #0, #16
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to ptr addrspace(256)
  %3 = load i16, ptr addrspace(256) %2
  %4 = zext i16 %3 to i64
  ret i64 %4
}

define i64 @load_unscaled_zext_i32_to_i64(i64 %a) {
; CHECK-LABEL: load_unscaled_zext_i32_to_i64
; CHECK:       ldur w[[REG:[0-9]+]], [x0, #-8]
; CHECK:       lsr w0, w[[REG]], #0
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to ptr addrspace(256)
  %3 = load i32, ptr addrspace(256) %2
  %4 = zext i32 %3 to i64
  ret i64 %4
}

define i32 @load_unscaled_sext_i8_to_i32(i64 %a) {
; CHECK-LABEL: load_unscaled_sext_i8_to_i32
; CHECK:       ldurb [[REG:w[0-9]+]], [x0, #-8]
; CHECK:       sxtb w0, [[REG]]
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to ptr addrspace(256)
  %3 = load i8, ptr addrspace(256) %2
  %4 = sext i8 %3 to i32
  ret i32 %4
}

define i32 @load_unscaled_sext_i16_to_i32(i64 %a) {
; CHECK-LABEL: load_unscaled_sext_i16_to_i32
; CHECK:       ldurh [[REG:w[0-9]+]], [x0, #-8]
; CHECK:       sxth w0, [[REG]]
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to ptr addrspace(256)
  %3 = load i16, ptr addrspace(256) %2
  %4 = sext i16 %3 to i32
  ret i32 %4
}

define i64 @load_unscaled_sext_i8_to_i64(i64 %a) {
; CHECK-LABEL: load_unscaled_sext_i8_to_i64
; CHECK:       ldurb [[REG:w[0-9]+]], [x0, #-8]
; CHECK:       sxtb x0, [[REG]]
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to ptr addrspace(256)
  %3 = load i8, ptr addrspace(256) %2
  %4 = sext i8 %3 to i64
  ret i64 %4
}

define i64 @load_unscaled_sext_i16_to_i64(i64 %a) {
; CHECK-LABEL: load_unscaled_sext_i16_to_i64
; CHECK:       ldurh [[REG:w[0-9]+]], [x0, #-8]
; CHECK:       sxth x0, [[REG]]
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to ptr addrspace(256)
  %3 = load i16, ptr addrspace(256) %2
  %4 = sext i16 %3 to i64
  ret i64 %4
}

define i64 @load_unscaled_sext_i32_to_i64(i64 %a) {
; CHECK-LABEL: load_unscaled_sext_i32_to_i64
; CHECK:       ldur [[REG:w[0-9]+]], [x0, #-8]
; CHECK:       sxtw x0, [[REG]]
  %1 = sub i64 %a, 8
  %2 = inttoptr i64 %1 to ptr addrspace(256)
  %3 = load i32, ptr addrspace(256) %2
  %4 = sext i32 %3 to i64
  ret i64 %4
}

