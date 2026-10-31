; air-emulate-f64: an integer conversion to double whose operand fits in 53
; bits is exact in every rounding, built inline from its leading zeros; a
; wider one keeps the soft-f64 library's conversion.
; RUN: opt -mtriple=air -passes=air-emulate-f64 -S < %s | FileCheck %s

target triple = "air"

; CHECK-LABEL: @from_i32(
; CHECK-NOT: __air_f64_from
; CHECK: call i64 @llvm.ctlz.i64
define double @from_i32(i32 %x) {
  %d = sitofp i32 %x to double
  ret double %d
}

; CHECK-LABEL: @from_u16(
; CHECK-NOT: __air_f64_from
; CHECK: call i64 @llvm.ctlz.i64
define double @from_u16(i16 %x) {
  %d = uitofp i16 %x to double
  ret double %d
}

; CHECK-LABEL: @from_masked_i64(
; CHECK-NOT: __air_f64_from
; CHECK: call i64 @llvm.ctlz.i64
define double @from_masked_i64(i64 %x) {
  %m = and i64 %x, 4294967295
  %d = uitofp i64 %m to double
  ret double %d
}

; CHECK-LABEL: @from_i64(
; CHECK: __air_f64_from_i64
define double @from_i64(i64 %x) {
  %d = sitofp i64 %x to double
  ret double %d
}
