; RUN: opt -S -passes=instcombine %s | FileCheck %s

; ============================================================
;  Test constant folding of overloaded @llvm.llrint intrinsic
; ============================================================

; LLVM intrinsic declarations (typed overloads)
declare i64 @llvm.llrint.f32(float)
declare i64 @llvm.llrint.f64(double)
declare i64 @llvm.llrint.f80(x86_fp80)
declare i64 @llvm.llrint.f128(fp128)

; ============================================================
; float overload
; ============================================================
define i64 @test_f32_pos() {
; CHECK-LABEL: @test_f32_pos(
; CHECK-NEXT: ret i64 4
  %v = call i64 @llvm.llrint.f32(float 3.5)
  ret i64 %v
}

define i64 @test_f32_neg() {
; CHECK-LABEL: @test_f32_neg(
; CHECK-NEXT: ret i64 -2
  %v = call i64 @llvm.llrint.f32(float -2.5)
  ret i64 %v
}

; ============================================================
; double overload
; ============================================================
define i64 @test_f64_pos() {
; CHECK-LABEL: @test_f64_pos(
; CHECK-NEXT: ret i64 4
  %v = call i64 @llvm.llrint.f64(double 3.5)
  ret i64 %v
}

define i64 @test_f64_neg() {
; CHECK-LABEL: @test_f64_neg(
; CHECK-NEXT: ret i64 -2
  %v = call i64 @llvm.llrint.f64(double -2.5)
  ret i64 %v
}

