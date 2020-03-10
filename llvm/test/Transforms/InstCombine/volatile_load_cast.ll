; RUN: opt < %s -instcombine -S | FileCheck %s

; Ensure that volatile loads followed by a bitcast don't get transformed into a
; volatile load of the bitcast-target type. This is unlikely to provide much in
; terms of optimizations, and might break the programmer's expectation for code
; generation, however brittle that expectation might be.
;
; See llvm.org/D75644 and llvm.org/D75505
target datalayout = "e-p:64:64-i32:32:32-i64:64:64-f32:32:32-f64:64:64"

define float @float_load(i32* %addr) {
; CHECK-LABEL: @float_load(
; CHECK:         %i32 = load volatile i32, i32* %addr, align 4
; CHECK-NEXT:    %float = bitcast i32 %i32 to float
; CHECK-NEXT:    ret float %float
  %i32 = load volatile i32, i32* %addr, align 4
  %float = bitcast i32 %i32 to float
  ret float %float
}

define i32 @i32_load(float* %addr) {
; CHECK-LABEL: @i32_load(
; CHECK:         %float = load volatile float, float* %addr, align 4
; CHECK-NEXT:    %i32 = bitcast float %float to i32
; CHECK-NEXT:    ret i32 %i32
  %float = load volatile float, float* %addr, align 4
  %i32 = bitcast float %float to i32
  ret i32 %i32
}

define double @double_load(i64* %addr) {
; CHECK-LABEL: @double_load(
; CHECK:         %i64 = load volatile i64, i64* %addr, align 8
; CHECK-NEXT:    %double = bitcast i64 %i64 to double
; CHECK-NEXT:    ret double %double
  %i64 = load volatile i64, i64* %addr, align 8
  %double = bitcast i64 %i64 to double
  ret double %double
}

define i64 @i64_load(double* %addr) {
; CHECK-LABEL: @i64_load(
; CHECK:         %double = load volatile double, double* %addr, align 8
; CHECK-NEXT:    %i64 = bitcast double %double to i64
; CHECK-NEXT:    ret i64 %i64
  %double = load volatile double, double* %addr, align 8
  %i64 = bitcast double %double to i64
  ret i64 %i64
}

define i8* @ptr_load(i64* %addr) {
; CHECK-LABEL: @ptr_load(
; CHECK:         %i64 = load volatile i64, i64* %addr, align 8
; CHECK-NEXT:    %ptr = inttoptr i64 %i64 to i8*
; CHECK-NEXT:    ret i8* %ptr
  %i64 = load volatile i64, i64* %addr, align 8
  %ptr = inttoptr i64 %i64 to i8*
  ret i8* %ptr
}
