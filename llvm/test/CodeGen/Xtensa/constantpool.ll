; RUN: llc -mtriple=xtensa -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

; Test placement of the i32,i64, float and double constants in constantpool

define dso_local i32 @const_i32() #0 {
; CHECK: .literal_position
; CHECK-NEXT: .literal .LCPI0_0, 74565
; CHECK-LABEL: const_i32:
; CHECK: l32r a2, .LCPI0_0
  %1 = alloca i32, align 4
  store i32 74565, ptr %1, align 4
  %2 = load i32, ptr %1, align 4
  ret i32 %2
}

define dso_local i64 @const_int64() #0 {
; CHECK: .literal_position
; CHECK-NEXT: .literal .LCPI1_0, 305419896
; CHECK-NEXT: .literal .LCPI1_1, -1859959449
; CHECK-LABEL: const_int64:
; CHECK: l32r a3, .LCPI1_0
; CHECK: l32r a2, .LCPI1_1
  %1 = alloca i64, align 8
  store i64 1311768467302729063, ptr %1, align 8
  %2 = load i64, ptr %1, align 8
  ret i64 %2
}
