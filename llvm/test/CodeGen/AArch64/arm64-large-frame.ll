; RUN: llc -verify-machineinstrs -mtriple=arm64-none-linux-gnu -frame-pointer=non-leaf -disable-post-ra < %s | FileCheck %s
declare void @use_addr(ptr)

@addr = global ptr null

define void @test_bigframe() {
; CHECK-LABEL: test_bigframe:
; CHECK: .cfi_startproc

  %var1 = alloca i8, i32 20000000
  %var2 = alloca i8, i32 16
  %var3 = alloca i8, i32 20000000

; CHECK:      sub sp, sp, #4095, lsl #12          // =16773120
; CHECK-NEXT: sub sp, sp, #4095, lsl #12          // =16773120
; CHECK-NEXT: sub sp, sp, #1575, lsl #12          // =6451200
; CHECK-NEXT: sub sp, sp, #2576
; CHECK-NEXT: .cfi_def_cfa_offset 40000032

; CHECK: add [[TMP:x[0-9]+]], sp, #4095, lsl #12
; CHECK: add [[TMP1:x[0-9]+]], [[TMP]], #787, lsl #12
; CHECK: add {{x[0-9]+}}, [[TMP1]], #3344
  store volatile ptr %var1, ptr @addr

  %var1plus2 = getelementptr i8, ptr %var1, i32 2
  store volatile ptr %var1plus2, ptr @addr

; CHECK: add [[TMP:x[0-9]+]], sp, #4095, lsl #12
; CHECK: add [[TMP1:x[0-9]+]], [[TMP]], #787, lsl #12
; CHECK: add {{x[0-9]+}}, [[TMP1]], #3328
  store volatile ptr %var2, ptr @addr

  %var2plus2 = getelementptr i8, ptr %var2, i32 2
  store volatile ptr %var2plus2, ptr @addr

  store volatile ptr %var3, ptr @addr

  %var3plus2 = getelementptr i8, ptr %var3, i32 2
  store volatile ptr %var3plus2, ptr @addr

; CHECK: add sp, sp, #4095, lsl #12
; CHECK: add sp, sp, #4095, lsl #12
; CHECK: add sp, sp, #1575, lsl #12
; CHECK: add sp, sp, #2576
; CHECK: .cfi_endproc
  ret void
}

define void @test_mediumframe() {
; CHECK-LABEL: test_mediumframe:
  %var1 = alloca i8, i32 1000000
  %var2 = alloca i8, i32 16
  %var3 = alloca i8, i32 1000000

; CHECK:      sub sp, sp, #488, lsl #12           // =1998848
; CHECK-NEXT: sub sp, sp, #1168
; CHECK-NEXT: .cfi_def_cfa_offset 2000032

  store volatile ptr %var1, ptr @addr
; CHECK: add     [[VAR1ADDR:x[0-9]+]], sp, #244, lsl #12
; CHECK: add     [[VAR1ADDR]], [[VAR1ADDR]], #592

; CHECK: add [[VAR2ADDR:x[0-9]+]], sp, #244, lsl #12
; CHECK: add [[VAR2ADDR]], [[VAR2ADDR]], #576

  store volatile ptr %var2, ptr @addr
; CHECK: add     sp, sp, #488, lsl #12
; CHECK: add     sp, sp, #1168
  ret void
}
