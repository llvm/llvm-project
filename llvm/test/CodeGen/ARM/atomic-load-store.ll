; RUN: llc < %s -mtriple=armv7-apple-ios -verify-machineinstrs | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=armv7-apple-ios -O0 | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=thumbv7-apple-ios | FileCheck %s -check-prefix=THUMBTWO
; RUN: llc < %s -mtriple=thumbv6-apple-ios | FileCheck %s -check-prefix=THUMBONE

define void @test1(i32* %ptr, i32 %val1) {
; ARM: test1
; ARM: dmb ish
; ARM-NEXT: str
; ARM-NEXT: dmb ish
; THUMBONE: test1
; THUMBONE: __sync_lock_test_and_set_4
; THUMBTWO: test1
; THUMBTWO: dmb ish
; THUMBTWO-NEXT: str
; THUMBTWO-NEXT: dmb ish
  store atomic i32 %val1, i32* %ptr seq_cst, align 4
  ret void
}

define i32 @test2(i32* %ptr) {
; ARM: test2
; ARM: ldr
; ARM-NEXT: dmb ish
; THUMBONE: test2
; THUMBONE: __sync_val_compare_and_swap_4
; THUMBTWO: test2
; THUMBTWO: ldr
; THUMBTWO-NEXT: dmb ish
  %val = load atomic i32* %ptr seq_cst, align 4
  ret i32 %val
}

define void @test3(i8* %ptr1, i8* %ptr2) {
; ARM: test3
; ARM: ldrb
; ARM: strb
; THUMBTWO: test3
; THUMBTWO: ldrb
; THUMBTWO: strb
; THUMBONE: test3
; THUMBONE: ldrb
; THUMBONE: strb
  %val = load atomic i8* %ptr1 unordered, align 1
  store atomic i8 %val, i8* %ptr2 unordered, align 1
  ret void
}

define void @test4(i8* %ptr1, i8* %ptr2) {
; THUMBONE: test4
; THUMBONE: ___sync_val_compare_and_swap_1
; THUMBONE: ___sync_lock_test_and_set_1
  %val = load atomic i8* %ptr1 seq_cst, align 1
  store atomic i8 %val, i8* %ptr2 seq_cst, align 1
  ret void
}
