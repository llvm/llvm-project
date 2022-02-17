; RUN: llc < %s -mtriple=armv7-apple-ios -verify-machineinstrs | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=armv7-apple-ios -O0 | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=thumbv7-apple-ios -verify-machineinstrs | FileCheck %s -check-prefix=THUMBTWO
; RUN: llc < %s -mtriple=thumbv6-apple-ios | FileCheck %s -check-prefix=THUMBONE
; RUN: llc < %s -mtriple=armv4-apple-ios | FileCheck %s -check-prefix=ARMV4
; RUN: llc < %s -mtriple=armv6-apple-ios | FileCheck %s -check-prefix=ARMV6
; RUN: llc < %s -mtriple=thumbv7m-apple-ios | FileCheck %s -check-prefix=THUMBM

define void @test1(i32* %ptr, i32 %val1) {
; ARM-LABEL: test1
; ARM: dmb {{ish$}}
; ARM-NEXT: str
; ARM-NEXT: dmb {{ish$}}
; THUMBONE-LABEL: test1
; THUMBONE: __sync_lock_test_and_set_4
; THUMBTWO-LABEL: test1
; THUMBTWO: dmb {{ish$}}
; THUMBTWO-NEXT: str
; THUMBTWO-NEXT: dmb {{ish$}}
; ARMV6-LABEL: test1
; ARMV6: mcr p15, #0, {{r[0-9]*}}, c7, c10, #5
; ARMV6: str
; ARMV6: mcr p15, #0, {{r[0-9]*}}, c7, c10, #5
; THUMBM-LABEL: test1
; THUMBM: dmb sy
; THUMBM: str
; THUMBM: dmb sy
  store atomic i32 %val1, i32* %ptr seq_cst, align 4
  ret void
}

define i32 @test2(i32* %ptr) {
; ARM-LABEL: test2
; ARM: ldr
; ARM-NEXT: dmb {{ish$}}
; THUMBONE-LABEL: test2
; THUMBONE: __sync_val_compare_and_swap_4
; THUMBTWO-LABEL: test2
; THUMBTWO: ldr
; THUMBTWO-NEXT: dmb {{ish$}}
; ARMV6-LABEL: test2
; ARMV6: ldr
; ARMV6: mcr p15, #0, {{r[0-9]*}}, c7, c10, #5
; THUMBM-LABEL: test2
; THUMBM: ldr
; THUMBM: dmb sy
  %val = load atomic i32, i32* %ptr seq_cst, align 4
  ret i32 %val
}

define void @test3(i8* %ptr1, i8* %ptr2) {
; ARM-LABEL: test3
; ARM-NOT: dmb
; ARM: ldrb
; ARM-NOT: dmb
; ARM: strb
; ARM-NOT: dmb
; ARM: bx lr

; THUMBTWO-LABEL: test3
; THUMBTWO-NOT: dmb
; THUMBTWO: ldrb
; THUMBTWO-NOT: dmb
; THUMBTWO: strb
; THUMBTWO-NOT: dmb
; THUMBTWO: bx lr

; THUMBONE-LABEL: test3
; THUMBONE-NOT: dmb
; THUMBONE: ldrb
; THUMBONE-NOT: dmb
; THUMBONE: strb
; THUMBONE-NOT: dmb

; ARMV6-LABEL: test3
; ARMV6-NOT: mcr
; THUMBM-LABEL: test3
; THUMBM-NOT: dmb sy
  %val = load atomic i8, i8* %ptr1 unordered, align 1
  store atomic i8 %val, i8* %ptr2 unordered, align 1
  ret void
}

define void @test4(i8* %ptr1, i8* %ptr2) {
; THUMBONE-LABEL: test4
; THUMBONE: ___sync_val_compare_and_swap_1
; THUMBONE: ___sync_lock_test_and_set_1
; ARMV6-LABEL: test4
; THUMBM-LABEL: test4
  %val = load atomic i8, i8* %ptr1 seq_cst, align 1
  store atomic i8 %val, i8* %ptr2 seq_cst, align 1
  ret void
}

define i64 @test_old_load_64bit(i64* %p) {
; ARMV4-LABEL: test_old_load_64bit
; ARMV4: ___atomic_load_8
  %1 = load atomic i64, i64* %p seq_cst, align 8
  ret i64 %1
}

define void @test_old_store_64bit(i64* %p, i64 %v) {
; ARMV4-LABEL: test_old_store_64bit
; ARMV4: ___atomic_store_8
  store atomic i64 %v, i64* %p seq_cst, align 8
  ret void
}
