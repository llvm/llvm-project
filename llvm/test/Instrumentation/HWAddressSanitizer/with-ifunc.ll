; Test -hwasan-with-ifunc flag.
;
; RUN: opt -hwasan -S < %s | \
; RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-IFUNC
; RUN: opt -hwasan -S -hwasan-with-ifunc=0 < %s | \
; RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-NOIFUNC
; RUN: opt -hwasan -S -hwasan-with-ifunc=1 < %s | \
; RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-IFUNC

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android22"

; CHECK-IFUNC: @__hwasan_shadow = external global [0 x i8]
; CHECK-NOIFUNC: @__hwasan_shadow_memory_dynamic_address = external global i64

define i32 @test_load(i32* %a) sanitize_hwaddress {
; First instrumentation in the function must be to load the dynamic shadow
; address into a local variable.
; CHECK-LABEL: @test_load
; CHECK: entry:

; CHECK-IFUNC:   %[[A:[^ ]*]] = call i64 asm "", "=r,0"([0 x i8]* @__hwasan_shadow)
; CHECK-IFUNC:   add i64 %{{.*}}, %[[A]]

; CHECK-NOIFUNC: load i64, i64* @__hwasan_shadow_memory_dynamic_address

entry:
  %x = load i32, i32* %a, align 4
  ret i32 %x
}
