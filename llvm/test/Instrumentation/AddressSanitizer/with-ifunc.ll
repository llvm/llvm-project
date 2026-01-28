; Test -asan-with-ifunc flag.
;
; RUN: opt -passes=asan -S -asan-with-ifunc=0 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-NOIFUNC
; RUN: opt -passes=asan -S -asan-with-ifunc=1 -asan-with-ifunc-suppress-remat=0 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-IFUNC
; RUN: opt -passes=asan -S -asan-with-ifunc=1 -asan-with-ifunc-suppress-remat=1 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-IFUNC-NOREMAT

; RUN: opt -passes=asan -S -asan-with-ifunc=1 -asan-with-ifunc-suppress-remat=0 -mtriple=armv7-linux-android < %s | FileCheck %s --check-prefixes=CHECK,CHECK-IFUNC

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7--linux-android"

; CHECK-IFUNC: @__asan_shadow = external global [0 x i8]
; CHECK-NOIFUNC: @__asan_shadow_memory_dynamic_address = external global i32

define i32 @test_load(ptr %a) sanitize_address {
; First instrumentation in the function must be to load the dynamic shadow
; address into a local variable.
; CHECK-LABEL: @test_load
; CHECK: entry:

; CHECK-IFUNC-NEXT: %[[A:[^ ]*]] = ptrtoint ptr %a to i32
; CHECK-IFUNC-NEXT: %[[B:[^ ]*]] = lshr i32 %[[A]], 3
; CHECK-IFUNC-NEXT: %[[C:[^ ]*]] = add i32 %[[B]], ptrtoint (ptr @__asan_shadow to i32)

; CHECK-IFUNC-NOREMAT-NEXT: %[[S:[^ ]*]] = call i32 asm "", "=r,0"(ptr @__asan_shadow)
; CHECK-IFUNC-NOREMAT-NEXT: %[[A:[^ ]*]] = ptrtoint ptr %a to i32
; CHECK-IFUNC-NOREMAT-NEXT: %[[B:[^ ]*]] = lshr i32 %[[A]], 3
; CHECK-IFUNC-NOREMAT-NEXT: %[[C:[^ ]*]] = add i32 %[[B]], %[[S]]

; CHECK-NOIFUNC-NEXT: %[[SHADOW:[^ ]*]] = load i32, ptr @__asan_shadow_memory_dynamic_address
; CHECK-NOIFUNC-NEXT: %[[A:[^ ]*]] = ptrtoint ptr %a to i32
; CHECK-NOIFUNC-NEXT: %[[B:[^ ]*]] = lshr i32 %[[A]], 3
; CHECK-NOIFUNC-NEXT: %[[C:[^ ]*]] = add i32 %[[B]], %[[SHADOW]]

entry:
  %x = load i32, ptr %a, align 4
  ret i32 %x
}
