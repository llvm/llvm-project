; Test -hwasan-with-ifunc flag.
;
; RUN: opt -hwasan -S < %s | \
; RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-TLS,CHECK-HISTORY
; RUN: opt -hwasan -S -hwasan-with-ifunc=0 -hwasan-with-tls=1 -hwasan-record-stack-history=1 < %s | \
; RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-TLS,CHECK-HISTORY
; RUN: opt -hwasan -S -hwasan-with-ifunc=0 -hwasan-with-tls=1 -hwasan-record-stack-history=0 < %s | \
; RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-TLS,CHECK-NOHISTORY
; RUN: opt -hwasan -S -hwasan-with-ifunc=0 -hwasan-with-tls=0 < %s | \
; RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-GLOBAL,CHECK-NOHISTORY
; RUN: opt -hwasan -S -hwasan-with-ifunc=1  -hwasan-with-tls=0 < %s | \
; RUN:     FileCheck %s --check-prefixes=CHECK,CHECK-IFUNC,CHECK-NOHISTORY

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

; CHECK-GLOBAL: load i64, i64* @__hwasan_shadow_memory_dynamic_address

; CHECK-TLS:   %[[A:[^ ]*]] = call i8* @llvm.thread.pointer()
; CHECK-TLS:   %[[B:[^ ]*]] = getelementptr i8, i8* %[[A]], i32 48
; CHECK-TLS:   %[[C:[^ ]*]] = bitcast i8* %[[B]] to i64*
; CHECK-TLS:   %[[D:[^ ]*]] = load i64, i64* %[[C]]
; CHECK-TLS:   %[[E:[^ ]*]] = or i64 %[[D]], 4294967295
; CHECK-TLS:   = add i64 %[[E]], 1

; "store i64" is only used to update stack history (this input IR intentionally does not use any i64)
; W/o any allocas, the history is not updated, even if it is enabled explicitly with -hwasan-record-stack-history=1
; CHECK-NOT: store i64

; CHECK: ret i32

entry:
  %x = load i32, i32* %a, align 4
  ret i32 %x
}

declare void @use(i32* %p)

define void @test_alloca() sanitize_hwaddress {
; First instrumentation in the function must be to load the dynamic shadow
; address into a local variable.
; CHECK-LABEL: @test_alloca
; CHECK: entry:

; CHECK-IFUNC:   %[[A:[^ ]*]] = call i64 asm "", "=r,0"([0 x i8]* @__hwasan_shadow)
; CHECK-IFUNC:   add i64 %{{.*}}, %[[A]]

; CHECK-GLOBAL: load i64, i64* @__hwasan_shadow_memory_dynamic_address

; CHECK-TLS:   %[[A:[^ ]*]] = call i8* @llvm.thread.pointer()
; CHECK-TLS:   %[[B:[^ ]*]] = getelementptr i8, i8* %[[A]], i32 48
; CHECK-TLS:   %[[C:[^ ]*]] = bitcast i8* %[[B]] to i64*
; CHECK-TLS:   %[[D:[^ ]*]] = load i64, i64* %[[C]]

; CHECK-NOHISTORY-NOT: store i64

; CHECK-HISTORY: %[[PTR:[^ ]*]] = inttoptr i64 %[[D]] to i64*
; CHECK-HISTORY: store i64 %{{.*}}, i64* %[[PTR]]
; CHECK-HISTORY: %[[D1:[^ ]*]] = ashr i64 %[[D]], 56
; CHECK-HISTORY: %[[D2:[^ ]*]] = shl nuw nsw i64 %[[D1]], 12
; CHECK-HISTORY: %[[D3:[^ ]*]] = xor i64 %[[D2]], -1
; CHECK-HISTORY: %[[D4:[^ ]*]] = add i64 %[[D]], 8
; CHECK-HISTORY: %[[D5:[^ ]*]] = and i64 %[[D4]], %[[D3]]
; CHECK-HISTORY: store i64 %[[D5]], i64* %[[C]]

; CHECK-TLS:   %[[E:[^ ]*]] = or i64 %[[D]], 4294967295
; CHECK-TLS:   = add i64 %[[E]], 1

; CHECK-NOHISTORY-NOT: store i64


entry:
  %x = alloca i32, align 4
  call void @use(i32* %x)
  ret void
}
