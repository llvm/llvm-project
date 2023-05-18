; Test basic address sanitizer instrumentation.
;
; RUN: opt < %s -passes=hwasan -hwasan-instrument-with-calls -S | FileCheck %s --check-prefixes=CHECK,ABORT
; RUN: opt < %s -passes=hwasan -hwasan-instrument-with-calls -hwasan-recover=1 -S | FileCheck %s --check-prefixes=CHECK,RECOVER

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux"

define i8 @test_load8(ptr %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load8(
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %a to i64
; ABORT: call void @__hwasan_load1(i64 %[[A]])
; RECOVER: call void @__hwasan_load1_noabort(i64 %[[A]])
; CHECK: %[[B:[^ ]*]] = load i8, ptr %a
; CHECK: ret i8 %[[B]]

entry:
  %b = load i8, ptr %a, align 4
  ret i8 %b
}

define i16 @test_load16(ptr %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load16(
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %a to i64
; ABORT: call void @__hwasan_load2(i64 %[[A]])
; RECOVER: call void @__hwasan_load2_noabort(i64 %[[A]])
; CHECK: %[[B:[^ ]*]] = load i16, ptr %a
; CHECK: ret i16 %[[B]]

entry:
  %b = load i16, ptr %a, align 4
  ret i16 %b
}

define i32 @test_load32(ptr %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load32(
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %a to i64
; ABORT: call void @__hwasan_load4(i64 %[[A]])
; RECOVER: call void @__hwasan_load4_noabort(i64 %[[A]])
; CHECK: %[[B:[^ ]*]] = load i32, ptr %a
; CHECK: ret i32 %[[B]]

entry:
  %b = load i32, ptr %a, align 4
  ret i32 %b
}

define i64 @test_load64(ptr %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load64(
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %a to i64
; ABORT: call void @__hwasan_load8(i64 %[[A]])
; RECOVER: call void @__hwasan_load8_noabort(i64 %[[A]])
; CHECK: %[[B:[^ ]*]] = load i64, ptr %a
; CHECK: ret i64 %[[B]]

entry:
  %b = load i64, ptr %a, align 8
  ret i64 %b
}

define i128 @test_load128(ptr %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load128(
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %a to i64
; ABORT: call void @__hwasan_load16(i64 %[[A]])
; RECOVER: call void @__hwasan_load16_noabort(i64 %[[A]])
; CHECK: %[[B:[^ ]*]] = load i128, ptr %a
; CHECK: ret i128 %[[B]]

entry:
  %b = load i128, ptr %a, align 16
  ret i128 %b
}

define i40 @test_load40(ptr %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load40(
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %a to i64
; ABORT: call void @__hwasan_loadN(i64 %[[A]], i64 5)
; RECOVER: call void @__hwasan_loadN_noabort(i64 %[[A]], i64 5)
; CHECK: %[[B:[^ ]*]] = load i40, ptr %a
; CHECK: ret i40 %[[B]]

entry:
  %b = load i40, ptr %a, align 4
  ret i40 %b
}

define void @test_store8(ptr %a, i8 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store8(
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %a to i64
; ABORT: call void @__hwasan_store1(i64 %[[A]])
; RECOVER: call void @__hwasan_store1_noabort(i64 %[[A]])
; CHECK: store i8 %b, ptr %a
; CHECK: ret void

entry:
  store i8 %b, ptr %a, align 4
  ret void
}

define void @test_store16(ptr %a, i16 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store16(
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %a to i64
; ABORT: call void @__hwasan_store2(i64 %[[A]])
; RECOVER: call void @__hwasan_store2_noabort(i64 %[[A]])
; CHECK: store i16 %b, ptr %a
; CHECK: ret void

entry:
  store i16 %b, ptr %a, align 4
  ret void
}

define void @test_store32(ptr %a, i32 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store32(
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %a to i64
; ABORT: call void @__hwasan_store4(i64 %[[A]])
; RECOVER: call void @__hwasan_store4_noabort(i64 %[[A]])
; CHECK: store i32 %b, ptr %a
; CHECK: ret void

entry:
  store i32 %b, ptr %a, align 4
  ret void
}

define void @test_store64(ptr %a, i64 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store64(
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %a to i64
; ABORT: call void @__hwasan_store8(i64 %[[A]])
; RECOVER: call void @__hwasan_store8_noabort(i64 %[[A]])
; CHECK: store i64 %b, ptr %a
; CHECK: ret void

entry:
  store i64 %b, ptr %a, align 8
  ret void
}

define void @test_store128(ptr %a, i128 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store128(
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %a to i64
; ABORT: call void @__hwasan_store16(i64 %[[A]])
; RECOVER: call void @__hwasan_store16_noabort(i64 %[[A]])
; CHECK: store i128 %b, ptr %a
; CHECK: ret void

entry:
  store i128 %b, ptr %a, align 16
  ret void
}

define void @test_store40(ptr %a, i40 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store40(
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %a to i64
; ABORT: call void @__hwasan_storeN(i64 %[[A]], i64 5)
; RECOVER: call void @__hwasan_storeN_noabort(i64 %[[A]], i64 5)
; CHECK: store i40 %b, ptr %a
; CHECK: ret void

entry:
  store i40 %b, ptr %a, align 4
  ret void
}

define i8 @test_load_noattr(ptr %a) {
; CHECK-LABEL: @test_load_noattr(
; CHECK-NEXT: entry:
; CHECK-NEXT: %[[B:[^ ]*]] = load i8, ptr %a
; CHECK-NEXT: ret i8 %[[B]]

entry:
  %b = load i8, ptr %a, align 4
  ret i8 %b
}

define i8 @test_load_notmyattr(ptr %a) sanitize_address {
; CHECK-LABEL: @test_load_notmyattr(
; CHECK-NEXT: entry:
; CHECK-NEXT: %[[B:[^ ]*]] = load i8, ptr %a
; CHECK-NEXT: ret i8 %[[B]]

entry:
  %b = load i8, ptr %a, align 4
  ret i8 %b
}

define i8 @test_load_addrspace(ptr addrspace(256) %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load_addrspace(
; CHECK-NEXT: entry:
; CHECK-NEXT: %[[B:[^ ]*]] = load i8, ptr addrspace(256) %a
; CHECK-NEXT: ret i8 %[[B]]

entry:
  %b = load i8, ptr addrspace(256) %a, align 4
  ret i8 %b
}

; CHECK: declare void @__hwasan_init()

; CHECK:      define internal void @hwasan.module_ctor() #[[#]] comdat {
; CHECK-NEXT:   call void @__hwasan_init()
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
