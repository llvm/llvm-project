; Test basic address sanitizer instrumentation.
; Generic code is covered by ../basic.ll, only the x86_64 specific code is
; tested here.
;
; RUN: opt < %s -passes=hwasan -hwasan-recover=0 -S | FileCheck %s  --check-prefixes=CHECK,ABORT
; RUN: opt < %s -passes=hwasan -hwasan-recover=1 -S | FileCheck %s  --check-prefixes=CHECK,RECOVER

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i8 @test_load8(ptr %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load8(
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %a to i64

; ABORT: call void @__hwasan_load1(i64 %[[A]])
; RECOVER: call void @__hwasan_load1_noabort(i64 %[[A]])

; CHECK: %[[G:[^ ]*]] = load i8, ptr %a, align 4
; CHECK: ret i8 %[[G]]

entry:
  %b = load i8, ptr %a, align 4
  ret i8 %b
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

; CHECK: store i8 %b, ptr %a, align 4
; CHECK: ret void

entry:
  store i8 %b, ptr %a, align 4
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

define void @test_store_unaligned(ptr %a, i64 %b) sanitize_hwaddress {
; CHECK-LABEL: @test_store_unaligned(
; CHECK: %[[A:[^ ]*]] = ptrtoint ptr %a to i64

; ABORT: call void @__hwasan_storeN(i64 %[[A]], i64 8)
; RECOVER: call void @__hwasan_storeN_noabort(i64 %[[A]], i64 8)

; CHECK: store i64 %b, ptr %a, align 4
; CHECK: ret void

entry:
  store i64 %b, ptr %a, align 4
  ret void
}
