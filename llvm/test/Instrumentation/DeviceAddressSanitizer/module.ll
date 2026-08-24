; RUN: opt < %s -passes=dasan -S | FileCheck %s
; RUN: opt < %s -passes=dasan,dasan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i8:8:8-i16:16:16-i32:32:32-i64:64:64-n8:16:32:64-S128-A5-G1"
target triple = "amdgcn-amd-amdhsa"

; CHECK-NOT: @__dasan_no_entry
; CHECK-NOT: @__dasan_instrumented
; CHECK-NOT: @__dasan_report

define void @nothing_to_check(i32 %x) sanitize_device_address {
; CHECK-LABEL: define void @nothing_to_check(
; CHECK-NEXT:    ret void
  ret void
}

; CHECK: !{i32 4, !"nosanitize_device_address", i32 1}
