; RUN: opt -passes='default<O0>' -S -mtriple=amdgcn-amd-amdhsa -amdgpu-internalize-symbols < %s | FileCheck %s
; RUN: opt -passes='default<O1>' -S -mtriple=amdgcn-amd-amdhsa -amdgpu-internalize-symbols < %s | FileCheck %s

; CHECK: define void @__dasan_report_load(
define void @__dasan_report_load(ptr %p, i64 %size, i64 %base, i64 %limit) {
  ret void
}

; CHECK: define weak void @__dasan_report_store_noabort(
define weak void @__dasan_report_store_noabort(ptr %p, i64 %size, i64 %base, i64 %limit) {
  ret void
}
