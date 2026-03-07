; RUN: opt < %s -passes=instrprof -S | FileCheck %s

; Test that weak functions on GPU targets get weak linkage for their
; __profd_ aliases to allow linker deduplication across TUs.
; Non-weak functions get external linkage (default for aliases).

target triple = "amdgcn-amd-amdhsa"

@__hip_cuid_abc123 = addrspace(1) global i8 0

; Weak function should have weak linkage on its profile data alias
; CHECK: @__profd_weak_func = weak protected alias
@__profn_weak_func = private constant [9 x i8] c"weak_func"

define weak void @weak_func() {
  call void @llvm.instrprof.increment(ptr @__profn_weak_func, i64 0, i32 1, i32 0)
  ret void
}

; Weak ODR function should have weak_odr linkage on its profile data alias
; CHECK: @__profd_weak_odr_func = weak_odr protected alias
@__profn_weak_odr_func = private constant [13 x i8] c"weak_odr_func"

define weak_odr void @weak_odr_func() {
  call void @llvm.instrprof.increment(ptr @__profn_weak_odr_func, i64 0, i32 1, i32 0)
  ret void
}

; Non-weak function should have external linkage (no linkage keyword shown)
; CHECK: @__profd_normal_func = protected alias
@__profn_normal_func = private constant [11 x i8] c"normal_func"

define void @normal_func() {
  call void @llvm.instrprof.increment(ptr @__profn_normal_func, i64 0, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)
