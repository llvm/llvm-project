;; Check that the prf_vnodes variable has the correct alignment
;; when pointers are 4 byte long.
; RUN: opt %s -passes=instrprof -S | FileCheck %s --check-prefix=ALIGN32

target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

@__profn_foo = private constant [3 x i8] c"foo"
@__profn_bar = private constant [3 x i8] c"bar"

define i32 @foo(ptr ) {
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 12884901887, i32 1, i32 0)
  %2 = ptrtoint ptr %0 to i64
  call void @llvm.instrprof.value.profile(ptr @__profn_foo, i64 12884901887, i64 %2, i32 0, i32 0)
  %3 = tail call i32 %0()
  ret i32 %3
}

$bar = comdat any

define i32 @bar(ptr ) comdat {
entry:
  call void @llvm.instrprof.increment(ptr @__profn_bar, i64 12884901887, i32 1, i32 0)
  %1 = ptrtoint ptr %0 to i64
  call void @llvm.instrprof.value.profile(ptr @__profn_bar, i64 12884901887, i64 %1, i32 0, i32 0)
  %2 = tail call i32 %0()
  ret i32 %2
}

; Function Attrs: nounwind
declare void @llvm.instrprof.increment(ptr, i64, i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.instrprof.value.profile(ptr, i64, i64, i32, i32) #0

attributes #0 = { nounwind }

; ALIGN32: @__llvm_prf_vnodes = private global {{.*}} section "__llvm_prf_vnds", align 4
