;; Counter increments under TSan should be atomic to prevent it reporting
;; them as data races. The !nosanitize attribute should also be added to
;; prevent unnecessary (expensive) calls into the runtime.

; RUN: opt < %s -passes=instrprof -S | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

@__profn_foo_with_tsan = private constant [13 x i8] c"foo_with_tsan"
@__profn_foo_without_tsan = private constant [16 x i8] c"foo_without_tsan"

; Function Attrs: noinline nounwind optnone sanitize_thread ssp uwtable
define void @foo_with_tsan() sanitize_thread #0 {
  ; CHECK-LABEL: define void @foo_with_tsan
  call void @llvm.instrprof.increment(ptr @__profn_foo_with_tsan, i64 0, i32 1, i32 0)
  ; CHECK-NEXT: atomicrmw add ptr @__profc_foo_with_tsan, i64 1 monotonic, align 8, !nosanitize
  ret void
}

define void @foo_without_tsan() #0 {
  ; CHECK-LABEL: define void @foo_without_tsan
  call void @llvm.instrprof.increment(ptr @__profn_foo_without_tsan, i64 0, i32 1, i32 0)
  ; CHECK-NEXT: [[REGISTER1:%[a-zA-Z0-9_.]+]] = load i64, ptr @__profc_foo_without_tsan, align 8, !nosanitize
  ; CHECK-NEXT: [[REGISTER2:%[a-zA-Z0-9_.]+]] = add i64 [[REGISTER1]], 1
  ; CHECK-NEXT: store i64 [[REGISTER2]], ptr @__profc_foo_without_tsan, align 8, !nosanitize
  ret void
}