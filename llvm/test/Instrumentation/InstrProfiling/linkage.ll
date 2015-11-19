;; Check that runtime symbols get appropriate linkage.

; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -instrprof -S | FileCheck %s --check-prefix=OTHER --check-prefix=COMMON
; RUN: opt < %s -mtriple=x86_64-unknown-linux -instrprof -S | FileCheck %s --check-prefix=LINUX --check-prefix=COMMON

@__llvm_profile_name_foo = hidden constant [3 x i8] c"foo"
@__llvm_profile_name_foo_weak = weak hidden constant [8 x i8] c"foo_weak"
@"__llvm_profile_name_linkage.ll:foo_internal" = internal constant [23 x i8] c"linkage.ll:foo_internal"
@__llvm_profile_name_foo_inline = linkonce_odr hidden constant [10 x i8] c"foo_inline"

; COMMON: @__llvm_profile_counters_foo = hidden global
; COMMON: @__llvm_profile_data_foo = hidden global
define void @foo() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__llvm_profile_name_foo, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; COMMON: @__llvm_profile_counters_foo_weak = weak hidden global
; COMMON: @__llvm_profile_data_foo_weak = weak hidden global
define weak void @foo_weak() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @__llvm_profile_name_foo_weak, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; COMMON: @"__llvm_profile_counters_linkage.ll:foo_internal" = internal global
; COMMON: @"__llvm_profile_data_linkage.ll:foo_internal" = internal global
define internal void @foo_internal() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @"__llvm_profile_name_linkage.ll:foo_internal", i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; COMMON: @__llvm_profile_counters_foo_inline = linkonce_odr hidden global
; COMMON: @__llvm_profile_data_foo_inline = linkonce_odr hidden global
define linkonce_odr void @foo_inline() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @__llvm_profile_name_foo_inline, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)

; OTHER: @__llvm_profile_runtime = external global i32
; LINUX-NOT: @__llvm_profile_runtime = external global i32

; OTHER: define linkonce_odr hidden i32 @__llvm_profile_runtime_user() {{.*}} {
; OTHER:   %[[REG:.*]] = load i32, i32* @__llvm_profile_runtime
; OTHER:   ret i32 %[[REG]]
; OTHER: }
; LINUX-NOT: define linkonce_odr hidden i32 @__llvm_profile_runtime_user() {{.*}} {
; LINUX-NOT:   %[[REG:.*]] = load i32, i32* @__llvm_profile_runtime
