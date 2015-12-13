; RUN: opt < %s -instrprof -S | FileCheck %s

target triple = "x86_64-apple-macosx10.10.0"

@__llvm_profile_name_foo = hidden constant [3 x i8] c"foo"
; CHECK: @__llvm_profile_name_foo = hidden constant [3 x i8] c"foo", section "__DATA,__llvm_prf_names", align 1
@__llvm_profile_name_bar = hidden constant [4 x i8] c"bar\00"
; CHECK: @__llvm_profile_name_bar = hidden constant [4 x i8] c"bar\00", section "__DATA,__llvm_prf_names", align 1
@__llvm_profile_name_baz = hidden constant [3 x i8] c"baz"
; CHECK: @__llvm_profile_name_baz = hidden constant [3 x i8] c"baz", section "__DATA,__llvm_prf_names", align 1

; CHECK: @__llvm_profile_counters_foo = hidden global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; CHECK: @__llvm_profile_data_foo = hidden {{.*}}, section "__DATA,__llvm_prf_data", align 8
define void @foo() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__llvm_profile_name_foo, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; CHECK: @__llvm_profile_counters_bar = hidden global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; CHECK: @__llvm_profile_data_bar = hidden {{.*}}, section "__DATA,__llvm_prf_data", align 8
define void @bar() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @__llvm_profile_name_bar, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; CHECK: @__llvm_profile_counters_baz = hidden global [3 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; CHECK: @__llvm_profile_data_baz = hidden {{.*}}, section "__DATA,__llvm_prf_data", align 8
define void @baz() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__llvm_profile_name_baz, i32 0, i32 0), i64 0, i32 3, i32 0)
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__llvm_profile_name_baz, i32 0, i32 0), i64 0, i32 3, i32 1)
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__llvm_profile_name_baz, i32 0, i32 0), i64 0, i32 3, i32 2)
  ret void
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)

; CHECK: @__llvm_profile_runtime = external global i32
; CHECK: @llvm.used = appending global {{.*}} @__llvm_profile_data_foo {{.*}} @__llvm_profile_data_bar {{.*}} @__llvm_profile_data_baz {{.*}} section "llvm.metadata"
