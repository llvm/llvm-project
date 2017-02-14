; RUN: opt < %s -instrprof -S | FileCheck %s
; RUN: opt < %s -passes=instrprof -S | FileCheck %s

target triple = "x86_64-apple-macosx10.10.0"

@__profn_foo = hidden constant [3 x i8] c"foo"
; CHECK-NOT: __profn_foo
@__profn_bar = hidden constant [4 x i8] c"bar\00"
; CHECK-NOT: __profn_bar
@__profn_baz = hidden constant [3 x i8] c"baz"
; CHECK-NOT: __profn_baz

; CHECK: @__profc_foo = hidden global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; CHECK: @__profd_foo = hidden {{.*}}, section "__DATA,__llvm_prf_data,regular,live_support", align 8
define void @foo() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; CHECK: @__profc_bar = hidden global [1 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; CHECK: @__profd_bar = hidden {{.*}}, section "__DATA,__llvm_prf_data,regular,live_support", align 8
define void @bar() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @__profn_bar, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

; CHECK: @__profc_baz = hidden global [3 x i64] zeroinitializer, section "__DATA,__llvm_prf_cnts", align 8
; CHECK: @__profd_baz = hidden {{.*}}, section "__DATA,__llvm_prf_data,regular,live_support", align 8
define void @baz() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_baz, i32 0, i32 0), i64 0, i32 3, i32 0)
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_baz, i32 0, i32 0), i64 0, i32 3, i32 1)
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_baz, i32 0, i32 0), i64 0, i32 3, i32 2)
  ret void
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)

; CHECK: @__llvm_profile_runtime = external global i32
; CHECK: @llvm.used = appending global {{.*}} @__profd_foo {{.*}} @__profd_bar {{.*}} @__profd_baz {{.*}} section "llvm.metadata"
