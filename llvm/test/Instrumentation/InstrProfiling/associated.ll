; RUN: opt < %s -instrprof -S | FileCheck %s
; RUN: opt < %s -passes=instrprof -S | FileCheck %s

@__profn_foo = hidden constant [3 x i8] c"foo"

; CHECK: @__profc_foo = hidden global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8, !associated !0
; CHECK: @__profd_foo = hidden global {{.*}}, section "__llvm_prf_data", align 8, !associated !0

define void @foo() {
  call void @llvm.instrprof.increment(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 0, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(i8*, i64, i32, i32)

; CHECK: !0 = !{[1 x i64]* @__profc_foo}
