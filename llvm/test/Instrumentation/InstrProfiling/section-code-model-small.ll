;; Check that globals are not marked large under x86-64 small code model.
; RUN: opt %s -mtriple=x86_64-unknown-linux -passes=instrprof -S | FileCheck %s

@__profn_foo = private constant [3 x i8] c"foo"

define i32 @foo(ptr) {
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 12884901887, i32 1, i32 0)
  %2 = ptrtoint ptr %0 to i64
  call void @llvm.instrprof.value.profile(ptr @__profn_foo, i64 12884901887, i64 %2, i32 0, i32 0)
  %3 = tail call i32 %0()
  ret i32 %3
}

; Function Attrs: nounwind
declare void @llvm.instrprof.increment(ptr, i64, i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.instrprof.value.profile(ptr, i64, i64, i32, i32) #0

attributes #0 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"Code Model", i32 1}

; CHECK: @__profc_foo =
; CHECK-NOT: code_model "large"
; CHECK: @__profvp_foo =
; CHECK-NOT: code_model "large"
; CHECK: @__profd_foo =
; CHECK-NOT: code_model "large"
; CHECK: @__llvm_prf_vnodes =
; CHECK-NOT: code_model "large"
; CHECK: @__llvm_prf_nm =
; CHECK-NOT: code_model "large"
