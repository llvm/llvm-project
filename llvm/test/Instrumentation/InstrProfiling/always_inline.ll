; Check that '__attribute__((always_inline)) inline' functions are inlined.

; RUN: opt < %s -mtriple=x86_64-apple-macosx10.10.0 -passes='instrprof,inline' -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

;; Don't record the address of an available_externally alwaysinline function.
; CHECK:      @__profd_foo = linkonce_odr hidden global
; CHECK-NOT:  @foo
; CHECK-SAME: , align 8

@__profn_foo = linkonce_odr hidden constant [3 x i8] c"foo"

; CHECK-LABEL: @main
; CHECK-NOT: call
define i32 @main() {
entry:
  %call = call i32 @foo()
  ret i32 %call
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32) #0

define available_externally i32 @foo() #1 {
entry:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 1, i32 0)
  ret i32 0
}

attributes #0 = { nounwind }
attributes #1 = { alwaysinline }

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"EnableValueProfiling", i32 1}
