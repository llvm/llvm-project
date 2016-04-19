;; No instrumentation should be emitted if there are no counter increments.

; RUN: opt < %s -instrprof -S | FileCheck %s
; RUN: opt < %s -passes=instrprof -S | FileCheck %s
; CHECK-NOT: @__profc
; CHECK-NOT: @__profd
; CHECK-NOT: @__llvm_profile_runtime

define void @foo() {
  ret void
}
