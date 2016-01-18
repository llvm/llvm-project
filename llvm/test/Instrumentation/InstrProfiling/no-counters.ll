;; No instrumentation should be emitted if there are no counter increments.

; RUN: opt < %s -instrprof -S | FileCheck %s
; CHECK-NOT: @__llvm_profile_counters
; CHECK-NOT: @__llvm_profile_data
; CHECK-NOT: @__llvm_profile_runtime

define void @foo() {
  ret void
}
