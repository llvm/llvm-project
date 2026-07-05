; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s

; Test that the cost heuristic for a folded load works also for a pointer operand.
define void @fun0(ptr %lhs, ptr %rhs_ptr) {
  %rhs = load ptr, ptr %rhs_ptr
  %c = icmp eq ptr %lhs, %rhs
  ret void
; CHECK: function 'fun0'
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %rhs = load
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %c = icmp
}
