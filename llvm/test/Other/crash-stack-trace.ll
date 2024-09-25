; REQUIRES: asserts

; RUN: not --crash opt -passes=trigger-crash-module %s -disable-output 2>&1 | \
; RUN: FileCheck %s --check-prefix=CHECK-MODULE

; CHECK-MODULE:      Stack dump:
; CHECK-MODULE-NEXT: 0. Program arguments:
; CHECK-MODULE-NEXT: 1. Running pass "trigger-crash-module" on module "{{.*}}crash-stack-trace.ll"

; RUN: not --crash opt -passes='sroa,trigger-crash-function' %s -disable-output 2>&1 | \
; RUN: FileCheck %s --check-prefix=CHECK-FUNCTION

; CHECK-FUNCTION:      Stack dump:
; CHECK-FUNCTION-NEXT: 0. Program arguments:
; CHECK-FUNCTION-NEXT: 1. Running pass "function(sroa<modify-cfg>,trigger-crash-function)" on module "{{.*}}crash-stack-trace.ll"
; CHECK-FUNCTION-NEXT: 2. Running pass "trigger-crash-function" on function "foo"

define void @foo() {
  ret void
}
