; REQUIRES: asserts, backtrace

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

; RUN: not --crash opt -passes='cgscc(trigger-crash-cgscc)' %s -disable-output 2>&1 | \
; RUN: FileCheck %s --check-prefix=CHECK-CGSCC

; CHECK-CGSCC:      Stack dump:
; CHECK-CGSCC-NEXT: 0. Program arguments:
; CHECK-CGSCC-NEXT: 1. Running pass "cgscc(trigger-crash-cgscc)" on module "{{.*}}crash-stack-trace.ll"

; RUN: not --crash opt -passes='function(loop(trigger-crash-loop))' %s -disable-output 2>&1 | \
; RUN: FileCheck %s --check-prefix=CHECK-LOOP

; CHECK-LOOP:      Stack dump:
; CHECK-LOOP-NEXT: 0. Program arguments:
; CHECK-LOOP-NEXT: 1. Running pass "function(loop(trigger-crash-loop))" on module "{{.*}}crash-stack-trace.ll"
; CHECK-LOOP-NEXT: 2. Running pass "loop(trigger-crash-loop)" on function "foo"

define void @foo() {
entry:
  br label %loop
loop:
  br label %loop
}
