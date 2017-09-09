; RUN: opt < %s -cost-model -cost-kind=latency -analyze -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s --check-prefix=LATENCY
; RUN: opt < %s -cost-model -cost-kind=code-size -analyze -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 | FileCheck %s --check-prefix=CODESIZE

; Tests if the interface TargetTransformInfo::getInstructionCost() works correctly.

define i64 @foo(i64 %arg) {

  ; LATENCY:  cost of 1 {{.*}} %I64 = add
  ; CODESIZE: cost of 1 {{.*}} %I64 = add
  %I64 = add i64 undef, undef

  ; LATENCY:  cost of 4 {{.*}} load
  ; CODESIZE: cost of 1 {{.*}} load
  load i64, i64* undef, align 4

  ; LATENCY:  cost of 1 {{.*}} ret
  ; CODESIZE: cost of 1 {{.*}} ret
  ret i64 undef
}
