; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mcpu=generic | FileCheck %s
;
; Verify that pass 'Constant Hoisting' is not run on optnone functions.
; Without optnone, Pass 'Constant Hoisting' would firstly hoist
; constant 0xBEEBEEBEC, and then rebase the other constant
; (i.e. constant 0xBEEBEEBF4) with respect to the previous one.
; With optnone, we check that constants are not coalesced.

define i64 @constant_hoisting_optnone() #0 {
; CHECK-LABEL: @constant_hoisting_optnone
; CHECK-DAG: movabsq {{.*#+}} imm = 0xBEEBEEBF4
; CHECK-DAG: movabsq {{.*#+}} imm = 0xBEEBEEBEC
; CHECK: ret
entry:
  %0 = load i64, ptr inttoptr (i64 51250129900 to ptr)
  %1 = load i64, ptr inttoptr (i64 51250129908 to ptr)
  %2 = add i64 %0, %1
  ret i64 %2
}

attributes #0 = { optnone noinline }
