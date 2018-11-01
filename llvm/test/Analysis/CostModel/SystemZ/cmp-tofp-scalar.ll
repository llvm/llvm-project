; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s
;
; Costs for conversion of i1 to fp.

define float @fun0(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = uitofp i1 %cmp to float
  ret float %v

; CHECK: fun0
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 5 for instruction:   %v = uitofp i1 %cmp to float
}

define double @fun1(i64 %val1, i64 %val2) {
  %cmp = icmp eq i64 %val1, %val2
  %v = uitofp i1 %cmp to double
  ret double %v

; CHECK: fun1
; CHECK: cost of 1 for instruction:   %cmp = icmp eq i64 %val1, %val2
; CHECK: cost of 5 for instruction:   %v = uitofp i1 %cmp to double
}
