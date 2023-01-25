; RUN: opt < %s -passes="print<cost-model>" -cost-kind=code-size 2>&1 -disable-output \
; RUN:   -mtriple=s390x-unknown-linux -mcpu=z13  | FileCheck %s
;
; Check that getInstructionCost() does not return TCC_Free for extensions of
; i1 returned from icmp.

define i64 @fun1(i64 %v) {
; CHECK-LABEL: 'fun1'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %cmp = icmp eq i64 %v, 0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %z = zext i1 %cmp to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   ret i64 %z
  %cmp = icmp eq i64 %v, 0
  %z = zext i1 %cmp to i64
  ret i64 %z
}

define i64 @fun2(i64 %v) {
; CHECK-LABEL: 'fun2'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %cmp = icmp eq i64 %v, 0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %z = sext i1 %cmp to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   ret i64 %z
  %cmp = icmp eq i64 %v, 0
  %z = sext i1 %cmp to i64
  ret i64 %z
}

define double @fun3(i64 %v) {
; CHECK-LABEL: 'fun3'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %cmp = icmp eq i64 %v, 0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %z = uitofp i1 %cmp to double
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   ret double %z
  %cmp = icmp eq i64 %v, 0
  %z = uitofp i1 %cmp to double
  ret double %z
}

define double @fun4(i64 %v) {
; CHECK-LABEL: 'fun4'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %cmp = icmp eq i64 %v, 0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %z = sitofp i1 %cmp to double
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   ret double %z
  %cmp = icmp eq i64 %v, 0
  %z = sitofp i1 %cmp to double
  ret double %z
}

define i64 @fun5(i1 %v) {
; CHECK-LABEL: 'fun5'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %z = zext i1 %v to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   ret i64 %z
  %z = zext i1 %v to i64
  ret i64 %z
}
