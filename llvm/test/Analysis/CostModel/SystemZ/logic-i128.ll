; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z13 \
; RUN:  | FileCheck %s -check-prefixes=CHECK,Z13
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z14 \
; RUN:  | FileCheck %s -check-prefixes=CHECK,Z14

define void @fun(i128 %a)  {
; CHECK-LABEL: 'fun'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction: %c0 = xor i128 %l0, -1
; Z13:   Cost Model: Found an estimated cost of 1 for instruction: %res0 = or i128 %a, %c0
; Z14:   Cost Model: Found an estimated cost of 0 for instruction: %res0 = or i128 %a, %c0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction: %c1 = xor i128 %l1, -1
; CHECK: Cost Model: Found an estimated cost of 0 for instruction: %res1 = and i128 %a, %c1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction: %c2 = and i128 %l2, %a
; Z13:   Cost Model: Found an estimated cost of 1 for instruction: %res2 = xor i128 %c2, -1
; Z14:   Cost Model: Found an estimated cost of 0 for instruction: %res2 = xor i128 %c2, -1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction: %c3 = or i128 %l3, %a
; CHECK: Cost Model: Found an estimated cost of 0 for instruction: %res3 = xor i128 %c3, -1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction: %c4 = xor i128 %l4, %a
; Z13:   Cost Model: Found an estimated cost of 1 for instruction: %res4 = xor i128 %c4, -1
; Z14:   Cost Model: Found an estimated cost of 0 for instruction: %res4 = xor i128 %c4, -1
;
  %l0 = load i128, ptr undef
  %c0 = xor i128 %l0, -1
  %res0 = or i128 %a, %c0
  store i128 %res0, ptr undef

  %l1 = load i128, ptr undef
  %c1 = xor i128 %l1, -1
  %res1 = and i128 %a, %c1
  store i128 %res1, ptr undef

  %l2 = load i128, ptr undef
  %c2 = and i128 %l2, %a
  %res2 = xor i128 %c2, -1
  store i128 %res2, ptr undef

  %l3 = load i128, ptr undef
  %c3 = or i128 %l3, %a
  %res3 = xor i128 %c3, -1
  store i128 %res3, ptr undef

  %l4 = load i128, ptr undef
  %c4 = xor i128 %l4, %a
  %res4 = xor i128 %c4, -1
  store i128 %res4, ptr undef

  ret void
}
