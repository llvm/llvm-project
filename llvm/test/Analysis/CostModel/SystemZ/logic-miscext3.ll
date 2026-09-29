; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z13 \
; RUN:  | FileCheck %s -check-prefixes=CHECK,Z13
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z15 \
; RUN:  | FileCheck %s -check-prefixes=CHECK,Z15

define void @fun0(i32 %a)  {
; CHECK-LABEL: function 'fun0'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %c0 = xor i32 %l0, -1
; Z13:   Cost Model: Found an estimated cost of 1 for instruction:   %res0 = or i32 %a, %c0
; Z15:   Cost Model: Found an estimated cost of 0 for instruction:   %res0 = or i32 %a, %c0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %c1 = xor i32 %l1, -1
; Z13:   Cost Model: Found an estimated cost of 1 for instruction:   %res1 = and i32 %a, %c1
; Z15:   Cost Model: Found an estimated cost of 0 for instruction:   %res1 = and i32 %a, %c1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %c2 = and i32 %l2, %a
; Z13:   Cost Model: Found an estimated cost of 1 for instruction:   %res2 = xor i32 %c2, -1
; Z15:   Cost Model: Found an estimated cost of 0 for instruction:   %res2 = xor i32 %c2, -1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %c3 = or i32 %l3, %a
; Z13:   Cost Model: Found an estimated cost of 1 for instruction:   %res3 = xor i32 %c3, -1
; Z15:   Cost Model: Found an estimated cost of 0 for instruction:   %res3 = xor i32 %c3, -1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %c4 = xor i32 %l4, %a
; Z13:   Cost Model: Found an estimated cost of 1 for instruction:   %res4 = xor i32 %c4, -1
; Z15:   Cost Model: Found an estimated cost of 0 for instruction:   %res4 = xor i32 %c4, -1

entry:
  %l0 = load i32, ptr undef
  %c0 = xor i32 %l0, -1
  %res0 = or i32 %a, %c0
  store i32 %res0, ptr undef

  %l1 = load i32, ptr undef
  %c1 = xor i32 %l1, -1
  %res1 = and i32 %a, %c1
  store i32 %res1, ptr undef

  %l2 = load i32, ptr undef
  %c2 = and i32 %l2, %a
  %res2 = xor i32 %c2, -1
  store i32 %res2, ptr undef

  %l3 = load i32, ptr undef
  %c3 = or i32 %l3, %a
  %res3 = xor i32 %c3, -1
  store i32 %res3, ptr undef

  %l4 = load i32, ptr undef
  %c4 = xor i32 %l4, %a
  %res4 = xor i32 %c4, -1
  store i32 %res4, ptr undef

  ret void
}

define void @fun1(i64 %a)  {
; CHECK-LABEL: function 'fun1'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %c0 = xor i64 %l0, -1
; Z13:   Cost Model: Found an estimated cost of 1 for instruction:   %res0 = or i64 %a, %c0
; Z15:   Cost Model: Found an estimated cost of 0 for instruction:   %res0 = or i64 %a, %c0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %c1 = xor i64 %l1, -1
; Z13:   Cost Model: Found an estimated cost of 1 for instruction:   %res1 = and i64 %a, %c1
; Z15:   Cost Model: Found an estimated cost of 0 for instruction:   %res1 = and i64 %a, %c1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %c2 = and i64 %l2, %a
; Z13:   Cost Model: Found an estimated cost of 1 for instruction:   %res2 = xor i64 %c2, -1
; Z15:   Cost Model: Found an estimated cost of 0 for instruction:   %res2 = xor i64 %c2, -1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %c3 = or i64 %l3, %a
; Z13:   Cost Model: Found an estimated cost of 1 for instruction:   %res3 = xor i64 %c3, -1
; Z15:   Cost Model: Found an estimated cost of 0 for instruction:   %res3 = xor i64 %c3, -1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %c4 = xor i64 %l4, %a
; Z13:   Cost Model: Found an estimated cost of 1 for instruction:   %res4 = xor i64 %c4, -1
; Z15:   Cost Model: Found an estimated cost of 0 for instruction:   %res4 = xor i64 %c4, -1
entry:
  %l0 = load i64, ptr undef
  %c0 = xor i64 %l0, -1
  %res0 = or i64 %a, %c0
  store i64 %res0, ptr undef

  %l1 = load i64, ptr undef
  %c1 = xor i64 %l1, -1
  %res1 = and i64 %a, %c1
  store i64 %res1, ptr undef

  %l2 = load i64, ptr undef
  %c2 = and i64 %l2, %a
  %res2 = xor i64 %c2, -1
  store i64 %res2, ptr undef

  %l3 = load i64, ptr undef
  %c3 = or i64 %l3, %a
  %res3 = xor i64 %c3, -1
  store i64 %res3, ptr undef

  %l4 = load i64, ptr undef
  %c4 = xor i64 %l4, %a
  %res4 = xor i64 %c4, -1
  store i64 %res4, ptr undef

  ret void
}
