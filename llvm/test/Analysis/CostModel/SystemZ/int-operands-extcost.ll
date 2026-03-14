; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z13 \
; RUN:  | FileCheck %s
;
; Test that i8/i16 operands get extra costs for extensions to i32 only in
; cases where this is needed.

define void @icmp() {
  %li8_0 = load i8, ptr undef
  %li8_1 = load i8, ptr undef
  icmp slt i8 %li8_0, %li8_1

  %a0 = add i8 %li8_0, 1
  %a1 = add i8 %li8_1, 1
  icmp slt i8 %a0, %a1

  icmp slt i8 %a0, 123

  %li16_0 = load i16, ptr undef
  %li16_1 = load i16, ptr undef
  icmp slt i16 %li16_0, %li16_1

  %a2 = add i16 %li16_0, 1
  %a3 = add i16 %li16_1, 1
  icmp slt i16 %a2, %a3

  icmp slt i16 %a2, 123

  ret void;
; CHECK: function 'icmp'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li8_0 = load i8, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li8_1 = load i8, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = icmp slt i8 %li8_0, %li8_1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %a0 = add i8 %li8_0, 1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %a1 = add i8 %li8_1, 1
; CHECK: Cost Model: Found an estimated cost of 3 for instruction:   %2 = icmp slt i8 %a0, %a1
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %3 = icmp slt i8 %a0, 123
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li16_0 = load i16, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li16_1 = load i16, ptr undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = icmp slt i16 %li16_0, %li16_1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %a2 = add i16 %li16_0, 1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %a3 = add i16 %li16_1, 1
; CHECK: Cost Model: Found an estimated cost of 3 for instruction:   %5 = icmp slt i16 %a2, %a3
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %6 = icmp slt i16 %a2, 123
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   ret void
}
