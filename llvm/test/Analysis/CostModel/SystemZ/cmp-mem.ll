; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s
;
; Test costs for i8 and i16 comparisons against memory with a small immediate.

define i32 @fun0(i8* %Src, i8* %Dst, i8 %Val) {
; CHECK: Printing analysis 'Cost Model Analysis' for function 'fun0':
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %Ld = load i8, i8* %Src
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %Cmp = icmp eq i8 %Ld, 123
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %Ret = zext i1 %Cmp to i32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   ret i32 %Ret
  %Ld = load i8, i8* %Src
  %Cmp = icmp eq i8 %Ld, 123
  %Ret = zext i1 %Cmp to i32
  ret i32 %Ret
}

define i32 @fun1(i16* %Src, i16* %Dst, i16 %Val) {
; CHECK: Printing analysis 'Cost Model Analysis' for function 'fun1':
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %Ld = load i16, i16* %Src
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %Cmp = icmp eq i16
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %Ret = zext i1 %Cmp to i32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   ret i32 %Ret
  %Ld = load i16, i16* %Src
  %Cmp = icmp eq i16 %Ld, 1234
  %Ret = zext i1 %Cmp to i32
  ret i32 %Ret
}
