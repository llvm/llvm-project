; RUN: opt < %s -passes='print<cost-model>' 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s
;
; Test that load and test results in 0 cost for the compare.

define i64 @fun0(i64* %Src, i64 %Arg) {
  %Ld1 = load i64, i64* %Src
  %Cmp = icmp eq i64 %Ld1, 0
  %S   = select i1 %Cmp, i64 %Arg, i64 %Ld1
  ret i64 %S
; CHECK: function 'fun0'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %Ld1 = load i64, i64* %Src
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %Cmp = icmp eq i64 %Ld1, 0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %S = select
}

define i32 @fun1(i32* %Src, i32 %Arg) {
  %Ld1 = load i32, i32* %Src
  %Cmp = icmp eq i32 %Ld1, 0
  %S   = select i1 %Cmp, i32 %Arg, i32 %Ld1
  ret i32 %S
; CHECK: function 'fun1'
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %Ld1 = load i32, i32* %Src
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %Cmp = icmp eq i32 %Ld1, 0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %S = select
}
