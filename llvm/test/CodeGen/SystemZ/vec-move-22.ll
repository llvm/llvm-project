; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test that a loaded value which is used both in a vector and scalar context
; is not transformed to a vlrep + vlgvg.

; CHECK-NOT: vlrep

define void @fun(i64 %arg, i64** %Addr, <2 x i64*>* %Dst) {
  %tmp10 = load i64*, i64** %Addr
  store i64 %arg, i64* %tmp10
  %tmp12 = insertelement <2 x i64*> undef, i64* %tmp10, i32 0
  %tmp13 = insertelement <2 x i64*> %tmp12, i64* %tmp10, i32 1
  store <2 x i64*> %tmp13, <2 x i64*>* %Dst
  ret void
}
