; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test that a loaded value which is used both in a vector and scalar context
; is not transformed to a vlrep + vlgvg.

; CHECK-NOT: vlrep

define void @fun(i64 %arg, ptr %Addr, ptr %Dst) {
  %tmp10 = load ptr, ptr %Addr
  store i64 %arg, ptr %tmp10
  %tmp12 = insertelement <2 x ptr> undef, ptr %tmp10, i32 0
  %tmp13 = insertelement <2 x ptr> %tmp12, ptr %tmp10, i32 1
  store <2 x ptr> %tmp13, ptr %Dst
  ret void
}
