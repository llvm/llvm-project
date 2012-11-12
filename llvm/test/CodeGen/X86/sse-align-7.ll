; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; CHECK:     movaps
; CHECK-NOT:     movaps

define void @bar(<2 x i64>* %p, <2 x i64> %x) nounwind {
  store <2 x i64> %x, <2 x i64>* %p
  ret void
}
