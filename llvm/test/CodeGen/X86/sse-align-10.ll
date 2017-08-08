; RUN: llc < %s -march=x86-64 | FileCheck %s

define <2 x i64> @bar(<2 x i64>* %p) nounwind {
; CHECK-LABEL: bar:
; CHECK: movups
; CHECK-NOT: movups
  %t = load <2 x i64>, <2 x i64>* %p, align 8
  ret <2 x i64> %t
}
