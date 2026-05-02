; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx2 | FileCheck %s

define <8 x i32> @count_lt1_ir(<8 x i32> %y, <8 x i32>* %p) {
; CHECK-LABEL: count_lt1_ir:
; CHECK:       # %bb.0:
; CHECK:         vpminud (%{{[a-z0-9]+}}), %ymm{{[0-9]+}}, %ymm{{[0-9]+}}
; CHECK:         retq
  %x = load <8 x i32>, <8 x i32>* %p
  %cmp = icmp ult <8 x i32> %x, %y
  %sext = sext <8 x i1> %cmp to <8 x i32>
  ret <8 x i32> %sext
}
