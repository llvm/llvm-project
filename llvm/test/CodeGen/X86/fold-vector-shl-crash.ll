; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s
; RUN: llc < %s -mtriple=i686--    | FileCheck %s

;CHECK-LABEL: test
define <2 x i256> @test() {
  %S = shufflevector <2 x i256> zeroinitializer, <2 x i256> <i256 -1, i256 -1>, <2 x i32> <i32 0, i32 2>
  %B = shl <2 x i256> %S, <i256 -1, i256 -1> ; DAG Combiner crashes here
  ret <2 x i256> %B
}
