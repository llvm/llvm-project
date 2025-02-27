; RUN: llc -mtriple=aarch64-none-linux-gnu < %s -mattr=+neon | FileCheck %s

; SDIVREM/UDIVREM DAG nodes are generated but expanded when lowering and
; should not generate select error.
define <2 x i32> @test_udivrem(<2 x i32> %x, < 2 x i32> %y, < 2 x i32>* %z) {
; CHECK-LABEL: test_udivrem
; CHECK-DAG: udivrem
; CHECK-NOT: LLVM ERROR: Cannot select
  %div = udiv <2 x i32> %x, %y
  store <2 x i32> %div, ptr %z
  %1 = urem <2 x i32> %x, %y
  ret <2 x i32> %1
}

define <4 x i32> @test_sdivrem(<4 x i32> %x,  ptr %y) {
; CHECK-LABEL: test_sdivrem
; CHECK-DAG: sdivrem
  %div = sdiv <4 x i32> %x,  < i32 20, i32 20, i32 20, i32 20 >
  store <4 x i32> %div, ptr %y
  %1 = srem <4 x i32> %x, < i32 20, i32 20, i32 20, i32 20 >
  ret <4 x i32> %1
}
