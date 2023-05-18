; RUN: llc -mtriple=aarch64-linux-gnu -o - %s | FileCheck %s

define i32 @test_consts(i32 %in) {
; CHECK-LABEL: test_consts:
; CHECK-NOT: bfxil
; CHECK-NOT: and
; CHECK-NOT: orr
; CHECK: ret

  %lo = and i32 %in, 65535
  %hi = and i32 %in, -65536
  %res = or i32 %lo, %hi
  ret i32 %res
}

define i32 @test_generic(i32 %in, i32 %mask1, i32 %mask2) {
; CHECK-LABEL: test_generic:
; CHECK: orr [[FULL_MASK:w[0-9]+]], w1, w2
; CHECK: and w0, w0, [[FULL_MASK]]

  %lo = and i32 %in, %mask1
  %hi = and i32 %in, %mask2
  %res = or i32 %lo, %hi
  ret i32 %res
}

; In this case the transformation isn't profitable, since %lo and %hi
; are used more than once.
define [3 x i32] @test_reuse(i32 %in, i32 %mask1, i32 %mask2) {
; CHECK-LABEL: test_reuse:
; CHECK-DAG: and w1, w0, w1
; CHECK-DAG: and w2, w0, w2
; CHECK-DAG: orr w0, w1, w2

  %lo = and i32 %in, %mask1
  %hi = and i32 %in, %mask2
  %recombine = or i32 %lo, %hi

  %res.tmp0 = insertvalue [3 x i32] undef, i32 %recombine, 0
  %res.tmp1 = insertvalue [3 x i32] %res.tmp0, i32 %lo, 1
  %res = insertvalue [3 x i32] %res.tmp1, i32 %hi, 2

  ret [3 x i32] %res
}
