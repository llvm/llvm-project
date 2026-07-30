; RUN: llc -mtriple=armv7-linux-gnueabihf %s -o - | FileCheck %s

define void @test_mismatched_setcc(<4 x i22> %l, <4 x i22> %r, ptr %addr) {
; CHECK-LABEL: test_mismatched_setcc:
; CHECK: vceq.i32 [[CMP128:q[0-9]+]], {{q[0-9]+}}, {{q[0-9]+}}
; CHECK: vmovn.i32 {{d[0-9]+}}, [[CMP128]]

  %tst = icmp eq <4 x i22> %l, %r
  store <4 x i1> %tst, ptr %addr
  ret void
}
