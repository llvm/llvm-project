; RUN: llc -mtriple=aarch64-linux-gnu %s -o - | FileCheck %s

define void @test_mismatched_setcc(<4 x i22> %l, <4 x i22> %r, <4 x i1>* %addr) {
; CHECK-LABEL: test_mismatched_setcc:
; CHECK: cmeq [[CMP128:v[0-9]+]].4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK: xtn {{v[0-9]+}}.4h, [[CMP128]].4s

  %tst = icmp eq <4 x i22> %l, %r
  store <4 x i1> %tst, <4 x i1>* %addr
  ret void
}
