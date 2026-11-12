; RUN: llc -mtriple=aarch64-none-linux-gnu < %s -mattr=+neon | FileCheck %s

define <4 x i32> @test1(<4 x i32> %a) {
  %rem = srem <4 x i32> %a, <i32 7, i32 7, i32 7, i32 7>
  ret <4 x i32> %rem
; For C constant X/C is simplified to X-X/C*C. The X/C division is lowered
; to MULHS due the simplification by multiplying by a magic number
; (TargetLowering::BuildSDIV).
; CHECK-LABEL: test1:
; CHECK: smull2 [[SMULL2:(v[0-9]+)]].2d, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
; CHECK: smull  [[SMULL:(v[0-9]+)]].2d, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
; CHECK: uzp2   [[UZP2:(v[0-9]+).4s]], [[SMULL]].4s, [[SMULL2]].4s
; CHECK: add    [[ADD:(v[0-9]+.4s)]], [[UZP2]], v0.4s
; CHECK: sshr   [[SSHR:(v[0-9]+.4s)]], [[ADD]], #2
}
