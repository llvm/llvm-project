;RUN: llc < %s -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -mattr=+neon | FileCheck %s

declare <1 x i64> @llvm.arm.neon.vqneg.v1i64(<1 x i64>)

declare <1 x i64> @llvm.arm.neon.vqabs.v1i64(<1 x i64>)

declare <1 x i64> @llvm.arm.neon.vabs.v1i64(<1 x i64>)

declare <1 x i64> @llvm.aarch64.neon.usqadd.v1i64(<1 x i64>, <1 x i64>)

declare <1 x i64> @llvm.aarch64.neon.suqadd.v1i64(<1 x i64>, <1 x i64>)

define <1 x i64> @test_vuqadd_s64(<1 x i64> %a, <1 x i64> %b) {
entry:
  ; CHECK: test_vuqadd_s64
  %vuqadd2.i = tail call <1 x i64> @llvm.aarch64.neon.suqadd.v1i64(<1 x i64> %a, <1 x i64> %b)
  ; CHECK: suqadd d{{[0-9]+}}, d{{[0-9]+}}
  ret <1 x i64> %vuqadd2.i
}

define <1 x i64> @test_vsqadd_u64(<1 x i64> %a, <1 x i64> %b) {
entry:
  ; CHECK: test_vsqadd_u64
  %vsqadd2.i = tail call <1 x i64> @llvm.aarch64.neon.usqadd.v1i64(<1 x i64> %a, <1 x i64> %b)
  ; CHECK: usqadd d{{[0-9]+}}, d{{[0-9]+}}
  ret <1 x i64> %vsqadd2.i
}

define <1 x i64> @test_vabs_s64(<1 x i64> %a) {
  ; CHECK: test_vabs_s64
entry:
  %vabs1.i = tail call <1 x i64> @llvm.arm.neon.vabs.v1i64(<1 x i64> %a)
  ; CHECK: abs d{{[0-9]+}}, d{{[0-9]+}}
  ret <1 x i64> %vabs1.i
}

define <1 x i64> @test_vqabs_s64(<1 x i64> %a) {
  ; CHECK: test_vqabs_s64
entry:
  %vqabs1.i = tail call <1 x i64> @llvm.arm.neon.vqabs.v1i64(<1 x i64> %a)
  ; CHECK: sqabs d{{[0-9]+}}, d{{[0-9]+}}
  ret <1 x i64> %vqabs1.i
}

define <1 x i64> @test_vqneg_s64(<1 x i64> %a) {
  ; CHECK: test_vqneg_s64
entry:
  %vqneg1.i = tail call <1 x i64> @llvm.arm.neon.vqneg.v1i64(<1 x i64> %a)
  ; CHECK: sqneg d{{[0-9]+}}, d{{[0-9]+}}
  ret <1 x i64> %vqneg1.i
}

define <1 x i64> @test_vneg_s64(<1 x i64> %a) {
  ; CHECK: test_vneg_s64
entry:
  %sub.i = sub <1 x i64> zeroinitializer, %a
  ; CHECK: neg d{{[0-9]+}}, d{{[0-9]+}}
  ret <1 x i64> %sub.i
}

