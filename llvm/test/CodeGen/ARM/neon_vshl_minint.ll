; RUN: llc < %s -mtriple=arm-none-eabi -mcpu=cortex-a8 2>&1 | FileCheck %s
; RUN: llc < %s -mtriple=thumb-none-eabi -mcpu=cortex-a8 2>&1 | FileCheck %s

define <1 x i64> @vshl_minint() #0 {
  entry:
    ; CHECK-LABEL: vshl_minint
    ; CHECK: vldr
    ; CHECK: vshl.u64
    %vshl.i = tail call <1 x i64> @llvm.arm.neon.vshiftu.v1i64(<1 x i64> undef, <1 x i64> <i64 -9223372036854775808>)
    ret <1 x i64> %vshl.i
}

declare <1 x i64> @llvm.arm.neon.vshiftu.v1i64(<1 x i64>, <1 x i64>)
