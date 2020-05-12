; RUN: opt < %s  -cost-model -analyze -mtriple=armv7-linux-gnueabihf -mcpu=cortex-a9 | FileCheck --check-prefix=COST %s
; To see the assembly output: llc -mcpu=cortex-a9 < %s | FileCheck --check-prefix=ASM %s
; ASM lines below are only for reference, tests on that direction should go to tests/CodeGen/ARM

; ModuleID = 'arm.ll'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv7--linux-gnueabihf"

%T216 = type <2 x i16>
%T232 = type <2 x i32>
%T264 = type <2 x i64>

%T416 = type <4 x i16>
%T432 = type <4 x i32>
%T464 = type <4 x i64>

define void @direct(%T432* %loadaddr, %T432* %loadaddr2, %T432* %storeaddr) {
; COST: function 'direct':
  %v0 = load %T432, %T432* %loadaddr
; ASM: vld1.64
  %v1 = load %T432, %T432* %loadaddr2
; ASM: vld1.64
  %r3 = add %T432 %v0, %v1 
; COST: cost of 1 for instruction: {{.*}} add <4 x i32>
; ASM: vadd.i32
  store %T432 %r3, %T432* %storeaddr
; ASM: vst1.64
  ret void
}

define void @ups1632(%T416* %loadaddr, %T416* %loadaddr2, %T432* %storeaddr) {
; COST: function 'ups1632':
  %v0 = load %T416, %T416* %loadaddr
; ASM: vldr
  %v1 = load %T416, %T416* %loadaddr2
; ASM: vldr
  %r1 = sext %T416 %v0 to %T432
  %r2 = sext %T416 %v1 to %T432
; COST: cost of 0 for instruction: {{.*}} sext <4 x i16> {{.*}} to <4 x i32>
  %r3 = add %T432 %r1, %r2 
; COST: cost of 1 for instruction: {{.*}} add <4 x i32>
; ASM: vaddl.s16
  store %T432 %r3, %T432* %storeaddr
; ASM: vst1.64
  ret void
}

define void @upu1632(%T416* %loadaddr, %T416* %loadaddr2, %T432* %storeaddr) {
; COST: function 'upu1632':
  %v0 = load %T416, %T416* %loadaddr
; ASM: vldr
  %v1 = load %T416, %T416* %loadaddr2
; ASM: vldr
  %r1 = zext %T416 %v0 to %T432
  %r2 = zext %T416 %v1 to %T432
; COST: cost of 0 for instruction: {{.*}} zext <4 x i16> {{.*}} to <4 x i32>
  %r3 = add %T432 %r1, %r2 
; COST: cost of 1 for instruction: {{.*}} add <4 x i32>
; ASM: vaddl.u16
  store %T432 %r3, %T432* %storeaddr
; ASM: vst1.64
  ret void
}

define void @ups3264(%T232* %loadaddr, %T232* %loadaddr2, %T264* %storeaddr) {
; COST: function 'ups3264':
  %v0 = load %T232, %T232* %loadaddr
; ASM: vldr
  %v1 = load %T232, %T232* %loadaddr2
; ASM: vldr
  %r3 = add %T232 %v0, %v1 
; ASM: vadd.i32
; COST: cost of 1 for instruction: {{.*}} add <2 x i32>
  %st = sext %T232 %r3 to %T264
; ASM: vmovl.s32
; COST: cost of 1 for instruction: {{.*}} sext <2 x i32> {{.*}} to <2 x i64>
  store %T264 %st, %T264* %storeaddr
; ASM: vst1.64
  ret void
}

define void @upu3264(%T232* %loadaddr, %T232* %loadaddr2, %T264* %storeaddr) {
; COST: function 'upu3264':
  %v0 = load %T232, %T232* %loadaddr
; ASM: vldr
  %v1 = load %T232, %T232* %loadaddr2
; ASM: vldr
  %r3 = add %T232 %v0, %v1 
; ASM: vadd.i32
; COST: cost of 1 for instruction: {{.*}} add <2 x i32>
  %st = zext %T232 %r3 to %T264
; ASM: vmovl.u32
; COST: cost of 1 for instruction: {{.*}} zext <2 x i32> {{.*}} to <2 x i64>
  store %T264 %st, %T264* %storeaddr
; ASM: vst1.64
  ret void
}

define void @dn3216(%T432* %loadaddr, %T432* %loadaddr2, %T416* %storeaddr) {
; COST: function 'dn3216':
  %v0 = load %T432, %T432* %loadaddr
; ASM: vld1.64
  %v1 = load %T432, %T432* %loadaddr2
; ASM: vld1.64
  %r3 = add %T432 %v0, %v1 
; ASM: vadd.i32
; COST: cost of 1 for instruction: {{.*}} add <4 x i32>
  %st = trunc %T432 %r3 to %T416
; ASM: vmovn.i32
; COST: cost of 1 for instruction: {{.*}} trunc <4 x i32> {{.*}} to <4 x i16>
  store %T416 %st, %T416* %storeaddr
; ASM: vstr
  ret void
}
