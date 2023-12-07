; We currently estimate the cost of sext/zext/trunc v8(v16)i32 <-> v8(v16)i8
; instructions as expensive. If lowering is improved the cost model needs to
; change.
; RUN: opt < %s -passes='print<cost-model>' -mtriple=arm-apple-ios6.0.0 -mcpu=cortex-a8 -disable-output 2>&1 | FileCheck %s --check-prefix=COST
%T0_5 = type <8 x i8>
%T1_5 = type <8 x i32>
; CHECK-LABEL: func_cvt5:
define void @func_cvt5(ptr %loadaddr, ptr %storeaddr) {
; CHECK: vmovl.s8
; CHECK: vmovl.s16
; CHECK: vmovl.s16
  %v0 = load %T0_5, ptr %loadaddr
; COST: func_cvt5
; COST: cost of 3 {{.*}} sext
  %r = sext %T0_5 %v0 to %T1_5
  store %T1_5 %r, ptr %storeaddr
  ret void
}
;; We currently estimate the cost of this instruction as expensive. If lowering
;; is improved the cost needs to change.
%TA0_5 = type <8 x i8>
%TA1_5 = type <8 x i32>
; CHECK-LABEL: func_cvt1:
define void @func_cvt1(ptr %loadaddr, ptr %storeaddr) {
; CHECK: vmovl.u8
; CHECK: vmovl.u16
; CHECK: vmovl.u16
  %v0 = load %TA0_5, ptr %loadaddr
; COST: func_cvt1
; COST: cost of 3 {{.*}} zext
  %r = zext %TA0_5 %v0 to %TA1_5
  store %TA1_5 %r, ptr %storeaddr
  ret void
}

%T0_51 = type <8 x i32>
%T1_51 = type <8 x i8>
; CHECK-LABEL: func_cvt51:
define void @func_cvt51(ptr %loadaddr, ptr %storeaddr) {
; CHECK: vmovn.i32
; CHECK: vmovn.i32
; CHECK: vmovn.i16
  %v0 = load %T0_51, ptr %loadaddr
; COST: func_cvt51
; COST: cost of 3 {{.*}} trunc
  %r = trunc %T0_51 %v0 to %T1_51
  store %T1_51 %r, ptr %storeaddr
  ret void
}

%TT0_5 = type <16 x i8>
%TT1_5 = type <16 x i32>
; CHECK-LABEL: func_cvt52:
define void @func_cvt52(ptr %loadaddr, ptr %storeaddr) {
; CHECK: vmovl.s16
; CHECK: vmovl.s16
; CHECK: vmovl.s16
; CHECK: vmovl.s16
  %v0 = load %TT0_5, ptr %loadaddr
; COST: func_cvt52
; COST: cost of 6 {{.*}} sext
  %r = sext %TT0_5 %v0 to %TT1_5
  store %TT1_5 %r, ptr %storeaddr
  ret void
}
;; We currently estimate the cost of this instruction as expensive. If lowering
;; is improved the cost needs to change.
%TTA0_5 = type <16 x i8>
%TTA1_5 = type <16 x i32>
; CHECK-LABEL: func_cvt12:
define void @func_cvt12(ptr %loadaddr, ptr %storeaddr) {
; CHECK: vmovl.u16
; CHECK: vmovl.u16
; CHECK: vmovl.u16
; CHECK: vmovl.u16
  %v0 = load %TTA0_5, ptr %loadaddr
; COST: func_cvt12
; COST: cost of 6 {{.*}} zext
  %r = zext %TTA0_5 %v0 to %TTA1_5
  store %TTA1_5 %r, ptr %storeaddr
  ret void
}

%TT0_51 = type <16 x i32>
%TT1_51 = type <16 x i8>
; CHECK-LABEL: func_cvt512:
define void @func_cvt512(ptr %loadaddr, ptr %storeaddr) {
; CHECK: vmovn.i32
; CHECK: vmovn.i32
; CHECK: vmovn.i32
; CHECK: vmovn.i32
; CHECK: vmovn.i16
; CHECK: vmovn.i16
  %v0 = load %TT0_51, ptr %loadaddr
; COST: func_cvt512
; COST: cost of 6 {{.*}} trunc
  %r = trunc %TT0_51 %v0 to %TT1_51
  store %TT1_51 %r, ptr %storeaddr
  ret void
}

; CHECK-LABEL: sext_v4i16_v4i64:
define void @sext_v4i16_v4i64(ptr %loadaddr, ptr %storeaddr) {
; CHECK: vmovl.s32
; CHECK: vmovl.s32
  %v0 = load <4 x i16>, ptr %loadaddr
; COST: sext_v4i16_v4i64
; COST: cost of 3 {{.*}} sext
  %r = sext <4 x i16> %v0 to <4 x i64>
  store <4 x i64> %r, ptr %storeaddr
  ret void
}

; CHECK-LABEL: zext_v4i16_v4i64:
define void @zext_v4i16_v4i64(ptr %loadaddr, ptr %storeaddr) {
; CHECK: vmovl.u32
; CHECK: vmovl.u32
  %v0 = load <4 x i16>, ptr %loadaddr
; COST: zext_v4i16_v4i64
; COST: cost of 3 {{.*}} zext
  %r = zext <4 x i16> %v0 to <4 x i64>
  store <4 x i64> %r, ptr %storeaddr
  ret void
}

; CHECK-LABEL: sext_v8i16_v8i64:
define void @sext_v8i16_v8i64(ptr %loadaddr, ptr %storeaddr) {
; CHECK: vmovl.s32
; CHECK: vmovl.s32
; CHECK: vmovl.s32
; CHECK: vmovl.s32
  %v0 = load <8 x i16>, ptr %loadaddr
; COST: sext_v8i16_v8i64
; COST: cost of 6 {{.*}} sext
  %r = sext <8 x i16> %v0 to <8 x i64>
  store <8 x i64> %r, ptr %storeaddr
  ret void
}

; CHECK-LABEL: zext_v8i16_v8i64:
define void @zext_v8i16_v8i64(ptr %loadaddr, ptr %storeaddr) {
; CHECK: vmovl.u32
; CHECK: vmovl.u32
; CHECK: vmovl.u32
; CHECK: vmovl.u32
  %v0 = load <8 x i16>, ptr %loadaddr
; COST: zext_v8i16_v8i64
; COST: cost of 6 {{.*}} zext
  %r = zext <8 x i16> %v0 to <8 x i64>
  store <8 x i64> %r, ptr %storeaddr
  ret void
}

