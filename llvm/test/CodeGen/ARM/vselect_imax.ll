; RUN: opt < %s  -cost-model -analyze -mtriple=thumbv7-apple-ios6.0.0 -march=arm -mcpu=cortex-a8 | FileCheck %s --check-prefix=COST
; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s
; Make sure that ARM backend with NEON handles vselect.

define void @vmax_v4i32(<4 x i32>* %m, <4 x i32> %a, <4 x i32> %b) {
; CHECK: vmax.s32 {{q[0-9]+}}, {{q[0-9]+}}, {{q[0-9]+}}
    %cmpres = icmp sgt <4 x i32> %a, %b
    %maxres = select <4 x i1> %cmpres, <4 x i32> %a,  <4 x i32> %b
    store <4 x i32> %maxres, <4 x i32>* %m
    ret void
}

%T0_10 = type <16 x i16>
%T1_10 = type <16 x i1>
; CHECK-LABEL: func_blend10:
define void @func_blend10(%T0_10* %loadaddr, %T0_10* %loadaddr2,
                           %T1_10* %blend, %T0_10* %storeaddr) {
  %v0 = load %T0_10, %T0_10* %loadaddr
  %v1 = load %T0_10, %T0_10* %loadaddr2
  %c = icmp slt %T0_10 %v0, %v1
; CHECK: vmin.s16
; CHECK: vmin.s16
; COST: func_blend10
; COST: cost of 2 {{.*}} select
  %r = select %T1_10 %c, %T0_10 %v0, %T0_10 %v1
  store %T0_10 %r, %T0_10* %storeaddr
  ret void
}
%T0_14 = type <8 x i32>
%T1_14 = type <8 x i1>
; CHECK-LABEL: func_blend14:
define void @func_blend14(%T0_14* %loadaddr, %T0_14* %loadaddr2,
                           %T1_14* %blend, %T0_14* %storeaddr) {
  %v0 = load %T0_14, %T0_14* %loadaddr
  %v1 = load %T0_14, %T0_14* %loadaddr2
  %c = icmp slt %T0_14 %v0, %v1
; CHECK: vmin.s32
; CHECK: vmin.s32
; COST: func_blend14
; COST: cost of 2 {{.*}} select
  %r = select %T1_14 %c, %T0_14 %v0, %T0_14 %v1
  store %T0_14 %r, %T0_14* %storeaddr
  ret void
}
%T0_15 = type <16 x i32>
%T1_15 = type <16 x i1>
; CHECK-LABEL: func_blend15:
define void @func_blend15(%T0_15* %loadaddr, %T0_15* %loadaddr2,
                           %T1_15* %blend, %T0_15* %storeaddr) {
; CHECK: vmin.s32
; CHECK: vmin.s32
  %v0 = load %T0_15, %T0_15* %loadaddr
  %v1 = load %T0_15, %T0_15* %loadaddr2
  %c = icmp slt %T0_15 %v0, %v1
; COST: func_blend15
; COST: cost of 4 {{.*}} select
  %r = select %T1_15 %c, %T0_15 %v0, %T0_15 %v1
  store %T0_15 %r, %T0_15* %storeaddr
  ret void
}

; We adjusted the cost model of the following selects. When we improve code
; lowering we also need to adjust the cost.
%T0_18 = type <4 x i64>
%T1_18 = type <4 x i1>
; CHECK-LABEL: func_blend18:
define void @func_blend18(%T0_18* %loadaddr, %T0_18* %loadaddr2,
                           %T1_18* %blend, %T0_18* %storeaddr) {
; CHECK: vbsl
; CHECK: vbsl
  %v0 = load %T0_18, %T0_18* %loadaddr
  %v1 = load %T0_18, %T0_18* %loadaddr2
  %c = icmp slt %T0_18 %v0, %v1
; COST: func_blend18
; COST: cost of 19 {{.*}} select
  %r = select %T1_18 %c, %T0_18 %v0, %T0_18 %v1
  store %T0_18 %r, %T0_18* %storeaddr
  ret void
}
%T0_19 = type <8 x i64>
%T1_19 = type <8 x i1>
; CHECK-LABEL: func_blend19:
define void @func_blend19(%T0_19* %loadaddr, %T0_19* %loadaddr2,
                           %T1_19* %blend, %T0_19* %storeaddr) {
; CHECK: vbsl
; CHECK: vbsl
; CHECK: vbsl
; CHECK: vbsl
  %v0 = load %T0_19, %T0_19* %loadaddr
  %v1 = load %T0_19, %T0_19* %loadaddr2
  %c = icmp slt %T0_19 %v0, %v1
; COST: func_blend19
; COST: cost of 50 {{.*}} select
  %r = select %T1_19 %c, %T0_19 %v0, %T0_19 %v1
  store %T0_19 %r, %T0_19* %storeaddr
  ret void
}
%T0_20 = type <16 x i64>
%T1_20 = type <16 x i1>
; CHECK-LABEL: func_blend20:
define void @func_blend20(%T0_20* %loadaddr, %T0_20* %loadaddr2,
                           %T1_20* %blend, %T0_20* %storeaddr) {
; CHECK: vbsl
; CHECK: vbsl
; CHECK: vbsl
; CHECK: vbsl
; CHECK: vbsl
; CHECK: vbsl
; CHECK: vbsl
; CHECK: vbsl
  %v0 = load %T0_20, %T0_20* %loadaddr
  %v1 = load %T0_20, %T0_20* %loadaddr2
  %c = icmp slt %T0_20 %v0, %v1
; COST: func_blend20
; COST: cost of 100 {{.*}} select
  %r = select %T1_20 %c, %T0_20 %v0, %T0_20 %v1
  store %T0_20 %r, %T0_20* %storeaddr
  ret void
}
