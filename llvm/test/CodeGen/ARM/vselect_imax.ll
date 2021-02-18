; RUN: opt < %s  -cost-model -analyze -mtriple=arm-apple-ios6.0.0 -mcpu=cortex-a8 | FileCheck %s --check-prefix=COST
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
; COST: cost of 2 {{.*}} icmp
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
; COST: cost of 2 {{.*}} icmp
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
; COST: cost of 4 {{.*}} icmp
; COST: cost of 4 {{.*}} select
  %r = select %T1_15 %c, %T0_15 %v0, %T0_15 %v1
  store %T0_15 %r, %T0_15* %storeaddr
  ret void
}

; We adjusted the cost model of the following selects. When we improve code
; lowering we also need to adjust the cost.
%T0_18 = type <4 x i64>
%T1_18 = type <4 x i1>
define void @func_blend18(%T0_18* %loadaddr, %T0_18* %loadaddr2,
                           %T1_18* %blend, %T0_18* %storeaddr) {
; CHECK-LABEL: func_blend18:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    .save {r4, r5, r6, r7, r11, lr}
; CHECK-NEXT:    push {r4, r5, r6, r7, r11, lr}
; CHECK-NEXT:    vld1.64 {d22, d23}, [r0:128]!
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1:128]!
; CHECK-NEXT:    vld1.64 {d18, d19}, [r1:128]
; CHECK-NEXT:    mov r1, #0
; CHECK-NEXT:    vld1.64 {d20, d21}, [r0:128]
; CHECK-NEXT:    vmov.32 r12, d18[0]
; CHECK-NEXT:    vmov.32 r2, d20[0]
; CHECK-NEXT:    vmov.32 lr, d18[1]
; CHECK-NEXT:    vmov.32 r0, d20[1]
; CHECK-NEXT:    vmov.32 r7, d16[0]
; CHECK-NEXT:    vmov.32 r5, d22[0]
; CHECK-NEXT:    vmov.32 r4, d22[1]
; CHECK-NEXT:    vmov.32 r6, d19[0]
; CHECK-NEXT:    subs r2, r2, r12
; CHECK-NEXT:    vmov.32 r2, d16[1]
; CHECK-NEXT:    sbcs r0, r0, lr
; CHECK-NEXT:    mov r0, #0
; CHECK-NEXT:    movlt r0, #1
; CHECK-NEXT:    cmp r0, #0
; CHECK-NEXT:    mvnne r0, #0
; CHECK-NEXT:    subs r7, r5, r7
; CHECK-NEXT:    vmov.32 r7, d21[0]
; CHECK-NEXT:    vmov.32 r5, d19[1]
; CHECK-NEXT:    sbcs r2, r4, r2
; CHECK-NEXT:    vmov.32 r4, d21[1]
; CHECK-NEXT:    mov r2, #0
; CHECK-NEXT:    movlt r2, #1
; CHECK-NEXT:    cmp r2, #0
; CHECK-NEXT:    mvnne r2, #0
; CHECK-NEXT:    subs r7, r7, r6
; CHECK-NEXT:    vmov.32 r6, d23[0]
; CHECK-NEXT:    vmov.32 r7, d17[0]
; CHECK-NEXT:    sbcs r5, r4, r5
; CHECK-NEXT:    mov r4, #0
; CHECK-NEXT:    movlt r4, #1
; CHECK-NEXT:    vmov.32 r5, d17[1]
; CHECK-NEXT:    subs r7, r6, r7
; CHECK-NEXT:    vmov.32 r7, d23[1]
; CHECK-NEXT:    sbcs r7, r7, r5
; CHECK-NEXT:    movlt r1, #1
; CHECK-NEXT:    cmp r1, #0
; CHECK-NEXT:    mvnne r1, #0
; CHECK-NEXT:    cmp r4, #0
; CHECK-NEXT:    vdup.32 d25, r1
; CHECK-NEXT:    mvnne r4, #0
; CHECK-NEXT:    vdup.32 d24, r2
; CHECK-NEXT:    vdup.32 d27, r4
; CHECK-NEXT:    vbit q8, q11, q12
; CHECK-NEXT:    vdup.32 d26, r0
; CHECK-NEXT:    vbit q9, q10, q13
; CHECK-NEXT:    vst1.64 {d16, d17}, [r3:128]!
; CHECK-NEXT:    vst1.64 {d18, d19}, [r3:128]
; CHECK-NEXT:    pop {r4, r5, r6, r7, r11, lr}
; CHECK-NEXT:    mov pc, lr
  %v0 = load %T0_18, %T0_18* %loadaddr
  %v1 = load %T0_18, %T0_18* %loadaddr2
  %c = icmp slt %T0_18 %v0, %v1
; COST: func_blend18
; COST: cost of 2 {{.*}} icmp
; COST: cost of 19 {{.*}} select
  %r = select %T1_18 %c, %T0_18 %v0, %T0_18 %v1
  store %T0_18 %r, %T0_18* %storeaddr
  ret void
}
%T0_19 = type <8 x i64>
%T1_19 = type <8 x i1>
define void @func_blend19(%T0_19* %loadaddr, %T0_19* %loadaddr2,
                           %T1_19* %blend, %T0_19* %storeaddr) {
; CHECK-LABEL: func_blend19:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    .save {r4, r5, r6, r7, r11, lr}
; CHECK-NEXT:    push {r4, r5, r6, r7, r11, lr}
; CHECK-NEXT:    add r2, r1, #48
; CHECK-NEXT:    add r5, r1, #32
; CHECK-NEXT:    vld1.64 {d16, d17}, [r2:128]
; CHECK-NEXT:    add r2, r0, #48
; CHECK-NEXT:    add r6, r0, #32
; CHECK-NEXT:    mov r7, #0
; CHECK-NEXT:    vld1.64 {d18, d19}, [r2:128]
; CHECK-NEXT:    vmov.32 r12, d16[0]
; CHECK-NEXT:    vmov.32 r2, d18[0]
; CHECK-NEXT:    vmov.32 lr, d16[1]
; CHECK-NEXT:    vmov.32 r4, d18[1]
; CHECK-NEXT:    vld1.64 {d28, d29}, [r0:128]!
; CHECK-NEXT:    vld1.64 {d26, d27}, [r5:128]
; CHECK-NEXT:    vld1.64 {d30, d31}, [r6:128]
; CHECK-NEXT:    vmov.32 r5, d17[0]
; CHECK-NEXT:    vld1.64 {d22, d23}, [r0:128]
; CHECK-NEXT:    vmov.32 r0, d17[1]
; CHECK-NEXT:    vld1.64 {d24, d25}, [r1:128]!
; CHECK-NEXT:    vld1.64 {d20, d21}, [r1:128]
; CHECK-NEXT:    mov r1, #0
; CHECK-NEXT:    subs r2, r2, r12
; CHECK-NEXT:    mov r12, #0
; CHECK-NEXT:    vmov.32 r2, d19[0]
; CHECK-NEXT:    sbcs r6, r4, lr
; CHECK-NEXT:    vmov.32 r4, d24[0]
; CHECK-NEXT:    vmov.32 r6, d19[1]
; CHECK-NEXT:    movlt r12, #1
; CHECK-NEXT:    cmp r12, #0
; CHECK-NEXT:    mvnne r12, #0
; CHECK-NEXT:    subs r2, r2, r5
; CHECK-NEXT:    vmov.32 r5, d28[0]
; CHECK-NEXT:    mov r2, #0
; CHECK-NEXT:    sbcs r0, r6, r0
; CHECK-NEXT:    vmov.32 r6, d28[1]
; CHECK-NEXT:    vmov.32 r0, d24[1]
; CHECK-NEXT:    movlt r2, #1
; CHECK-NEXT:    cmp r2, #0
; CHECK-NEXT:    mvnne r2, #0
; CHECK-NEXT:    vdup.32 d7, r2
; CHECK-NEXT:    vdup.32 d6, r12
; CHECK-NEXT:    subs r5, r5, r4
; CHECK-NEXT:    vmov.32 r4, d25[1]
; CHECK-NEXT:    vmov.32 r5, d25[0]
; CHECK-NEXT:    sbcs r0, r6, r0
; CHECK-NEXT:    mov r6, #0
; CHECK-NEXT:    vmov.32 r0, d29[0]
; CHECK-NEXT:    movlt r6, #1
; CHECK-NEXT:    cmp r6, #0
; CHECK-NEXT:    mvnne r6, #0
; CHECK-NEXT:    subs r0, r0, r5
; CHECK-NEXT:    vmov.32 r5, d21[0]
; CHECK-NEXT:    vmov.32 r0, d29[1]
; CHECK-NEXT:    sbcs r0, r0, r4
; CHECK-NEXT:    vmov.32 r4, d23[0]
; CHECK-NEXT:    mov r0, #0
; CHECK-NEXT:    movlt r0, #1
; CHECK-NEXT:    cmp r0, #0
; CHECK-NEXT:    mvnne r0, #0
; CHECK-NEXT:    vdup.32 d1, r0
; CHECK-NEXT:    mov r0, #0
; CHECK-NEXT:    vdup.32 d0, r6
; CHECK-NEXT:    vmov.32 r6, d22[0]
; CHECK-NEXT:    vbit q12, q14, q0
; CHECK-NEXT:    subs r5, r4, r5
; CHECK-NEXT:    vmov.32 r4, d23[1]
; CHECK-NEXT:    vmov.32 r5, d21[1]
; CHECK-NEXT:    sbcs r5, r4, r5
; CHECK-NEXT:    vmov.32 r4, d20[1]
; CHECK-NEXT:    vmov.32 r5, d20[0]
; CHECK-NEXT:    movlt r0, #1
; CHECK-NEXT:    cmp r0, #0
; CHECK-NEXT:    mvnne r0, #0
; CHECK-NEXT:    vdup.32 d5, r0
; CHECK-NEXT:    add r0, r3, #32
; CHECK-NEXT:    subs r6, r6, r5
; CHECK-NEXT:    vmov.32 r5, d26[0]
; CHECK-NEXT:    vmov.32 r6, d22[1]
; CHECK-NEXT:    sbcs r6, r6, r4
; CHECK-NEXT:    mov r4, #0
; CHECK-NEXT:    vmov.32 r6, d30[0]
; CHECK-NEXT:    movlt r4, #1
; CHECK-NEXT:    subs r6, r6, r5
; CHECK-NEXT:    vmov.32 r5, d30[1]
; CHECK-NEXT:    vmov.32 r6, d26[1]
; CHECK-NEXT:    sbcs r6, r5, r6
; CHECK-NEXT:    vmov.32 r5, d31[0]
; CHECK-NEXT:    vmov.32 r6, d27[0]
; CHECK-NEXT:    movlt r1, #1
; CHECK-NEXT:    subs r6, r5, r6
; CHECK-NEXT:    vmov.32 r5, d31[1]
; CHECK-NEXT:    vmov.32 r6, d27[1]
; CHECK-NEXT:    sbcs r6, r5, r6
; CHECK-NEXT:    movlt r7, #1
; CHECK-NEXT:    cmp r7, #0
; CHECK-NEXT:    mvnne r7, #0
; CHECK-NEXT:    cmp r1, #0
; CHECK-NEXT:    mvnne r1, #0
; CHECK-NEXT:    vdup.32 d3, r7
; CHECK-NEXT:    cmp r4, #0
; CHECK-NEXT:    vdup.32 d2, r1
; CHECK-NEXT:    mvnne r4, #0
; CHECK-NEXT:    vbit q13, q15, q1
; CHECK-NEXT:    vdup.32 d4, r4
; CHECK-NEXT:    vbit q10, q11, q2
; CHECK-NEXT:    vbit q8, q9, q3
; CHECK-NEXT:    vst1.64 {d26, d27}, [r0:128]
; CHECK-NEXT:    add r0, r3, #48
; CHECK-NEXT:    vst1.64 {d24, d25}, [r3:128]!
; CHECK-NEXT:    vst1.64 {d16, d17}, [r0:128]
; CHECK-NEXT:    vst1.64 {d20, d21}, [r3:128]
; CHECK-NEXT:    pop {r4, r5, r6, r7, r11, lr}
; CHECK-NEXT:    mov pc, lr
  %v0 = load %T0_19, %T0_19* %loadaddr
  %v1 = load %T0_19, %T0_19* %loadaddr2
  %c = icmp slt %T0_19 %v0, %v1
; COST: func_blend19
; COST: cost of 4 {{.*}} icmp
; COST: cost of 50 {{.*}} select
  %r = select %T1_19 %c, %T0_19 %v0, %T0_19 %v1
  store %T0_19 %r, %T0_19* %storeaddr
  ret void
}
%T0_20 = type <16 x i64>
%T1_20 = type <16 x i1>
define void @func_blend20(%T0_20* %loadaddr, %T0_20* %loadaddr2,
                           %T1_20* %blend, %T0_20* %storeaddr) {
; CHECK-LABEL: func_blend20:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    .save {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-NEXT:    push {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-NEXT:    .pad #4
; CHECK-NEXT:    sub sp, sp, #4
; CHECK-NEXT:    .vsave {d8, d9, d10, d11}
; CHECK-NEXT:    vpush {d8, d9, d10, d11}
; CHECK-NEXT:    .pad #8
; CHECK-NEXT:    sub sp, sp, #8
; CHECK-NEXT:    mov r8, r1
; CHECK-NEXT:    mov r9, r0
; CHECK-NEXT:    vld1.64 {d16, d17}, [r8:128]!
; CHECK-NEXT:    add r10, r0, #64
; CHECK-NEXT:    vld1.64 {d18, d19}, [r9:128]!
; CHECK-NEXT:    vmov.32 r2, d16[0]
; CHECK-NEXT:    str r3, [sp, #4] @ 4-byte Spill
; CHECK-NEXT:    vmov.32 r6, d18[0]
; CHECK-NEXT:    vmov.32 r4, d16[1]
; CHECK-NEXT:    vmov.32 r7, d18[1]
; CHECK-NEXT:    vmov.32 r5, d17[0]
; CHECK-NEXT:    subs r2, r6, r2
; CHECK-NEXT:    mov r6, #0
; CHECK-NEXT:    vmov.32 r2, d19[0]
; CHECK-NEXT:    sbcs r7, r7, r4
; CHECK-NEXT:    movlt r6, #1
; CHECK-NEXT:    vmov.32 r7, d17[1]
; CHECK-NEXT:    subs r2, r2, r5
; CHECK-NEXT:    vmov.32 r2, d19[1]
; CHECK-NEXT:    sbcs r2, r2, r7
; CHECK-NEXT:    mov r2, #0
; CHECK-NEXT:    movlt r2, #1
; CHECK-NEXT:    cmp r2, #0
; CHECK-NEXT:    mvnne r2, #0
; CHECK-NEXT:    cmp r6, #0
; CHECK-NEXT:    vdup.32 d21, r2
; CHECK-NEXT:    mvnne r6, #0
; CHECK-NEXT:    vdup.32 d20, r6
; CHECK-NEXT:    mov r2, #32
; CHECK-NEXT:    add r6, r1, #64
; CHECK-NEXT:    vld1.64 {d24, d25}, [r10:128], r2
; CHECK-NEXT:    vbit q8, q9, q10
; CHECK-NEXT:    vld1.64 {d28, d29}, [r6:128], r2
; CHECK-NEXT:    vmov.32 r4, d29[0]
; CHECK-NEXT:    vmov.32 r5, d25[0]
; CHECK-NEXT:    vld1.64 {d0, d1}, [r9:128]
; CHECK-NEXT:    vld1.64 {d2, d3}, [r8:128]
; CHECK-NEXT:    vld1.64 {d22, d23}, [r6:128]!
; CHECK-NEXT:    vld1.64 {d20, d21}, [r6:128]
; CHECK-NEXT:    vmov.32 r6, d0[0]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r10:128]!
; CHECK-NEXT:    vmov.32 r9, d23[0]
; CHECK-NEXT:    vmov.32 r11, d19[0]
; CHECK-NEXT:    vmov.32 r8, d23[1]
; CHECK-NEXT:    subs r4, r5, r4
; CHECK-NEXT:    vmov.32 r5, d25[1]
; CHECK-NEXT:    vmov.32 r4, d29[1]
; CHECK-NEXT:    sbcs r4, r5, r4
; CHECK-NEXT:    vmov.32 r5, d24[0]
; CHECK-NEXT:    mov r4, #0
; CHECK-NEXT:    movlt r4, #1
; CHECK-NEXT:    cmp r4, #0
; CHECK-NEXT:    mvnne r4, #0
; CHECK-NEXT:    vdup.32 d5, r4
; CHECK-NEXT:    vmov.32 r4, d28[0]
; CHECK-NEXT:    subs r4, r5, r4
; CHECK-NEXT:    vmov.32 r5, d24[1]
; CHECK-NEXT:    vmov.32 r4, d28[1]
; CHECK-NEXT:    sbcs r4, r5, r4
; CHECK-NEXT:    vmov.32 r5, d1[0]
; CHECK-NEXT:    mov r4, #0
; CHECK-NEXT:    movlt r4, #1
; CHECK-NEXT:    cmp r4, #0
; CHECK-NEXT:    mvnne r4, #0
; CHECK-NEXT:    vdup.32 d4, r4
; CHECK-NEXT:    vmov.32 r4, d3[0]
; CHECK-NEXT:    subs r4, r5, r4
; CHECK-NEXT:    vmov.32 r5, d1[1]
; CHECK-NEXT:    vmov.32 r4, d3[1]
; CHECK-NEXT:    sbcs r4, r5, r4
; CHECK-NEXT:    add r5, r1, #32
; CHECK-NEXT:    vld1.64 {d26, d27}, [r5:128]
; CHECK-NEXT:    add r5, r1, #48
; CHECK-NEXT:    mov r4, #0
; CHECK-NEXT:    add r1, r1, #80
; CHECK-NEXT:    vld1.64 {d30, d31}, [r5:128]
; CHECK-NEXT:    movlt r4, #1
; CHECK-NEXT:    cmp r4, #0
; CHECK-NEXT:    vbif q12, q14, q2
; CHECK-NEXT:    vmov.32 r5, d2[0]
; CHECK-NEXT:    mvnne r4, #0
; CHECK-NEXT:    vdup.32 d29, r4
; CHECK-NEXT:    vmov.32 r4, d31[1]
; CHECK-NEXT:    subs r5, r6, r5
; CHECK-NEXT:    vmov.32 r6, d0[1]
; CHECK-NEXT:    vmov.32 r5, d2[1]
; CHECK-NEXT:    sbcs r5, r6, r5
; CHECK-NEXT:    add r6, r0, #48
; CHECK-NEXT:    mov r5, #0
; CHECK-NEXT:    vld1.64 {d6, d7}, [r6:128]
; CHECK-NEXT:    movlt r5, #1
; CHECK-NEXT:    cmp r5, #0
; CHECK-NEXT:    mvnne r5, #0
; CHECK-NEXT:    vmov.32 r7, d7[0]
; CHECK-NEXT:    vdup.32 d28, r5
; CHECK-NEXT:    vmov.32 r5, d31[0]
; CHECK-NEXT:    vbsl q14, q0, q1
; CHECK-NEXT:    vmov.32 r6, d7[1]
; CHECK-NEXT:    vmov.32 r2, d6[0]
; CHECK-NEXT:    subs r5, r7, r5
; CHECK-NEXT:    vmov.32 r7, d6[1]
; CHECK-NEXT:    sbcs r4, r6, r4
; CHECK-NEXT:    vmov.32 r6, d30[0]
; CHECK-NEXT:    vmov.32 r5, d30[1]
; CHECK-NEXT:    mov r4, #0
; CHECK-NEXT:    movlt r4, #1
; CHECK-NEXT:    cmp r4, #0
; CHECK-NEXT:    mvnne r4, #0
; CHECK-NEXT:    vdup.32 d3, r4
; CHECK-NEXT:    vmov.32 r4, d26[1]
; CHECK-NEXT:    subs r2, r2, r6
; CHECK-NEXT:    sbcs r2, r7, r5
; CHECK-NEXT:    add r5, r0, #32
; CHECK-NEXT:    mov r2, #0
; CHECK-NEXT:    vld1.64 {d0, d1}, [r5:128]
; CHECK-NEXT:    movlt r2, #1
; CHECK-NEXT:    cmp r2, #0
; CHECK-NEXT:    mvnne r2, #0
; CHECK-NEXT:    vmov.32 r6, d0[0]
; CHECK-NEXT:    vdup.32 d2, r2
; CHECK-NEXT:    add r0, r0, #80
; CHECK-NEXT:    vmov.32 r2, d26[0]
; CHECK-NEXT:    vbit q15, q3, q1
; CHECK-NEXT:    vmov.32 r5, d0[1]
; CHECK-NEXT:    vmov.32 r7, d1[0]
; CHECK-NEXT:    vld1.64 {d2, d3}, [r10:128]
; CHECK-NEXT:    vld1.64 {d6, d7}, [r1:128]
; CHECK-NEXT:    vld1.64 {d8, d9}, [r0:128]
; CHECK-NEXT:    vmov.32 r1, d7[1]
; CHECK-NEXT:    vmov.32 r10, d19[1]
; CHECK-NEXT:    vmov.32 lr, d6[0]
; CHECK-NEXT:    vmov.32 r3, d8[0]
; CHECK-NEXT:    vmov.32 r12, d8[1]
; CHECK-NEXT:    subs r2, r6, r2
; CHECK-NEXT:    vmov.32 r6, d1[1]
; CHECK-NEXT:    sbcs r2, r5, r4
; CHECK-NEXT:    vmov.32 r5, d27[0]
; CHECK-NEXT:    vmov.32 r4, d27[1]
; CHECK-NEXT:    mov r2, #0
; CHECK-NEXT:    movlt r2, #1
; CHECK-NEXT:    cmp r2, #0
; CHECK-NEXT:    mvnne r2, #0
; CHECK-NEXT:    subs r5, r7, r5
; CHECK-NEXT:    vmov.32 r7, d7[0]
; CHECK-NEXT:    sbcs r4, r6, r4
; CHECK-NEXT:    vmov.32 r6, d2[0]
; CHECK-NEXT:    mov r4, #0
; CHECK-NEXT:    vmov.32 r5, d2[1]
; CHECK-NEXT:    movlt r4, #1
; CHECK-NEXT:    cmp r4, #0
; CHECK-NEXT:    mvnne r4, #0
; CHECK-NEXT:    vdup.32 d5, r4
; CHECK-NEXT:    vdup.32 d4, r2
; CHECK-NEXT:    vmov.32 r2, d20[0]
; CHECK-NEXT:    vbit q13, q0, q2
; CHECK-NEXT:    vmov.32 r4, d20[1]
; CHECK-NEXT:    subs r0, r6, r2
; CHECK-NEXT:    vmov.32 r2, d9[1]
; CHECK-NEXT:    sbcs r0, r5, r4
; CHECK-NEXT:    vmov.32 r4, d9[0]
; CHECK-NEXT:    mov r0, #0
; CHECK-NEXT:    vmov.32 r6, d18[0]
; CHECK-NEXT:    movlt r0, #1
; CHECK-NEXT:    cmp r0, #0
; CHECK-NEXT:    mvnne r0, #0
; CHECK-NEXT:    vmov.32 r5, d18[1]
; CHECK-NEXT:    subs r4, r4, r7
; CHECK-NEXT:    vmov.32 r7, d21[1]
; CHECK-NEXT:    sbcs r1, r2, r1
; CHECK-NEXT:    vmov.32 r4, d22[1]
; CHECK-NEXT:    vmov.32 r1, d22[0]
; CHECK-NEXT:    mov r2, #0
; CHECK-NEXT:    movlt r2, #1
; CHECK-NEXT:    cmp r2, #0
; CHECK-NEXT:    mvnne r2, #0
; CHECK-NEXT:    vdup.32 d11, r2
; CHECK-NEXT:    vmov.32 r2, d3[1]
; CHECK-NEXT:    subs r1, r6, r1
; CHECK-NEXT:    vmov.32 r6, d21[0]
; CHECK-NEXT:    sbcs r1, r5, r4
; CHECK-NEXT:    vmov.32 r4, d3[0]
; CHECK-NEXT:    vmov.32 r5, d6[1]
; CHECK-NEXT:    mov r1, #0
; CHECK-NEXT:    movlt r1, #1
; CHECK-NEXT:    cmp r1, #0
; CHECK-NEXT:    mvnne r1, #0
; CHECK-NEXT:    subs r4, r4, r6
; CHECK-NEXT:    sbcs r2, r2, r7
; CHECK-NEXT:    mov r2, #0
; CHECK-NEXT:    movlt r2, #1
; CHECK-NEXT:    subs r4, r11, r9
; CHECK-NEXT:    sbcs r4, r10, r8
; CHECK-NEXT:    mov r4, #0
; CHECK-NEXT:    movlt r4, #1
; CHECK-NEXT:    subs r3, r3, lr
; CHECK-NEXT:    sbcs r3, r12, r5
; CHECK-NEXT:    mov r3, #0
; CHECK-NEXT:    movlt r3, #1
; CHECK-NEXT:    cmp r3, #0
; CHECK-NEXT:    mvnne r3, #0
; CHECK-NEXT:    cmp r4, #0
; CHECK-NEXT:    mvnne r4, #0
; CHECK-NEXT:    vdup.32 d10, r3
; CHECK-NEXT:    vdup.32 d1, r4
; CHECK-NEXT:    vorr q2, q5, q5
; CHECK-NEXT:    cmp r2, #0
; CHECK-NEXT:    vdup.32 d0, r1
; CHECK-NEXT:    vbsl q2, q4, q3
; CHECK-NEXT:    mvnne r2, #0
; CHECK-NEXT:    vbif q9, q11, q0
; CHECK-NEXT:    ldr r1, [sp, #4] @ 4-byte Reload
; CHECK-NEXT:    vdup.32 d7, r2
; CHECK-NEXT:    vdup.32 d6, r0
; CHECK-NEXT:    add r0, r1, #80
; CHECK-NEXT:    vbit q10, q1, q3
; CHECK-NEXT:    vst1.64 {d4, d5}, [r0:128]
; CHECK-NEXT:    add r0, r1, #32
; CHECK-NEXT:    vst1.64 {d26, d27}, [r0:128]
; CHECK-NEXT:    add r0, r1, #48
; CHECK-NEXT:    vst1.64 {d30, d31}, [r0:128]
; CHECK-NEXT:    add r0, r1, #64
; CHECK-NEXT:    vst1.64 {d16, d17}, [r1:128]!
; CHECK-NEXT:    vst1.64 {d28, d29}, [r1:128]
; CHECK-NEXT:    mov r1, #32
; CHECK-NEXT:    vst1.64 {d24, d25}, [r0:128], r1
; CHECK-NEXT:    vst1.64 {d18, d19}, [r0:128]!
; CHECK-NEXT:    vst1.64 {d20, d21}, [r0:128]
; CHECK-NEXT:    add sp, sp, #8
; CHECK-NEXT:    vpop {d8, d9, d10, d11}
; CHECK-NEXT:    add sp, sp, #4
; CHECK-NEXT:    pop {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-NEXT:    mov pc, lr
  %v0 = load %T0_20, %T0_20* %loadaddr
  %v1 = load %T0_20, %T0_20* %loadaddr2
  %c = icmp slt %T0_20 %v0, %v1
; COST: func_blend20
; COST: cost of 8 {{.*}} icmp
; COST: cost of 100 {{.*}} select
  %r = select %T1_20 %c, %T0_20 %v0, %T0_20 %v1
  store %T0_20 %r, %T0_20* %storeaddr
  ret void
}
