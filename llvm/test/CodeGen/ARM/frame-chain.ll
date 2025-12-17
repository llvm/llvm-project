; RUN: llc -mtriple arm-arm-none-eabi -filetype asm -o - %s -frame-pointer=all | FileCheck %s --check-prefixes=FP,LEAF-FP
; RUN: llc -mtriple arm-arm-none-eabi -filetype asm -o - %s -frame-pointer=all -mattr=+aapcs-frame-chain | FileCheck %s --check-prefixes=FP-AAPCS,LEAF-FP-AAPCS
; RUN: llc -mtriple arm-arm-none-eabi -filetype asm -o - %s -frame-pointer=non-leaf | FileCheck %s --check-prefixes=FP,LEAF-NOFP
; RUN: llc -mtriple arm-arm-none-eabi -filetype asm -o - %s -frame-pointer=non-leaf -mattr=+aapcs-frame-chain | FileCheck %s --check-prefixes=FP-AAPCS,LEAF-NOFP-AAPCS
; RUN: llc -mtriple arm-arm-none-eabi -filetype asm -o - %s -frame-pointer=none | FileCheck %s --check-prefixes=NOFP,LEAF-NOFP
; RUN: llc -mtriple arm-arm-none-eabi -filetype asm -o - %s -frame-pointer=none -mattr=+aapcs-frame-chain | FileCheck %s --check-prefixes=NOFP-AAPCS,LEAF-NOFP-AAPCS
; RUN: llc -mtriple arm-arm-none-eabi -filetype asm -o - %s -frame-pointer=reserved | FileCheck %s --check-prefixes=NOFP,LEAF-NOFP
; RUN: llc -mtriple arm-arm-none-eabi -filetype asm -o - %s -frame-pointer=reserved -mattr=+aapcs-frame-chain | FileCheck %s --check-prefixes=NOFP-AAPCS,LEAF-NOFP-AAPCS

define dso_local noundef i32 @leaf(i32 noundef %0) {
; LEAF-FP-LABEL: leaf:
; LEAF-FP:       @ %bb.0:
; LEAF-FP-NEXT:    .save {r11, lr}
; LEAF-FP-NEXT:    push {r11, lr}
; LEAF-FP-NEXT:    .setfp r11, sp
; LEAF-FP-NEXT:    mov r11, sp
; LEAF-FP-NEXT:    push {r0}
; LEAF-FP-NEXT:    add r0, r0, #4
; LEAF-FP-NEXT:    mov sp, r11
; LEAF-FP-NEXT:    pop {r11, lr}
; LEAF-FP-NEXT:    mov pc, lr
;
; LEAF-FP-AAPCS-LABEL: leaf:
; LEAF-FP-AAPCS:       @ %bb.0:
; LEAF-FP-AAPCS-NEXT:    .save {r11, lr}
; LEAF-FP-AAPCS-NEXT:    push {r11, lr}
; LEAF-FP-AAPCS-NEXT:    .setfp r11, sp
; LEAF-FP-AAPCS-NEXT:    mov r11, sp
; LEAF-FP-AAPCS-NEXT:    push {r0}
; LEAF-FP-AAPCS-NEXT:    add r0, r0, #4
; LEAF-FP-AAPCS-NEXT:    mov sp, r11
; LEAF-FP-AAPCS-NEXT:    pop {r11, lr}
; LEAF-FP-AAPCS-NEXT:    mov pc, lr
;
; LEAF-NOFP-LABEL: leaf:
; LEAF-NOFP:       @ %bb.0:
; LEAF-NOFP-NEXT:    .pad #4
; LEAF-NOFP-NEXT:    sub sp, sp, #4
; LEAF-NOFP-NEXT:    str r0, [sp]
; LEAF-NOFP-NEXT:    add r0, r0, #4
; LEAF-NOFP-NEXT:    add sp, sp, #4
; LEAF-NOFP-NEXT:    mov pc, lr
;
; LEAF-NOFP-AAPCS-LABEL: leaf:
; LEAF-NOFP-AAPCS:       @ %bb.0:
; LEAF-NOFP-AAPCS-NEXT:    .pad #4
; LEAF-NOFP-AAPCS-NEXT:    sub sp, sp, #4
; LEAF-NOFP-AAPCS-NEXT:    str r0, [sp]
; LEAF-NOFP-AAPCS-NEXT:    add r0, r0, #4
; LEAF-NOFP-AAPCS-NEXT:    add sp, sp, #4
; LEAF-NOFP-AAPCS-NEXT:    mov pc, lr
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = add nsw i32 %3, 4
  ret i32 %4
}

define dso_local noundef i32 @non_leaf(i32 noundef %0) {
; FP-LABEL: non_leaf:
; FP:       @ %bb.0:
; FP-NEXT:    .save {r11, lr}
; FP-NEXT:    push {r11, lr}
; FP-NEXT:    .setfp r11, sp
; FP-NEXT:    mov r11, sp
; FP-NEXT:    .pad #8
; FP-NEXT:    sub sp, sp, #8
; FP-NEXT:    str r0, [sp, #4]
; FP-NEXT:    bl leaf
; FP-NEXT:    add r0, r0, #1
; FP-NEXT:    mov sp, r11
; FP-NEXT:    pop {r11, lr}
; FP-NEXT:    mov pc, lr
;
; FP-AAPCS-LABEL: non_leaf:
; FP-AAPCS:       @ %bb.0:
; FP-AAPCS-NEXT:    .save {r11, lr}
; FP-AAPCS-NEXT:    push {r11, lr}
; FP-AAPCS-NEXT:    .setfp r11, sp
; FP-AAPCS-NEXT:    mov r11, sp
; FP-AAPCS-NEXT:    .pad #8
; FP-AAPCS-NEXT:    sub sp, sp, #8
; FP-AAPCS-NEXT:    str r0, [sp, #4]
; FP-AAPCS-NEXT:    bl leaf
; FP-AAPCS-NEXT:    add r0, r0, #1
; FP-AAPCS-NEXT:    mov sp, r11
; FP-AAPCS-NEXT:    pop {r11, lr}
; FP-AAPCS-NEXT:    mov pc, lr
;
; NOFP-LABEL: non_leaf:
; NOFP:       @ %bb.0:
; NOFP-NEXT:    .save {r11, lr}
; NOFP-NEXT:    push {r11, lr}
; NOFP-NEXT:    .pad #8
; NOFP-NEXT:    sub sp, sp, #8
; NOFP-NEXT:    str r0, [sp, #4]
; NOFP-NEXT:    bl leaf
; NOFP-NEXT:    add r0, r0, #1
; NOFP-NEXT:    add sp, sp, #8
; NOFP-NEXT:    pop {r11, lr}
; NOFP-NEXT:    mov pc, lr
;
; NOFP-AAPCS-LABEL: non_leaf:
; NOFP-AAPCS:       @ %bb.0:
; NOFP-AAPCS-NEXT:    .save {r11, lr}
; NOFP-AAPCS-NEXT:    push {r11, lr}
; NOFP-AAPCS-NEXT:    .pad #8
; NOFP-AAPCS-NEXT:    sub sp, sp, #8
; NOFP-AAPCS-NEXT:    str r0, [sp, #4]
; NOFP-AAPCS-NEXT:    bl leaf
; NOFP-AAPCS-NEXT:    add r0, r0, #1
; NOFP-AAPCS-NEXT:    add sp, sp, #8
; NOFP-AAPCS-NEXT:    pop {r11, lr}
; NOFP-AAPCS-NEXT:    mov pc, lr
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = call noundef i32 @leaf(i32 noundef %3)
  %5 = add nsw i32 %4, 1
  ret i32 %5
}

declare ptr @llvm.stacksave()
define dso_local void @required_fp(i32 %0, i32 %1) {
; LEAF-FP-LABEL: required_fp:
; LEAF-FP:       @ %bb.0:
; LEAF-FP-NEXT:    .save {r4, r5, r11, lr}
; LEAF-FP-NEXT:    push {r4, r5, r11, lr}
; LEAF-FP-NEXT:    .setfp r11, sp, #8
; LEAF-FP-NEXT:    add r11, sp, #8
; LEAF-FP-NEXT:    .pad #24
; LEAF-FP-NEXT:    sub sp, sp, #24
; LEAF-FP-NEXT:    str r1, [r11, #-16]
; LEAF-FP-NEXT:    mov r1, #7
; LEAF-FP-NEXT:    add r1, r1, r0, lsl #2
; LEAF-FP-NEXT:    str r0, [r11, #-12]
; LEAF-FP-NEXT:    bic r1, r1, #7
; LEAF-FP-NEXT:    str sp, [r11, #-24]
; LEAF-FP-NEXT:    sub sp, sp, r1
; LEAF-FP-NEXT:    mov r1, #0
; LEAF-FP-NEXT:    str r0, [r11, #-32]
; LEAF-FP-NEXT:    str r1, [r11, #-28]
; LEAF-FP-NEXT:    sub sp, r11, #8
; LEAF-FP-NEXT:    pop {r4, r5, r11, lr}
; LEAF-FP-NEXT:    mov pc, lr
;
; LEAF-FP-AAPCS-LABEL: required_fp:
; LEAF-FP-AAPCS:       @ %bb.0:
; LEAF-FP-AAPCS-NEXT:    .save {r4, r5, r11, lr}
; LEAF-FP-AAPCS-NEXT:    push {r4, r5, r11, lr}
; LEAF-FP-AAPCS-NEXT:    .setfp r11, sp, #8
; LEAF-FP-AAPCS-NEXT:    add r11, sp, #8
; LEAF-FP-AAPCS-NEXT:    .pad #24
; LEAF-FP-AAPCS-NEXT:    sub sp, sp, #24
; LEAF-FP-AAPCS-NEXT:    str r1, [r11, #-16]
; LEAF-FP-AAPCS-NEXT:    mov r1, #7
; LEAF-FP-AAPCS-NEXT:    add r1, r1, r0, lsl #2
; LEAF-FP-AAPCS-NEXT:    str r0, [r11, #-12]
; LEAF-FP-AAPCS-NEXT:    bic r1, r1, #7
; LEAF-FP-AAPCS-NEXT:    str sp, [r11, #-24]
; LEAF-FP-AAPCS-NEXT:    sub sp, sp, r1
; LEAF-FP-AAPCS-NEXT:    mov r1, #0
; LEAF-FP-AAPCS-NEXT:    str r0, [r11, #-32]
; LEAF-FP-AAPCS-NEXT:    str r1, [r11, #-28]
; LEAF-FP-AAPCS-NEXT:    sub sp, r11, #8
; LEAF-FP-AAPCS-NEXT:    pop {r4, r5, r11, lr}
; LEAF-FP-AAPCS-NEXT:    mov pc, lr
;
; LEAF-NOFP-LABEL: required_fp:
; LEAF-NOFP:       @ %bb.0:
; LEAF-NOFP-NEXT:    .save {r4, r5, r11}
; LEAF-NOFP-NEXT:    push {r4, r5, r11}
; LEAF-NOFP-NEXT:    .setfp r11, sp, #8
; LEAF-NOFP-NEXT:    add r11, sp, #8
; LEAF-NOFP-NEXT:    .pad #20
; LEAF-NOFP-NEXT:    sub sp, sp, #20
; LEAF-NOFP-NEXT:    str r1, [r11, #-16]
; LEAF-NOFP-NEXT:    mov r1, #7
; LEAF-NOFP-NEXT:    add r1, r1, r0, lsl #2
; LEAF-NOFP-NEXT:    str r0, [r11, #-12]
; LEAF-NOFP-NEXT:    bic r1, r1, #7
; LEAF-NOFP-NEXT:    str sp, [r11, #-20]
; LEAF-NOFP-NEXT:    sub sp, sp, r1
; LEAF-NOFP-NEXT:    mov r1, #0
; LEAF-NOFP-NEXT:    str r0, [r11, #-28]
; LEAF-NOFP-NEXT:    str r1, [r11, #-24]
; LEAF-NOFP-NEXT:    sub sp, r11, #8
; LEAF-NOFP-NEXT:    pop {r4, r5, r11}
; LEAF-NOFP-NEXT:    mov pc, lr
;
; LEAF-NOFP-AAPCS-LABEL: required_fp:
; LEAF-NOFP-AAPCS:       @ %bb.0:
; LEAF-NOFP-AAPCS-NEXT:    .save {r4, r5, r11, lr}
; LEAF-NOFP-AAPCS-NEXT:    push {r4, r5, r11, lr}
; LEAF-NOFP-AAPCS-NEXT:    .setfp r11, sp, #8
; LEAF-NOFP-AAPCS-NEXT:    add r11, sp, #8
; LEAF-NOFP-AAPCS-NEXT:    .pad #24
; LEAF-NOFP-AAPCS-NEXT:    sub sp, sp, #24
; LEAF-NOFP-AAPCS-NEXT:    str r1, [r11, #-16]
; LEAF-NOFP-AAPCS-NEXT:    mov r1, #7
; LEAF-NOFP-AAPCS-NEXT:    add r1, r1, r0, lsl #2
; LEAF-NOFP-AAPCS-NEXT:    str r0, [r11, #-12]
; LEAF-NOFP-AAPCS-NEXT:    bic r1, r1, #7
; LEAF-NOFP-AAPCS-NEXT:    str sp, [r11, #-24]
; LEAF-NOFP-AAPCS-NEXT:    sub sp, sp, r1
; LEAF-NOFP-AAPCS-NEXT:    mov r1, #0
; LEAF-NOFP-AAPCS-NEXT:    str r0, [r11, #-32]
; LEAF-NOFP-AAPCS-NEXT:    str r1, [r11, #-28]
; LEAF-NOFP-AAPCS-NEXT:    sub sp, r11, #8
; LEAF-NOFP-AAPCS-NEXT:    pop {r4, r5, r11, lr}
; LEAF-NOFP-AAPCS-NEXT:    mov pc, lr
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %7 = load i32, ptr %3, align 4
  %8 = zext i32 %7 to i64
  %9 = call ptr @llvm.stacksave()
  store ptr %9, ptr %5, align 8
  %10 = alloca i32, i64 %8, align 4
  store i64 %8, ptr %6, align 8
  ret void
}

define dso_local noundef i32 @leaf_tailcall(i32 noundef %0) {
; LEAF-FP-LABEL: leaf_tailcall:
; LEAF-FP:       @ %bb.0:
; LEAF-FP-NEXT:    .save {r11, lr}
; LEAF-FP-NEXT:    push {r11, lr}
; LEAF-FP-NEXT:    .setfp r11, sp
; LEAF-FP-NEXT:    mov r11, sp
; LEAF-FP-NEXT:    pop {r11, lr}
; LEAF-FP-NEXT:    b leaf
;
; LEAF-FP-AAPCS-LABEL: leaf_tailcall:
; LEAF-FP-AAPCS:       @ %bb.0:
; LEAF-FP-AAPCS-NEXT:    .save {r11, lr}
; LEAF-FP-AAPCS-NEXT:    push {r11, lr}
; LEAF-FP-AAPCS-NEXT:    .setfp r11, sp
; LEAF-FP-AAPCS-NEXT:    mov r11, sp
; LEAF-FP-AAPCS-NEXT:    pop {r11, lr}
; LEAF-FP-AAPCS-NEXT:    b leaf
;
; LEAF-NOFP-LABEL: leaf_tailcall:
; LEAF-NOFP:       @ %bb.0:
; LEAF-NOFP-NEXT:    b leaf
;
; LEAF-NOFP-AAPCS-LABEL: leaf_tailcall:
; LEAF-NOFP-AAPCS:       @ %bb.0:
; LEAF-NOFP-AAPCS-NEXT:    b leaf
  %a = tail call noundef i32 @leaf(i32 noundef %0)
  ret i32 %a
}
