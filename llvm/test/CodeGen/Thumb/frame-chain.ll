; RUN: llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=all --verify-machineinstrs | FileCheck %s --check-prefixes=FP,LEAF-FP
; RUN: llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=all -mattr=+aapcs-frame-chain --verify-machineinstrs | FileCheck %s --check-prefixes=FP-AAPCS,LEAF-FP
; RUN: llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=all -mattr=+aapcs-frame-chain-leaf --verify-machineinstrs | FileCheck %s --check-prefixes=FP-AAPCS,LEAF-FP-AAPCS
; RUN: llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=non-leaf --verify-machineinstrs | FileCheck %s --check-prefixes=FP,LEAF-NOFP
; RUN: llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=non-leaf -mattr=+aapcs-frame-chain --verify-machineinstrs | FileCheck %s --check-prefixes=FP-AAPCS,LEAF-NOFP
; RUN: llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=non-leaf -mattr=+aapcs-frame-chain-leaf --verify-machineinstrs | FileCheck %s --check-prefixes=FP-AAPCS,LEAF-NOFP-AAPCS
; RUN: llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=none --verify-machineinstrs | FileCheck %s --check-prefixes=NOFP,LEAF-NOFP
; RUN: llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=none -mattr=+aapcs-frame-chain --verify-machineinstrs | FileCheck %s --check-prefixes=NOFP-AAPCS,LEAF-NOFP
; RUN: llc -mtriple thumbv6m-arm-none-eabi -filetype asm -o - %s -frame-pointer=none -mattr=+aapcs-frame-chain-leaf --verify-machineinstrs | FileCheck %s --check-prefixes=NOFP-AAPCS,LEAF-NOFP-AAPCS

define dso_local noundef i32 @leaf(i32 noundef %0) {
; LEAF-FP-LABEL: leaf:
; LEAF-FP:       @ %bb.0:
; LEAF-FP-NEXT:    .pad #4
; LEAF-FP-NEXT:    sub sp, #4
; LEAF-FP-NEXT:    str r0, [sp]
; LEAF-FP-NEXT:    adds r0, r0, #4
; LEAF-FP-NEXT:    add sp, #4
; LEAF-FP-NEXT:    bx lr
;
; LEAF-FP-AAPCS-LABEL: leaf:
; LEAF-FP-AAPCS:       @ %bb.0:
; LEAF-FP-AAPCS-NEXT:    .save {lr}
; LEAF-FP-AAPCS-NEXT:    push {lr}
; LEAF-FP-AAPCS-NEXT:    mov lr, r11
; LEAF-FP-AAPCS-NEXT:    .save {r11}
; LEAF-FP-AAPCS-NEXT:    push {lr}
; LEAF-FP-AAPCS-NEXT:    .setfp r11, sp
; LEAF-FP-AAPCS-NEXT:    mov r11, sp
; LEAF-FP-AAPCS-NEXT:    .pad #4
; LEAF-FP-AAPCS-NEXT:    sub sp, #4
; LEAF-FP-AAPCS-NEXT:    str r0, [sp]
; LEAF-FP-AAPCS-NEXT:    adds r0, r0, #4
; LEAF-FP-AAPCS-NEXT:    add sp, #4
; LEAF-FP-AAPCS-NEXT:    pop {r1}
; LEAF-FP-AAPCS-NEXT:    mov r11, r1
; LEAF-FP-AAPCS-NEXT:    pop {pc}
;
; LEAF-NOFP-LABEL: leaf:
; LEAF-NOFP:       @ %bb.0:
; LEAF-NOFP-NEXT:    .pad #4
; LEAF-NOFP-NEXT:    sub sp, #4
; LEAF-NOFP-NEXT:    str r0, [sp]
; LEAF-NOFP-NEXT:    adds r0, r0, #4
; LEAF-NOFP-NEXT:    add sp, #4
; LEAF-NOFP-NEXT:    bx lr
;
; LEAF-NOFP-AAPCS-LABEL: leaf:
; LEAF-NOFP-AAPCS:       @ %bb.0:
; LEAF-NOFP-AAPCS-NEXT:    .pad #4
; LEAF-NOFP-AAPCS-NEXT:    sub sp, #4
; LEAF-NOFP-AAPCS-NEXT:    str r0, [sp]
; LEAF-NOFP-AAPCS-NEXT:    adds r0, r0, #4
; LEAF-NOFP-AAPCS-NEXT:    add sp, #4
; LEAF-NOFP-AAPCS-NEXT:    bx lr
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  %4 = add nsw i32 %3, 4
  ret i32 %4
}

define dso_local noundef i32 @non_leaf(i32 noundef %0) {
; FP-LABEL: non_leaf:
; FP:       @ %bb.0:
; FP-NEXT:    .save {r7, lr}
; FP-NEXT:    push {r7, lr}
; FP-NEXT:    .setfp r7, sp
; FP-NEXT:    add r7, sp, #0
; FP-NEXT:    .pad #8
; FP-NEXT:    sub sp, #8
; FP-NEXT:    str r0, [sp, #4]
; FP-NEXT:    bl leaf
; FP-NEXT:    adds r0, r0, #1
; FP-NEXT:    add sp, #8
; FP-NEXT:    pop {r7, pc}
;
; FP-AAPCS-LABEL: non_leaf:
; FP-AAPCS:       @ %bb.0:
; FP-AAPCS-NEXT:    .save {lr}
; FP-AAPCS-NEXT:    push {lr}
; FP-AAPCS-NEXT:    mov lr, r11
; FP-AAPCS-NEXT:    .save {r11}
; FP-AAPCS-NEXT:    push {lr}
; FP-AAPCS-NEXT:    .setfp r11, sp
; FP-AAPCS-NEXT:    mov r11, sp
; FP-AAPCS-NEXT:    .pad #8
; FP-AAPCS-NEXT:    sub sp, #8
; FP-AAPCS-NEXT:    str r0, [sp, #4]
; FP-AAPCS-NEXT:    bl leaf
; FP-AAPCS-NEXT:    adds r0, r0, #1
; FP-AAPCS-NEXT:    add sp, #8
; FP-AAPCS-NEXT:    pop {r1}
; FP-AAPCS-NEXT:    mov r11, r1
; FP-AAPCS-NEXT:    pop {pc}
;
; NOFP-LABEL: non_leaf:
; NOFP:       @ %bb.0:
; NOFP-NEXT:    .save {r7, lr}
; NOFP-NEXT:    push {r7, lr}
; NOFP-NEXT:    .pad #8
; NOFP-NEXT:    sub sp, #8
; NOFP-NEXT:    str r0, [sp, #4]
; NOFP-NEXT:    bl leaf
; NOFP-NEXT:    adds r0, r0, #1
; NOFP-NEXT:    add sp, #8
; NOFP-NEXT:    pop {r7, pc}
;
; NOFP-AAPCS-LABEL: non_leaf:
; NOFP-AAPCS:       @ %bb.0:
; NOFP-AAPCS-NEXT:    .save {r7, lr}
; NOFP-AAPCS-NEXT:    push {r7, lr}
; NOFP-AAPCS-NEXT:    .pad #8
; NOFP-AAPCS-NEXT:    sub sp, #8
; NOFP-AAPCS-NEXT:    str r0, [sp, #4]
; NOFP-AAPCS-NEXT:    bl leaf
; NOFP-AAPCS-NEXT:    adds r0, r0, #1
; NOFP-AAPCS-NEXT:    add sp, #8
; NOFP-AAPCS-NEXT:    pop {r7, pc}
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  %4 = call noundef i32 @leaf(i32 noundef %3)
  %5 = add nsw i32 %4, 1
  ret i32 %5
}

declare i8* @llvm.stacksave()
define dso_local void @required_fp(i32 %0, i32 %1) {
; FP-LABEL: required_fp:
; FP:       @ %bb.0:
; FP-NEXT:    .save {r4, r6, r7, lr}
; FP-NEXT:    push {r4, r6, r7, lr}
; FP-NEXT:    .setfp r7, sp, #8
; FP-NEXT:    add r7, sp, #8
; FP-NEXT:    .pad #24
; FP-NEXT:    sub sp, #24
; FP-NEXT:    mov r6, sp
; FP-NEXT:    mov r2, r6
; FP-NEXT:    str r1, [r2, #16]
; FP-NEXT:    str r0, [r2, #20]
; FP-NEXT:    mov r1, sp
; FP-NEXT:    str r1, [r2, #8]
; FP-NEXT:    lsls r1, r0, #2
; FP-NEXT:    adds r1, r1, #7
; FP-NEXT:    movs r3, #7
; FP-NEXT:    bics r1, r3
; FP-NEXT:    mov r3, sp
; FP-NEXT:    subs r1, r3, r1
; FP-NEXT:    mov sp, r1
; FP-NEXT:    movs r1, #0
; FP-NEXT:    str r1, [r6, #4]
; FP-NEXT:    str r0, [r2]
; FP-NEXT:    subs r4, r7, #7
; FP-NEXT:    subs r4, #1
; FP-NEXT:    mov sp, r4
; FP-NEXT:    pop {r4, r6, r7, pc}
;
; FP-AAPCS-LABEL: required_fp:
; FP-AAPCS:       @ %bb.0:
; FP-AAPCS-NEXT:    .save {lr}
; FP-AAPCS-NEXT:    push {lr}
; FP-AAPCS-NEXT:    mov lr, r11
; FP-AAPCS-NEXT:    .save {r11}
; FP-AAPCS-NEXT:    push {lr}
; FP-AAPCS-NEXT:    .setfp r11, sp
; FP-AAPCS-NEXT:    mov r11, sp
; FP-AAPCS-NEXT:    .save {r4, r6}
; FP-AAPCS-NEXT:    push {r4, r6}
; FP-AAPCS-NEXT:    .pad #24
; FP-AAPCS-NEXT:    sub sp, #24
; FP-AAPCS-NEXT:    mov r6, sp
; FP-AAPCS-NEXT:    mov r2, r6
; FP-AAPCS-NEXT:    str r1, [r2, #16]
; FP-AAPCS-NEXT:    str r0, [r2, #20]
; FP-AAPCS-NEXT:    mov r1, sp
; FP-AAPCS-NEXT:    str r1, [r2, #8]
; FP-AAPCS-NEXT:    lsls r1, r0, #2
; FP-AAPCS-NEXT:    adds r1, r1, #7
; FP-AAPCS-NEXT:    movs r3, #7
; FP-AAPCS-NEXT:    bics r1, r3
; FP-AAPCS-NEXT:    mov r3, sp
; FP-AAPCS-NEXT:    subs r1, r3, r1
; FP-AAPCS-NEXT:    mov sp, r1
; FP-AAPCS-NEXT:    movs r1, #0
; FP-AAPCS-NEXT:    str r1, [r6, #4]
; FP-AAPCS-NEXT:    str r0, [r2]
; FP-AAPCS-NEXT:    mov r4, r11
; FP-AAPCS-NEXT:    subs r4, #8
; FP-AAPCS-NEXT:    mov sp, r4
; FP-AAPCS-NEXT:    pop {r4, r6}
; FP-AAPCS-NEXT:    pop {r0}
; FP-AAPCS-NEXT:    mov r11, r0
; FP-AAPCS-NEXT:    pop {pc}
;
; NOFP-LABEL: required_fp:
; NOFP:       @ %bb.0:
; NOFP-NEXT:    .save {r4, r6, r7, lr}
; NOFP-NEXT:    push {r4, r6, r7, lr}
; NOFP-NEXT:    .setfp r7, sp, #8
; NOFP-NEXT:    add r7, sp, #8
; NOFP-NEXT:    .pad #24
; NOFP-NEXT:    sub sp, #24
; NOFP-NEXT:    mov r6, sp
; NOFP-NEXT:    mov r2, r6
; NOFP-NEXT:    str r1, [r2, #16]
; NOFP-NEXT:    str r0, [r2, #20]
; NOFP-NEXT:    mov r1, sp
; NOFP-NEXT:    str r1, [r2, #8]
; NOFP-NEXT:    lsls r1, r0, #2
; NOFP-NEXT:    adds r1, r1, #7
; NOFP-NEXT:    movs r3, #7
; NOFP-NEXT:    bics r1, r3
; NOFP-NEXT:    mov r3, sp
; NOFP-NEXT:    subs r1, r3, r1
; NOFP-NEXT:    mov sp, r1
; NOFP-NEXT:    movs r1, #0
; NOFP-NEXT:    str r1, [r6, #4]
; NOFP-NEXT:    str r0, [r2]
; NOFP-NEXT:    subs r4, r7, #7
; NOFP-NEXT:    subs r4, #1
; NOFP-NEXT:    mov sp, r4
; NOFP-NEXT:    pop {r4, r6, r7, pc}
;
; NOFP-AAPCS-LABEL: required_fp:
; NOFP-AAPCS:       @ %bb.0:
; NOFP-AAPCS-NEXT:    .save {lr}
; NOFP-AAPCS-NEXT:    push {lr}
; NOFP-AAPCS-NEXT:    mov lr, r11
; NOFP-AAPCS-NEXT:    .save {r11}
; NOFP-AAPCS-NEXT:    push {lr}
; NOFP-AAPCS-NEXT:    .setfp r11, sp
; NOFP-AAPCS-NEXT:    mov r11, sp
; NOFP-AAPCS-NEXT:    .save {r4, r6}
; NOFP-AAPCS-NEXT:    push {r4, r6}
; NOFP-AAPCS-NEXT:    .pad #24
; NOFP-AAPCS-NEXT:    sub sp, #24
; NOFP-AAPCS-NEXT:    mov r6, sp
; NOFP-AAPCS-NEXT:    mov r2, r6
; NOFP-AAPCS-NEXT:    str r1, [r2, #16]
; NOFP-AAPCS-NEXT:    str r0, [r2, #20]
; NOFP-AAPCS-NEXT:    mov r1, sp
; NOFP-AAPCS-NEXT:    str r1, [r2, #8]
; NOFP-AAPCS-NEXT:    lsls r1, r0, #2
; NOFP-AAPCS-NEXT:    adds r1, r1, #7
; NOFP-AAPCS-NEXT:    movs r3, #7
; NOFP-AAPCS-NEXT:    bics r1, r3
; NOFP-AAPCS-NEXT:    mov r3, sp
; NOFP-AAPCS-NEXT:    subs r1, r3, r1
; NOFP-AAPCS-NEXT:    mov sp, r1
; NOFP-AAPCS-NEXT:    movs r1, #0
; NOFP-AAPCS-NEXT:    str r1, [r6, #4]
; NOFP-AAPCS-NEXT:    str r0, [r2]
; NOFP-AAPCS-NEXT:    mov r4, r11
; NOFP-AAPCS-NEXT:    subs r4, #8
; NOFP-AAPCS-NEXT:    mov sp, r4
; NOFP-AAPCS-NEXT:    pop {r4, r6}
; NOFP-AAPCS-NEXT:    pop {r0}
; NOFP-AAPCS-NEXT:    mov r11, r0
; NOFP-AAPCS-NEXT:    pop {pc}
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i8*, align 8
  %6 = alloca i64, align 8
  store i32 %0, i32* %3, align 4
  store i32 %1, i32* %4, align 4
  %7 = load i32, i32* %3, align 4
  %8 = zext i32 %7 to i64
  %9 = call i8* @llvm.stacksave()
  store i8* %9, i8** %5, align 8
  %10 = alloca i32, i64 %8, align 4
  store i64 %8, i64* %6, align 8
  ret void
}
