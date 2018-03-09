; RUN: llc -mtriple=armv7 %s -o - | FileCheck %s
; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s --check-prefix=CHECK-T2

define i1 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: mov r2, #0
; CHECK: cmp r0, r1
; CHECK: movwne r2, #1
; CHECK: mov r0, r2
; CHECK-T2: mov{{.*}} r2, #0
; CHECK-T2: cmp r0, r1
; CHECK-T2: movne r2, #1
; CHECK-T2: mov r0, r2
    %tmp = icmp ne i32 %a, %b
    ret i1 %tmp
}

define i1 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: mov r2, #0
; CHECK: cmp r0, r1
; CHECK: movweq r2, #1
; CHECK: mov r0, r2
; CHECK-T2: mov{{.*}} r2, #0
; CHECK-T2: cmp r0, r1
; CHECK-T2: moveq r2, #1
; CHECK-T2: mov r0, r2
    %tmp = icmp eq i32 %a, %b
    ret i1 %tmp
}

define i1 @f6(i32 %a, i32 %b) {
; CHECK-LABEL: f6:
; CHECK: mov r2, #0
; CHECK: cmp {{.*}}, r1, lsl #5
; CHECK: movweq r2, #1
; CHECK: mov r0, r2
; CHECK-T2: mov{{.*}} r2, #0
; CHECK-T2: cmp.w r0, r1, lsl #5
; CHECK-T2: moveq r2, #1
; CHECK-T2: mov r0, r2
    %tmp = shl i32 %b, 5
    %tmp1 = icmp eq i32 %a, %tmp
    ret i1 %tmp1
}

define i1 @f7(i32 %a, i32 %b) {
; CHECK-LABEL: f7:
; CHECK: mov r2, #0
; CHECK: cmp r0, r1, lsr #6
; CHECK: movwne r2, #1
; CHECK: mov r0, r2
; CHECK-T2: mov{{.*}} r2, #0
; CHECK-T2: cmp.w r0, r1, lsr #6
; CHECK-T2: movne r2, #1
; CHECK-T2: mov r0, r2
    %tmp = lshr i32 %b, 6
    %tmp1 = icmp ne i32 %a, %tmp
    ret i1 %tmp1
}

define i1 @f8(i32 %a, i32 %b) {
; CHECK-LABEL: f8:
; CHECK: mov r2, #0
; CHECK: cmp r0, r1, asr #7
; CHECK: movweq r2, #1
; CHECK: mov r0, r2
; CHECK-T2: mov{{.*}} r2, #0
; CHECK-T2: cmp.w r0, r1, asr #7
; CHECK-T2: moveq r2, #1
; CHECK-T2: mov r0, r2
    %tmp = ashr i32 %b, 7
    %tmp1 = icmp eq i32 %a, %tmp
    ret i1 %tmp1
}

define i1 @f9(i32 %a) {
; CHECK-LABEL: f9:
; CHECK: mov r1, #0
; CHECK: cmp r0, r0, ror #8
; CHECK: movwne r1, #1
; CHECK: mov r0, r1
; CHECK-T2: mov{{.*}} r1, #0
; CHECK-T2: cmp.w r0, r0, ror #8
; CHECK-T2: movne r1, #1
; CHECK-T2: mov r0, r1
    %l8 = shl i32 %a, 24
    %r8 = lshr i32 %a, 8
    %tmp = or i32 %l8, %r8
    %tmp1 = icmp ne i32 %a, %tmp
    ret i1 %tmp1
}

; CHECK-LABEL: swap_cmp_shl
; CHECK: mov r2, #0
; CHECK: cmp r1, r0, lsl #11
; CHECK: movwlt r2, #1
; CHECK-T2: mov{{.*}} r2, #0
; CHECK-T2: cmp.w r1, r0, lsl #11
; CHECK-T2: movlt r2, #1
define arm_aapcscc i32 @swap_cmp_shl(i32 %a, i32 %b) {
entry:
  %shift = shl i32 %a, 11
  %cmp = icmp sgt i32 %shift, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: swap_cmp_lshr
; CHECK: mov r2, #0
; CHECK: cmp r1, r0, lsr #11
; CHECK: movwhi r2, #1
; CHECK-T2: mov{{.*}} r2, #0
; CHECK-T2: cmp.w r1, r0, lsr #11
; CHECK-T2: movhi r2, #1
define arm_aapcscc i32 @swap_cmp_lshr(i32 %a, i32 %b) {
entry:
  %shift = lshr i32 %a, 11
  %cmp = icmp ult i32 %shift, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: swap_cmp_ashr
; CHECK: mov r2, #0
; CHECK: cmp r1, r0, asr #11
; CHECK: movwle r2, #1
; CHECK-T2: mov{{.*}} r2, #0
; CHECK-T2: cmp.w r1, r0, asr #11
; CHECK-T2: movle r2, #1
define arm_aapcscc i32 @swap_cmp_ashr(i32 %a, i32 %b) {
entry:
  %shift = ashr i32 %a, 11
  %cmp = icmp sge i32 %shift, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK-LABEL: swap_cmp_rotr
; CHECK: mov r2, #0
; CHECK: cmp r1, r0, ror #11
; CHECK: movwls r2, #1
; CHECK-T2: mov{{.*}} r2, #0
; CHECK-T2: cmp.w r1, r0, ror #11
; CHECK-T2: movls r2, #1
define arm_aapcscc i32 @swap_cmp_rotr(i32 %a, i32 %b) {
entry:
  %lsr = lshr i32 %a, 11
  %lsl = shl i32 %a, 21
  %ror = or i32 %lsr, %lsl
  %cmp = icmp uge i32 %ror, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}
