; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s -check-prefix=CHECK --check-prefix=CHECK-LE
; RUN: llc -mtriple=armv7-eabi %s -o - | FileCheck %s -check-prefix=CHECK --check-prefix=CHECK-V7-LE
; RUN: llc -mtriple=armeb-eabi %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-BE
; RUN: llc -mtriple=armebv7-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-V7-BE
; RUN: llc -mtriple=thumbv6-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-V6-THUMB
; RUN: llc -mtriple=thumbv6t2-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-T2-DSP
; RUN: llc -mtriple=thumbv7-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-T2-DSP
; RUN: llc -mtriple=thumbebv7-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-V7-THUMB-BE
; RUN: llc -mtriple=thumbv6m-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-V6M-THUMB
; RUN: llc -mtriple=thumbv7m-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-V7M-THUMB
; RUN: llc -mtriple=thumbv7em-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-T2-DSP
; RUN: llc -mtriple=armv5te-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-V5TE
; Check generated signed and unsigned multiply accumulate long.

define i64 @MACLongTest1(i32 %a, i32 %b, i64 %c) {
;CHECK-LABEL: MACLongTest1:
;CHECK-V6-THUMB-NOT: umlal
;CHECK-LE: umlal [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], [[LHS:r[0-9]+]], [[RHS:r[0-9]+]]
;CHECK-LE: mov r0, [[RDLO]]
;CHECK-LE: mov r1, [[RDHI]]
;CHECK-BE: umlal [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], [[LHS:r[0-9]+]], [[RHS:r[0-9]+]]
;CHECK-BE: mov r0, [[RDHI]]
;CHECK-BE: mov r1, [[RDLO]]
;CHECK-T2-DSP: umlal [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], [[LHS:r[0-9]+]], [[RHS:r[0-9]+]]
;CHECK-T2-DSP-NEXT: mov r0, [[RDLO]]
;CHECK-T2-DSP-NEXT: mov r1, [[RDHI]]
;CHECK-V7-THUMB-BE: umlal [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], [[LHS:r[0-9]+]], [[RHS:r[0-9]+]]
;CHECK-V7-THUMB-BE: mov r0, [[RDHI]]
;CHECK-V7-THUMB-BE: mov r1, [[RDLO]]
  %conv = zext i32 %a to i64
  %conv1 = zext i32 %b to i64
  %mul = mul i64 %conv1, %conv
  %add = add i64 %mul, %c
  ret i64 %add
}

define i64 @MACLongTest2(i32 %a, i32 %b, i64 %c)  {
;CHECK-LABEL: MACLongTest2:
;CHECK-LE: smlal [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], [[LHS:r[0-9]+]], [[RHS:r[0-9]+]]
;CHECK-LE: mov r0, [[RDLO]]
;CHECK-LE: mov r1, [[RDHI]]
;CHECK-BE: smlal [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], [[LHS:r[0-9]+]], [[RHS:r[0-9]+]]
;CHECK-BE: mov r0, [[RDHI]]
;CHECK-BE: mov r1, [[RDLO]]
;CHECK-T2-DSP: smlal [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], [[LHS:r[0-9]+]], [[RHS:r[0-9]+]]
;CHECK-T2-DSP-NEXT: mov r0, [[RDLO]]
;CHECK-T2-DSP-NEXT: mov r1, [[RDHI]]
;CHECK-V7-THUMB-BE: smlal [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], [[LHS:r[0-9]+]], [[RHS:r[0-9]+]]
;CHECK-V7-THUMB-BE: mov r0, [[RDHI]]
;CHECK-V7-THUMB-BE: mov r1, [[RDLO]]
  %conv = sext i32 %a to i64
  %conv1 = sext i32 %b to i64
  %mul = mul nsw i64 %conv1, %conv
  %add = add nsw i64 %mul, %c
  ret i64 %add
}

; Two things to check here: the @earlyclobber constraint (on <= v5) and the "$Rd = $R" ones.
;    + Without @earlyclobber the v7 code is natural. With it, the first two
;      registers must be distinct from the third.
;    + Without "$Rd = $R", this can be satisfied without a mov before the umlal
;      by trying to use 6 different registers in the MachineInstr. The natural
;      evolution of this attempt currently leaves only two movs in the final
;      function, both after the umlal. With it, *some* move has to happen
;      before the umlal.
define i64 @MACLongTest3(i32 %a, i32 %b, i32 %c) {
;CHECK-LABEL: MACLongTest3:
;CHECK-LE: mov [[RDHI:r[0-9]+]], #0
;CHECK-LE: umlal [[RDLO:r[0-9]+]], [[RDHI]], r1, r0
;CHECK-LE: mov r0, [[RDLO]]
;CHECK-LE: mov r1, [[RDHI]]
;CHECK-BE: mov [[RDHI:r[0-9]+]], #0
;CHECK-BE: umlal [[RDLO:r[0-9]+]], [[RDHI]], r1, r0
;CHECK-BE: mov r0, [[RDHI]]
;CHECK-BE: mov r1, [[RDLO]]
;CHECK-T2-DSP: umlal
;CHECK-V6-THUMB-NOT: umlal
  %conv = zext i32 %b to i64
  %conv1 = zext i32 %a to i64
  %mul = mul i64 %conv, %conv1
  %conv2 = zext i32 %c to i64
  %add = add i64 %mul, %conv2
  ret i64 %add
}

define i64 @MACLongTest4(i32 %a, i32 %b, i32 %c) {
;CHECK-LABEL: MACLongTest4:
;CHECK-V6-THUMB-NOT: smlal
;CHECK-T2-DSP: smlal
;CHECK-LE: asr [[RDHI:r[0-9]+]], [[RDLO:r[0-9]+]], #31
;CHECK-LE: smlal [[RDLO]], [[RDHI]], r1, r0
;CHECK-LE: mov r0, [[RDLO]]
;CHECK-LE: mov r1, [[RDHI]]
;CHECK-BE: asr [[RDHI:r[0-9]+]], [[RDLO:r[0-9]+]], #31
;CHECK-BE: smlal [[RDLO]], [[RDHI]], r1, r0
;CHECK-BE: mov r0, [[RDHI]]
;CHECK-BE: mov r1, [[RDLO]]
  %conv = sext i32 %b to i64
  %conv1 = sext i32 %a to i64
  %mul = mul nsw i64 %conv, %conv1
  %conv2 = sext i32 %c to i64
  %add = add nsw i64 %mul, %conv2
  ret i64 %add
}

define i64 @MACLongTest6(i32 %a, i32 %b, i32 %c, i32 %d) {
;CHECK-LABEL: MACLongTest6:
;CHECK-V6-THUMB-NOT: smull
;CHECK-V6-THUMB-NOT: smlal
;CHECK-LE: smull   r12, lr, r1, r0
;CHECK-LE: smlal   r12, lr, r3, r2
;CHECK-V7: smull   [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], r1, r0
;CHECK-V7: smlal   [[RDLO]], [[RDHI]], [[Rn:r[0-9]+]], [[Rm:r[0-9]+]]
;CHECK-T2-DSP: smull   [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], r1, r0
;CHECK-T2-DSP: smlal   [[RDLO]], [[RDHI]], [[Rn:r[0-9]+]], [[Rm:r[0-9]+]]
  %conv = sext i32 %a to i64
  %conv1 = sext i32 %b to i64
  %mul = mul nsw i64 %conv1, %conv
  %conv2 = sext i32 %c to i64
  %conv3 = sext i32 %d to i64
  %mul4 = mul nsw i64 %conv3, %conv2
  %add = add nsw i64 %mul4, %mul
  ret i64 %add
}

define i64 @MACLongTest7(i64 %acc, i32 %lhs, i32 %rhs) {
;CHECK-LABEL: MACLongTest7:
;CHECK-NOT: smlal
;CHECK-V6-THUMB2-NOT: smlal
;CHECK-V7-THUMB-NOT: smlal
;CHECK-V6-THUMB-NOT: smlal
  %conv = sext i32 %lhs to i64
  %conv1 = sext i32 %rhs to i64
  %mul = mul nsw i64 %conv1, %conv
  %shl = shl i64 %mul, 32
  %shr = lshr i64 %mul, 32
  %or = or i64 %shl, %shr
  %add = add i64 %or, %acc
  ret i64 %add
}

define i64 @MACLongTest8(i64 %acc, i32 %lhs, i32 %rhs) {
;CHECK-LABEL: MACLongTest8:
;CHECK-NOT: smlal
;CHECK-V6-THUMB2-NOT: smlal
;CHECK-V7-THUMB-NOT: smlal
;CHECK-V6-THUMB-NOT: smlal
  %conv = zext i32 %lhs to i64
  %conv1 = zext i32 %rhs to i64
  %mul = mul nuw i64 %conv1, %conv
  %and = and i64 %mul, 4294967295
  %shl = shl i64 %mul, 32
  %or = or i64 %and, %shl
  %add = add i64 %or, %acc
  ret i64 %add
}

define i64 @MACLongTest9(i32 %lhs, i32 %rhs, i32 %lo, i32 %hi) {
;CHECK-LABEL: MACLongTest9:
;CHECK-V7-LE: umaal [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], [[LHS:r[0-9]+]], [[RHS:r[0-9]+]]
;CHECK-V7-LE: mov r0, [[RDLO]]
;CHECK-V7-LE: mov r1, [[RDHI]]
;CHECK-V7-BE: umaal [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], [[LHS:r[0-9]+]], [[RHS:r[0-9]+]]
;CHECK-V7-BE: mov r0, [[RDHI]]
;CHECK-V7-BE: mov r1, [[RDLO]]
;CHECK-T2-DSP: umaal [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], [[LHS:r[0-9]+]], [[RHS:r[0-9]+]]
;CHECK-T2-DSP-NEXT: mov r0, [[RDLO]]
;CHECK-T2-DSP-NEXT: mov r1, [[RDHI]]
;CHECK-V7-THUMB-BE: umaal [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], [[LHS:r[0-9]+]], [[RHS:r[0-9]+]]
;CHECK-V7-THUMB-BE: mov r0, [[RDHI]]
;CHECK-V7-THUMB-BE: mov r1, [[RDLO]]
;CHECK-NOT:umaal
;CHECK-V6-THUMB-NOT: umaal
;CHECK-V6M-THUMB-NOT: umaal
;CHECK-V7M-THUMB-NOT: umaal
  %conv = zext i32 %lhs to i64
  %conv1 = zext i32 %rhs to i64
  %mul = mul nuw i64 %conv1, %conv
  %conv2 = zext i32 %lo to i64
  %add = add i64 %mul, %conv2
  %conv3 = zext i32 %hi to i64
  %add2 = add i64 %add, %conv3
  ret i64 %add2
}

define i64 @MACLongTest10(i32 %lhs, i32 %rhs, i32 %lo, i32 %hi) {
;CHECK-LABEL: MACLongTest10:
;CHECK-V7-LE: umaal [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], [[LHS:r[0-9]+]], [[RHS:r[0-9]+]]
;CHECK-V7-LE: mov r0, [[RDLO]]
;CHECK-V7-LE: mov r1, [[RDHI]]
;CHECK-V7-BE: umaal [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], [[LHS:r[0-9]+]], [[RHS:r[0-9]+]]
;CHECK-V7-BE: mov r0, [[RDHI]]
;CHECK-V7-BE: mov r1, [[RDLO]]
;CHECK-T2-DSP: umaal r2, r3, r1, r0
;CHECK-T2-DSP-NEXT: mov r0, r2
;CHECK-T2-DSP-NEXT: mov r1, r3
;CHECK-V7-THUMB-BE: umaal [[RDLO:r[0-9]+]], [[RDHI:r[0-9]+]], [[LHS:r[0-9]+]], [[RHS:r[0-9]+]]
;CHECK-V7-THUMB-BE: mov r0, [[RDHI]]
;CHECK-V7-THUMB-BE: mov r1, [[RDLO]]
;CHECK-NOT:umaal
;CHECK-V6-THUMB-NOT:umaal
;CHECK-V6M-THUMB-NOT: umaal
;CHECK-V7M-THUMB-NOT: umaal
  %conv = zext i32 %lhs to i64
  %conv1 = zext i32 %rhs to i64
  %mul = mul nuw i64 %conv1, %conv
  %conv2 = zext i32 %lo to i64
  %conv3 = zext i32 %hi to i64
  %add = add i64 %conv2, %conv3
  %add2 = add i64 %add, %mul
  ret i64 %add2
}

define i64 @MACLongTest11(i16 %a, i16 %b, i64 %c)  {
;CHECK-LABEL: MACLongTest11:
;CHECK-T2-DSP-NOT: sxth
;CHECK-T2-DSP: smlalbb r2, r3
;CHECK-T2-DSP-NEXT: mov r0, r2
;CHECK-T2-DSP-NEXT: mov r1, r3
;CHECK-V5TE-NOT: sxth
;CHECK-V5TE: smlalbb r2, r3
;CHECK-V5TE-NEXT: mov r0, r2
;CHECK-V5TE-NEXT: mov r1, r3
;CHECK-V7-LE-NOT: sxth
;CHECK-V7-LE: smlalbb r2, r3
;CHECK-V7-LE-NEXT: mov r0, r2
;CHECK-V7-LE-NEXT: mov r1, r3
;CHECK-V7-THUMB-BE: smlalbb r3, r2
;CHECK-V7-THUMB-BE-NEXT: mov r0, r2
;CHECK-V7-THUMB-BE-NEXT: mov r1, r3
;CHECK-LE-NOT: smlalbb
;CHECK-BE-NOT: smlalbb
;CHECK-V6M-THUMB-NOT: smlalbb
;CHECK-V7M-THUMB-NOT: smlalbb
  %conv = sext i16 %a to i32
  %conv1 = sext i16 %b to i32
  %mul = mul nsw i32 %conv1, %conv
  %conv2 = sext i32 %mul to i64
  %add = add nsw i64 %conv2, %c
  ret i64 %add
}

define i64 @MACLongTest12(i16 %b, i32 %t, i64 %c)  {
;CHECK-LABEL: MACLongTest12:
;CHECK-T2-DSP-NOT: sxth
;CHECK-T2-DSP-NOT: {{asr|lsr}}
;CHECK-T2-DSP: smlalbt r2, r3, r0, r1
;CHECK-T2-DSP-NEXT: mov r0, r2
;CHECK-T2-DSP-NEXT: mov r1, r3
;CHECK-T2-DSP-NOT: sxth
;CHECK-V5TE-NOT: sxth
;CHECK-V5TE-NOT: {{asr|lsr}}
;CHECK-V5TE: smlalbt r2, r3, r0, r1
;CHECK-V5TE-NEXT: mov r0, r2
;CHECK-V5TE-NEXT: mov r1, r3
;CHECK-V7-LE-NOT: sxth
;CHECK-V7-LE-NOT: {{asr|lsr}}
;CHECK-V7-LE: smlalbt r2, r3, r0, r1
;CHECK-V7-LE-NEXT: mov r0, r2
;CHECK-V7-LE-NEXT: mov r1, r3
;CHECK-V7-THUMB-BE: smlalbt r3, r2,
;CHECK-V7-THUMB-BE-NEXT: mov r0, r2
;CHECK-V7-THUMB-BE-NEXT: mov r1, r3
;CHECK-LE-NOT: smlalbt
;CHECK-BE-NOT: smlalbt
;CHECK-V6M-THUMB-NOT: smlalbt
;CHECK-V7M-THUMB-NOT: smlalbt
  %conv0 = sext i16 %b to i32
  %conv1 = ashr i32 %t, 16
  %mul = mul nsw i32 %conv0, %conv1
  %conv2 = sext i32 %mul to i64
  %add = add nsw i64 %conv2, %c
  ret i64 %add
}

define i64 @MACLongTest13(i32 %t, i16 %b, i64 %c)  {
;CHECK-LABEL: MACLongTest13:
;CHECK-T2-DSP-NOT: sxth
;CHECK-T2-DSP-NOT: {{asr|lsr}}
;CHECK-T2-DSP: smlaltb r2, r3, r0, r1
;CHECK-T2-DSP-NEXT: mov r0, r2
;CHECK-T2-DSP-NEXT: mov r1, r3
;CHECK-V5TE-NOT: sxth
;CHECK-V5TE-NOT: {{asr|lsr}}
;CHECK-V5TE: smlaltb r2, r3, r0, r1
;CHECK-V5TE-NEXT: mov r0, r2
;CHECK-V5TE-NEXT: mov r1, r3
;CHECK-V7-LE-NOT: sxth
;CHECK-V7-LE-NOT: {{asr|lsr}}
;CHECK-V7-LE: smlaltb r2, r3, r0, r1
;CHECK-V7-LE-NEXT: mov r0, r2
;CHECK-V7-LE-NEXT: mov r1, r3
;CHECK-V7-THUMB-BE: smlaltb r3, r2, r0, r1
;CHECK-V7-THUMB-BE-NEXT: mov r0, r2
;CHECK-V7-THUMB-BE-NEXT: mov r1, r3
;CHECK-LE-NOT: smlaltb
;CHECK-BE-NOT: smlaltb
;CHECK-V6M-THUMB-NOT: smlaltb
;CHECK-V7M-THUMB-NOT: smlaltb
  %conv0 = ashr i32 %t, 16
  %conv1= sext i16 %b to i32
  %mul = mul nsw i32 %conv0, %conv1
  %conv2 = sext i32 %mul to i64
  %add = add nsw i64 %conv2, %c
  ret i64 %add
}

define i64 @MACLongTest14(i32 %a, i32 %b, i64 %c)  {
;CHECK-LABEL: MACLongTest14:
;CHECK-T2-DSP-NOT: {{asr|lsr}}
;CHECK-T2-DSP: smlaltt r2, r3,
;CHECK-T2-DSP-NEXT: mov r0, r2
;CHECK-T2-DSP-NEXT: mov r1, r3
;CHECK-V5TE-NOT: {{asr|lsr}}
;CHECK-V5TE: smlaltt r2, r3,
;CHECK-V5TE-NEXT: mov r0, r2
;CHECK-V5TE-NEXT: mov r1, r3
;CHECK-V7-LE-NOT: {{asr|lsr}}
;CHECK-V7-LE: smlaltt r2, r3,
;CHECK-V7-LE-NEXT: mov r0, r2
;CHECK-V7-LE-NEXT: mov r1, r3
;CHECK-V7-THUMB-BE: smlaltt r3, r2,
;CHECK-V7-THUMB-BE-NEXT: mov r0, r2
;CHECK-V7-THUMB-BE-NEXT: mov r1, r3
;CHECK-LE-NOT: smlaltt
;CHECK-BE-NOT: smlaltt
;CHECK-V6M-THUMB-NOT: smlaltt
;CHECK-V7M-THUMB-NOT: smlaltt
  %conv0 = ashr i32 %a, 16
  %conv1 = ashr i32 %b, 16
  %mul = mul nsw i32 %conv1, %conv0
  %conv2 = sext i32 %mul to i64
  %add = add nsw i64 %conv2, %c
  ret i64 %add
}

@global_b = external global i16, align 2
;CHECK-LABEL: MACLongTest15
;CHECK-T2-DSP-NOT: {{asr|lsr}}
;CHECK-T2-DSP: mov r1, r3
;CHECK-T2-DSP: smlaltb r2, r1, r0, r3
;CHECK-T2-DSP-NEXT: mov r0, r2
;CHECK-V5TE-NOT: {{asr|lsr}}
;CHECK-V5TE: mov r1, r3
;CHECK-V5TE: smlaltb r2, r1, r0, r3
;CHECK-V5TE-NEXT: mov r0, r2
;CHECK-V7-LE-NOT: {{asr|lsr}}
;CHECK-V7-LE: mov r1, r3
;CHECK-V7-LE: smlaltb r2, r1, r0, r3
;CHECK-V7-LE-NEXT: mov r0, r2
;CHECK-V7-THUMB-BE: mov r1, r3
;CHECK-V7-THUMB-BE: smlaltb r1, r2, r0, r3
;CHECK-V7-THUMB-BE-NEXT: mov r0, r2
;CHECK-LE-NOT: smlaltb
;CHECK-BE-NOT: smlaltb
;CHECK-V6M-THUMB-NOT: smlaltb
;CHECK-V7M-THUMB-NOT: smlaltb
define i64 @MACLongTest15(i32 %t, i64 %acc) {
entry:
  %0 = load i16, ptr @global_b, align 2
  %conv = sext i16 %0 to i32
  %shr = ashr i32 %t, 16
  %mul = mul nsw i32 %shr, %conv
  %conv1 = sext i32 %mul to i64
  %add = add nsw i64 %conv1, %acc
  ret i64 %add
}

;CHECK-LABEL: MACLongTest16
;CHECK-T2-DSP-NOT: {{asr|lsr}}
;CHECK-T2-DSP: mov r1, r3
;CHECK-T2-DSP: smlalbt r2, r1, r3, r0
;CHECK-T2-DSP-NEXT: mov r0, r2
;CHECK-V5TE-NOT: {{asr|lsr}}
;CHECK-V5TE: mov r1, r3
;CHECK-V5TE: smlalbt r2, r1, r3, r0
;CHECK-V5TE-NEXT: mov r0, r2
;CHECK-V7-LE: mov r1, r3
;CHECK-V7-LE: smlalbt r2, r1, r3, r0
;CHECK-V7-LE-NEXT: mov r0, r2
;CHECK-V7-THUMB-BE: mov r1, r3
;CHECK-V7-THUMB-BE: smlalbt r1, r2, r3, r0
;CHECK-V7-THUMB-BE-NEXT: mov r0, r2
;CHECK-LE-NOT: smlalbt
;CHECK-BE-NOT: smlalbt
;CHECK-V6M-THUMB-NOT: smlalbt
;CHECK-V7M-THUMB-NOT: smlalbt
define i64 @MACLongTest16(i32 %t, i64 %acc) {
entry:
  %0 = load i16, ptr @global_b, align 2
  %conv = sext i16 %0 to i32
  %shr = ashr i32 %t, 16
  %mul = mul nsw i32 %conv, %shr
  %conv1 = sext i32 %mul to i64
  %add = add nsw i64 %conv1, %acc
  ret i64 %add
}
