; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s -check-prefix=CHECK --check-prefix=CHECK-LE
; RUN: llc -mtriple=armv7-eabi %s -o - | FileCheck %s --check-prefix=CHECK-V7-LE
; RUN: llc -mtriple=armeb-eabi %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-BE
; RUN: llc -mtriple=armebv7-eabi %s -o - | FileCheck %s -check-prefix=CHECK-V7-BE
; Check generated signed and unsigned multiply accumulate long.

define i64 @MACLongTest1(i32 %a, i32 %b, i64 %c) {
;CHECK-LABEL: MACLongTest1:
;CHECK: umlal
  %conv = zext i32 %a to i64
  %conv1 = zext i32 %b to i64
  %mul = mul i64 %conv1, %conv
  %add = add i64 %mul, %c
  ret i64 %add
}

define i64 @MACLongTest2(i32 %a, i32 %b, i64 %c)  {
;CHECK-LABEL: MACLongTest2:
;CHECK: smlal
  %conv = sext i32 %a to i64
  %conv1 = sext i32 %b to i64
  %mul = mul nsw i64 %conv1, %conv
  %add = add nsw i64 %mul, %c
  ret i64 %add
}

define i64 @MACLongTest3(i32 %a, i32 %b, i32 %c) {
;CHECK-LABEL: MACLongTest3:
;CHECK: umlal
  %conv = zext i32 %b to i64
  %conv1 = zext i32 %a to i64
  %mul = mul i64 %conv, %conv1
  %conv2 = zext i32 %c to i64
  %add = add i64 %mul, %conv2
  ret i64 %add
}

define i64 @MACLongTest4(i32 %a, i32 %b, i32 %c) {
;CHECK-LABEL: MACLongTest4:
;CHECK: smlal
  %conv = sext i32 %b to i64
  %conv1 = sext i32 %a to i64
  %mul = mul nsw i64 %conv, %conv1
  %conv2 = sext i32 %c to i64
  %add = add nsw i64 %mul, %conv2
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
define i64 @MACLongTest5(i64 %c, i32 %a, i32 %b) {
; CHECK-V7-LE-LABEL: MACLongTest5:
; CHECK-V7-LE-LABEL: umlal r0, r1, r0, r0
; CHECK-V7-BE-LABEL: MACLongTest5:
; CHECK-V7-BE-LABEL: umlal r1, r0, r1, r1

; CHECK-LABEL: MACLongTest5:
; CHECK-LE: mov [[RDLO:r[0-9]+]], r0
; CHECK-LE: umlal [[RDLO]], r1, r0, r0
; CHECK-LE: mov r0, [[RDLO]]
; CHECK-BE: mov [[RDLO:r[0-9]+]], r1
; CHECK-BE: umlal [[RDLO]], r0, r1, r1
; CHECK-BE: mov r1, [[RDLO]]

  %conv.trunc = trunc i64 %c to i32
  %conv = zext i32 %conv.trunc to i64
  %conv1 = zext i32 %b to i64
  %mul = mul i64 %conv, %conv
  %add = add i64 %mul, %c
  ret i64 %add
}

define i64 @MACLongTest6(i32 %a, i32 %b, i32 %c, i32 %d) {
;CHECK-LABEL: MACLongTest6:
;CHECK: smull   r12, lr, r1, r0
;CHECK: smlal   r12, lr, r3, r2
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
;CHECK-V7-LE:umaal
;CHECK-V7-BE:umaal
;CHECK-NOT:umaal
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
;CHECK-V7-LE:umaal
;CHECK-V7-BE:umaal
;CHECK-NOT:umaal
  %conv = zext i32 %lhs to i64
  %conv1 = zext i32 %rhs to i64
  %mul = mul nuw i64 %conv1, %conv
  %conv2 = zext i32 %lo to i64
  %conv3 = zext i32 %hi to i64
  %add = add i64 %conv2, %conv3
  %add2 = add i64 %add, %mul
  ret i64 %add2
}
