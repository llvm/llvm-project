; RUN: llc < %s -mtriple=arm-apple-darwin -mcpu=cortex-a8 | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -mtriple=thumb-apple-darwin -mcpu=cortex-a8 | FileCheck %s -check-prefix=T2
; rdar://8662825

define i32 @t1(i32 %a, i32 %b, i32 %c) nounwind {
; ARM-LABEL: t1:
; ARM: mov r0, r1
; ARM: suble r0, r0, #-2147483647

; T2-LABEL: t1:
; T2: mov r0, r1
; T2: mvn r1, #-2147483648
; T2: addle r0, r1
  %tmp1 = icmp sgt i32 %c, 10
  %tmp2 = select i1 %tmp1, i32 0, i32 2147483647
  %tmp3 = add i32 %tmp2, %b
  ret i32 %tmp3
}

define i32 @t2(i32 %a, i32 %b, i32 %c, i32 %d) nounwind {
; ARM-LABEL: t2:
; ARM: mov r0, r1
; ARM: suble r0, r0, #10

; T2-LABEL: t2:
; T2: mov r0, r1
; T2: suble r0, #10
  %tmp1 = icmp sgt i32 %c, 10
  %tmp2 = select i1 %tmp1, i32 0, i32 10
  %tmp3 = sub i32 %b, %tmp2
  ret i32 %tmp3
}

define i32 @t3(i32 %a, i32 %b, i32 %x, i32 %y) nounwind {
; ARM-LABEL: t3:
; ARM: andge r3, r3, r2
; ARM: mov r0, r3

; T2-LABEL: t3:
; T2: andge r3, r2
; T2: mov r0, r3
  %cond = icmp slt i32 %a, %b
  %z = select i1 %cond, i32 -1, i32 %x
  %s = and i32 %z, %y
 ret i32 %s
}

define i32 @t4(i32 %a, i32 %b, i32 %x, i32 %y) nounwind {
; ARM-LABEL: t4:
; ARM: orrge r3, r3, r2
; ARM: mov r0, r3

; T2-LABEL: t4:
; T2: orrge r3, r2
; T2: mov r0, r3
  %cond = icmp slt i32 %a, %b
  %z = select i1 %cond, i32 0, i32 %x
  %s = or i32 %z, %y
 ret i32 %s
}

define i32 @t5(i32 %a, i32 %b, i32 %c) nounwind {
entry:
; ARM-LABEL: t5:
; ARM-NOT: moveq
; ARM: orreq r2, r2, #1

; T2-LABEL: t5:
; T2-NOT: moveq
; T2: orreq r2, r2, #1
  %tmp1 = icmp eq i32 %a, %b
  %tmp2 = zext i1 %tmp1 to i32
  %tmp3 = or i32 %tmp2, %c
  ret i32 %tmp3
}

define i32 @t6(i32 %a, i32 %b, i32 %c, i32 %d) nounwind {
; ARM-LABEL: t6:
; ARM-NOT: movge
; ARM: eorlt r3, r3, r2

; T2-LABEL: t6:
; T2-NOT: movge
; T2: eorlt r3, r2
  %cond = icmp slt i32 %a, %b
  %tmp1 = select i1 %cond, i32 %c, i32 0
  %tmp2 = xor i32 %tmp1, %d
  ret i32 %tmp2
}

define i32 @t7(i32 %a, i32 %b, i32 %c) nounwind {
entry:
; ARM-LABEL: t7:
; ARM-NOT: lsleq
; ARM: andeq r2, r2, r2, lsl #1

; T2-LABEL: t7:
; T2-NOT: lsleq.w
; T2: andeq.w r2, r2, r2, lsl #1
  %tmp1 = shl i32 %c, 1
  %cond = icmp eq i32 %a, %b
  %tmp2 = select i1 %cond, i32 %tmp1, i32 -1
  %tmp3 = and i32 %c, %tmp2
  ret i32 %tmp3
}

; Fold ORRri into movcc.
define i32 @t8(i32 %a, i32 %b) nounwind {
; ARM-LABEL: t8:
; ARM: cmp r0, r1
; ARM: orrge r0, r1, #1

; T2-LABEL: t8:
; T2: cmp r0, r1
; T2: orrge r0, r1, #1
  %x = or i32 %b, 1
  %cond = icmp slt i32 %a, %b
  %tmp1 = select i1 %cond, i32 %a, i32 %x
  ret i32 %tmp1
}

; Fold ANDrr into movcc.
define i32 @t9(i32 %a, i32 %b, i32 %c) nounwind {
; ARM-LABEL: t9:
; ARM: cmp r0, r1
; ARM: andge r0, r1, r2

; T2-LABEL: t9:
; T2: cmp r0, r1
; T2: andge.w r0, r1, r2
  %x = and i32 %b, %c
  %cond = icmp slt i32 %a, %b
  %tmp1 = select i1 %cond, i32 %a, i32 %x
  ret i32 %tmp1
}

; Fold EORrs into movcc.
define i32 @t10(i32 %a, i32 %b, i32 %c, i32 %d) nounwind {
; ARM-LABEL: t10:
; ARM: cmp r0, r1
; ARM: eorge r0, r1, r2, lsl #7

; T2-LABEL: t10:
; T2: cmp r0, r1
; T2: eorge.w r0, r1, r2, lsl #7
  %s = shl i32 %c, 7
  %x = xor i32 %b, %s
  %cond = icmp slt i32 %a, %b
  %tmp1 = select i1 %cond, i32 %a, i32 %x
  ret i32 %tmp1
}

; Fold ORRri into movcc, reversing the condition.
define i32 @t11(i32 %a, i32 %b) nounwind {
; ARM-LABEL: t11:
; ARM: cmp r0, r1
; ARM: orrlt r0, r1, #1

; T2-LABEL: t11:
; T2: cmp r0, r1
; T2: orrlt r0, r1, #1
  %x = or i32 %b, 1
  %cond = icmp slt i32 %a, %b
  %tmp1 = select i1 %cond, i32 %x, i32 %a
  ret i32 %tmp1
}

; Fold ADDri12 into movcc
define i32 @t12(i32 %a, i32 %b) nounwind {
; ARM-LABEL: t12:
; ARM: cmp r0, r1
; ARM: addge r0, r1,

; T2-LABEL: t12:
; T2: cmp r0, r1
; T2: addwge r0, r1, #3000
  %x = add i32 %b, 3000
  %cond = icmp slt i32 %a, %b
  %tmp1 = select i1 %cond, i32 %a, i32 %x
  ret i32 %tmp1
}

; Handle frame index operands.
define void @pr13628() nounwind uwtable align 2 {
  %x3 = alloca i8, i32 256, align 8
  %x4 = load i8, i8* undef, align 1
  %x5 = icmp ne i8 %x4, 0
  %x6 = select i1 %x5, i8* %x3, i8* null
  call void @bar(i8* %x6) nounwind
  ret void
}
declare void @bar(i8*)

; Fold zext i1 into predicated add
define i32 @t13(i32 %c, i32 %a) nounwind readnone ssp {
entry:
; ARM: t13
; ARM: cmp r1, #10
; ARM: addgt r0, r0, #1

; T2: t13
; T2: cmp r1, #10
; T2: addgt r0, #1
  %cmp = icmp sgt i32 %a, 10
  %conv = zext i1 %cmp to i32
  %add = add i32 %conv, %c
  ret i32 %add
}

; Fold sext i1 into predicated sub
define i32 @t14(i32 %c, i32 %a) nounwind readnone ssp {
entry:
; ARM: t14
; ARM: cmp r1, #10
; ARM: subgt r0, r0, #1

; T2: t14
; T2: cmp r1, #10
; T2: subgt r0, #1
  %cmp = icmp sgt i32 %a, 10
  %conv = sext i1 %cmp to i32
  %add = add i32 %conv, %c
  ret i32 %add
}

; Fold the xor into the select.
define i32 @t15(i32 %p) {
entry:
; ARM-LABEL: t15:
; ARM: mov     [[REG:r[0-9]+]], #3
; ARM: cmp     r0, #8
; ARM: movwgt  [[REG:r[0-9]+]], #0

; T2-LABEL: t15:
; T2: movs    [[REG:r[0-9]+]], #3
; T2: cmp     [[REG:r[0-9]+]], #8
; T2: it      gt
; T2: movgt   [[REG:r[0-9]+]], #0
  %cmp = icmp sgt i32 %p, 8
  %a = select i1 %cmp, i32 1, i32 2
  %xor = xor i32 %a, 1
  ret i32 %xor
}

define i32 @t16(i32 %x, i32 %y) {
entry:
; ARM-LABEL: t16:
; ARM: and r0, {{r[0-9]+}}, {{r[0-9]+}}

; T2-LABEL: t16:
; T2: ands r0, {{r[0-9]+}}
  %cmp = icmp eq i32 %x, 0
  %cond = select i1 %cmp, i32 5, i32 2
  %cmp1 = icmp eq i32 %y, 0
  %cond2 = select i1 %cmp1, i32 3, i32 4
  %and = and i32 %cond2, %cond
  ret i32 %and
}

define i32 @t17(i32 %x, i32 %y) #0 {
entry:
; ARM-LABEL: t17:
; ARM: and r0, {{r[0-9]+}}, {{r[0-9]+}}

; T2-LABEL: t17:
; T2: ands r0, {{r[0-9]+}}
  %cmp = icmp eq i32 %x, -1
  %cond = select i1 %cmp, i32 5, i32 2
  %cmp1 = icmp eq i32 %y, -1
  %cond2 = select i1 %cmp1, i32 3, i32 4
  %and = and i32 %cond2, %cond
  ret i32 %and
}

define i32 @t18(i32 %x, i32 %y) #0 {
entry:
; ARM-LABEL: t18:
; ARM: and r0, {{r[0-9]+}}, {{r[0-9]+}}

; T2-LABEL: t18:
; T2: and{{s|.w}} r0, {{r[0-9]+}}
  %cmp = icmp ne i32 %x, 0
  %cond = select i1 %cmp, i32 5, i32 2
  %cmp1 = icmp ne i32 %x, -1
  %cond2 = select i1 %cmp1, i32 3, i32 4
  %and = and i32 %cond2, %cond
  ret i32 %and
}

define i32 @t19(i32 %x, i32 %y) #0 {
entry:
; ARM-LABEL: t19:
; ARM: orr r0, {{r[0-9]+}}, {{r[0-9]+}}

; T2-LABEL: t19:
; T2: orrs r0, {{r[0-9]+}}
  %cmp = icmp ne i32 %x, 0
  %cond = select i1 %cmp, i32 5, i32 2
  %cmp1 = icmp ne i32 %y, 0
  %cond2 = select i1 %cmp1, i32 3, i32 4
  %or = or i32 %cond2, %cond
  ret i32 %or
}

define i32 @t20(i32 %x, i32 %y) #0 {
entry:
; ARM-LABEL: t20:
; ARM: orr r0, {{r[0-9]+}}, {{r[0-9]+}}

; T2-LABEL: t20:
; T2: orrs r0, {{r[0-9]+}}
  %cmp = icmp ne i32 %x, -1
  %cond = select i1 %cmp, i32 5, i32 2
  %cmp1 = icmp ne i32 %y, -1
  %cond2 = select i1 %cmp1, i32 3, i32 4
  %or = or i32 %cond2, %cond
  ret i32 %or
}

define  <2 x i32> @t21(<2 x i32> %lhs, <2 x i32> %rhs) {
; CHECK-LABEL: t21:
; CHECK-NOT: eor
; CHECK: mvn
; CHECK-NOT: eor
  %tst = icmp eq <2 x i32> %lhs, %rhs
  %ntst = xor <2 x i1> %tst, <i1 1 , i1 undef>
  %btst = sext <2 x i1> %ntst to <2 x i32>
  ret <2 x i32> %btst
}
