; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s --check-prefix=ARM

; RUN: llc -mtriple=arm-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - \
; RUN:  | FileCheck %s --check-prefix=ARMT2

; RUN: llc -mtriple=thumb-eabi -mcpu=cortex-m0 %s -o - \
; RUN:  | FileCheck %s --check-prefix=THUMB1

; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - \
; RUN:  | FileCheck %s --check-prefix=THUMB2

; RUN: llc -mtriple=thumbv8m.base-eabi %s -o - \
; RUN:  | FileCheck %s --check-prefix=V8MBASE

define i32 @t1(i32 %c) nounwind readnone {
entry:
; ARM-LABEL: t1:
; ARM: mov [[R1:r[0-9]+]], #101
; ARM: orr [[R1b:r[0-9]+]], [[R1]], #256
; ARM: movgt {{r[0-1]}}, #123

; ARMT2-LABEL: t1:
; ARMT2: movw [[R:r[0-1]]], #357
; ARMT2: movwgt [[R]], #123

; THUMB1-LABEL: t1:
; THUMB1: mov     r1, r0
; THUMB1: movs    r2, #255
; THUMB1: adds    r2, #102
; THUMB1: movs    r0, #123
; THUMB1: cmp     r1, #1
; THUMB1: bgt

; THUMB2-LABEL: t1:
; THUMB2: movw [[R:r[0-1]]], #357
; THUMB2: movgt [[R]], #123

  %0 = icmp sgt i32 %c, 1
  %1 = select i1 %0, i32 123, i32 357
  ret i32 %1
}

define i32 @t2(i32 %c) nounwind readnone {
entry:
; ARM-LABEL: t2:
; ARM: mov [[R:r[0-9]+]], #101
; ARM: orr [[R]], [[R]], #256
; ARM: movle [[R]], #123

; ARMT2-LABEL: t2:
; ARMT2: mov [[R:r[0-1]]], #123
; ARMT2: movwgt [[R]], #357

; THUMB1-LABEL: t2:
; THUMB1: cmp r{{[0-9]+}}, #1
; THUMB1: bgt

; THUMB2-LABEL: t2:
; THUMB2: mov{{(s|\.w)}} [[R:r[0-1]]], #123
; THUMB2: movwgt [[R]], #357

  %0 = icmp sgt i32 %c, 1
  %1 = select i1 %0, i32 357, i32 123
  ret i32 %1
}

define i32 @t3(i32 %a) nounwind readnone {
entry:
; ARM-LABEL: t3:
; ARM: rsbs r1, r0, #0
; ARM: adc  r0, r0, r1

; ARMT2-LABEL: t3:
; ARMT2: clz r0, r0
; ARMT2: lsr r0, r0, #5

; THUMB1-LABEL: t3:
; THUMB1: movs r1, #0
; THUMB1: subs r1, r1, r0
; THUMB1: adcs r0, r1

; THUMB2-LABEL: t3:
; THUMB2: clz r0, r0
; THUMB2: lsrs r0, r0, #5
  %0 = icmp eq i32 %a, 160
  %1 = zext i1 %0 to i32
  ret i32 %1
}

define i32 @t4(i32 %a, i32 %b, i32 %x) nounwind {
entry:
; ARM-LABEL: t4:
; ARM: ldr
; ARM: mov{{lt|ge}}

; ARMT2-LABEL: t4:
; ARMT2: movwlt [[R0:r[0-9]+]], #65365
; ARMT2: movtlt [[R0]], #65365

; THUMB1-LABEL: t4:
; THUMB1: cmp r{{[0-9]+}}, r{{[0-9]+}}
; THUMB1: b{{lt|ge}}

; THUMB2-LABEL: t4:
; THUMB2: mvnlt [[R0:r[0-9]+]], #11141290
  %0 = icmp slt i32 %a, %b
  %1 = select i1 %0, i32 4283826005, i32 %x
  ret i32 %1
}

; rdar://9758317
define i32 @t5(i32 %a) nounwind {
entry:
; ARM-LABEL: t5:
; ARM-NOT: mov
; ARM: sub  r0, r0, #1
; ARM-NOT: mov
; ARM: rsbs r1, r0, #0
; ARM: adc  r0, r0, r1

; THUMB1-LABEL: t5:
; THUMB1-NOT: bne
; THUMB1: movs r0, #0
; THUMB1: subs r0, r0, r1
; THUMB1: adcs r0, r1

; THUMB2-LABEL: t5:
; THUMB2-NOT: mov
; THUMB2: subs r0, #1
; THUMB2: clz  r0, r0
; THUMB2: lsrs r0, r0, #5

  %cmp = icmp eq i32 %a, 1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @t6(i32 %a) nounwind {
entry:
; ARM-LABEL: t6:
; ARM-NOT: mov
; ARM: cmp r0, #0
; ARM: movne r0, #1

; THUMB1-LABEL: t6:
; THUMB1: cmp r{{[0-9]+}}, #0
; THUMB1: bne

; THUMB2-LABEL: t6:
; THUMB2-NOT: mov
; THUMB2: cmp r0, #0
; THUMB2: it ne
; THUMB2: movne r0, #1
  %tobool = icmp ne i32 %a, 0
  %lnot.ext = zext i1 %tobool to i32
  ret i32 %lnot.ext
}

define i32 @t7(i32 %a, i32 %b) nounwind readnone {
entry:
; ARM-LABEL: t7:
; ARM: subs r0, r0, r1
; ARM: movne   r0, #1
; ARM: lsl     r0, r0, #2

; ARMT2-LABEL: t7:
; ARMT2: subs r0, r0, r1
; ARMT2: movwne r0, #1
; ARMT2: lsl     r0, r0, #2

; THUMB1-LABEL: t7:
; THUMB1: subs r0, r0, r1
; THUMB1: subs r1, r0, #1
; THUMB1: sbcs r0, r1
; THUMB1: lsls r0, r0, #2

; THUMB2-LABEL: t7:
; THUMB2: subs r0, r0, r1
; THUMB2: it ne
; THUMB2: movne r0, #1
; THUMB2: lsls    r0, r0, #2
  %0 = icmp ne i32 %a, %b
  %1 = select i1 %0, i32 4, i32 0
  ret i32 %1
}

define void @t8(i32 %a) {
entry:

; ARM scheduler emits icmp/zext before both calls, so isn't relevant

; ARMT2-LABEL: t8:
; ARMT2: bl t7
; ARMT2: mov r1, r0
; ARMT2: sub r0, r4, #5
; ARMT2: clz r0, r0
; ARMT2: lsr r0, r0, #5

; THUMB1-LABEL: t8:
; THUMB1: bl t7
; THUMB1: mov r1, r0
; THUMB1: subs r2, r4, #5
; THUMB1: movs r0, #0
; THUMB1: subs r0, r0, r2
; THUMB1: adcs r0, r2

; THUMB2-LABEL: t8:
; THUMB2: bl t7
; THUMB2: mov r1, r0
; THUMB2: subs r0, r4, #5
; THUMB2: clz r0, r0
; THUMB2: lsrs r0, r0, #5

  %cmp = icmp eq i32 %a, 5
  %conv = zext i1 %cmp to i32
  %call = tail call i32 @t7(i32 9, i32 %a)
  tail call i32 @t7(i32 %conv, i32 %call)
  ret void
}

define void @t9(i8* %a, i8 %b) {
entry:

; ARM scheduler emits icmp/zext before both calls, so isn't relevant

; ARMT2-LABEL: t9:
; ARMT2: bl f
; ARMT2: uxtb r0, r4
; ARMT2: cmp  r0, r0
; ARMT2: add  r1, r4, #1
; ARMT2: mov  r2, r0
; ARMT2: add  r2, r2, #1
; ARMT2: add  r1, r1, #1
; ARMT2: uxtb r3, r2
; ARMT2: cmp  r3, r0

; THUMB1-LABEL: t9:
; THUMB1: bl f
; THUMB1: sxtb r1, r4
; THUMB1: uxtb r0, r1
; THUMB1: cmp  r0, r0
; THUMB1: adds r1, r1, #1
; THUMB1: mov  r2, r0
; THUMB1: adds r1, r1, #1
; THUMB1: adds r2, r2, #1
; THUMB1: uxtb r3, r2
; THUMB1: cmp  r3, r0

; THUMB2-LABEL: t9:
; THUMB2: bl f
; THUMB2: uxtb r0, r4
; THUMB2: cmp  r0, r0
; THUMB2: adds r1, r4, #1
; THUMB2: mov  r2, r0
; THUMB2: adds r2, #1
; THUMB2: adds r1, #1
; THUMB2: uxtb r3, r2
; THUMB2: cmp  r3, r0

  %0 = load i8, i8* %a
  %conv = sext i8 %0 to i32
  %conv119 = zext i8 %0 to i32
  %conv522 = and i32 %conv, 255
  %cmp723 = icmp eq i32 %conv522, %conv119
  tail call void @f(i1 zeroext %cmp723)
  br i1 %cmp723, label %while.body, label %while.end

while.body:                                       ; preds = %entry, %while.body
  %ref.025 = phi i8 [ %inc9, %while.body ], [ %0, %entry ]
  %in.024 = phi i32 [ %inc, %while.body ], [ %conv, %entry ]
  %inc = add i32 %in.024, 1
  %inc9 = add i8 %ref.025, 1
  %conv1 = zext i8 %inc9 to i32
  %cmp = icmp slt i32 %conv1, %conv119
  %conv5 = and i32 %inc, 255
  br i1 %cmp, label %while.body, label %while.end

while.end:
  ret void
}

declare void @f(i1 zeroext)


define i1 @t10() {
entry:
  %q = alloca i32
  %p = alloca i32
  store i32 -3, i32* %q
  store i32 -8, i32* %p
  %0 = load i32, i32* %q
  %1 = load i32, i32* %p
  %div = sdiv i32 %0, %1
  %mul = mul nsw i32 %div, %1
  %rem = srem i32 %0, %1
  %add = add nsw i32 %mul, %rem
  %cmp = icmp eq i32 %add, %0
  ret i1 %cmp

; ARM-LABEL: t10:
; ARM: rsbs r1, r0, #0
; ARM: adc  r0, r0, r1

; ARMT2-LABEL: t10:
; ARMT2: clz r0, r0
; ARMT2: lsr r0, r0, #5

; THUMB1-LABEL: t10:
; THUMB1: movs r0, #0
; THUMB1: subs r0, r0, r1
; THUMB1: adcs r0, r1

; THUMB2-LABEL: t10:
; THUMB2: clz r0, r0
; THUMB2: lsrs r0, r0, #5

; V8MBASE-LABEL: t10:
; V8MBASE-NOT: movs r0, #0
; V8MBASE: movs r0, #7
}

define i1 @t11() {
entry:
  %bit = alloca i32
  %load = load i32, i32* %bit
  %clear = and i32 %load, -4096
  %set = or i32 %clear, 33
  store i32 %set, i32* %bit
  %load1 = load i32, i32* %bit
  %clear2 = and i32 %load1, -33550337
  %set3 = or i32 %clear2, 40960
  %clear5 = and i32 %set3, 4095
  %rem = srem i32 %clear5, 10
  %clear9 = and i32 %set3, -4096
  %set10 = or i32 %clear9, %rem
  store i32 %set10, i32* %bit
  %clear12 = and i32 %set10, 4095
  %cmp = icmp eq i32 %clear12, 3
  ret i1 %cmp

; ARM-LABEL: t11:
; ARM: rsbs r1, r0, #0
; ARM: adc  r0, r0, r1

; ARMT2-LABEL: t11:
; ARMT2: clz r0, r0
; ARMT2: lsr r0, r0, #5

; THUMB1-LABEL: t11:
; THUMB1-NOT: movs r0, #0
; THUMB1: movs r0, #5

; THUMB2-LABEL: t11:
; THUMB2: clz r0, r0
; THUMB2: lsrs r0, r0, #5

; V8MBASE-LABEL: t11:
; V8MBASE-NOT: movs r0, #0
; V8MBASE: movw	r0, #40960
}
