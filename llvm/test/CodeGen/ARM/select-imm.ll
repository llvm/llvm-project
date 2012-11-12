; RUN: llc < %s -march=arm                  | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -march=arm -mattr=+thumb2   | FileCheck %s --check-prefix=ARMT2
; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s --check-prefix=THUMB2

define i32 @t1(i32 %c) nounwind readnone {
entry:
; ARM: t1:
; ARM: mov [[R1:r[0-9]+]], #101
; ARM: orr [[R1b:r[0-9]+]], [[R1]], #256
; ARM: movgt r0, #123

; ARMT2: t1:
; ARMT2: movw r0, #357
; ARMT2: movgt r0, #123

; THUMB2: t1:
; THUMB2: movw r0, #357
; THUMB2: movgt r0, #123

  %0 = icmp sgt i32 %c, 1
  %1 = select i1 %0, i32 123, i32 357
  ret i32 %1
}

define i32 @t2(i32 %c) nounwind readnone {
entry:
; ARM: t2:
; ARM: mov r0, #123
; ARM: movgt r0, #101
; ARM: orrgt r0, r0, #256

; ARMT2: t2:
; ARMT2: mov r0, #123
; ARMT2: movwgt r0, #357

; THUMB2: t2:
; THUMB2: mov{{(s|\.w)}} r0, #123
; THUMB2: movwgt r0, #357

  %0 = icmp sgt i32 %c, 1
  %1 = select i1 %0, i32 357, i32 123
  ret i32 %1
}

define i32 @t3(i32 %a) nounwind readnone {
entry:
; ARM: t3:
; ARM: mov r0, #0
; ARM: moveq r0, #1

; ARMT2: t3:
; ARMT2: mov r0, #0
; ARMT2: moveq r0, #1

; THUMB2: t3:
; THUMB2: mov{{(s|\.w)}} r0, #0
; THUMB2: moveq r0, #1
  %0 = icmp eq i32 %a, 160
  %1 = zext i1 %0 to i32
  ret i32 %1
}

define i32 @t4(i32 %a, i32 %b, i32 %x) nounwind {
entry:
; ARM: t4:
; ARM: ldr
; ARM: mov{{lt|ge}}

; ARMT2: t4:
; ARMT2: movwlt [[R0:r[0-9]+]], #65365
; ARMT2: movtlt [[R0]], #65365

; THUMB2: t4:
; THUMB2: mvnlt [[R0:r[0-9]+]], #11141290
  %0 = icmp slt i32 %a, %b
  %1 = select i1 %0, i32 4283826005, i32 %x
  ret i32 %1
}

; rdar://9758317
define i32 @t5(i32 %a) nounwind {
entry:
; ARM: t5:
; ARM-NOT: mov
; ARM: cmp r0, #1
; ARM-NOT: mov
; ARM: movne r0, #0

; THUMB2: t5:
; THUMB2-NOT: mov
; THUMB2: cmp r0, #1
; THUMB2: it ne
; THUMB2: movne r0, #0
  %cmp = icmp eq i32 %a, 1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @t6(i32 %a) nounwind {
entry:
; ARM: t6:
; ARM-NOT: mov
; ARM: cmp r0, #0
; ARM: movne r0, #1

; THUMB2: t6:
; THUMB2-NOT: mov
; THUMB2: cmp r0, #0
; THUMB2: it ne
; THUMB2: movne r0, #1
  %tobool = icmp ne i32 %a, 0
  %lnot.ext = zext i1 %tobool to i32
  ret i32 %lnot.ext
}
