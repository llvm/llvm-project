; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB

define i32 @t1(i1 %c) nounwind readnone {
entry:
; ARM: t1
; ARM: movw r{{[1-9]}}, #10
; ARM: cmp r0, #0
; ARM: moveq r{{[1-9]}}, #20
; ARM: mov r0, r{{[1-9]}}
; THUMB: t1
; THUMB: movs r{{[1-9]}}, #10
; THUMB: movt r{{[1-9]}}, #0
; THUMB: cmp r0, #0
; THUMB: it eq
; THUMB: moveq r{{[1-9]}}, #20
; THUMB: mov r0, r{{[1-9]}}
  %0 = select i1 %c, i32 10, i32 20
  ret i32 %0
}

define i32 @t2(i1 %c, i32 %a) nounwind readnone {
entry:
; ARM: t2
; ARM: cmp r0, #0
; ARM: moveq r{{[1-9]}}, #20
; ARM: mov r0, r{{[1-9]}}
; THUMB: t2
; THUMB: cmp r0, #0
; THUMB: it eq
; THUMB: moveq r{{[1-9]}}, #20
; THUMB: mov r0, r{{[1-9]}}
  %0 = select i1 %c, i32 %a, i32 20
  ret i32 %0
}

define i32 @t3(i1 %c, i32 %a, i32 %b) nounwind readnone {
entry:
; ARM: t3
; ARM: cmp r0, #0
; ARM: movne r{{[1-9]}}, r{{[1-9]}}
; ARM: mov r0, r{{[1-9]}}
; THUMB: t3
; THUMB: cmp r0, #0
; THUMB: it ne
; THUMB: movne r{{[1-9]}}, r{{[1-9]}}
; THUMB: mov r0, r{{[1-9]}}
  %0 = select i1 %c, i32 %a, i32 %b
  ret i32 %0
}

define i32 @t4(i1 %c) nounwind readnone {
entry:
; ARM: t4
; ARM: mvn r{{[1-9]}}, #9
; ARM: cmp r0, #0
; ARM: mvneq r{{[1-9]}}, #0
; ARM: mov r0, r{{[1-9]}}
; THUMB: t4
; THUMB: movw r{{[1-9]}}, #65526
; THUMB: movt r{{[1-9]}}, #65535
; THUMB: cmp r0, #0
; THUMB: it eq
; THUMB: mvneq r{{[1-9]}}, #0
; THUMB: mov r0, r{{[1-9]}}
  %0 = select i1 %c, i32 -10, i32 -1
  ret i32 %0
}

define i32 @t5(i1 %c, i32 %a) nounwind readnone {
entry:
; ARM: t5
; ARM: cmp r0, #0
; ARM: mvneq r{{[1-9]}}, #1
; ARM: mov r0, r{{[1-9]}}
; THUMB: t5
; THUMB: cmp r0, #0
; THUMB: it eq
; THUMB: mvneq r{{[1-9]}}, #1
; THUMB: mov r0, r{{[1-9]}}
  %0 = select i1 %c, i32 %a, i32 -2
  ret i32 %0
}

; Check one large negative immediates.
define i32 @t6(i1 %c, i32 %a) nounwind readnone {
entry:
; ARM: t6
; ARM: cmp r0, #0
; ARM: mvneq r{{[1-9]}}, #978944
; ARM: mov r0, r{{[1-9]}}
; THUMB: t6
; THUMB: cmp r0, #0
; THUMB: it eq
; THUMB: mvneq r{{[1-9]}}, #978944
; THUMB: mov r0, r{{[1-9]}}
  %0 = select i1 %c, i32 %a, i32 -978945
  ret i32 %0
}
