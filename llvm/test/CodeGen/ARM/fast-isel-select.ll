; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB
; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv8-apple-ios | FileCheck %s --check-prefix=THUMB

define i32 @t1(i1 %c) nounwind readnone {
entry:
; ARM: t1
; ARM: movw r{{[1-9]}}, #10
; ARM: tst r0, #1
; ARM: moveq r{{[1-9]}}, #20
; ARM: mov r0, r{{[1-9]}}
; THUMB: t1
; THUMB: movs r{{[1-9]}}, #10
; THUMB: tst.w r0, #1
; THUMB: it eq
; THUMB: moveq r{{[1-9]}}, #20
; THUMB: mov r0, r{{[1-9]}}
  %0 = select i1 %c, i32 10, i32 20
  ret i32 %0
}

define i32 @t2(i1 %c, i32 %a) nounwind readnone {
entry:
; ARM: t2
; ARM: tst r0, #1
; ARM: moveq r{{[1-9]}}, #20
; ARM: mov r0, r{{[1-9]}}
; THUMB: t2
; THUMB: tst.w r0, #1
; THUMB: it eq
; THUMB: moveq r{{[1-9]}}, #20
; THUMB: mov r0, r{{[1-9]}}
  %0 = select i1 %c, i32 %a, i32 20
  ret i32 %0
}

define i32 @t3(i1 %c, i32 %a, i32 %b) nounwind readnone {
entry:
; ARM: t3
; ARM: tst r0, #1
; ARM: movne r2, r1
; ARM: add r0, r2, r1
; THUMB: t3
; THUMB: tst.w r0, #1
; THUMB: it ne
; THUMB: movne r2, r1
; THUMB: add.w r0, r2, r1
  %0 = select i1 %c, i32 %a, i32 %b
  %1 = add i32 %0, %a
  ret i32 %1
}

define i32 @t4(i1 %c) nounwind readnone {
entry:
; ARM: t4
; ARM: mvn r{{[1-9]}}, #9
; ARM: tst r0, #1
; ARM: mvneq r{{[1-9]}}, #0
; ARM: mov r0, r{{[1-9]}}
; THUMB-LABEL: t4
; THUMB: mvn [[REG:r[1-9]+]], #9
; THUMB: tst.w r0, #1
; THUMB: it eq
; THUMB: mvneq [[REG]], #0
; THUMB: mov r0, [[REG]]
  %0 = select i1 %c, i32 -10, i32 -1
  ret i32 %0
}

define i32 @t5(i1 %c, i32 %a) nounwind readnone {
entry:
; ARM: t5
; ARM: tst r0, #1
; ARM: mvneq r{{[1-9]}}, #1
; ARM: mov r0, r{{[1-9]}}
; THUMB: t5
; THUMB: tst.w r0, #1
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
; ARM: tst r0, #1
; ARM: mvneq r{{[1-9]}}, #978944
; ARM: mov r0, r{{[1-9]}}
; THUMB: t6
; THUMB: tst.w r0, #1
; THUMB: it eq
; THUMB: mvneq r{{[1-9]}}, #978944
; THUMB: mov r0, r{{[1-9]}}
  %0 = select i1 %c, i32 %a, i32 -978945
  ret i32 %0
}
