; RUN: llc -relocation-model=static    -mtriple=armv7a--none-eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=ARM_RO_ABS --check-prefix=ARM_RW_ABS
; RUN: llc -relocation-model=ropi      -mtriple=armv7a--none-eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=ARM_RO_PC  --check-prefix=ARM_RW_ABS
; RUN: llc -relocation-model=rwpi      -mtriple=armv7a--none-eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=ARM_RO_ABS --check-prefix=ARM_RW_SB
; RUN: llc -relocation-model=ropi-rwpi -mtriple=armv7a--none-eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=ARM_RO_PC  --check-prefix=ARM_RW_SB

; RUN: llc -relocation-model=static    -mtriple=thumbv7m--none-eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB2_RO_ABS --check-prefix=THUMB2_RW_ABS
; RUN: llc -relocation-model=ropi      -mtriple=thumbv7m--none-eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB2_RO_PC  --check-prefix=THUMB2_RW_ABS
; RUN: llc -relocation-model=rwpi      -mtriple=thumbv7m--none-eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB2_RO_ABS  --check-prefix=THUMB2_RW_SB
; RUN: llc -relocation-model=ropi-rwpi -mtriple=thumbv7m--none-eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB2_RO_PC  --check-prefix=THUMB2_RW_SB

; RUN: llc -relocation-model=static    -mtriple=thumbv6m--none-eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB1_RO_ABS --check-prefix=THUMB1_RW_ABS
; RUN: llc -relocation-model=ropi      -mtriple=thumbv6m--none-eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB1_RO_PC  --check-prefix=THUMB1_RW_ABS
; RUN: llc -relocation-model=rwpi      -mtriple=thumbv6m--none-eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB1_RO_ABS --check-prefix=THUMB1_RW_SB
; RUN: llc -relocation-model=ropi-rwpi -mtriple=thumbv6m--none-eabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB1_RO_PC  --check-prefix=THUMB1_RW_SB

; RUN: llc -relocation-model=rwpi      -mtriple=armv7a--none-eabi   -mattr=no-movt < %s | FileCheck %s --check-prefix=CHECK --check-prefix=NO_MOVT_ARM_RO_ABS --check-prefix=NO_MOVT_ARM_RW_SB
; RUN: llc -relocation-model=ropi-rwpi -mtriple=armv7a--none-eabi   -mattr=no-movt < %s | FileCheck %s --check-prefix=CHECK --check-prefix=NO_MOVT_ARM_RO_PC  --check-prefix=NO_MOVT_ARM_RW_SB

; RUN: llc -relocation-model=rwpi      -mtriple=thumbv7m--none-eabi -mattr=no-movt < %s | FileCheck %s --check-prefix=CHECK --check-prefix=NO_MOVT_THUMB2_RO_ABS  --check-prefix=NO_MOVT_THUMB2_RW_SB
; RUN: llc -relocation-model=ropi-rwpi -mtriple=thumbv7m--none-eabi -mattr=no-movt < %s | FileCheck %s --check-prefix=CHECK --check-prefix=NO_MOVT_THUMB2_RO_PC  --check-prefix=NO_MOVT_THUMB2_RW_SB

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

@a = external global i32, align 4
@b = external constant i32, align 4

define i32 @read() {
entry:
  %0 = load i32, ptr @a, align 4
  ret i32 %0
; CHECK-LABEL: read:

; ARM_RW_ABS: movw    r[[REG:[0-9]]], :lower16:a
; ARM_RW_ABS: movt    r[[REG]], :upper16:a
; ARM_RW_ABS: ldr     r0, [r[[REG]]]

; ARM_RW_SB: movw    r[[REG:[0-9]]], :lower16:a(sbrel)
; ARM_RW_SB: movt    r[[REG]], :upper16:a(sbrel)
; ARM_RW_SB: ldr     r0, [r9, r[[REG]]]

; NO_MOVT_ARM_RW_SB: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; NO_MOVT_ARM_RW_SB: ldr     r0, [r9, r[[REG]]]

; THUMB2_RW_ABS: movw    r[[REG:[0-9]]], :lower16:a
; THUMB2_RW_ABS: movt    r[[REG]], :upper16:a
; THUMB2_RW_ABS: ldr     r0, [r[[REG]]]

; THUMB2_RW_SB: movw    r[[REG:[0-9]]], :lower16:a(sbrel)
; THUMB2_RW_SB: movt    r[[REG]], :upper16:a(sbrel)
; THUMB2_RW_SB: ldr.w   r0, [r9, r[[REG]]]

; NO_MOVT_THUMB2_RW_SB: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; NO_MOVT_THUMB2_RW_SB: ldr.w   r0, [r9, r[[REG]]]

; THUMB1_RW_ABS: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; THUMB1_RW_ABS: ldr     r0, [r[[REG]]]

; THUMB1_RW_SB: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; THUMB1_RW_SB: mov     r[[REG_SB:[0-9]+]], r9
; THUMB1_RW_SB: ldr     r0, [r[[REG_SB]], r[[REG]]]

; CHECK: {{(bx lr|pop)}}

; NO_MOVT_ARM_RW_SB: [[LCPI]]
; NO_MOVT_ARM_RW_SB: .long   a(sbrel)

; NO_MOVT_THUMB2_RW_SB: [[LCPI]]
; NO_MOVT_THUMB2_RW_SB: .long   a(sbrel)

; THUMB1_RW_ABS: [[LCPI]]
; THUMB1_RW_ABS-NEXT: .long a

; THUMB1_RW_SB: [[LCPI]]
; THUMB1_RW_SB: .long   a(sbrel)
}

define void @write(i32 %v)  {
entry:
  store i32 %v, ptr @a, align 4
  ret void
; CHECK-LABEL: write:

; ARM_RW_ABS: movw    r[[REG:[0-9]]], :lower16:a
; ARM_RW_ABS: movt    r[[REG]], :upper16:a
; ARM_RW_ABS: str     r0, [r[[REG:[0-9]]]]

; ARM_RW_SB: movw    r[[REG:[0-9]]], :lower16:a
; ARM_RW_SB: movt    r[[REG]], :upper16:a
; ARM_RW_SB: str     r0, [r9, r[[REG:[0-9]]]]

; NO_MOVT_ARM_RW_SB: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; NO_MOVT_ARM_RW_SB: str     r0, [r9, r[[REG]]]

; THUMB2_RW_ABS: movw    r[[REG:[0-9]]], :lower16:a
; THUMB2_RW_ABS: movt    r[[REG]], :upper16:a
; THUMB2_RW_ABS: str     r0, [r[[REG]]]

; THUMB2_RW_SB: movw    r[[REG:[0-9]]], :lower16:a(sbrel)
; THUMB2_RW_SB: movt    r[[REG]], :upper16:a(sbrel)
; THUMB2_RW_SB: str.w   r0, [r9, r[[REG]]]

; NO_MOVT_THUMB2_RW_SB: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; NO_MOVT_THUMB2_RW_SB: str.w   r0, [r9, r[[REG]]]

; THUMB1_RW_ABS: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; THUMB1_RW_ABS: str     r0, [r[[REG]]]

; THUMB1_RW_SB: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; THUMB1_RW_SB: mov     r[[REG_SB:[0-9]+]], r9
; THUMB1_RW_SB: str     r0, [r[[REG_SB]], r[[REG]]]

; CHECK: {{(bx lr|pop)}}

; NO_MOVT_ARM_RW_SB: [[LCPI]]
; NO_MOVT_ARM_RW_SB: .long   a(sbrel)

; NO_MOVT_THUMB2_RW_SB: [[LCPI]]
; NO_MOVT_THUMB2_RW_SB: .long   a(sbrel)

; THUMB1_RW_ABS: [[LCPI]]
; THUMB1_RW_ABS-NEXT: .long a

; THUMB1_RW_SB: [[LCPI]]
; THUMB1_RW_SB: .long   a(sbrel)
}

define i32 @read_const()  {
entry:
  %0 = load i32, ptr @b, align 4
  ret i32 %0
; CHECK-LABEL: read_const:

; ARM_RO_ABS: movw    r[[reg:[0-9]]], :lower16:b
; ARM_RO_ABS: movt    r[[reg]], :upper16:b
; ARM_RO_ABS: ldr     r0, [r[[reg]]]

; NO_MOVT_ARM_RO_ABS: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; NO_MOVT_ARM_RO_ABS: ldr     r0, [r[[REG]]]

; ARM_RO_PC: movw    r[[REG:[0-9]]], :lower16:(b-([[LPC:.LPC[0-9]+_[0-9]+]]+8))
; ARM_RO_PC: movt    r[[REG]], :upper16:(b-([[LPC]]+8))
; ARM_RO_PC: [[LPC]]:
; ARM_RO_PC-NEXT: ldr     r0, [pc, r[[REG]]]

; NO_MOVT_ARM_RO_PC: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; NO_MOVT_ARM_RO_PC: [[LPC:.LPC[0-9]+_[0-9]+]]:
; NO_MOVT_ARM_RO_PC: ldr     r0, [pc, r[[REG]]]

; THUMB2_RO_ABS: movw    r[[REG:[0-9]]], :lower16:b
; THUMB2_RO_ABS: movt    r[[REG]], :upper16:b
; THUMB2_RO_ABS: ldr     r0, [r[[REG]]]

; NO_MOVT_THUMB2_RO_ABS: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; NO_MOVT_THUMB2_RO_ABS: ldr     r0, [r[[REG]]]

; THUMB2_RO_PC: movw    r[[REG:[0-9]]], :lower16:(b-([[LPC:.LPC[0-9]+_[0-9]+]]+4))
; THUMB2_RO_PC: movt    r[[REG]], :upper16:(b-([[LPC]]+4))
; THUMB2_RO_PC: [[LPC]]:
; THUMB2_RO_PC-NEXT: add     r[[REG]], pc
; THUMB2_RO_PC: ldr     r0, [r[[REG]]]

; NO_MOVT_THUMB2_RO_PC: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; NO_MOVT_THUMB2_RO_PC: [[LPC:.LPC[0-9]+_[0-9]+]]:
; NO_MOVT_THUMB2_RO_PC-NEXT: add     r[[REG]], pc
; NO_MOVT_THUMB2_RO_PC: ldr     r0, [r[[REG]]]


; THUMB1_RO_ABS: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; THUMB1_RO_ABS: ldr     r0, [r[[REG]]]

; THUMB1_RO_PC: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; THUMB1_RO_PC: [[LPC:.LPC[0-9]+_[0-9]+]]:
; THUMB1_RO_PC-NEXT: add     r[[REG]], pc
; THUMB1_RO_PC: ldr     r0, [r[[REG]]]

; CHECK: {{(bx lr|pop)}}

; NO_MOVT_ARM_RO_ABS: [[LCPI]]
; NO_MOVT_ARM_RO_ABS-NEXT: .long b

; NO_MOVT_THUMB2_RO_ABS: [[LCPI]]
; NO_MOVT_THUMB2_RO_ABS-NEXT: .long b

; THUMB1_RO_ABS: [[LCPI]]
; THUMB1_RO_ABS-NEXT: .long b

; NO_MOVT_ARM_RO_PC: [[LCPI]]
; NO_MOVT_ARM_RO_PC-NEXT: .long b-([[LPC]]+8)

; NO_MOVT_THUMB2_RO_PC: [[LCPI]]
; NO_MOVT_THUMB2_RO_PC-NEXT: .long b-([[LPC]]+4)

; THUMB1_RO_PC: [[LCPI]]
; THUMB1_RO_PC-NEXT: .long b-([[LPC]]+4)
}

define ptr @take_addr()  {
entry:
  ret ptr @a
; CHECK-LABEL: take_addr:

; ARM_RW_ABS: movw    r[[REG:[0-9]]], :lower16:a
; ARM_RW_ABS: movt    r[[REG]], :upper16:a

; ARM_RW_SB: movw    r[[REG:[0-9]]], :lower16:a(sbrel)
; ARM_RW_SB: movt    r[[REG]], :upper16:a(sbrel)
; ARM_RW_SB: add     r0, r9, r[[REG]]

; NO_MOVT_ARM_RW_SB: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; NO_MOVT_ARM_RW_SB: add     r0, r9, r[[REG]]

; THUMB2_RW_ABS: movw    r[[REG:[0-9]]], :lower16:a
; THUMB2_RW_ABS: movt    r[[REG]], :upper16:a

; THUMB2_RW_SB: movw    r[[REG:[0-9]]], :lower16:a(sbrel)
; THUMB2_RW_SB: movt    r[[REG]], :upper16:a(sbrel)
; THUMB2_RW_SB: add     r0, r9

; NO_MOVT_THUMB2_RW_SB: ldr     r0, [[LCPI:.LCPI[0-9]+_[0-9]+]]
; NO_MOVT_THUMB2_RW_SB: add     r0, r9

; THUMB1_RW_ABS: ldr     r0, [[LCPI:.LCPI[0-9]+_[0-9]+]]

; THUMB1_RW_SB: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; THUMB1_RW_SB: mov     r[[REG_SB:[0-9]+]], r9
; THUMB1_RW_SB: adds    r[[REG]], r[[REG_SB]], r[[REG]]

; CHECK: {{(bx lr|pop)}}

; NO_MOVT_ARM_RW_SB: [[LCPI]]
; NO_MOVT_ARM_RW_SB: .long   a(sbrel)

; NO_MOVT_THUMB2_RW_SB: [[LCPI]]
; NO_MOVT_THUMB2_RW_SB: .long   a(sbrel)

; THUMB1_RW_ABS: [[LCPI]]
; THUMB1_RW_ABS-NEXT: .long a

; THUMB1_RW_SB: [[LCPI]]
; THUMB1_RW_SB: .long   a(sbrel)
}

define ptr @take_addr_const()  {
entry:
  ret ptr @b
; CHECK-LABEL: take_addr_const:

; ARM_RO_ABS: movw    r[[REG:[0-9]]], :lower16:b
; ARM_RO_ABS: movt    r[[REG]], :upper16:b

; NO_MOVT_ARM_RO_ABS: ldr     r0, [[LCPI:.LCPI[0-9]+_[0-9]+]]

; ARM_RO_PC: movw    r[[REG:[0-9]]], :lower16:(b-([[LPC:.LPC[0-9]+_[0-9]+]]+8))
; ARM_RO_PC: movt    r[[REG]], :upper16:(b-([[LPC]]+8))
; ARM_RO_PC: [[LPC]]:
; ARM_RO_PC-NEXT: add     r0, pc, r[[REG:[0-9]]]

; NO_MOVT_ARM_RO_PC: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; NO_MOVT_ARM_RO_PC: [[LPC:.LPC[0-9]+_[0-9]+]]:
; NO_MOVT_ARM_RO_PC-NEXT: add     r0, pc, r[[REG]]

; THUMB2_RO_ABS: movw    r[[REG:[0-9]]], :lower16:b
; THUMB2_RO_ABS: movt    r[[REG]], :upper16:b

; NO_MOVT_THUMB2_RO_ABS: ldr     r0, [[LCPI:.LCPI[0-9]+_[0-9]+]]

; THUMB2_RO_PC: movw    r0, :lower16:(b-([[LPC:.LPC[0-9]+_[0-9]+]]+4))
; THUMB2_RO_PC: movt    r0, :upper16:(b-([[LPC]]+4))
; THUMB2_RO_PC: [[LPC]]:
; THUMB2_RO_PC-NEXT: add     r0, pc

; NO_MOVT_THUMB2_RO_PC: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; NO_MOVT_THUMB2_RO_PC: [[LPC:.LPC[0-9]+_[0-9]+]]:
; NO_MOVT_THUMB2_RO_PC-NEXT: add     r[[REG]], pc

; THUMB1_RO_ABS: ldr     r0, [[LCPI:.LCPI[0-9]+_[0-9]+]]

; THUMB1_RO_PC: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; THUMB1_RO_PC: [[LPC:.LPC[0-9]+_[0-9]+]]:
; THUMB1_RO_PC-NEXT: add     r[[REG]], pc

; CHECK: {{(bx lr|pop)}}

; NO_MOVT_ARM_RO_ABS: [[LCPI]]
; NO_MOVT_ARM_RO_ABS-NEXT: .long b

; NO_MOVT_THUMB2_RO_ABS: [[LCPI]]
; NO_MOVT_THUMB2_RO_ABS-NEXT: .long b

; THUMB1_RO_ABS: [[LCPI]]
; THUMB1_RO_ABS-NEXT: .long b

; NO_MOVT_ARM_RO_PC: [[LCPI]]
; NO_MOVT_ARM_RO_PC-NEXT: .long b-([[LPC]]+8)

; NO_MOVT_THUMB2_RO_PC: [[LCPI]]
; NO_MOVT_THUMB2_RO_PC-NEXT: .long b-([[LPC]]+4)

; THUMB1_RO_PC: [[LCPI]]
; THUMB1_RO_PC-NEXT: .long b-([[LPC]]+4)
}

define ptr @take_addr_func()  {
entry:
  ret ptr @take_addr_func
; CHECK-LABEL: take_addr_func:

; ARM_RO_ABS: movw    r[[REG:[0-9]]], :lower16:take_addr_func
; ARM_RO_ABS: movt    r[[REG]], :upper16:take_addr_func

; NO_MOVT_ARM_RO_ABS: ldr     r0, [[LCPI:.LCPI[0-9]+_[0-9]+]]

; ARM_RO_PC: movw    r[[REG:[0-9]]], :lower16:(take_addr_func-([[LPC:.LPC[0-9]+_[0-9]+]]+8))
; ARM_RO_PC: movt    r[[REG]], :upper16:(take_addr_func-([[LPC]]+8))
; ARM_RO_PC: [[LPC]]:
; ARM_RO_PC-NEXT: add     r0, pc, r[[REG:[0-9]]]

; NO_MOVT_ARM_RO_PC: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; NO_MOVT_ARM_RO_PC: [[LPC:.LPC[0-9]+_[0-9]+]]:
; NO_MOVT_ARM_RO_PC-NEXT: add     r0, pc, r[[REG]]

; THUMB2_RO_ABS: movw    r[[REG:[0-9]]], :lower16:take_addr_func
; THUMB2_RO_ABS: movt    r[[REG]], :upper16:take_addr_func

; NO_MOVT_THUMB2_RO_ABS: ldr     r0, [[LCPI:.LCPI[0-9]+_[0-9]+]]

; THUMB2_RO_PC: movw    r0, :lower16:(take_addr_func-([[LPC:.LPC[0-9]+_[0-9]+]]+4))
; THUMB2_RO_PC: movt    r0, :upper16:(take_addr_func-([[LPC]]+4))
; THUMB2_RO_PC: [[LPC]]:
; THUMB2_RO_PC-NEXT: add     r0, pc

; NO_MOVT_THUMB2_RO_PC: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; NO_MOVT_THUMB2_RO_PC: [[LPC:.LPC[0-9]+_[0-9]+]]:
; NO_MOVT_THUMB2_RO_PC-NEXT: add     r[[REG]], pc

; THUMB1_RO_ABS: ldr     r0, [[LCPI:.LCPI[0-9]+_[0-9]+]]

; THUMB1_RO_PC: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; THUMB1_RO_PC: [[LPC:.LPC[0-9]+_[0-9]+]]:
; THUMB1_RO_PC-NEXT: add     r[[REG]], pc

; CHECK: {{(bx lr|pop)}}

; NO_MOVT_ARM_RO_ABS: [[LCPI]]
; NO_MOVT_ARM_RO_ABS-NEXT: .long take_addr_func

; NO_MOVT_THUMB2_RO_ABS: [[LCPI]]
; NO_MOVT_THUMB2_RO_ABS-NEXT: .long take_addr_func

; THUMB1_RO_ABS: [[LCPI]]
; THUMB1_RO_ABS-NEXT: .long take_addr_func

; NO_MOVT_ARM_RO_PC: [[LCPI]]
; NO_MOVT_ARM_RO_PC-NEXT: .long take_addr_func-([[LPC]]+8)

; NO_MOVT_THUMB2_RO_PC: [[LCPI]]
; NO_MOVT_THUMB2_RO_PC-NEXT: .long take_addr_func-([[LPC]]+4)

; THUMB1_RO_PC: [[LCPI]]
; THUMB1_RO_PC-NEXT: .long take_addr_func-([[LPC]]+4)
}

define ptr @block_addr() {
entry:
  br label %lab1

lab1:
  ret ptr blockaddress(@block_addr, %lab1)

; CHECK-LABEL: block_addr:

; ARM_RO_ABS: [[LTMP:.Ltmp[0-9]+]]:
; ARM_RO_ABS: ldr     r0, [[LCPI:.LCPI[0-9]+_[0-9]+]]

; ARM_RO_PC: [[LTMP:.Ltmp[0-9]+]]:
; ARM_RO_PC: ldr     r[[REG:[0-9]]], [[LCPI:.LCPI[0-9]+_[0-9]+]]
; ARM_RO_PC: [[LPC:.LPC[0-9]+_[0-9]+]]:
; ARM_RO_PC: add     r0, pc, r[[REG]]

; THUMB2_RO_ABS: [[LTMP:.Ltmp[0-9]+]]:
; THUMB2_RO_ABS: ldr     r0, [[LCPI:.LCPI[0-9]+_[0-9]+]]

; THUMB2_RO_PC: [[LTMP:.Ltmp[0-9]+]]:
; THUMB2_RO_PC: ldr     r0, [[LCPI:.LCPI[0-9]+_[0-9]+]]
; THUMB2_RO_PC: [[LPC:.LPC[0-9]+_[0-9]+]]:
; THUMB2_RO_PC: add     r0, pc

; THUMB1_RO_ABS: [[LTMP:.Ltmp[0-9]+]]:
; THUMB1_RO_ABS: ldr     r0, [[LCPI:.LCPI[0-9]+_[0-9]+]]

; THUMB1_RO_PC: [[LTMP:.Ltmp[0-9]+]]:
; THUMB1_RO_PC: ldr     r0, [[LCPI:.LCPI[0-9]+_[0-9]+]]
; THUMB1_RO_PC: [[LPC:.LPC[0-9]+_[0-9]+]]:
; THUMB1_RO_PC: add     r0, pc

; CHECK: bx lr

; ARM_RO_ABS: [[LCPI]]
; ARM_RO_ABS-NEXT: .long   [[LTMP]]

; ARM_RO_PC: [[LCPI]]
; ARM_RO_PC-NEXT: .long   [[LTMP]]-([[LPC]]+8)

; THUMB2_RO_ABS: [[LCPI]]
; THUMB2_RO_ABS-NEXT: .long   [[LTMP]]

; THUMB2_RO_PC: [[LCPI]]
; THUMB2_RO_PC-NEXT: .long   [[LTMP]]-([[LPC]]+4)

; THUMB1_RO_ABS: [[LCPI]]
; THUMB1_RO_ABS-NEXT: .long   [[LTMP]]

; THUMB1_RO_PC: [[LCPI]]
; THUMB1_RO_PC-NEXT: .long   [[LTMP]]-([[LPC]]+4)
}
