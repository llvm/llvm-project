; RUN: llc <%s -mtriple=arm-none-eabi 2>&1 | FileCheck %s -check-prefix=ARM_NONE

; RUN: llc <%s -mtriple=arm-none-eabi -relocation-model=rwpi 2>&1 \
; RUN:   | FileCheck %s -check-prefix=RWPI

; RUN: llc <%s -mtriple=arm-none-eabi --frame-pointer=all 2>&1 \
; RUN:   | FileCheck %s -check-prefix=NO_FP_ELIM

; RUN: llc <%s -mtriple=armv6-apple-ios2 2>&1 | FileCheck %s -check-prefix=IOS2
; RUN: llc <%s -mtriple=armv6k-apple-ios2 2>&1 | FileCheck %s -check-prefix=IOS2
; RUN: llc <%s -mtriple=armv6k-apple-ios3 2>&1 | FileCheck %s -check-prefix=IOS3
; RUN: llc <%s -mtriple=armv7-apple-ios2 2>&1 | FileCheck %s -check-prefix=IOS3

; RUN: llc <%s -mtriple=thumbv6m-none-eabi 2>&1 | FileCheck %s -check-prefix=THUMB
; RUN: llc <%s -mtriple=thumbv8m.base-none-eabi 2>&1 | FileCheck %s -check-prefix=THUMB
; RUN: llc <%s -mtriple=thumbv7m-none-eabi 2>&1 | FileCheck %s -check-prefix=THUMB

; ARM_NONE: warning: inline asm clobber list contains reserved registers: SP, PC
; ARM_NONE: warning: inline asm clobber list contains reserved registers: R11
; RWPI: warning: inline asm clobber list contains reserved registers: R9, SP, PC
; RWPI: warning: inline asm clobber list contains reserved registers: R11
; NO_FP_ELIM: warning: inline asm clobber list contains reserved registers: R11, SP, PC
; NO_FP_ELIM: warning: inline asm clobber list contains reserved registers: R11
; IOS2: warning: inline asm clobber list contains reserved registers: R9, SP, PC
; IOS3: warning: inline asm clobber list contains reserved registers: SP, PC
; THUMB: warning: inline asm clobber list contains reserved registers: SP, PC

define void @foo() nounwind {
  call void asm sideeffect "movs r7, #1",
    "~{r9},~{r11},~{r12},~{lr},~{sp},~{pc},~{r10}"()
  ret void
}

define i32 @bar(i32 %i) {
  %vla = alloca i32, i32 %i, align 4
  tail call void asm sideeffect "movs r7, #1", "~{r11}"()
  %1 = load volatile i32, ptr %vla, align 4
  ret i32 %1
}

; r14 is an alias for lr.
define void @clobber_r14() nounwind {
; ARM_NONE-LABEL: clobber_r14:
; ARM_NONE:       .save {r11, lr}
; ARM_NONE:       push {r11, lr}
; ARM_NONE-NEXT:  @APP
; ARM_NONE-NEXT:  @NO_APP
; ARM_NONE-NEXT:  pop {r11, lr}
; THUMB-LABEL: clobber_r14:
; THUMB:       push {r7, lr}
; THUMB-NEXT:  @APP
; THUMB-NEXT:  @NO_APP
; THUMB-NEXT:  pop {r7, pc}
  tail call void asm sideeffect "", "~{r14}"()
  ret void
}

; r14 is an alias for lr.
define i32 @read_r14() nounwind {
start:
; ARM_NONE-LABEL: read_r14:
; ARM_NONE:       push {r11, lr}
; ARM_NONE-NEXT:  @APP
; ARM_NONE-NEXT:  @NO_APP
; ARM_NONE-NEXT:  mov r0, lr
; ARM_NONE-NEXT:  pop {r11, lr}
; THUMB-LABEL: read_r14:
; THUMB:       push {r7, lr}
; THUMB-NEXT:  @APP
; THUMB-NEXT:  @NO_APP
; THUMB-NEXT:  mov r0, lr
; THUMB-NEXT:  pop {r7, pc}
  %1 = tail call i32 asm sideeffect alignstack "", "=&{r14},~{cc},~{memory}"()
  ret i32 %1
}
