; RUN: llc <%s -mtriple=arm-none-eabi 2>&1 | FileCheck %s -check-prefix=CHECK

; RUN: llc <%s -mtriple=arm-none-eabi -relocation-model=rwpi 2>&1 \
; RUN:   | FileCheck %s -check-prefix=RWPI

; RUN: llc <%s -mtriple=arm-none-eabi --frame-pointer=all 2>&1 \
; RUN:   | FileCheck %s -check-prefix=NO_FP_ELIM

; RUN: llc <%s -mtriple=armv6-apple-ios2 2>&1 | FileCheck %s -check-prefix=IOS2
; RUN: llc <%s -mtriple=armv6k-apple-ios2 2>&1 | FileCheck %s -check-prefix=IOS2
; RUN: llc <%s -mtriple=armv6k-apple-ios3 2>&1 | FileCheck %s -check-prefix=IOS3
; RUN: llc <%s -mtriple=armv7-apple-ios2 2>&1 | FileCheck %s -check-prefix=IOS3

; CHECK: warning: inline asm clobber list contains reserved registers: SP, PC
; CHECK: warning: inline asm clobber list contains reserved registers: R11
; RWPI: warning: inline asm clobber list contains reserved registers: R9, SP, PC
; RWPI: warning: inline asm clobber list contains reserved registers: R11
; NO_FP_ELIM: warning: inline asm clobber list contains reserved registers: R11, SP, PC
; NO_FP_ELIM: warning: inline asm clobber list contains reserved registers: R11
; IOS2: warning: inline asm clobber list contains reserved registers: R9, SP, PC
; IOS3: warning: inline asm clobber list contains reserved registers: SP, PC

define void @foo() nounwind {
  call void asm sideeffect "mov r7, #1",
    "~{r9},~{r11},~{r12},~{lr},~{sp},~{pc},~{r10}"()
  ret void
}

define i32 @bar(i32 %i) {
  %vla = alloca i32, i32 %i, align 4
  tail call void asm sideeffect "mov r7, #1", "~{r11}"()
  %1 = load volatile i32, ptr %vla, align 4
  ret i32 %1
}

; r14 is an alias for lr.
define void @clobber_r14() nounwind {
; CHECK-LABEL: clobber_r14:
; CHECK:       .save {r11, lr}
; CHECK:       push {r11, lr}
; CHECK-NEXT:  @APP
; CHECK-NEXT:  @NO_APP
; CHECK-NEXT:  pop {r11, lr}
  tail call void asm sideeffect "", "~{r14}"()
  ret void
}

; r14 is an alias for lr.
define i32 @read_r14() nounwind {
start:
; CHECK-LABEL: read_r14:
; CHECK:       push {r11, lr}
; CHECK-NEXT:  @APP
; CHECK-NEXT:  @NO_APP
; CHECK-NEXT:  mov r0, lr
; CHECK-NEXT:  pop {r11, lr}
  %1 = tail call i32 asm sideeffect alignstack "", "=&{r14},~{cc},~{memory}"()
  ret i32 %1
}
