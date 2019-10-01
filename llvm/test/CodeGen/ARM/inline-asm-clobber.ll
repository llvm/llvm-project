; RUN: llc <%s -mtriple=arm-none-eabi 2>&1 | FileCheck %s -check-prefix=CHECK

; RUN: llc <%s -mtriple=arm-none-eabi -relocation-model=rwpi 2>&1 \
; RUN:   | FileCheck %s -check-prefix=RWPI

; RUN: llc <%s -mtriple=arm-none-eabi --frame-pointer=all 2>&1 \
; RUN:   | FileCheck %s -check-prefix=NO_FP_ELIM

; CHECK: warning: inline asm clobber list contains reserved registers: SP, PC
; CHECK: warning: inline asm clobber list contains reserved registers: R11
; RWPI: warning: inline asm clobber list contains reserved registers: R9, SP, PC
; RWPI: warning: inline asm clobber list contains reserved registers: R11
; NO_FP_ELIM: warning: inline asm clobber list contains reserved registers: R11, SP, PC
; NO_FP_ELIM: warning: inline asm clobber list contains reserved registers: R11

define void @foo() nounwind {
  call void asm sideeffect "mov r7, #1",
    "~{r9},~{r11},~{r12},~{lr},~{sp},~{pc},~{r10}"()
  ret void
}

define i32 @bar(i32 %i) {
  %vla = alloca i32, i32 %i, align 4
  tail call void asm sideeffect "mov r7, #1", "~{r11}"()
  %1 = load volatile i32, i32* %vla, align 4
  ret i32 %1
}
