; RUN: llc -mtriple armv7a-none-eabi < %s --frame-pointer=all  | FileCheck %s --check-prefixes CHECK,FPALL-ARM
; RUN: llc -mtriple armv6m-none-eabi < %s --frame-pointer=all  | FileCheck %s --check-prefixes CHECK,FPALL-THUMB1
; RUN: llc -mtriple armv7m-none-eabi < %s --frame-pointer=all  | FileCheck %s --check-prefixes CHECK,FPALL-THUMB2
; RUN: llc -mtriple armv7a-none-eabi < %s --frame-pointer=none | FileCheck %s --check-prefixes CHECK,FPNONE
; RUN: llc -mtriple armv6m-none-eabi < %s --frame-pointer=none | FileCheck %s --check-prefixes CHECK,FPNONE
; RUN: llc -mtriple armv7m-none-eabi < %s --frame-pointer=none | FileCheck %s --check-prefixes CHECK,FPNONE

; When the AAPCS frame chain is enabled, check that r11 is either used as a
; frame pointer, which must point to an ABI-compatible frame record, or not
; used at all, so that it continues to point to a valid frame record for the
; calling function.

define i32 @foo(i32 %a) "target-features"="+aapcs-frame-chain" {
; CHECK-LABEL: foo:
; FPALL-ARM: add r11, sp, 
; FPALL-THUMB1: mov r11, sp
; FPALL-THUMB2: add.w r11, sp, 
; FPNONE-NOT: r11
entry:
  tail call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r12},~{lr}"()
  ret i32 %a
}

