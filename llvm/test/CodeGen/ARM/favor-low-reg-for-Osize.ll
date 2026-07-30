; REQUIRES: asserts
; RUN: llc -debug-only=regalloc < %s 2>%t | FileCheck %s --check-prefix=CHECK
; RUN: FileCheck %s < %t --check-prefix=DEBUG

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n8:16:32-S64"
target triple = "thumbv7m--linux-gnueabi"


; DEBUG:         AllocationOrder(GPR) = [ $r0 $r1 $r2 $r3 $r4 $r5 $r6 $r7 $r12 $lr $r8 $r9 $r10 $r11 ]

define i32 @test_minsize(i32 %x) optsize minsize {
; CHECK-LABEL: test_minsize:
entry:
; CHECK: mov     r4, r0
  tail call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3}"()
; CHECK: mov     r0, r4
  ret i32 %x
}

; DEBUG: AllocationOrder(GPR) = [ $r0 $r1 $r2 $r3 $r12 $lr $r4 $r5 $r6 $r7 $r8 $r9 $r10 $r11 ]

define i32 @test_optsize(i32 %x) optsize {
; CHECK-LABEL: test_optsize:
entry:
; CHECK: mov     r12, r0
  tail call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3}"()
; CHECK: mov     r0, r12
  ret i32 %x
}
