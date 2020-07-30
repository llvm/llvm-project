; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:16:16-i32:16:32-a:16-n8:16"
target triple = "msp430---elf"

define void @test_no_clobber() {
entry:
; CHECK-LABEL: test_no_clobber
; CHECK-NOT: push
  call void asm sideeffect "", ""()
; CHECK-NOT: pop
  ret void
; CHECK: -- End function
}

define void @test_1() {
entry:
; CHECK-LABEL: test_1:
; CHECK: push r8
; CHECK: push r6
; CHECK: push r4
  call void asm sideeffect "", "~{r4},~{r6},~{r8}"()
; CHECK: pop r4
; CHECK: pop r6
; CHECK: pop r8
  ret void
}

define void @test_2() {
entry:
; CHECK-LABEL: test_2:
; CHECK: push r9
; CHECK: push r7
; CHECK: push r5
  call void asm sideeffect "", "~{r5},~{r7},~{r9}"()
; CHECK: pop r5
; CHECK: pop r7
; CHECK: pop r9
  ret void
}

; The r10 register is special because the sequence
;   pop r10
;   ret
; can be replaced with
;   jmp __mspabi_func_epilog_1
; or other such function (depending on previous instructions).
; Still, it is not replaced *yet*.
define void @test_r10() {
entry:
; CHECK-LABEL: test_r10:
; CHECK: push r10
  call void asm sideeffect "", "~{r10}"()
; CHECK: pop r10
  ret void
}
