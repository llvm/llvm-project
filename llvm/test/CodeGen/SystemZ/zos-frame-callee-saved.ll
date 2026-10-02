; A function without calls and without stack objects that uses callee-saved
; registers must get a stack frame of its own. Otherwise the registers are
; stored relative to the unchanged stack pointer, i.e. into the register save
; area of the caller's DSA, overwriting the caller's saved return address.
;
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

define void @leaf() {
; CHECK-LABEL: leaf DS 0H
; CHECK:         stmg 6,9,1936(4)
; CHECK-NEXT:    L#stack_update0 DS 0H
; CHECK-NEXT:    aghi 4,-128
; CHECK:         lmg 7,9,2072(4)
; CHECK-NEXT:    aghi 4,128
; CHECK-NEXT:    b 2(7)
  call void asm sideeffect "", "~{r8},~{r9}"()
  ret void
}

; A true leaf (no callee-saved registers) still gets no stack frame.
define i64 @trueleaf(i64 %a) {
; CHECK-LABEL: trueleaf DS 0H
; CHECK-NOT:     stmg
; CHECK-NOT:     aghi 4,
; CHECK:         b 2(7)
  %r = add i64 %a, 1
  ret i64 %r
}
