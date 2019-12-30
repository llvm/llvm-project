; RUN: llc -mtriple thumbv7em-apple-unknown-eabi-macho %s -o - -O0 | FileCheck %s

; CHECK: add r7, sp, #{{[1-9]+}}

define void @foo1() {
  call void asm sideeffect "", "~{r4}"()
  call void @foo2()
  ret void
}

declare void @foo2()

; CHECK: _bar:
; CHECK-NEXT: .cfi_startproc
; CHECK-NEXT: @ %bb.0:
; CHECK-NEXT: subw    sp, sp, #3800
; CHECK-NEXT: .cfi_def_cfa_offset 3800
; CHECK-NEXT: addw    sp, sp, #3800
; CHECK-NEXT: bx      lr
; CHECK-NEXT: .cfi_endproc

define void @bar() {
  %a1 = alloca [3800 x i8], align 4
  %p = getelementptr inbounds [3800 x i8], [3800 x i8]* %a1, i32 0, i32 0
  ret void
}
