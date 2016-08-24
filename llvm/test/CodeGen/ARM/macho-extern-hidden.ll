; RUN: llc < %s -mtriple=thumbv7em-apple-unknown-macho | FileCheck %s

; CHECK: movw   r0, :lower16:(L_bar$non_lazy_ptr-(LPC0_0+4))
; CHECK: movt   r0, :upper16:(L_bar$non_lazy_ptr-(LPC0_0+4))

@bar = external hidden global i32
define i32 @foo() {
  %tmp = load i32, i32* @bar, align 4
  ret i32 %tmp
}
