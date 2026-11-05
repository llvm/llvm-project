; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -O2 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -O2 -enable-subreg-liveness | FileCheck %s

; A partial def that preserves the bits it does not write also reads them.  If
; that read is not recorded as an implicit use of the super-register, copy
; propagation removes the only thing keeping the incoming value alive and the
; byte/word fixup widens the load, zero-extending and dropping bits 8..63.

define i64 @f(i64 %x) {
  %a = add i64 %x, 8
  %p = inttoptr i64 %a to ptr
  %m = and i64 %x, -256
  %b = load i8, ptr %p, align 1
  %z = zext i8 %b to i64
  %r = or i64 %m, %z
  ret i64 %r
}

; CHECK-LABEL: f:
; CHECK:         movq %rdi, %rax
; CHECK-NEXT:    movb 8({{%rax|%rdi}}), %al
; CHECK-NEXT:    retq
; CHECK-NOT:     movzbl
