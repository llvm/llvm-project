; RUN: llc -verify-machineinstrs < %s -mtriple=i386-apple-darwin   -frame-pointer=all | FileCheck %s -check-prefix=X86
; RUN: llc -verify-machineinstrs < %s -mtriple=x86_64-apple-darwin -frame-pointer=all | FileCheck %s -check-prefix=X64

; Tail call should not use ebp / rbp after it's popped. Use esp / rsp.

define void @t1(ptr nocapture %value) nounwind {
entry:
; X86-LABEL: t1:
; X86: jmpl *4(%esp)

; X64-LABEL: t1:
; X64: jmpq *%rdi
  tail call void %value() nounwind
  ret void
}

define void @t2(i32 %a, ptr nocapture %value) nounwind {
entry:
; X86-LABEL: t2:
; X86: jmpl *8(%esp)

; X64-LABEL: t2:
; X64: jmpq *%rsi
  tail call void %value() nounwind
  ret void
}

define void @t3(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, ptr nocapture %value) nounwind {
entry:
; X86-LABEL: t3:
; X86: jmpl *28(%esp)

; X64-LABEL: t3:
; X64: jmpq *8(%rsp)
  tail call void %value() nounwind
  ret void
}

define void @t4(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, ptr nocapture %value) nounwind {
entry:
; X86-LABEL: t4:
; X86: jmpl *32(%esp)

; X64-LABEL: t4:
; X64: jmpq *16(%rsp)
  tail call void %value() nounwind
  ret void
}
