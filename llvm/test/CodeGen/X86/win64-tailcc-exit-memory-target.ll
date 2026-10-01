; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; A tail call target held in a stack slot must be loaded in the body: after the
; epilogue moves RSP the slot is below it, where Win64 gives no guarantee that
; it survives, and its offset from RSP has changed.

declare void @use(ptr)

define tailcc void @mem(ptr %p, i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h) {
; CHECK-LABEL: mem:
; CHECK:         movq 32(%rsp), %rax
; CHECK:         .seh_startepilogue
; CHECK-NEXT:    addq $104, %rsp
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    rex64 jmpq *%rax # TAILCALL
  %slot = alloca ptr
  store ptr %p, ptr %slot
  call void @use(ptr %slot)
  %fp = load ptr, ptr %slot
  musttail call tailcc void %fp(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f)
  ret void
}
