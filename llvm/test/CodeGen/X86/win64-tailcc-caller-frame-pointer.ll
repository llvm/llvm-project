; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-linux-gnu < %s | FileCheck %s --check-prefix=LINUX

; A Win64 frame that stays live across a call to a guaranteed-TCO convention
; must be found through a frame pointer: the callee may tail-call with a
; different stack-argument size, which changes the RSP the unwinder recovers.

declare tailcc void @tailcc_callee()
declare swifttailcc void @swifttailcc_callee()
declare void @c_callee()

define void @call_tailcc() {
; CHECK-LABEL: call_tailcc:
; CHECK:         .seh_setframe %rbp
; CHECK:         callq tailcc_callee
  call tailcc void @tailcc_callee()
  ret void
}

define void @call_swifttailcc() {
; CHECK-LABEL: call_swifttailcc:
; CHECK:         .seh_setframe %rbp
; CHECK:         callq swifttailcc_callee
  call swifttailcc void @swifttailcc_callee()
  ret void
}

define void @call_tailcc_indirect(ptr %fp) {
; CHECK-LABEL: call_tailcc_indirect:
; CHECK:         .seh_setframe %rbp
; CHECK:         callq *%rcx
  call tailcc void %fp()
  ret void
}

define void @call_c() {
; CHECK-LABEL: call_c:
; CHECK-NOT:     .seh_setframe
; CHECK:         callq c_callee
  call void @c_callee()
  ret void
}

define tailcc void @only_tail_calls() {
; CHECK-LABEL: only_tail_calls:
; CHECK-NOT:     .seh_setframe
; CHECK:         jmp tailcc_callee # TAILCALL
  musttail call tailcc void @tailcc_callee()
  ret void
}

; Other targets are unchanged.
; LINUX-LABEL: call_tailcc:
; LINUX-NOT:     rbp
; LINUX-LABEL: call_tailcc_indirect:
; LINUX-NOT:     rbp
