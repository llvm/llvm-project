; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -relocation-model=static | FileCheck %s

@target_ptr = external global ptr

define preserve_nonecc void @dyn_tail(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9, i64 %a10, i64 %a11, i64 %a12, i64 %a13, i64 %a14, i64 %a15, i64 %a16, i64 %a17, i64 %a18, i64 %a19, i64 %a20, i64 %a21, i64 %a22, i64 %a23, i64 %a24, i64 %a25, i64 %a26, i64 %a27, i64 %a28, i64 %a29, i64 %a30, i64 %a31, i64 %a32, i64 %a33, i64 %a34, i64 %a35) {
; CHECK-LABEL: dyn_tail:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq target_ptr@GOTPCREL(%rip), %r10
; CHECK-NEXT:    movq (%r10), %r10
; CHECK-NEXT:    jmpq *%r10 # TAILCALL
entry:
  %target = load ptr, ptr @target_ptr, align 8
  musttail call preserve_nonecc void %target(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9, i64 %a10, i64 %a11, i64 %a12, i64 %a13, i64 %a14, i64 %a15, i64 %a16, i64 %a17, i64 %a18, i64 %a19, i64 %a20, i64 %a21, i64 %a22, i64 %a23, i64 %a24, i64 %a25, i64 %a26, i64 %a27, i64 %a28, i64 %a29, i64 %a30, i64 %a31, i64 %a32, i64 %a33, i64 %a34, i64 %a35)
  ret void
}

define preserve_nonecc void @dyn_tail_free_regs() {
; CHECK-LABEL: dyn_tail_free_regs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq target_ptr@GOTPCREL(%rip), %rax
; CHECK-NEXT:    jmpq *(%rax) # TAILCALL
entry:
  %target = load ptr, ptr @target_ptr, align 8
  musttail call preserve_nonecc void %target()
  ret void
}

define preserve_nonecc void @dyn_tail_arg(ptr %target, i64 %a0) {
; CHECK-LABEL: dyn_tail_arg:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movq %r12, %rax
; CHECK-NEXT:    jmpq *%rax # TAILCALL
entry:
  musttail call preserve_nonecc void %target(ptr %target, i64 %a0)
  ret void
}
