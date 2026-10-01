; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; The value stored over the caller's return address is live across a call, so
; it is in a callee-saved register. That store must be the last body
; instruction, so the register cannot be restored by a mov before it. Either the
; epilogue pops it (largest tail call), or a copy of its saved value is popped
; from just below the return address.

declare tailcc void @g(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare tailcc void @g2(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare void @use(ptr)

define tailcc void @gpr(i64 %a, i64 %b, i32 %c) {
; CHECK-LABEL: gpr:
; CHECK:         movq %rsi, 120(%rsp)
; CHECK-NEXT:    .seh_startepilogue
; CHECK-NEXT:    addq $32, %rsp
; CHECK-NEXT:    popq %rbx
; CHECK-NEXT:    popq %rdi
; CHECK-NEXT:    popq %rsi
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    jmp g2 # TAILCALL
;
; CHECK:         movq 32(%rsp), %rbx
; CHECK-NEXT:    movq 40(%rsp), %rdi
; CHECK-NEXT:    movq 48(%rsp), %rax
; CHECK-NEXT:    movq %rax, 64(%rsp)
; CHECK-NEXT:    movq %rsi, 120(%rsp)
; CHECK-NEXT:    .seh_startepilogue
; CHECK-NEXT:    addq $64, %rsp
; CHECK-NEXT:    popq %rsi
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    jmp g # TAILCALL
  call void @use(ptr null)
  switch i32 %c, label %ret [ i32 0, label %big
                              i32 1, label %mid ]
big:
  musttail call tailcc void @g2(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
mid:
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
ret:
  ret void
}

; With a frame pointer, the register and the frame register are both popped.
define tailcc void @gpr_fp(i64 %a, i64 %b, i32 %c) "frame-pointer"="all" {
; CHECK-LABEL: gpr_fp:
; CHECK:         movq 24(%rbp), %rax
; CHECK-NEXT:    movq %rax, 40(%rbp)
; CHECK-NEXT:    movq 32(%rbp), %rax
; CHECK-NEXT:    movq %rax, 48(%rbp)
; CHECK-NEXT:    movq %rsi, 104(%rbp)
; CHECK-NEXT:    .seh_startepilogue
; CHECK-NEXT:    leaq 40(%rbp), %rsp
; CHECK-NEXT:    popq %rsi
; CHECK-NEXT:    popq %rbp
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    jmp g # TAILCALL
  call void @use(ptr null)
  switch i32 %c, label %ret [ i32 0, label %big
                              i32 1, label %mid ]
big:
  musttail call tailcc void @g2(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
mid:
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
ret:
  ret void
}

declare tailcc void @gd(i64, i64, i64, i64, i64, double, double, double, double)
declare tailcc void @gd2(i64, i64, i64, i64, i64, double, double, double, double, double, double)

; A callee-saved XMM source is copied to a volatile register before its restore.
define tailcc void @xmm(i64 %a, double %d, i32 %c) {
; CHECK-LABEL: xmm:
; CHECK:         movaps %xmm6, %xmm0
; CHECK-NEXT:    movaps 32(%rsp), %xmm6
; CHECK-NEXT:    movsd %xmm0, 120(%rsp)
; CHECK-NEXT:    .seh_startepilogue
  call void @use(ptr null)
  switch i32 %c, label %ret [ i32 0, label %big
                              i32 1, label %mid ]
big:
  musttail call tailcc void @gd2(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, double %d, double %d, double %d, double %d, double %d, double %d)
  ret void
mid:
  musttail call tailcc void @gd(i64 %a, i64 %a, i64 %a, i64 %a, i64 %a, double %d, double %d, double %d, double %d)
  ret void
ret:
  ret void
}
