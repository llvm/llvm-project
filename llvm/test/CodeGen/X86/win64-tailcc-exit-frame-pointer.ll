; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; As win64-tailcc-exit-sequences.ll, with a frame pointer. The frame register
; has to stay intact until the epilogue, so it is restored by the pop that ends
; it. A copy of the saved value is stored just below the new return address
; first, so that RSP ends up on the return address whatever its offset is.

declare tailcc void @g(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare tailcc void @h(i64, i64, i64, i64, i64, i64, i64)
declare void @use(ptr)

define tailcc void @exits(i64 %a, i64 %b, i32 %c) "frame-pointer"="all" {
; CHECK-LABEL: exits:
; CHECK:         .seh_endprologue
;
; No adjustment for the largest tail call: the standard epilogue, preceded by
; the store that overwrites the return address.
; CHECK:         movq %rsi, 88(%rbp)
; CHECK-NEXT:    .seh_startepilogue
; CHECK-NEXT:    addq $56, %rsp
; CHECK-NEXT:    popq %rbx
; CHECK-NEXT:    popq %rdi
; CHECK-NEXT:    popq %rsi
; CHECK-NEXT:    popq %rbp
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    jmp g # TAILCALL
;
; The smaller tail call restores the other registers with movs and pops rbp
; from a copy placed below its return address.
; CHECK:         movq 8(%rbp), %rbx
; CHECK-NEXT:    movq 16(%rbp), %rdi
; CHECK-NEXT:    movq 24(%rbp), %rsi
; CHECK-NEXT:    movq 32(%rbp), %rax
; CHECK-NEXT:    movq %rax, 64(%rbp)
; CHECK-NEXT:    .seh_startepilogue
; CHECK-NEXT:    leaq 64(%rbp), %rsp
; CHECK-NEXT:    popq %rbp
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    jmp h # TAILCALL
;
; CHECK:         movq 32(%rbp), %rax
; CHECK-NEXT:    movq %rax, 80(%rbp)
; CHECK-NEXT:    .seh_startepilogue
; CHECK-NEXT:    leaq 80(%rbp), %rsp
; CHECK-NEXT:    popq %rbp
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    retq $40
  %x = alloca [2 x i64], align 16
  call void @use(ptr %x)
  switch i32 %c, label %ret [ i32 0, label %big
                              i32 1, label %small ]
big:
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
small:
  musttail call tailcc void @h(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a)
  ret void
ret:
  ret void
}

; A shrinking tail call: the new return address is above the old one, and the
; epilogue ends with the pop of the frame register that lands RSP on it.
define tailcc void @shrink(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h, i64 %i, i64 %j, i64 %k, i64 %l) "frame-pointer"="all" {
; CHECK-LABEL: shrink:
; CHECK:         movq 48(%rbp), %rax
; CHECK-NEXT:    movq %rax, 96(%rbp)
; CHECK-NEXT:    .seh_startepilogue
; CHECK-NEXT:    leaq 96(%rbp), %rsp
; CHECK-NEXT:    popq %rbp
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    jmp h # TAILCALL
  call void @use(ptr null)
  musttail call tailcc void @h(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g)
  ret void
}
