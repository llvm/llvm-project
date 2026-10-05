; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; A callee-pop function can pop more bytes than ret's 16-bit immediate holds.
; The Windows unwinder can't follow a pop/add/push sequence, so instead the
; return address is copied to just past the bytes to pop, RSP is moved to it by
; an add (or lea) that starts a recognised epilogue, and a plain ret finishes.
; Stack argument area here: 8996 stack arguments plus the 32 byte home area,
; padded, is 72008 bytes.

declare void @use(ptr)

declare void @useint(i64)

declare tailcc void @h(i64, i64, i64, i64, i64, i64, i64)

define tailcc void @big([9000 x i64] %a) {
; CHECK-LABEL: big:
; CHECK:         movq (%rsp), %rax
; CHECK-NEXT:    movq %rax, 72008(%rsp)
; CHECK-NEXT:    addq $72008, %rsp
; CHECK-NEXT:    retq
  ret void
}

; With a frame pointer, the frame register is popped from a copy placed just
; below the moved return address.
define tailcc void @big_fp([9000 x i64] %a) "frame-pointer"="all" {
; CHECK-LABEL: big_fp:
; CHECK:         movq 8(%rbp), %rax
; CHECK-NEXT:    movq %rax, 72016(%rbp)
; CHECK-NEXT:    movq (%rbp), %rax
; CHECK-NEXT:    movq %rax, 72008(%rbp)
; CHECK-NEXT:    .seh_startepilogue
; CHECK-NEXT:    leaq 72008(%rbp), %rsp
; CHECK-NEXT:    popq %rbp
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    retq
  call void @use(ptr null)
  ret void
}

; Callee-saved registers are restored with movs first.
define tailcc void @big_csr(i64 %x, [9000 x i64] %a) {
; CHECK-LABEL: big_csr:
; CHECK:         movq 32(%rsp), %rsi
; CHECK-NEXT:    movq 40(%rsp), %rax
; CHECK-NEXT:    movq %rax, 72048(%rsp)
; CHECK-NEXT:    .seh_startepilogue
; CHECK-NEXT:    addq $72048, %rsp
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    retq
  call void @use(ptr null)
  call void @useint(i64 %x)
  ret void
}

; The scratch register used for the copies is not one that holds the result.
define tailcc i64 @big_ret(i64 %x, [9000 x i64] %a) "frame-pointer"="all" {
; CHECK-LABEL: big_ret:
; CHECK:         movq %rsi, %rax
; CHECK:         movq %rcx, 72032(%rbp)
; CHECK:         movq %rcx, 72024(%rbp)
; CHECK:         leaq 72024(%rbp), %rsp
; CHECK-NEXT:    popq %rbp
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    retq
  call void @use(ptr null)
  ret i64 %x
}

; A tail call that shrinks the area by a lot needs nothing new.
define tailcc void @big_shrink([9000 x i64] %a) {
; CHECK-LABEL: big_shrink:
; CHECK:         movq (%rsp), %rax
; CHECK-NEXT:    movq %rax, 71952(%rsp)
; CHECK:         addq $71952, %rsp
; CHECK-NEXT:    jmp h # TAILCALL
  musttail call tailcc void @h(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7)
  ret void
}
