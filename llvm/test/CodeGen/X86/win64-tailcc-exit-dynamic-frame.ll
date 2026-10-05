; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; A frame with a dynamic alloca and an over-aligned object: RSP is not known
; statically, but the frame pointer is derived before the stack is realigned, so
; the exit is the same as for any other frame-pointer frame.

declare tailcc void @g(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)

declare tailcc void @h(i64, i64, i64, i64, i64, i64, i64)

declare void @use(ptr)

define tailcc void @dyn(i64 %a, i64 %b, i32 %c) {
; CHECK-LABEL: dyn:
; CHECK:         leaq 32(%rsp), %rbp
; CHECK-NEXT:    .seh_setframe %rbp, 32
; CHECK:         jmp g # TAILCALL
; CHECK:         .seh_startepilogue
; CHECK-NEXT:    leaq 64(%rbp), %rsp
; CHECK-NEXT:    popq %rbp
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    jmp h # TAILCALL
; CHECK:         .seh_startepilogue
; CHECK-NEXT:    leaq 80(%rbp), %rsp
; CHECK-NEXT:    popq %rbp
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    retq $40
  %x = alloca i8, i64 %a, align 16
  %y = alloca [4 x i64], align 64
  call void @use(ptr %x)
  call void @use(ptr %y)
  switch i32 %c, label %ret [
    i32 0, label %big
    i32 1, label %small
  ]

big:
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void

small:
  musttail call tailcc void @h(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a)
  ret void

ret:
  ret void
}
