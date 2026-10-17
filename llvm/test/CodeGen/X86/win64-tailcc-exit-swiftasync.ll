; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s

; On Win64 there is no extended Swift async frame (the async context stored just
; below the saved frame pointer): the context is kept in an ordinary stack slot.
; A swiftasync function therefore gets the same exits as any other frame.

declare swifttailcc void @g(ptr swiftasync, i64, i64, i64, i64, i64, i64, i64, i64, i64)

declare swifttailcc void @h(ptr swiftasync, i64, i64, i64, i64, i64, i64)

declare void @use(ptr)

declare ptr @llvm.swift.async.context.addr()

define swifttailcc void @async_fn(ptr swiftasync %ctx, i64 %a, i64 %b, i32 %c) "frame-pointer"="all" {
; CHECK-LABEL: async_fn:
; CHECK:         movq %r14, -8(%rbp)
;
; CHECK:         .seh_startepilogue
; CHECK-NEXT:    addq $48, %rsp
; CHECK-NEXT:    popq %rdi
; CHECK-NEXT:    popq %rsi
; CHECK-NEXT:    popq %rbp
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    jmp g # TAILCALL
;
; CHECK:         movq 16(%rbp), %rax
; CHECK-NEXT:    movq %rax, 48(%rbp)
; CHECK-NEXT:    .seh_startepilogue
; CHECK-NEXT:    leaq 48(%rbp), %rsp
; CHECK-NEXT:    popq %rbp
; CHECK-NEXT:    .seh_endepilogue
; CHECK-NEXT:    jmp h # TAILCALL
  %addr = call ptr @llvm.swift.async.context.addr()
  call void @use(ptr %addr)
  switch i32 %c, label %ret [
    i32 0, label %big
    i32 1, label %small
  ]

big:
  musttail call swifttailcc void @g(ptr swiftasync %ctx, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a)
  ret void

small:
  musttail call swifttailcc void @h(ptr swiftasync %ctx, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a)
  ret void

ret:
  ret void
}
