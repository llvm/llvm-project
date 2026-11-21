; RUN: llc -mtriple=x86_64-pc-windows-msvc %s -o - | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-uefi %s -o - | FileCheck %s

; A nounwind function has no unwind info to keep correct, so its tail call exits
; use the ordinary epilogue: the registers are popped, and then RSP is moved by
; whatever remains of the reserve.

declare tailcc void @g(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare tailcc void @h(i64, i64, i64, i64, i64, i64, i64)
declare void @use(ptr)

define tailcc void @nounwind_reserve(i64 %a, i64 %b, i1 %c) nounwind {
; CHECK-LABEL: nounwind_reserve:
; CHECK-NOT:     .seh_
; CHECK:         subq $48, %rsp
; CHECK:         addq $64, %rsp
; CHECK-NEXT:    popq %rbx
; CHECK-NEXT:    popq %rdi
; CHECK-NEXT:    popq %rsi
; CHECK-NEXT:    jmp g # TAILCALL
; CHECK:         addq $64, %rsp
; CHECK-NEXT:    popq %rbx
; CHECK-NEXT:    popq %rdi
; CHECK-NEXT:    popq %rsi
; CHECK-NEXT:    addq $32, %rsp
; CHECK-NEXT:    jmp h # TAILCALL
  %p = alloca [4 x i64]
  call void @use(ptr %p)
  br i1 %c, label %big, label %small
big:
  musttail call tailcc void @g(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b)
  ret void
small:
  musttail call tailcc void @h(i64 %a, i64 %b, i64 %a, i64 %b, i64 %a, i64 %b, i64 %a)
  ret void
}
