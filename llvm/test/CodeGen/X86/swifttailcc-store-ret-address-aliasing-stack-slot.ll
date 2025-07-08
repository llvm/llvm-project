; RUN: llc %s -o - | FileCheck %s

target triple = "x86_64-apple-macosx"

declare swifttailcc void @tc_fn(ptr swiftasync, i64, ptr, i8, ptr swiftself)

declare void @foo()

; FIXME: Currently the return address is written to the stack before loading the
; argument from an aliasing stack slot.
define swifttailcc void @test(ptr %0, ptr swiftasync %1, i64 %2, i64 %3, ptr %4, ptr %5, i64 %6, ptr %7, i8 %8) {
; CHECK-LABEL: test:
; CHECK:       ## %bb.0: ## %entry
; CHECK-NEXT:    pushq %r15
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    pushq %rbx
; CHECK-NEXT:    .cfi_def_cfa_offset 24
; CHECK-NEXT:    pushq %rax
; CHECK-NEXT:    .cfi_def_cfa_offset 32
; CHECK-NEXT:    .cfi_offset %rbx, -24
; CHECK-NEXT:    .cfi_offset %r15, -16
; CHECK-NEXT:    movq %r9, %r13
; CHECK-NEXT:    movq %r8, %rbx
; CHECK-NEXT:    movq {{[0-9]+}}(%rsp), %r15
; CHECK-NEXT:    callq _foo
; CHECK-NEXT:    movq %r14, (%rax)
; CHECK-NEXT:    movl [[OFF:[0-9]+]](%rsp), %edx
; CHECK-NEXT:    movq {{[0-9]+}}(%rsp), %rcx
; CHECK-NEXT:    movq %rcx, [[OFF]](%rsp)
; CHECK-NEXT:    movq %rax, %r14
; CHECK-NEXT:    movq %r13, %rdi
; CHECK-NEXT:    movq %r15, %rsi
; CHECK-NEXT:    movq %rbx, %r13
; CHECK-NEXT:    addq $8, %rsp
; CHECK-NEXT:    popq %rbx
; CHECK-NEXT:    popq %r15
; CHECK-NEXT:    addq $16, %rsp
; CHECK-NEXT:    jmp _tc_fn ## TAILCALL
entry:
  %res = tail call ptr @foo()
  store ptr %1, ptr %res, align 8
  musttail call swifttailcc void @tc_fn(ptr swiftasync %res, i64 %6, ptr %7, i8 %8, ptr swiftself %5)
  ret void
}
