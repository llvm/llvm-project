; LTO
; This doesn't currently work with gold, because it does not apply defsym
; renaming to symbols in the same module (apparently by design for consistency
; with GNU ld). Because regular LTO hands back a single object file to gold,
; it doesn't perform the desired defsym renaming. This isn't an issue with
; ThinLTO which hands back multiple native objects to gold. For regular
; LTO defsym handling, gold will need a fix (not the gold plugin).
; RUN-TODO: llvm-as %s -o %t.o
; RUN-TODO: llvm-as %S/Inputs/wrap-bar.ll -o %t1.o
; RUN-TODO: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext %t.o %t1.o -shared -o %t.so -wrap=bar
; RUN-TODO: llvm-objdump -d %t.so | FileCheck %s
; RUN-TODO: llvm-readobj --symbols %t.so | FileCheck -check-prefix=BIND %s

; ThinLTO
; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %S/Inputs/wrap-bar.ll -o %t1.o
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext %t.o %t1.o -shared -o %t.so -wrap=bar
; RUN: llvm-objdump -d %t.so | FileCheck %s -check-prefix=THIN
; RUN: llvm-readobj --symbols %t.so | FileCheck -check-prefix=BIND %s

; Make sure that calls in foo() are not eliminated and that bar is
; routed to __wrap_bar and __real_bar is routed to bar.

; CHECK:      <foo>:
; CHECK-NEXT: pushq	%rax
; CHECK-NEXT: callq{{.*}}<__wrap_bar>
; CHECK-NEXT: callq{{.*}}<bar>

; THIN:      <foo>:
; THIN-NEXT: pushq	%rax
; THIN-NEXT: callq{{.*}}<__wrap_bar>
; THIN-NEXT: popq  %rax
; THIN-NEXT: jmp{{.*}}<bar>

; Check that bar and __wrap_bar retain their original binding.
; BIND:      Name: bar
; BIND-NEXT: Value:
; BIND-NEXT: Size:
; BIND-NEXT: Binding: Local
; BIND:      Name: __wrap_bar
; BIND-NEXT: Value:
; BIND-NEXT: Size:
; BIND-NEXT: Binding: Local

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @bar()
declare void @__real_bar()

define void @foo() {
  call void @bar()
  call void @__real_bar()
  ret void
}
