; RUN: llc -mtriple=x86_64-apple-macosx -O0 %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=x86_64-apple-macosx -O0 -filetype=obj %s -o - | \
; RUN:   llvm-dwarfdump --eh-frame - | FileCheck %s --check-prefix=UNWIND

; Verify Swift async prologue CFA updates after the context push.

; ASM-LABEL: foo:
; ASM:        pushq   %rbp
; ASM-NEXT:   .cfi_def_cfa_offset 16
; ASM-NEXT:   .cfi_offset %rbp, -16
; ASM-NEXT:   pushq   %r14
; ASM-NEXT:   .cfi_adjust_cfa_offset 8
; ASM-NEXT:   leaq    8(%rsp), %rbp
; ASM-NEXT:   .cfi_def_cfa %rbp, 16
; ASM-NOT:    .cfi_def_cfa_register

; UNWIND:      0x0: CFA=RSP+8:
; UNWIND-NEXT: 0x6: CFA=RSP+16:
; UNWIND-NEXT: 0x8: CFA=RSP+24:
; UNWIND-NEXT: 0xd: CFA=RBP+16:

define void @foo(ptr swiftasync %ctx) "frame-pointer"="all" {
  call void asm sideeffect "int3", ""()
  ret void
}

; A frame with locals emits a real stack adjustment after the FP is set up.
; The CFA must already be %rbp-relative before that subq, so no CFI directive
; appears between the .cfi_def_cfa and the subq.
; ASM-LABEL: with_locals:
; ASM:        pushq   %r14
; ASM-NEXT:   .cfi_adjust_cfa_offset 8
; ASM-NEXT:   leaq    8(%rsp), %rbp
; ASM-NEXT:   .cfi_def_cfa %rbp, 16
; ASM-NEXT:   subq    ${{[0-9]+}}, %rsp

define void @with_locals(ptr swiftasync %ctx) "frame-pointer"="all" {
  %a = alloca [128 x i8]
  call void asm sideeffect "int3", "r"(ptr %a)
  ret void
}
