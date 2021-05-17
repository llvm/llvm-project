; RUN: llc -mtriple i686-pc-linux < %s | FileCheck %s --check-prefix=X86
; RUN: llc -mtriple x86_64-pc-linux < %s | FileCheck %s --check-prefix=X86_64
; RUN: llc -mtriple x86_64-pc-linux-gnux32 < %s | FileCheck %s --check-prefix=X86

; This test checks that the compiler emits a .note.gnu.property section for
; modules with "cf-protection" module flags.

; X86:      .section        .note.gnu.property,"a",@note
; X86-NEXT: .p2align 2
; X86-NEXT: .long    4
; X86-NEXT: .long    12
; X86-NEXT: .long    5
; X86-NEXT: .asciz   "GNU"
; X86-NEXT: .long    3221225474
; X86-NEXT: .long    4
; X86-NEXT: .long    3
; X86-NEXT: .p2align 2

; X86_64:      .section        .note.gnu.property,"a",@note
; X86_64-NEXT: .p2align 3
; X86_64-NEXT: .long    4
; X86_64-NEXT: .long    16
; X86_64-NEXT: .long    5
; X86_64-NEXT: .asciz   "GNU"
; X86_64-NEXT: .long    3221225474
; X86_64-NEXT: .long    4
; X86_64-NEXT: .long    3
; X86_64-NEXT: .p2align 3

!llvm.module.flags = !{!0, !1}

!0 = !{i32 4, !"cf-protection-return", i32 1}
!1 = !{i32 4, !"cf-protection-branch", i32 1}
