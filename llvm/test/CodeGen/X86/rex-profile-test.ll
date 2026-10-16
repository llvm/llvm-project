;; Test that the UEFI and Windows targets set the rex64 correctly.
; RUN: llc -mtriple x86_64-uefi %s -o - | FileCheck %s -check-prefix=REX
; RUN: llc -mtriple x86_64-windows-msvc %s -o - | FileCheck %s -check-prefix=REX
; RUN: llc -mtriple x86_64-unknown-linux %s -o - | FileCheck %s -check-prefix=NOREX

define void @test_tailjmp(ptr %fptr) {
; REX-LABEL:    test_tailjmp:           # @test_tailjmp
; REX:          # %bb.0:                # %entry
; REX-NEXT:     rex64   jmpq    *%rcx   # TAILCALL
;
; NOREX-LABEL:  test_tailjmp:           # @test_tailjmp
; NOREX:        .cfi_startproc
; NOREX-NEXT:   # %bb.0:                # %entry
; NOREX-NEXT:   jmpq	*%rdi           # TAILCALL
entry:
  tail call void %fptr()
  ret void
}
