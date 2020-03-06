; RUN: llc -mtriple=arm64-linux-gnu < %s | FileCheck %s
; RUN: llc -mtriple=arm64-linux-gnu -filetype=obj < %s | llvm-objdump - -r -d --triple=arm64-linux-gnu | FileCheck --check-prefix=CHECK-ELF %s

define void @foo() nounwind {
  ret void
}

  ; Check source looks ELF-like: no leading underscore, comments with //
; CHECK: foo: // @foo
; CHECK:     ret

  ; Similarly make sure ELF output works and is vaguely sane: aarch64 target
  ; machine with correct section & symbol names.
; CHECK-ELF: file format elf64-aarch64

; CHECK-ELF: Disassembly of section .text
; CHECK-ELF-LABEL: <foo>:
; CHECK-ELF:    ret
