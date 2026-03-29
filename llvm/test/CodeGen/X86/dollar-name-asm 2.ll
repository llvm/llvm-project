; RUN: llc < %s -mtriple=x86_64 | FileCheck %s --check-prefix=ATT
; RUN: llc < %s -mtriple=x86_64 -output-asm-variant=1 | FileCheck %s --check-prefix=INTEL

module asm "mov ($foo), %eax"

; ATT:   movl ($foo), %eax
; INTEL: mov eax, dword ptr [$foo]
