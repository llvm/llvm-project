; RUN: llc -combiner-topological-sorting < %s -mtriple=x86_64 | FileCheck %s --check-prefix=ATT
; RUN: llc -combiner-topological-sorting < %s -mtriple=x86_64 -output-asm-variant=1 | FileCheck %s --check-prefix=INTEL

module asm "mov ($foo), %eax"

; ATT:   .att_syntax{{$}}
; ATT:   movl ($foo), %eax
; INTEL: .intel_syntax noprefix{{$}}
; INTEL: mov eax, dword ptr [$foo]
