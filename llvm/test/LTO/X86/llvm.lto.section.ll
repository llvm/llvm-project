; RUN: rm -rf %t && mkdir -p %t
; RUN: llvm-as -o %t/llvm.lto.section.bc %s

; RUN: llvm-mc -I=%t -filetype=obj -triple=x86_64-pc-win32 -o %t/llvm.lto.section.coff.bco %p/Inputs/llvm.lto.section.s
; RUN: llvm-nm %t/llvm.lto.section.coff.bco | FileCheck %s
; RUN: llvm-lto -exported-symbol=main -exported-symbol=_main -o %t/llvm.lto.section.coff.o %t/llvm.lto.section.coff.bco
; RUN: llvm-nm %t/llvm.lto.section.coff.o | FileCheck %s

; RUN: llvm-mc -I=%t -filetype=obj -triple=x86_64-unknown-linux-gnu -o %t/llvm.lto.section.elf.bco %p/Inputs/llvm.lto.section.s
; RUN: llvm-nm %t/llvm.lto.section.elf.bco | FileCheck %s
; RUN: llvm-lto -exported-symbol=main -exported-symbol=_main -o %t/llvm.lto.section.elf.o %t/llvm.lto.section.elf.bco
; RUN: llvm-nm %t/llvm.lto.section.elf.o | FileCheck %s


; RUN: llvm-mc -I=%t -filetype=obj -triple=x86_64-apple-darwin11 -o %t/bcsection.macho.bco %p/Inputs/bcsection.macho.s
; RUN: llvm-nm %t/bcsection.macho.bco | FileCheck %s
; RUN: llvm-lto -exported-symbol=main -exported-symbol=_main -o %t/bcsection.macho.o %t/bcsection.macho.bco
; RUN: llvm-nm %t/bcsection.macho.o | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK: main
define i32 @main() {
  ret i32 0
}
