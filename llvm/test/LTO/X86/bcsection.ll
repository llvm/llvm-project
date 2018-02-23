; RUN: llvm-as -o %T/bcsection.bc %s

; RUN: llvm-mc -I=%T -filetype=obj -triple=x86_64-pc-win32 -o %T/bcsection.coff.bco %p/Inputs/bcsection.s
; RUN: llvm-nm %T/bcsection.coff.bco | FileCheck %s
; RUN: llvm-lto -exported-symbol=main -exported-symbol=_main -o %T/bcsection.coff.o %T/bcsection.coff.bco
; RUN: llvm-nm %T/bcsection.coff.o | FileCheck %s

; RUN: llvm-mc -I=%T -filetype=obj -triple=x86_64-unknown-linux-gnu -o %T/bcsection.elf.bco %p/Inputs/bcsection.s
; RUN: llvm-nm %T/bcsection.elf.bco | FileCheck %s
; RUN: llvm-lto -exported-symbol=main -exported-symbol=_main -o %T/bcsection.elf.o %T/bcsection.elf.bco
; RUN: llvm-nm %T/bcsection.elf.o | FileCheck %s

; RUN: llvm-mc -I=%T -filetype=obj -triple=x86_64-apple-darwin11 -o %T/bcsection.macho.bco %p/Inputs/bcsection.macho.s
; RUN: llvm-nm %T/bcsection.macho.bco | FileCheck %s
; RUN: llvm-lto -exported-symbol=main -exported-symbol=_main -o %T/bcsection.macho.o %T/bcsection.macho.bco
; RUN: llvm-nm %T/bcsection.macho.o | FileCheck %s

; REQUIRES: default_triple

; CHECK: main
define i32 @main() {
  ret i32 0
}
