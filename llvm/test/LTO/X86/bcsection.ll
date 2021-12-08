; RUN: rm -rf %t && mkdir -p %t
; RUN: llvm-as -o %t/bcsection.bc %s

; RUN: llvm-mc -I=%t -filetype=obj -triple=x86_64-pc-win32 -o %t/bcsection.coff.bco %p/Inputs/bcsection.s
; RUN: llvm-nm %t/bcsection.coff.bco | FileCheck %s
; RUN: llvm-lto -exported-symbol=main -exported-symbol=_main -o %t/bcsection.coff.o %t/bcsection.coff.bco
; RUN: llvm-nm %t/bcsection.coff.o | FileCheck %s

; RUN: llvm-mc -I=%t -filetype=obj -triple=x86_64-unknown-linux-gnu -o %t/bcsection.elf.bco %p/Inputs/bcsection.s
; RUN: llvm-nm %t/bcsection.elf.bco | FileCheck %s
; RUN: llvm-lto -exported-symbol=main -exported-symbol=_main -o %t/bcsection.elf.o %t/bcsection.elf.bco
; RUN: llvm-nm %t/bcsection.elf.o | FileCheck %s

; RUN: llvm-mc -I=%t -filetype=obj -triple=x86_64-apple-darwin11 -o %t/bcsection.macho.bco %p/Inputs/bcsection.macho.s
; RUN: llvm-nm %t/bcsection.macho.bco | FileCheck %s
; RUN: llvm-lto -exported-symbol=main -exported-symbol=_main -o %t/bcsection.macho.o %t/bcsection.macho.bco
; RUN: llvm-nm %t/bcsection.macho.o | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK: main
define i32 @main() {
  ret i32 0
}
