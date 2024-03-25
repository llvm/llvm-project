; RUN: rm -rf %t && mkdir -p %t
; RUN: llvm-as -o %t/bcsection.bc %s

; RUN: llvm-mc -I=%t -filetype=obj -triple=x86_64-pc-win32 -o %t/bcsection.coff.bco %p/Inputs/bcsection.s
; RUN: llvm-nm %t/bcsection.coff.bco | FileCheck %s --allow-empty
; RUN: not llvm-lto -exported-symbol=main -exported-symbol=_main -o %t/bcsection.coff.o %t/bcsection.coff.bco

; RUN: llvm-mc -I=%t -filetype=obj -triple=x86_64-unknown-linux-gnu -o %t/bcsection.elf.bco %p/Inputs/bcsection.s
; RUN: llvm-nm %t/bcsection.elf.bco | FileCheck %s --allow-empty
; RUN: not llvm-lto -exported-symbol=main -exported-symbol=_main -o %t/bcsection.elf.o %t/bcsection.elf.bco

target triple = "x86_64-unknown-linux-gnu"

;; The .llvmbc section is not intended for use with LTO, so there should be nothing here
; CHECK-NOT: main
define i32 @main() {
  ret i32 0
}
