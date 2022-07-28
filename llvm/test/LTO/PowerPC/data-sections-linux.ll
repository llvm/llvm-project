; RUN: rm -rf %t
; RUN: mkdir %t
; RUN: llvm-as %s -o %t/bc.bc
; RUN: llvm-lto -exported-symbol var -O0 %t/bc.bc -o %t/default.o
; RUN: llvm-lto -exported-symbol var -O0 --data-sections=1 %t/bc.bc -o \
; RUN:   %t/data-sections.o
; RUN: llvm-lto -exported-symbol var -O0 --data-sections=0 %t/bc.bc -o \
; RUN:   %t/no-data-sections.o
; RUN: llvm-objdump -t %t/default.o | FileCheck %s
; RUN: llvm-objdump -t %t/data-sections.o | FileCheck %s
; RUN: llvm-objdump -t %t/no-data-sections.o | FileCheck --check-prefix \
; RUN:   CHECK-NO-DATA-SECTIONS %s

target triple = "powerpc64le-unknown-linux-gnu"

@var = global i32 0

; CHECK:                  0000000000000000 g O .bss.var [[#%x,]] var

; CHECK-NO-DATA-SECTIONS: 0000000000000000 g O .bss [[#%x,]] var
