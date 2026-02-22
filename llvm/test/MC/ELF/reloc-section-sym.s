# RUN: llvm-mc -filetype=obj -triple x86_64 %s -o %t.all
# RUN: llvm-mc -filetype=obj -triple x86_64 --reloc-section-sym=all %s -o %t.all2
# RUN: llvm-mc -filetype=obj -triple x86_64 --reloc-section-sym=internal %s -o %t.internal
# RUN: llvm-mc -filetype=obj -triple x86_64 --reloc-section-sym=none %s -o %t.none
# RUN: llvm-mc -filetype=obj -triple x86_64 --reloc-section-sym=internal --save-temp-labels %s -o %t.internal-stl
# RUN: llvm-readelf -rs %t.all | FileCheck %s --check-prefix=ALL
# RUN: cmp %t.all %t.all2
# RUN: llvm-readelf -rs %t.internal | FileCheck %s --check-prefixes=INT,INTERNAL
# RUN: llvm-readelf -rs %t.none | FileCheck %s --check-prefix=NONE
# RUN: llvm-readelf -rs %t.internal-stl | FileCheck %s --check-prefixes=INT,INTERNAL2

.text
  nop
local:
  nop
.Ltemp:
  nop

.section .text1,"ax"
  call local
  call .Ltemp

.section .rodata,"a"
.long local + 16
.long .Ltemp + 16

## =all (default): all eligible local binding symbols are converted to section symbols.
# ALL:      .rela.text1
# ALL:      R_X86_64_PLT32 {{.*}} .text - 3
# ALL-NEXT: R_X86_64_PLT32 {{.*}} .text - 2
# ALL:      .rela.rodata
# ALL:      R_X86_64_32 {{.*}} .text + 11
# ALL-NEXT: R_X86_64_32 {{.*}} .text + 12
# ALL:      SECTION LOCAL  DEFAULT {{.*}} .text
# ALL:      NOTYPE  LOCAL  DEFAULT {{.*}} local

## =internal: internal symbols (.L) are converted to section symbols.
## With --save-temp-labels, .Ltemp is still converted but remains in the symbol table.
# INT:      .rela.text1
# INT:      R_X86_64_PLT32 {{.*}} local - 4
# INT-NEXT: R_X86_64_PLT32 {{.*}} .text - 2
# INT:      .rela.rodata
# INT:      R_X86_64_32 {{.*}} local + 10
# INT-NEXT: R_X86_64_32 {{.*}} .text + 12
# INT:            SECTION LOCAL  DEFAULT {{.*}} .text
# INTERNAL:       NOTYPE  LOCAL  DEFAULT {{.*}} local
# INTERNAL-NOT:   {{.}}
# INTERNAL2:      NOTYPE  LOCAL  DEFAULT {{.*}} local
# INTERNAL2-NEXT: NOTYPE  LOCAL  DEFAULT {{.*}} .Ltemp

## =none: no symbol-to-section conversion.
# NONE:      .rela.text1
# NONE:      R_X86_64_PLT32 {{.*}} local - 4
# NONE-NEXT: R_X86_64_PLT32 {{.*}} .Ltemp - 4
# NONE:      .rela.rodata
# NONE:      R_X86_64_32 {{.*}} local + 10
# NONE-NEXT: R_X86_64_32 {{.*}} .Ltemp + 10
# NONE-NOT:  SECTION
# NONE:      NOTYPE  LOCAL  DEFAULT {{.*}} local
# NONE-NEXT: NOTYPE  LOCAL  DEFAULT {{.*}} .Ltemp
