# RUN: llvm-mc -filetype=obj -triple x86_64 %s -o %t
# RUN: llvm-readelf -Srs %t | FileCheck %s

## Test that we can forward reference a section.

mov .rodata, %rsi
mov .debug_info, %rsi

.section .rodata,"a"
.section .debug_info,"G",@progbits,11,comdat; .long x1
.section .debug_info,"G",@progbits,22,comdat; .long x2
.section .debug_info,"",@progbits; .long x0

# CHECK:      Relocation section '.rela.debug_info' at offset {{.*}} contains 1
# CHECK:      Relocation section '.rela.debug_info' at offset {{.*}} contains 1
# CHECK:      Relocation section '.rela.debug_info' at offset {{.*}} contains 1

# CHECK:      Symbol table '.symtab' contains 8 entries:
# CHECK-NEXT:    Num:    Value          Size Type    Bind   Vis       Ndx Name
# CHECK-NEXT:  0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
# CHECK-NEXT:  0000000000000000     0 SECTION LOCAL  DEFAULT     4 .rodata
# CHECK-NEXT:  0000000000000000     0 SECTION LOCAL  DEFAULT    11 .debug_info
# CHECK-NEXT:  0000000000000000     0 NOTYPE  LOCAL  DEFAULT     5 11
# CHECK-NEXT:  0000000000000000     0 NOTYPE  LOCAL  DEFAULT     8 22
# CHECK-NEXT:  0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND x1
# CHECK-NEXT:  0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND x2
# CHECK-NEXT:  0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND x0
