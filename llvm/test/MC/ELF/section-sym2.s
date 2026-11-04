# RUN: llvm-mc -filetype=obj -triple x86_64 %s -o %t
# RUN: llvm-readelf -SrsX %t | FileCheck %s

## Test that we can forward reference a section.

mov .rodata, %rsi
mov data, %rsi
mov .debug_info, %rsi
mov .debug_abbrev, %rsi

.section .rodata,"a"
.pushsection data, 2; .long 2; .popsection
.section data; .long 1
.section .debug_info,"G",@progbits,11,comdat; .long x1
.section .debug_info,"G",@progbits,22,comdat; .long x2
.section .debug_info,"",@progbits; .long x0

.text
mov data, %rdi

# CHECK:      Relocation section '.rela.text'
# CHECK:      R_X86_64_32S {{.*}} data + 0
# CHECK:      R_X86_64_32S {{.*}} data + 0

# CHECK:      Relocation section '.rela.debug_info' at offset {{.*}} contains 1
# CHECK:      Relocation section '.rela.debug_info' at offset {{.*}} contains 1
# CHECK:      Relocation section '.rela.debug_info' at offset {{.*}} contains 1

# CHECK:      Symbol table '.symtab' contains 10 entries:
# CHECK-NEXT:    Num:
# CHECK-NEXT:  0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND
# CHECK-NEXT:  0000000000000000     0 SECTION LOCAL  DEFAULT [[#]] (.rodata) .rodata
# CHECK-NEXT:  0000000000000000     0 SECTION LOCAL  DEFAULT [[#]] (data) data
# CHECK-NEXT:  0000000000000000     0 SECTION LOCAL  DEFAULT [[#]] (.debug_info) .debug_info
# CHECK-NEXT:  0000000000000000     0 NOTYPE  LOCAL  DEFAULT [[#]] (.group) 11
# CHECK-NEXT:  0000000000000000     0 NOTYPE  LOCAL  DEFAULT [[#]] (.group) 22
# CHECK-NEXT:  0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND .debug_abbrev
# CHECK-NEXT:  0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND x1
# CHECK-NEXT:  0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND x2
# CHECK-NEXT:  0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND x0
