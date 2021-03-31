# RUN: llvm-mc -filetype=obj -triple=x86_64 %s | llvm-readelf -s - | FileCheck %s

// Test which symbols should be in the symbol table

# CHECK:      Symbol table '.symtab' contains 5 entries:
# CHECK-NEXT: Num:    Value          Size Type    Bind   Vis       Ndx Name
# CHECK-NEXT:   0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT   UND 
# CHECK-NEXT:   1: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT     4 .Lsym8
# CHECK-NEXT:   2: 0000000000000000     0 OBJECT  GLOBAL DEFAULT   UND sym6
# CHECK-NEXT:   3: 0000000000000000     0 NOTYPE  GLOBAL HIDDEN    UND hidden
# CHECK-NEXT:   4: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND undef

.Lsym2:
.Lsym3:
.Lsym4 = .Lsym2 - .Lsym3
        .long	.Lsym4

	.type	.Lsym5,@object
        .type   sym6,@object
        .long sym6

	.section	.rodata.str1.1,"aMS",@progbits,1
.Lsym7:
.Lsym8:

        .text
        movsd   .Lsym8(%rip), %xmm1

.hidden hidden

test2_a = undef
test2_b = undef + 1
