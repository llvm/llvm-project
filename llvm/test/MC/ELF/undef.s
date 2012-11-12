// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test which symbols should be in the symbol table

        .long	.Lsym1
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

// CHECK:      ('_symbols', [
// CHECK-NEXT:  # Symbol 0
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 1
// CHECK-NEXT:  (('st_name', 0x0000000d) # '.Lsym8'
// CHECK:       # Symbol 2
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 3
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 4
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 5
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 6
// CHECK-NEXT:  (('st_name', 0x00000001) # '.Lsym1'
// CHECK:       # Symbol 7
// CHECK-NEXT:  (('st_name', 0x00000008) # 'sym6'
// CHECK-NEXT:   ('st_bind', 0x1)
// CHECK-NEXT:   ('st_type', 0x1)
// CHECK-NEXT:   ('st_other', 0x00)
// CHECK-NEXT:   ('st_shndx', 0x0000)
// CHECK-NEXT:   ('st_value', 0x0000000000000000)
// CHECK-NEXT:   ('st_size', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:  ])
