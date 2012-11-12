// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// CHECK:       ('_relocations', [
// CHECK-NEXT:    # Relocation 0
// CHECK-NEXT:    (('r_offset', 0x000000000000000c)
// CHECK-NEXT:     ('r_sym', 0x00000005)
// CHECK-NEXT:     ('r_type', 0x00000002)
// CHECK-NEXT:     ('r_addend', 0x0000000000000008)
// CHECK-NEXT:    ),
// CHECK-NEXT:   ])

// CHECK:         # Symbol 5
// CHECK-NEXT:    (('st_name', 0x00000005) # 'baz'
// CHECK-NEXT:     ('st_bind', 0x1)
// CHECK-NEXT:     ('st_type', 0x0)
// CHECK-NEXT:     ('st_other', 0x00)
// CHECK-NEXT:     ('st_shndx', 0x0000)
// CHECK-NEXT:     ('st_value', 0x0000000000000000)
// CHECK-NEXT:     ('st_size', 0x0000000000000000)
// CHECK-NEXT:    ),

.zero 4
.data

.zero 1
.align 4
foo:
.zero 8
.long baz - foo
