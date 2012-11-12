// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that this produces a R_X86_64_GOT32 and that we have an undefined
// reference to _GLOBAL_OFFSET_TABLE_.

        movl	foo@GOT, %eax
        movl	foo@GOTPCREL(%rip), %eax

// CHECK:      ('_relocations', [
// CHECK-NEXT:   # Relocation 0
// CHECK-NEXT:    (('r_offset',
// CHECK-NEXT:     ('r_sym',
// CHECK-NEXT:     ('r_type', 0x00000003)
// CHECK-NEXT:     ('r_addend',
// CHECK-NEXT:    ),
// CHECK-NEXT:   # Relocation 1
// CHECK-NEXT:    (('r_offset',
// CHECK-NEXT:     ('r_sym',
// CHECK-NEXT:     ('r_type', 0x00000009)
// CHECK-NEXT:     ('r_addend',
// CHECK-NEXT:    ),
// CHECK-NEXT:   ])

// CHECK:     (('st_name', 0x00000005) # '_GLOBAL_OFFSET_TABLE_'
// CHECK-NEXT: ('st_bind', 0x1)
