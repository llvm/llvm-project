// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that weak symbols always produce relocations

	.weak	foo
foo:
bar:
        call    foo

//CHECK:        # Relocation 0
//CHECK-NEXT:   (('r_offset', 0x0000000000000001)
//CHECK-NEXT:    ('r_sym', 0x00000005)
//CHECK-NEXT:    ('r_type', 0x00000002)
//CHECK-NEXT:    ('r_addend', 0xfffffffffffffffc)
//CHECK-NEXT:   ),
