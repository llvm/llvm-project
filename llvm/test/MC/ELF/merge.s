// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

// Test that PIC relocations with local symbols in a mergeable section are done
// with a reference to the symbol. Not sure if this is a linker limitation,
// but this matches the behavior of gas.

// Non-PIC relocations with 0 offset don't use the symbol.


        movsd   .Lfoo(%rip), %xmm1
        movl	$.Lfoo, %edi
        movl	$.Lfoo+2, %edi
        jmp	foo@PLT
        movq 	foo@GOTPCREL, %rax
        movq    zed, %rax

        .section        .sec1,"aM",@progbits,16
.Lfoo:
zed:
        .global zed

        .section	bar,"ax",@progbits
foo:

// Relocation 0 refers to symbol 1
// CHECK:       ('_relocations', [
// CHECK-NEXT:   # Relocation 0
// CHECK-NEXT:   (('r_offset',
// CHECK-NEXT:    ('r_sym', 0x00000001)
// CHECK-NEXT:    ('r_type', 0x00000002
// CHECK-NEXT:    ('r_addend',
// CHECK-NEXT:   ),

// Relocation 1 refers to symbol 6
// CHECK-NEXT:  # Relocation 1
// CHECK-NEXT: (('r_offset',
// CHECK-NEXT:  ('r_sym', 0x00000006)
// CHECK-NEXT:  ('r_type', 0x0000000a)
// CHECK-NEXT:  ('r_addend',
// CHECK-NEXT: ),

// Relocation 2 refers to symbol 1
// CHECK-NEXT:   # Relocation 2
// CHECK-NEXT:   (('r_offset',
// CHECK-NEXT:    ('r_sym', 0x00000001)
// CHECK-NEXT:    ('r_type', 0x0000000a
// CHECK-NEXT:    ('r_addend',
// CHECK-NEXT:   ),

// Relocation 3 refers to symbol 2
// CHECK-NEXT:   # Relocation 3
// CHECK-NEXT:   (('r_offset',
// CHECK-NEXT:    ('r_sym', 0x00000002)
// CHECK-NEXT:    ('r_type', 0x00000004
// CHECK-NEXT:    ('r_addend',
// CHECK-NEXT:   ),

// Relocation 4 refers to symbol 2
// CHECK-NEXT:   # Relocation 4
// CHECK-NEXT:   (('r_offset',
// CHECK-NEXT:    ('r_sym', 0x00000002)
// CHECK-NEXT:    ('r_type', 0x00000009
// CHECK-NEXT:    ('r_addend',
// CHECK-NEXT:   ),

// Relocation 5 refers to symbol 8
// CHECK-NEXT:   # Relocation 5
// CHECK-NEXT:   (('r_offset', 0x0000000000000023)
// CHECK-NEXT:    ('r_sym', 0x00000008)
// CHECK-NEXT:    ('r_type', 0x0000000b)
// CHECK-NEXT:    ('r_addend', 0x0000000000000000)
// CHECK-NEXT:   ),
// CHECK-NEXT:  ])

// Section 5 is "sec1"
// CHECK: # Section 5
// CHECK-NEXT:  (('sh_name', 0x00000035) # '.sec1'

// Symbol number 1 is .Lfoo
// CHECK:      # Symbol 1
// CHECK-NEXT: (('st_name', 0x00000001) # '.Lfoo'

// Symbol number 2 is foo
// CHECK:      # Symbol 2
// CHECK-NEXT: (('st_name', 0x00000007) # 'foo'

// Symbol number 6 is section 5
// CHECK:        # Symbol 6
// CHECK-NEXT:    (('st_name', 0x00000000) # ''
// CHECK-NEXT:     ('st_bind', 0x0)
// CHECK-NEXT:     ('st_type', 0x3)
// CHECK-NEXT:     ('st_other', 0x00)
// CHECK-NEXT:     ('st_shndx', 0x0005)

// Symbol number 8 is zed
// CHECK:        # Symbol 8
// CHECK-NEXT:    (('st_name', 0x0000000b) # 'zed'
