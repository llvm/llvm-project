// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that this produces a R_X86_64_PLT32 with bar.

        .globl foo
foo:
bar = foo
        .section zed, "", @progbits
        call bar@PLT


// Test that this produres a relocation with bar2

    .weak    foo2
foo2:
    .weak    bar2
    .set    bar2,foo2
    .quad    bar2

// CHECK:       # Relocation 0
// CHECK-NEXT:  (('r_offset', 0x0000000000000001)
// CHECK-NEXT:   ('r_sym', 0x00000001)
// CHECK-NEXT:   ('r_type', 0x00000004)
// CHECK-NEXT:   ('r_addend', 0xfffffffffffffffc)
// CHECK-NEXT:  ),

// CHECK:      # Relocation 1
// CHECK-NEXT: (('r_offset', 0x0000000000000005)
// CHECK-NEXT:  ('r_sym', 0x00000006)
// CHECK-NEXT:  ('r_type', 0x00000001)
// CHECK-NEXT:  ('r_addend', 0x0000000000000000)
// CHECK-NEXT: ),

// CHECK:       # Symbol 1
// CHECK-NEXT:  (('st_name', 0x00000005) # 'bar'
// CHECK-NEXT:   ('st_bind', 0x0)
// CHECK-NEXT:   ('st_type', 0x0)
// CHECK-NEXT:   ('st_other', 0x00)
// CHECK-NEXT:   ('st_shndx', 0x0001)
// CHECK-NEXT:   ('st_value', 0x0000000000000000)
// CHECK-NEXT:   ('st_size', 0x0000000000000000)
// CHECK-NEXT:  ),

// CHECK:      # Symbol 6
// CHECK-NEXT: (('st_name', 0x0000000e) # 'bar2'
// CHECK-NEXT:  ('st_bind', 0x2)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0004)
// CHECK-NEXT:  ('st_value', 0x0000000000000005)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
