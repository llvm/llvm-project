// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump   | FileCheck %s

// Test that like gnu as we create text, data and bss by default. Also test
// that shstrtab, symtab and strtab are listed in that order.

// CHECK:      ('sh_name', 0x00000001) # '.text'
// CHECK-NEXT: ('sh_type', 0x00000001)
// CHECK-NEXT: ('sh_flags', 0x0000000000000006)
// CHECK-NEXT: ('sh_addr', 0x0000000000000000)
// CHECK-NEXT: ('sh_offset', 0x0000000000000040)
// CHECK-NEXT: ('sh_size', 0x0000000000000000)
// CHECK-NEXT: ('sh_link', 0x00000000)
// CHECK-NEXT: ('sh_info', 0x00000000)
// CHECK-NEXT: ('sh_addralign', 0x0000000000000004)
// CHECK-NEXT: ('sh_entsize', 0x0000000000000000)

// CHECK:      ('sh_name', 0x00000026) # '.data'
// CHECK-NEXT: ('sh_type', 0x00000001)
// CHECK-NEXT: ('sh_flags', 0x0000000000000003)
// CHECK-NEXT: ('sh_addr', 0x0000000000000000)
// CHECK-NEXT: ('sh_offset', 0x0000000000000040)
// CHECK-NEXT: ('sh_size', 0x0000000000000000)
// CHECK-NEXT: ('sh_link', 0x00000000)
// CHECK-NEXT: ('sh_info', 0x00000000)
// CHECK-NEXT: ('sh_addralign', 0x0000000000000004)
// CHECK-NEXT: ('sh_entsize', 0x0000000000000000)

// CHECK:      ('sh_name', 0x00000007) # '.bss'
// CHECK-NEXT: ('sh_type', 0x00000008)
// CHECK-NEXT: ('sh_flags', 0x0000000000000003)
// CHECK-NEXT: ('sh_addr', 0x0000000000000000)
// CHECK-NEXT: ('sh_offset', 0x0000000000000040)
// CHECK-NEXT: ('sh_size', 0x0000000000000000)
// CHECK-NEXT: ('sh_link', 0x00000000)
// CHECK-NEXT: ('sh_info', 0x00000000)
// CHECK-NEXT: ('sh_addralign', 0x0000000000000004)
// CHECK-NEXT: ('sh_entsize', 0x0000000000000000)

// CHECK:      ('sh_name', 0x0000000c) # '.shstrtab'
// CHECK-NEXT: ('sh_type', 0x00000003)
// CHECK-NEXT:    ('sh_flags', 0x0000000000000000)
// CHECK-NEXT:    ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:    ('sh_offset', 0x0000000000000040)
// CHECK-NEXT:    ('sh_size', 0x000000000000002c)
// CHECK-NEXT:    ('sh_link', 0x00000000)
// CHECK-NEXT:    ('sh_info', 0x00000000)
// CHECK-NEXT:    ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT:    ('sh_entsize', 0x0000000000000000)

// CHECK: ('sh_name', 0x0000001e) # '.symtab'
// CHECK-NEXT:    ('sh_type', 0x00000002)
// CHECK-NEXT:    ('sh_flags', 0x0000000000000000)
// CHECK-NEXT:    ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:    ('sh_offset',
// CHECK-NEXT:    ('sh_size', 0x0000000000000060)
// CHECK-NEXT:    ('sh_link', 0x00000006)
// CHECK-NEXT:    ('sh_info', 0x00000004)
// CHECK-NEXT:    ('sh_addralign', 0x0000000000000008)
// CHECK-NEXT:    ('sh_entsize', 0x0000000000000018)

// CHECK: ('sh_name', 0x00000016) # '.strtab'
// CHECK-NEXT:    ('sh_type', 0x00000003)
// CHECK-NEXT:    ('sh_flags', 0x0000000000000000)
// CHECK-NEXT:    ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:    ('sh_offset',
// CHECK-NEXT:    ('sh_size', 0x0000000000000001)
// CHECK-NEXT:    ('sh_link', 0x00000000)
// CHECK-NEXT:    ('sh_info', 0x00000000)
// CHECK-NEXT:    ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT:    ('sh_entsize', 0x0000000000000000)
