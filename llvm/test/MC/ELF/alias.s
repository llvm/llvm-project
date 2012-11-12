// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

foo:
bar = foo

        .globl	foo2
foo2 = bar2

foo3:
	.globl	bar3
bar3 = foo3

// Test that bar4 is also a function
        .type	foo4,@function
foo4:
bar4 = foo4

        .long foo2
// CHECK:       # Symbol 1
// CHECK-NEXT:  (('st_name', 0x00000005) # 'bar'
// CHECK-NEXT:   ('st_bind', 0x0)
// CHECK-NEXT:   ('st_type', 0x0)
// CHECK-NEXT:   ('st_other', 0x00)
// CHECK-NEXT:   ('st_shndx', 0x0001)
// CHECK-NEXT:   ('st_value', 0x0000000000000000)
// CHECK-NEXT:   ('st_size', 0x0000000000000000)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Symbol 2
// CHECK-NEXT: (('st_name', 0x0000001d) # 'bar4'
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x2)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT:  # Symbol 3
// CHECK-NEXT:  (('st_name', 0x00000001) # 'foo'
// CHECK-NEXT:   ('st_bind', 0x0)
// CHECK-NEXT:   ('st_type', 0x0)
// CHECK-NEXT:   ('st_other', 0x00)
// CHECK-NEXT:   ('st_shndx', 0x0001)
// CHECK-NEXT:   ('st_value', 0x0000000000000000)
// CHECK-NEXT:   ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT:  # Symbol 4
// CHECK-NEXT:  (('st_name', 0x0000000e) # 'foo3'
// CHECK-NEXT:   ('st_bind', 0x0)
// CHECK-NEXT:   ('st_type', 0x0)
// CHECK-NEXT:   ('st_other', 0x00)
// CHECK-NEXT:   ('st_shndx', 0x0001)
// CHECK-NEXT:   ('st_value', 0x0000000000000000)
// CHECK-NEXT:   ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 5
// CHECK-NEXT: (('st_name', 0x00000018) # 'foo4'
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x2)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0001)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 6
// CHECK-NEXT: (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 7
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 8
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK:       # Symbol 9
// CHECK-NEXT:  (('st_name', 0x00000013) # 'bar3'
// CHECK-NEXT:   ('st_bind', 0x1)
// CHECK-NEXT:   ('st_type', 0x0)
// CHECK-NEXT:   ('st_other', 0x00)
// CHECK-NEXT:   ('st_shndx', 0x0001)
// CHECK-NEXT:   ('st_value', 0x0000000000000000)
// CHECK-NEXT:   ('st_size', 0x0000000000000000)
// CHECK:       # Symbol 10
// CHECK-NEXT:  (('st_name', 0x00000009) # 'bar2'
// CHECK-NEXT:   ('st_bind', 0x1)
// CHECK-NEXT:   ('st_type', 0x0)
// CHECK-NEXT:   ('st_other', 0x00)
// CHECK-NEXT:   ('st_shndx', 0x0000)
// CHECK-NEXT:   ('st_value', 0x0000000000000000)
// CHECK-NEXT:   ('st_size', 0x0000000000000000)
