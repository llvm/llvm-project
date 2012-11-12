// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that these names are accepted.

.section	.note.GNU-stack,"",@progbits
.section	.note.GNU-stack2,"",%progbits
.section	.note.GNU-,"",@progbits
.section	-.note.GNU,"",@progbits

// CHECK: ('sh_name', 0x00000038) # '.note.GNU-stack'
// CHECK: ('sh_name', 0x0000008f) # '.note.GNU-stack2'
// CHECK: ('sh_name', 0x000000a0) # '.note.GNU-'
// CHECK: ('sh_name', 0x00000084) # '-.note.GNU'

// Test that the defaults are used

.section	.init
.section	.fini
.section	.rodata
.section	zed, ""

// CHECK:      (('sh_name', 0x00000012) # '.init'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x0000000000000006)
// CHECK-NEXT:  ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:  ('sh_offset', 0x0000000000000050)
// CHECK-NEXT:  ('sh_size', 0x0000000000000000)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT:  ('sh_entsize', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Section 11
// CHECK-NEXT: (('sh_name', 0x00000048) # '.fini'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x0000000000000006)
// CHECK-NEXT:  ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:  ('sh_offset', 0x0000000000000050)
// CHECK-NEXT:  ('sh_size', 0x0000000000000000)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT:  ('sh_entsize', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Section 12
// CHECK-NEXT: (('sh_name', 0x00000076) # '.rodata'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x0000000000000002)
// CHECK-NEXT:  ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:  ('sh_offset', 0x0000000000000050)
// CHECK-NEXT:  ('sh_size', 0x0000000000000000)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT:  ('sh_entsize', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Section 13
// CHECK-NEXT: (('sh_name', 0x00000058) # 'zed'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x0000000000000000)
// CHECK-NEXT:  ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:  ('sh_offset', 0x0000000000000050)
// CHECK-NEXT:  ('sh_size', 0x0000000000000000)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT:  ('sh_entsize', 0x0000000000000000)
// CHECK-NEXT: ),

.section	.note.test,"",@note
// CHECK:       (('sh_name', 0x00000007) # '.note.test'
// CHECK-NEXT:   ('sh_type', 0x00000007)
// CHECK-NEXT:   ('sh_flags', 0x0000000000000000)
// CHECK-NEXT:   ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:   ('sh_offset', 0x0000000000000050)
// CHECK-NEXT:   ('sh_size', 0x0000000000000000)
// CHECK-NEXT:   ('sh_link', 0x00000000)
// CHECK-NEXT:   ('sh_info', 0x00000000)
// CHECK-NEXT:   ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT:   ('sh_entsize', 0x0000000000000000)
// CHECK-NEXT:  ),

// Test that we can parse these
foo:
bar:
.section        .text.foo,"axG",@progbits,foo,comdat
.section        .text.bar,"axMG",@progbits,42,bar,comdat

// Test that the default values are not used

.section .eh_frame,"a",@unwind

// CHECK:       (('sh_name', 0x0000004e) # '.eh_frame'
// CHECK-NEXT:   ('sh_type', 0x70000001)
// CHECK-NEXT:   ('sh_flags', 0x0000000000000002)
// CHECK-NEXT:   ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:   ('sh_offset', 0x0000000000000050)
// CHECK-NEXT:   ('sh_size', 0x0000000000000000)
// CHECK-NEXT:   ('sh_link', 0x00000000)
// CHECK-NEXT:   ('sh_info', 0x00000000)
// CHECK-NEXT:   ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT:   ('sh_entsize', 0x0000000000000000)
// CHECK-NEXT:  ),

// Test that we handle the strings like gas
.section bar-"foo"
.section "foo"

// CHECK: ('sh_name', 0x000000ab) # 'bar-"foo"'
// CHECK: ('sh_name', 0x00000034) # 'foo'
