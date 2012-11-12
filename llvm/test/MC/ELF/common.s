// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck %s


	.text

// Test that this produces a regular local symbol.
	.type	common1,@object
	.local	common1
	.comm	common1,1,1

// CHECK: ('st_name', 0x00000001) # 'common1'
// CHECK-NEXT: ('st_bind', 0x0)
// CHECK-NEXT: ('st_type', 0x1)
// CHECK-NEXT: ('st_other', 0x00)
// CHECK-NEXT: ('st_shndx',
// CHECK-NEXT: ('st_value', 0x0000000000000000)
// CHECK-NEXT: ('st_size', 0x0000000000000001)


// Same as common1, but with directives in a different order.
	.local	common2
	.type	common2,@object
	.comm	common2,1,1

// CHECK: ('st_name', 0x00000009) # 'common2'
// CHECK-NEXT: ('st_bind', 0x0)
// CHECK-NEXT: ('st_type', 0x1)
// CHECK-NEXT: ('st_other', 0x00)
// CHECK-NEXT: ('st_shndx',
// CHECK-NEXT: ('st_value', 0x0000000000000001)
// CHECK-NEXT: ('st_size', 0x0000000000000001)

        .local	common6
        .comm	common6,8,16

// CHECK:      # Symbol 3
// CHECK-NEXT: (('st_name', 0x00000011) # 'common6'
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x1)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0004)
// CHECK-NEXT:  ('st_value', 0x0000000000000010)
// CHECK-NEXT:  ('st_size', 0x0000000000000008)
// CHECK-NEXT: ),

// Test that without an explicit .local we produce a global.
	.type	common3,@object
	.comm	common3,4,4

// CHECK: ('st_name', 0x00000019) # 'common3'
// CHECK-NEXT: ('st_bind', 0x1)
// CHECK-NEXT: ('st_type', 0x1)
// CHECK-NEXT: ('st_other', 0x00)
// CHECK-NEXT: ('st_shndx', 0xfff2)
// CHECK-NEXT: ('st_value', 0x0000000000000004)
// CHECK-NEXT: ('st_size', 0x0000000000000004)


// Test that without an explicit .local we produce a global, even if the first
// occurrence is not in a directive.
	.globl	foo
	.type	foo,@function
foo:
	movsbl	common4+3(%rip), %eax


	.type	common4,@object
	.comm	common4,40,16

// CHECK: ('st_name', 0x00000025) # 'common4'
// CHECK-NEXT: ('st_bind', 0x1)
// CHECK-NEXT: ('st_type', 0x1)
// CHECK-NEXT: ('st_other', 0x00)
// CHECK-NEXT: ('st_shndx', 0xfff2)
// CHECK-NEXT: ('st_value', 0x0000000000000010)
// CHECK-NEXT: ('st_size', 0x0000000000000028)

        .comm	common5,4,4

// CHECK:      # Symbol 9
// CHECK-NEXT: (('st_name', 0x0000002d) # 'common5'
// CHECK-NEXT:  ('st_bind', 0x1)
// CHECK-NEXT:  ('st_type', 0x1)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0xfff2)
// CHECK-NEXT:  ('st_value', 0x0000000000000004)
// CHECK-NEXT:  ('st_size', 0x0000000000000004)
// CHECK-NEXT: ),
