// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj --symbols | FileCheck %s


	.text

// Test that this produces a regular local symbol.
	.type	common1,@object
	.local	common1
	.comm	common1,1,1

// CHECK:        Symbol {
// CHECK:          Name: common1
// CHECK-NEXT:     Value: 0x0
// CHECK-NEXT:     Size: 1
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section:
// CHECK-NEXT:   }


// Same as common1, but with directives in a different order.
	.local	common2
	.type	common2,@object
	.comm	common2,1,1

// CHECK:        Symbol {
// CHECK:          Name: common2
// CHECK-NEXT:     Value: 0x1
// CHECK-NEXT:     Size: 1
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section:
// CHECK-NEXT:   }


        .local	common6
        .comm	common6,8,16

// CHECK:        Symbol {
// CHECK:          Name: common6
// CHECK-NEXT:     Value: 0x10
// CHECK-NEXT:     Size: 8
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: .bss
// CHECK-NEXT:   }


// Test that without an explicit .local we produce a global.
	.type	common3,@object
	.comm	common3,4,4

// CHECK:        Symbol {
// CHECK:          Name: common3
// CHECK-NEXT:     Value: 0x4
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Common (0xFFF2)
// CHECK-NEXT:   }


// Test that without an explicit .local we produce a global, even if the first
// occurrence is not in a directive.
	.globl	foo
	.type	foo,@function
foo:
	movsbl	common4+3(%rip), %eax


	.type	common4,@object
	.comm	common4,40,16

// CHECK:        Symbol {
// CHECK:          Name: common4
// CHECK-NEXT:     Value: 0x10
// CHECK-NEXT:     Size: 40
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Common (0xFFF2)
// CHECK-NEXT:   }


        .comm	common5,4,4

// CHECK:        Symbol {
// CHECK:          Name: common5
// CHECK-NEXT:     Value: 0x4
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other: 0
// CHECK-NEXT:     Section: Common (0xFFF2)
// CHECK-NEXT:   }
