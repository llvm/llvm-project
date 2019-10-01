// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S | FileCheck %s

// Test local common construction.
// Unlike gas, common symbols are created when found, not at the end of .bss.
// In this example it causes .bss to have size 8 instead of 9.

	.local	vimvardict
	.comm	vimvardict,1,8
	.bss
        .zero 1
	.align	8

// CHECK:        Section {
// CHECK:          Name: .bss
// CHECK-NEXT:     Type:
// CHECK-NEXT:     Flags [
// CHECK:          ]
// CHECK-NEXT:     Address:
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 8
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment:
// CHECK-NEXT:     EntrySize:
// CHECK-NEXT:   }
