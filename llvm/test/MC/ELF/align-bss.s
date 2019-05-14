// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S | FileCheck %s

// Test that the bss section is correctly aligned

	.local	foo
	.comm	foo,2048,16

// CHECK:        Section {
// CHECK:          Name: .bss
// CHECK-NEXT:     Type: SHT_NOBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x40
// CHECK-NEXT:     Size: 2048
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 16
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }
