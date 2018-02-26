// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -sd | FileCheck %s

// Test that two subsequent .loc directives generate two
// distinct line table entries.

// CHECK:        Section {
// CHECK:          Name: .debug_line
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 57
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 35000000 04001D00 00000101 01FB0E0D  |5...............|
// CHECK-NEXT:       0010: 00010101 01000000 01000001 00666F6F  |.............foo|
// CHECK-NEXT:       0020: 2E630000 00000000 09020000 00000000  |.c..............|
// CHECK-NEXT:       0030: 00000113 02010001 01                 |.........|
// CHECK-NEXT:     )
// CHECK-NEXT:   }

	.section	.debug_line,"",@progbits
	.text

	.file 1 "foo.c"
	.loc 1 1 0
	.loc 1 2 0
	nop
