// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s | FileCheck %s

// Test that we don't regress on the size of the line info section. We used
// to handle negative line diffs incorrectly which manifested as very
// large integers being passed to DW_LNS_advance_line.

// FIXME: This size is the same as gnu as, but we can probably do a bit better.
// FIXME2: We need a debug_line dumper so that we can test the actual contents.

// CHECK:        Section {
// CHECK:          Index:
// CHECK:          Name: .debug_line
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 61
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:   }

	.section	.debug_line,"",@progbits
	.text
foo:
	.file 1 "Driver.ii"
	.loc 1 2 0
        nop
	.loc 1 4 0
        nop
	.loc 1 3 0
        nop
