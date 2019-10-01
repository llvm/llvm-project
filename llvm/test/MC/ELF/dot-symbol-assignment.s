// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj --sections --section-data | FileCheck %s

one:
	.quad 0xffffffffffffffff

. = . + 16
two:
	.quad 0xeeeeeeeeeeeeeeee

. = 0x20
three:
	.quad 0xdddddddddddddddd

        .align 4
        . = three + 9

// CHECK:        Section {
// CHECK:          Name: .text
// CHECK-NEXT:     Type:
// CHECK-NEXT:     Flags [
// CHECK:          SectionData (
// CHECK-NEXT:     0000: FFFFFFFF FFFFFFFF 00000000 00000000
// CHECK-NEXT:     0010: 00000000 00000000 EEEEEEEE EEEEEEEE
// CHECK-NEXT:     0020: DDDDDDDD DDDDDDDD 00 |
// CHECK-NEXT:     )
