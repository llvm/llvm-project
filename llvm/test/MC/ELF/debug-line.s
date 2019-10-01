// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S --sd | FileCheck %s

// Test that .debug_line is populated.  TODO: This test should really be using
// llvm-dwarfdump, but it cannot parse this particular object file.  The content
// of .debug_line was checked using GNU binutils:

// $ objdump --dwarf=decodedline debug-line.o
// [...]
// File name                            Line number    Starting address
// foo.c                                          4                   0
// foo.c                                          5                 0x4
// foo.c                                          6                 0x5

// CHECK:        Section {
// CHECK:          Name: .debug_line
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 58
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 1
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     SectionData (

// CHECK-NEXT:       0000: 36000000 04001D00 00000101 01FB0E0D
// CHECK-NEXT:       0010: 00010101 01000000 01000001 00666F6F  
// CHECK-NEXT:       0020: 2E630000 00000000 09020000 00000000  
// CHECK-NEXT:       0030: 0000154B 21020800 0101              
// CHECK-NEXT:     )
// CHECK-NEXT:   }

	.section	.debug_line,"",@progbits
	.text

	.file 1 "foo.c"
	.loc 1 4 0
	subq	$8, %rsp

// Test that .loc works with values, not just instructions.

	.loc 1 5 0
	.byte 0xc3

	.loc 1 6 0
l:
	.quad l
