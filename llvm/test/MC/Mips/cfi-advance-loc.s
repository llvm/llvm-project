// RUN: llvm-mc -filetype=obj -triple mipsel-pc-Linux-gnu %s -o - | llvm-readobj  -S --sd | FileCheck %s -check-prefix=CHECK-LE
// RUN: llvm-mc -filetype=obj -triple mips-pc-linux-gnu %s -o - | llvm-readobj -S --sd | FileCheck %s -check-prefix=CHECK-BE

// test that this produces a correctly encoded cfi_advance_loc for both endians.

f:
	.cfi_startproc
	nop
	.zero 252
	// DW_CFA_advance_loc2: 256 to 00000100
	.cfi_def_cfa_offset 8
	nop
	.cfi_endproc

g:
	.cfi_startproc
	nop
	.zero 65532
	// DW_CFA_advance_loc4: 65536 to 00010104
	.cfi_def_cfa_offset 8
	nop
	.cfi_endproc

// CHECK-LE:       Section {
// CHECK-LE:        Index: 7
// CHECK-LE:        Name: .eh_frame (44)
// CHECK-LE-NEXT:   Type: SHT_PROGBITS (0x1)
// CHECK-LE-NEXT:   Flags [ (0x2)
// CHECK-LE-NEXT:     SHF_ALLOC (0x2)
// CHECK-LE-NEXT:   ]
// CHECK-LE-NEXT:   Address: 0x0
// CHECK-LE-NEXT:   Offset: 0x10180
// CHECK-LE-NEXT:   Size: 68
// CHECK-LE-NEXT:   Link: 0
// CHECK-LE-NEXT:   Info: 0
// CHECK-LE-NEXT:   AddressAlignment: 4
// CHECK-LE-NEXT:   EntrySize: 0
// CHECK-LE-NEXT:   SectionData (
// CHECK-LE-NEXT:     0000: 10000000 00000000 017A5200 017C1F01
// CHECK-LE-NEXT:     0010: 0B0C1D00 14000000 18000000 00000000
// CHECK-LE-NEXT:     0020: 04010000 00030001 0E080000 14000000
// CHECK-LE-NEXT:     0030: 30000000 04010000 04000100 00040000
// CHECK-LE-NEXT:     0040: 01000E08
// CHECK-LE-NEXT:   )
// CHECK-LE-NEXT: }

// CHECK-BE:      Section {
// CHECK-BE:        Index: 7
// CHECK-BE:        Name: .eh_frame (44)
// CHECK-BE-NEXT:   Type: SHT_PROGBITS (0x1)
// CHECK-BE-NEXT:   Flags [ (0x2)
// CHECK-BE-NEXT:     SHF_ALLOC (0x2)
// CHECK-BE-NEXT:   ]
// CHECK-BE-NEXT:   Address: 0x0
// CHECK-BE-NEXT:   Offset: 0x10180
// CHECK-BE-NEXT:   Size: 68
// CHECK-BE-NEXT:   Link: 0
// CHECK-BE-NEXT:   Info: 0
// CHECK-BE-NEXT:   AddressAlignment: 4
// CHECK-BE-NEXT:   EntrySize: 0
// CHECK-BE-NEXT:   SectionData (
// CHECK-BE-NEXT:     0000: 00000010 00000000 017A5200 017C1F01
// CHECK-BE-NEXT:     0010: 0B0C1D00 00000014 00000018 00000000
// CHECK-BE-NEXT:     0020: 00000104 00030100 0E080000 00000014
// CHECK-BE-NEXT:     0030: 00000030 00000104 00010004 00040001
// CHECK-BE-NEXT:     0040: 00000E08
// CHECK-BE-NEXT:   )
// CHECK-BE-NEXT: }
