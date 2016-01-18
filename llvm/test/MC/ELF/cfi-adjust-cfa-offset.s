// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -s -sr -sd | FileCheck %s

f:
	.cfi_startproc
	subq	$8, %rsp
	.cfi_def_cfa_offset 16
        nop
        .cfi_adjust_cfa_offset 4
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc

        .cfi_startproc
	nop
	.cfi_adjust_cfa_offset 4
	.cfi_endproc

        .cfi_startproc
	nop
	.cfi_adjust_cfa_offset 4
	.cfi_endproc

// CHECK:        Section {
// CHECK:          Index:
// CHECK:          Name: .eh_frame
// CHECK-NEXT:     Type: SHT_PROGBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x50
// CHECK-NEXT:     Size: 96
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 8
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     Relocations [
// CHECK-NEXT:     ]
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 14000000 00000000 017A5200 01781001
// CHECK-NEXT:       0010: 1B0C0708 90010000 18000000 1C000000
// CHECK-NEXT:       0020: 00000000 0A000000 00440E10 410E1444
// CHECK-NEXT:       0030: 0E080000 10000000 38000000 00000000
// CHECK-NEXT:       0040: 01000000 00410E0C 14000000 4C000000
// CHECK-NEXT:       0050: 00000000 01000000 00410E0C 00000000
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index:
// CHECK-NEXT:     Name: .rela.eh_frame
// CHECK-NEXT:     Type: SHT_RELA
// CHECK-NEXT:     Flags [
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 72
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment: 8
// CHECK-NEXT:     EntrySize: 24
// CHECK-NEXT:     Relocations [
// CHECK-NEXT:       0x20 R_X86_64_PC32 .text 0x0
// CHECK-NEXT:       0x3C R_X86_64_PC32 .text 0x
// CHECK-NEXT:       0x50 R_X86_64_PC32 .text 0x
// CHECK-NEXT:     ]
// CHECK:        }
