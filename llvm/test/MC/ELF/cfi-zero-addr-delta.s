// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S --sr --sd - | FileCheck %s

// Test that we don't produce a DW_CFA_advance_loc 0

f:
	.cfi_startproc
        nop
	.cfi_def_cfa_offset 16
        nop
	.cfi_remember_state
	.cfi_def_cfa_offset 8
        nop
	.cfi_restore_state
        nop
	.cfi_endproc

// CHECK:        Section {
// CHECK:          Name: .eh_frame
// CHECK-NEXT:     Type: SHT_X86_64_UNWIND
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x48
// CHECK-NEXT:     Size: 56
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 8
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     Relocations [
// CHECK-NEXT:     ]
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 14000000 00000000 017A5200 01781001
// CHECK-NEXT:       0010: 1B0C0708 90010000 1C000000 1C000000
// CHECK-NEXT:       0020: 00000000 04000000 00410E10 410A0E08
// CHECK-NEXT:       0030: 410B0000 00000000
// CHECK-NEXT:     )
// CHECK-NEXT:   }

// CHECK:        Section {
// CHECK:          Name: .rela.eh_frame
// CHECK-NEXT:     Type: SHT_RELA
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_INFO_LINK
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 24
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment: 8
// CHECK-NEXT:     EntrySize: 24
// CHECK-NEXT:     Relocations [
// CHECK-NEXT:       0x20 R_X86_64_PC32 .text 0x0
// CHECK-NEXT:     ]
