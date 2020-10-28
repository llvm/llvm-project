// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S --sr --sd - | FileCheck %s

f:
	.cfi_startproc
        nop
	.cfi_register %rbp, %rax
        nop
	.cfi_endproc

// CHECK:        Section {
// CHECK:          Index:
// CHECK:          Name: .eh_frame
// CHECK-NEXT:     Type: SHT_X86_64_UNWIND
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x0
// CHECK-NEXT:     Offset: 0x48
// CHECK-NEXT:     Size: 48
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 8
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     Relocations [
// CHECK-NEXT:     ]
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:       0000: 14000000 00000000 017A5200 01781001
// CHECK-NEXT:       0010: 1B0C0708 90010000 14000000 1C000000
// CHECK-NEXT:       0020: 00000000 02000000 00410906 00000000
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
// CHECK-NEXT:     Size: 24
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment: 8
// CHECK-NEXT:     EntrySize: 24
// CHECK-NEXT:     Relocations [
// CHECK-NEXT:       0x20 R_X86_64_PC32 .text 0x0
// CHECK-NEXT:     ]
// CHECK:        }
