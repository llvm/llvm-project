// RUN: llvm-mc -filetype=asm -triple x86_64-pc-linux-gnu %s -o - | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S --sr --sd | FileCheck --check-prefix=READOBJ %s

f:
	.cfi_startproc
	nop
	.cfi_llvm_def_aspace_cfa %rcx, 0, 6
	nop
	.cfi_endproc

// ASM: f:
// ASM-NEXT: .cfi_startproc
// ASM-NEXT: nop
// ASM-NEXT: .cfi_llvm_def_aspace_cfa %rcx, 0, 6
// FIXME Why emit an extra empty line?
// ASM-EMPTY:
// ASM-NEXT: nop
// ASM-NEXT: .cfi_endproc

// READOBJ:        Section {
// READOBJ:          Name: .eh_frame
// READOBJ-NEXT:     Type: SHT_X86_64_UNWIND
// READOBJ-NEXT:     Flags [
// READOBJ-NEXT:       SHF_ALLOC
// READOBJ-NEXT:     ]
// READOBJ-NEXT:     Address: 0x0
// READOBJ-NEXT:     Offset: 0x48
// READOBJ-NEXT:     Size: 48
// READOBJ-NEXT:     Link: 0
// READOBJ-NEXT:     Info: 0
// READOBJ-NEXT:     AddressAlignment: 8
// READOBJ-NEXT:     EntrySize: 0
// READOBJ-NEXT:     Relocations [
// READOBJ-NEXT:     ]
// READOBJ-NEXT:     SectionData (
// READOBJ-NEXT:       0000: 14000000 00000000 017A5200 01781001
// READOBJ-NEXT:       0010: 1B0C0708 90010000 14000000 1C000000
// READOBJ-NEXT:       0020: 00000000 02000000 00413002 00060000
// READOBJ-NEXT:     )
// READOBJ-NEXT:   }

// READOBJ:        Section {
// READOBJ:          Name: .rela.eh_frame
// READOBJ-NEXT:     Type: SHT_RELA
// READOBJ-NEXT:     Flags [
// READOBJ-NEXT:     ]
// READOBJ-NEXT:     Address: 0x0
// READOBJ-NEXT:     Offset:
// READOBJ-NEXT:     Size: 24
// READOBJ-NEXT:     Link:
// READOBJ-NEXT:     Info:
// READOBJ-NEXT:     AddressAlignment: 8
// READOBJ-NEXT:     EntrySize: 24
// READOBJ-NEXT:     Relocations [
// READOBJ-NEXT:       0x20 R_X86_64_PC32 .text 0x0
// READOBJ-NEXT:     ]
// READOBJ:        }
