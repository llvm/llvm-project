// RUN: llvm-mc -filetype=asm -mcpu=gfx900 -triple amdgcn-amd-amdhsa %s -o - | FileCheck --check-prefix=ASM %s
// RUN: llvm-mc -filetype=obj -mcpu=gfx900 -triple amdgcn-amd-amdhsa %s -o - | llvm-readobj -S --sr --sd | FileCheck --check-prefix=READOBJ %s

f:
	.cfi_sections .debug_frame
	.cfi_startproc
	s_nop 0
	.cfi_endproc

// ASM: f:
// ASM-NEXT: .cfi_sections .debug_frame
// FIXME Why emit an extra empty line?
// ASM-EMPTY:
// ASM-NEXT: .cfi_startproc
// ASM-NEXT: s_nop 0
// FIXME Why emit an extra empty line?
// ASM-EMPTY:
// ASM-NEXT: .cfi_endproc

// READOBJ:        Section {
// READOBJ:          Name: .debug_frame
// READOBJ-NEXT:     Type: SHT_PROGBITS
// READOBJ-NEXT:     Flags [
// READOBJ-NEXT:     ]
// READOBJ-NEXT:     Address: 0x0
// READOBJ-NEXT:     Offset: 0x48
// READOBJ-NEXT:     Size: 56
// READOBJ-NEXT:     Link: 0
// READOBJ-NEXT:     Info: 0
// READOBJ-NEXT:     AddressAlignment: 8
// READOBJ-NEXT:     EntrySize: 0
// READOBJ-NEXT:     Relocations [
// READOBJ-NEXT:     ]
// READOBJ-NEXT:     SectionData (
// READOBJ-NEXT:       0000: 1C000000 FFFFFFFF 045B6C6C 766D3A76  |.........[llvm:v|
// READOBJ-NEXT:       0010: 302E305D 00080004 04100000 00000000  |0.0]............|
// READOBJ-NEXT:       0020: 14000000 00000000 00000000 00000000  |................|
// READOBJ-NEXT:       0030: 04000000 00000000                    |........|
// READOBJ-NEXT:     )
// READOBJ-NEXT:   }

// READOBJ:        Section {
// READOBJ:          Name: .rela.debug_frame
// READOBJ-NEXT:     Type: SHT_RELA
// READOBJ-NEXT:     Flags [
// READOBJ-NEXT:     ]
// READOBJ-NEXT:     Address: 0x0
// READOBJ-NEXT:     Offset:
// READOBJ-NEXT:     Size: 48
// READOBJ-NEXT:     Link:
// READOBJ-NEXT:     Info:
// READOBJ-NEXT:     AddressAlignment: 8
// READOBJ-NEXT:     EntrySize: 24
// READOBJ-NEXT:     Relocations [
// READOBJ-NEXT:       0x24 R_AMDGPU_ABS32 .debug_frame 0x0
// READOBJ-NEXT:       0x28 R_AMDGPU_ABS64 .text 0x0
// READOBJ-NEXT:     ]
// READOBJ:        }
