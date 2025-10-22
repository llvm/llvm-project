# RUN: llvm-mc -triple amdgcn-amd-amdhsa %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple amdgcn-amd-amdhsa %s | llvm-readobj -S --sr --sd - | FileCheck %s

# REQUIRES: amdgpu-registered-target

# ASM: .cfi_llvm_register_pair 16, 62, 32, 63, 32
# ASM-NEXT: s_nop 0

f:
  .cfi_startproc
  s_nop 0
  .cfi_llvm_register_pair 16, 62, 32, 63, 32
  s_nop 0
  .cfi_endproc

// CHECK:        Section {
// CHECK:          Index:
// CHECK:          Name: .eh_frame
// CHECK-NEXT:     Type: SHT_PROGBITS
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
// CHECK-NEXT:       0000: 10000000 00000000 017A5200 04041001  |.........zR.....|
// CHECK-NEXT:       0010: 1B000000 20000000 18000000 00000000  |.... ...........|
// CHECK-NEXT:       0020: 08000000 00411010 08903E93 04903F93  |.....A....>...?.|
// CHECK-NEXT:       0030: 04000000 00000000                    |........|
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Index:
// CHECK-NEXT:     Name: .rela.eh_frame
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
// CHECK-NEXT:       0x1C R_AMDGPU_REL32 .text
// CHECK-NEXT:     ]
// CHECK:        }
