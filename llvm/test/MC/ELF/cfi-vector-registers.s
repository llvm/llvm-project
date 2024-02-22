# RUN: llvm-mc -triple amdgcn-amd-amdhsa %s | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -filetype=obj -triple amdgcn-amd-amdhsa %s | llvm-readobj -S --sr --sd - | FileCheck %s

# REQUIRES: amdgpu-registered-target

# ASM: .cfi_llvm_vector_registers 16, 1663, 0, 32, 1663, 1, 32
# ASM-NEXT: s_nop 0

f:
  .cfi_startproc
  s_nop 0
  .cfi_llvm_vector_registers 16, 1663, 0, 32, 1663, 1, 32
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
// CHECK-NEXT:     Size: 64
// CHECK-NEXT:     Link: 0
// CHECK-NEXT:     Info: 0
// CHECK-NEXT:     AddressAlignment: 8
// CHECK-NEXT:     EntrySize: 0
// CHECK-NEXT:     Relocations [
// CHECK-NEXT:     ]
// CHECK-NEXT:     SectionData (
// CHECK-NEXT:        0000: 18000000 00000000 017A525B 6C6C766D
// CHECK-NEXT:        0010: 3A76302E 305D0004 0410011B 20000000
// CHECK-NEXT:        0020: 20000000 00000000 08000000 00411010
// CHECK-NEXT:        0030: 0C90FF0C 9D200090 FF0C9D20 20000000
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
// CHECK-NEXT:       0x24 R_AMDGPU_REL32 .text
// CHECK-NEXT:     ]
// CHECK:        }
