# RUN: llvm-mc -triple arm64-linux -filetype obj -o - %s | llvm-readobj -u | FileCheck %s

# CHECK:      [0x0] CIE length=16
# CHECK-NEXT:   version: 1
# CHECK-NEXT:   augmentation: zR
# CHECK-NEXT:   code_alignment_factor: 1
# CHECK-NEXT:   data_alignment_factor: -4
# CHECK-NEXT:   return_address_register: 30

# CHECK:        Program:
# CHECK-NEXT: DW_CFA_def_cfa: reg31 +0

# CHECK:      [0x14] FDE length=16 cie=[0x0]
# CHECK-NEXT:   initial_location: 0x0
# CHECK-NEXT:   address_range: 0x4 (end : 0x4)

# CHECK:        Program:
# CHECK-NEXT: DW_CFA_nop:
# CHECK-NEXT: DW_CFA_nop:
# CHECK-NEXT: DW_CFA_nop:

foo:
.cfi_startproc
ret
.cfi_endproc
