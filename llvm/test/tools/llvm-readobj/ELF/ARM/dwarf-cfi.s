# RUN: llvm-mc -triple arm-linux -filetype obj -o - %s | llvm-readobj -u - | FileCheck %s

# CHECK:      [0x0] CIE length=16
# CHECK-NEXT:   version: 1
# CHECK-NEXT:   augmentation: zR
# CHECK-NEXT:   code_alignment_factor: 1
# CHECK-NEXT:   data_alignment_factor: -4
# CHECK-NEXT:   return_address_register: 14

# CHECK:        Program:
# CHECK-NEXT: DW_CFA_def_cfa: reg13 +0

## FIXME Use getEHFrameSection() so that the address is decoded correctly.
# CHECK:      [0x14] FDE length=16 cie=[0x0]
# CHECK-NEXT:   initial_location: 0x1c
# CHECK-NEXT:   address_range: 0x4 (end : 0x20)

# CHECK:        Program:
# CHECK-NEXT: DW_CFA_nop:
# CHECK-NEXT: DW_CFA_nop:
# CHECK-NEXT: DW_CFA_nop:

.cpu cortex-a8

foo:
.cfi_startproc
bx lr
.cfi_endproc
