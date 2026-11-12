// Test the bits of .eh_frame on CSKY that are already implemented correctly.

# RUN: llvm-mc -filetype=obj -triple=csky %s | llvm-dwarfdump -eh-frame - \
# RUN:    | FileCheck  %s

func:
  .cfi_startproc
  jmp16 r15
  .cfi_endproc

# CHECK: 00000000 00000010 00000000 CIE
# CHECK:   Version:               1
# CHECK:   Augmentation:          "zR"
# CHECK:   Code alignment factor: 1
# CHECK:   Data alignment factor: -4
# CHECK:   Return address column: 15
# CHECK:   Augmentation data:     1B
# CHECK:   DW_CFA_def_cfa: R14 +0

# CHECK:   CFA=R14
#
# CHECK: 00000014 00000010 00000018 FDE cie=00000000 pc=00000000...00000002
# CHECK:   DW_CFA_nop:
# CHECK:   DW_CFA_nop:
# CHECK:   DW_CFA_nop:
# CHECK:   0x0: CFA=R14
