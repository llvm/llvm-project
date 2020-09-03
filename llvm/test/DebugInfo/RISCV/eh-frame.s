# RUN: llvm-mc -filetype=obj -triple=riscv32 < %s | llvm-dwarfdump -eh-frame - \
# RUN:    | FileCheck --check-prefixes=CHECK,RV32 %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 < %s | llvm-dwarfdump -eh-frame - \
# RUN:    | FileCheck --check-prefixes=CHECK,RV64 %s

func:
  .cfi_startproc
  ret
  .cfi_endproc

# CHECK: 00000000 00000010 00000000 CIE
# CHECK:   Version:               1
# CHECK:   Augmentation:          "zR"
# CHECK:   Code alignment factor: 1

# TODO: gas uses -4 for the data alignment factor for both RV32 and RV64. They
# do so on the basis that on RV64F, F registers may only be 4 bytes
# (DWARF2_CIE_DATA_ALIGNMENT).

# RV32:    Data alignment factor: -4
# RV64:    Data alignment factor: -8

# CHECK:   Return address column: 1

# Check the pointer encoding for address pointers used in FDE. This is set by
# FDECFIEncoding and should be DW_EH_PE_pcrel | DW_EH_PE_sdata4 (0x1b).

# CHECK:   Augmentation data:     1B
# CHECK:   DW_CFA_def_cfa: reg2 +0
#
# CHECK: 00000014 00000010 00000018 FDE cie=00000000 pc=00000000...00000004
# CHECK:   DW_CFA_nop:
# CHECK:   DW_CFA_nop:
# CHECK:   DW_CFA_nop:
