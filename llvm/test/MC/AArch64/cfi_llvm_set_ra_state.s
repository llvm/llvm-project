# RUN: llvm-mc -triple aarch64-pc-linux-gnu %s -filetype=asm -o - | FileCheck %s --check-prefix=ASM

# RUN: llvm-mc -triple aarch64-pc-linux-gnu %s -filetype=obj -o %t.o
# RUN: llvm-dwarfdump --eh-frame %t.o | FileCheck %s --check-prefix=DWARF
# RUN: llvm-readobj --hex-dump=.eh_frame %t.o | FileCheck %s --check-prefix=HEX

  .cfi_startproc
  .cfi_llvm_set_ra_state 1, 0
  .cfi_llvm_set_ra_state 2, 0
  .cfi_llvm_set_ra_state 0, -4
  .cfi_endproc

# ASM:      .cfi_startproc
# ASM-NEXT: .cfi_llvm_set_ra_state 1, 0
# ASM-NEXT: .cfi_llvm_set_ra_state 2, 0
# ASM-NEXT: .cfi_llvm_set_ra_state 0, -4
# ASM-NEXT: .cfi_endproc

# DWARF:      DW_CFA_AARCH64_set_ra_state: 1 0
# DWARF-NEXT: DW_CFA_AARCH64_set_ra_state: 2 0
# DWARF-NEXT: DW_CFA_AARCH64_set_ra_state: 0 -4

# Verify the raw encoding:
#
#   opcode 0x2b, then ULEB128(state), then SLEB128(offset).
#
#   state=1, offset= 0  => 2b 01 00
#   state=2, offset= 0  => 2b 02 00
#   state=0, offset=-4  => 2b 00 7c
#
# HEX: 002b0100 2b02002b 007c
