# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d - | FileCheck --check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s

.long foo

jump foo, x31
# RELOC: R_RISCV_CALL_PLT foo 0x0
# INSTR: auipc t6, 0
# INSTR: jr  t6

# Ensure that jumps to symbols whose names coincide with register names work.

jump zero, x1
# RELOC: R_RISCV_CALL_PLT zero 0x0
# INSTR: auipc ra, 0
# INSTR: ret

1:
jump 1b, x31
# INSTR: auipc t6, 0
# INSTR: jr  t6
