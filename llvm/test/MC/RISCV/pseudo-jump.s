# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d - | FileCheck --check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
# RUN: llvm-mc -triple riscv32 < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=FIXUP %s

.long foo

jump foo, x31
# RELOC: R_RISCV_CALL_PLT foo 0x0
# INSTR: auipc t6, 0
# INSTR: jr  t6
# FIXUP: fixup A - offset: 0, value: foo, kind: fixup_riscv_call

# Ensure that jumps to symbols whose names coincide with register names work.

jump zero, x1
# RELOC: R_RISCV_CALL_PLT zero 0x0
# INSTR: auipc ra, 0
# INSTR: ret
# FIXUP: fixup A - offset: 0, value: zero, kind: fixup_riscv_call

1:
jump 1b, x31
# INSTR: auipc t6, 0
# INSTR: jr  t6
# FIXUP: fixup A - offset: 0, value: .Ltmp0, kind: fixup_riscv_call
