# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d - | FileCheck --check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s

.long foo

call foo
# RELOC: R_RISCV_CALL_PLT foo 0x0
# INSTR: auipc ra, 0
# INSTR: jalr  ra
call bar
# RELOC: R_RISCV_CALL_PLT bar 0x0
# INSTR: auipc ra, 0
# INSTR: jalr  ra

# Ensure that calls to functions whose names coincide with register names work.

call zero
# RELOC: R_RISCV_CALL_PLT zero 0x0
# INSTR: auipc ra, 0
# INSTR: jalr  ra

call f1
# RELOC: R_RISCV_CALL_PLT f1 0x0
# INSTR: auipc ra, 0
# INSTR: jalr  ra

call ra
# RELOC: R_RISCV_CALL_PLT ra 0x0
# INSTR: auipc ra, 0
# INSTR: jalr  ra

call mstatus
# RELOC: R_RISCV_CALL_PLT mstatus 0x0
# INSTR: auipc ra, 0
# INSTR: jalr  ra

# Ensure that calls to procedure linkage table symbols work.

call foo@plt
# RELOC: R_RISCV_CALL_PLT foo 0x0
# INSTR: auipc ra, 0
# INSTR: jalr  ra

# Ensure that an explicit register operand can be parsed.

call a0, foo
# RELOC: R_RISCV_CALL_PLT foo 0x0
# INSTR: auipc a0, 0
# INSTR: jalr  a0

call a0, foo@plt
# RELOC: R_RISCV_CALL_PLT foo 0x0
# INSTR: auipc a0, 0
# INSTR: jalr  a0
