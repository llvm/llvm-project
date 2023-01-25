# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-objdump -d - | FileCheck --check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
# RUN: llvm-mc -triple riscv32 < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=FIXUP %s

.long foo

call foo
# RELOC: R_RISCV_CALL_PLT foo 0x0
# INSTR: auipc ra, 0
# INSTR: jalr  ra
# FIXUP: fixup A - offset: 0, value: foo, kind: fixup_riscv_call
call bar
# RELOC: R_RISCV_CALL_PLT bar 0x0
# INSTR: auipc ra, 0
# INSTR: jalr  ra
# FIXUP: fixup A - offset: 0, value: bar, kind: fixup_riscv_call

# Ensure that calls to functions whose names coincide with register names work.

call zero
# RELOC: R_RISCV_CALL_PLT zero 0x0
# INSTR: auipc ra, 0
# INSTR: jalr  ra
# FIXUP: fixup A - offset: 0, value: zero, kind: fixup_riscv_call

call f1
# RELOC: R_RISCV_CALL_PLT f1 0x0
# INSTR: auipc ra, 0
# INSTR: jalr  ra
# FIXUP: fixup A - offset: 0, value: f1, kind: fixup_riscv_call

call ra
# RELOC: R_RISCV_CALL_PLT ra 0x0
# INSTR: auipc ra, 0
# INSTR: jalr  ra
# FIXUP: fixup A - offset: 0, value: ra, kind: fixup_riscv_call

call mstatus
# RELOC: R_RISCV_CALL_PLT mstatus 0x0
# INSTR: auipc ra, 0
# INSTR: jalr  ra
# FIXUP: fixup A - offset: 0, value: mstatus, kind: fixup_riscv_call

# Ensure that calls to procedure linkage table symbols work.

call foo@plt
# RELOC: R_RISCV_CALL_PLT foo 0x0
# INSTR: auipc ra, 0
# INSTR: jalr  ra
# FIXUP: fixup A - offset: 0, value: foo@plt, kind: fixup_riscv_call_plt

# Ensure that an explicit register operand can be parsed.

call a0, foo
# RELOC: R_RISCV_CALL_PLT foo 0x0
# INSTR: auipc a0, 0
# INSTR: jalr  a0
# FIXUP: fixup A - offset: 0, value: foo, kind: fixup_riscv_call

call a0, foo@plt
# RELOC: R_RISCV_CALL_PLT foo 0x0
# INSTR: auipc a0, 0
# INSTR: jalr  a0
# FIXUP: fixup A - offset: 0, value: foo@plt, kind: fixup_riscv_call_plt
