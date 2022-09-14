# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:   | llvm-objdump -d - | FileCheck --check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
# RUN: llvm-mc -triple riscv32 < %s -show-encoding \
# RUN:   | FileCheck -check-prefix=FIXUP %s

# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:   | llvm-objdump -d - | FileCheck --check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
# RUN: llvm-mc -triple riscv64 < %s -show-encoding \
# RUN:   | FileCheck -check-prefix=FIXUP %s

.long foo

tail foo
# RELOC: R_RISCV_CALL_PLT foo 0x0
# INSTR: auipc t1, 0
# INSTR: jr  t1
# FIXUP: fixup A - offset: 0, value: foo, kind:
tail bar
# RELOC: R_RISCV_CALL_PLT bar 0x0
# INSTR: auipc t1, 0
# INSTR: jr  t1
# FIXUP: fixup A - offset: 0, value: bar, kind:

# Ensure that tail calls to functions whose names coincide with register names
# work.

tail zero
# RELOC: R_RISCV_CALL_PLT zero 0x0
# INSTR: auipc t1, 0
# INSTR: jr  t1
# FIXUP: fixup A - offset: 0, value: zero, kind:

tail f1
# RELOC: R_RISCV_CALL_PLT f1 0x0
# INSTR: auipc t1, 0
# INSTR: jr  t1
# FIXUP: fixup A - offset: 0, value: f1, kind:

tail ra
# RELOC: R_RISCV_CALL_PLT ra 0x0
# INSTR: auipc t1, 0
# INSTR: jr  t1
# FIXUP: fixup A - offset: 0, value: ra, kind:

tail foo@plt
# RELOC: R_RISCV_CALL_PLT foo 0x0
# INSTR: auipc t1, 0
# INSTR: jr  t1
# FIXUP: fixup A - offset: 0, value: foo@plt, kind:
