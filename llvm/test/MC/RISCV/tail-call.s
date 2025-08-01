# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:   | llvm-objdump -d - | FileCheck --check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s

# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:   | llvm-objdump -d - | FileCheck --check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s

# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-zicfilp < %s \
# RUN:   | llvm-objdump -d - | FileCheck --check-prefix=INSTR-ZICFILP %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-zicfilp < %s \
# RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s

# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+experimental-zicfilp < %s \
# RUN:   | llvm-objdump -d - | FileCheck --check-prefix=INSTR-ZICFILP %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+experimental-zicfilp < %s \
# RUN:   | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s

.long foo

tail foo
# RELOC: R_RISCV_CALL_PLT foo 0x0
# INSTR: auipc t1, 0
# INSTR: jr  t1
# INSTR-ZICFILP: auipc t2, 0
# INSTR-ZICFILP: jr  t2

tail bar
# RELOC: R_RISCV_CALL_PLT bar 0x0
# INSTR: auipc t1, 0
# INSTR: jr  t1
# INSTR-ZICFILP: auipc t2, 0
# INSTR-ZICFILP: jr  t2

# Ensure that tail calls to functions whose names coincide with register names
# work.

tail zero
# RELOC: R_RISCV_CALL_PLT zero 0x0
# INSTR: auipc t1, 0
# INSTR: jr  t1
# INSTR-ZICFILP: auipc t2, 0
# INSTR-ZICFILP: jr  t2

tail f1
# RELOC: R_RISCV_CALL_PLT f1 0x0
# INSTR: auipc t1, 0
# INSTR: jr  t1
# INSTR-ZICFILP: auipc t2, 0
# INSTR-ZICFILP: jr  t2

tail ra
# RELOC: R_RISCV_CALL_PLT ra 0x0
# INSTR: auipc t1, 0
# INSTR: jr  t1
# INSTR-ZICFILP: auipc t2, 0
# INSTR-ZICFILP: jr  t2

tail foo@plt
# RELOC: R_RISCV_CALL_PLT foo 0x0
# INSTR: auipc t1, 0
# INSTR: jr  t1
# INSTR-ZICFILP: auipc t2, 0
# INSTR-ZICFILP: jr  t2
